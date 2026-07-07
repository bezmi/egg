// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

#pragma once

/// @file mg_level.hpp
/// One level of the FAS geometric-multigrid hierarchy and its host builder.
///
/// A coarse level is deliberately tiny: a coarsened BlockLayout plus per-DOF
/// index arrays. Every coarse free DOF is synth-eligible by construction (its
/// patch is rebuilt in-kernel from block_off/nstride with an identity target,
/// see structured_patch.hpp::patch_eval_synth), so the coarse GroupViewT
/// carries no metric table, no entity SoA, and no partitions — metric_kernel
/// and interior_update_kernel (sweep.hpp) run on it unchanged. Coarse images
/// of boundary-constrained and singular nodes are frozen (Dirichlet at their
/// restricted positions). Coarse images of conforming block-INTERFACE nodes
/// relax too when the guards in the builder hold: each level
/// carries a BlockTopologyDevice derived from its parent's tables
/// (mg_topology.hpp) whose ghost fills + owner→copy broadcasts run as the
/// level's halo hook, so a face DOF's synth patch reads a current ghost
/// plane exactly like the fine smoother's frozen-halo cadence. Face nodes
/// whose neighbourhood is not fully derivable (block edges/corners, singular
/// fans, non-conforming remnants) keep the freeze — correctness never
/// depends on unfreezing.
///
/// Coarsening is factor 2 per axis in logical space where the interior node
/// count is odd and ≥ kMinCoarsenNodes, factor 1 (semi-coarsening) elsewhere;
/// a block with no coarsenable axis is carried unchanged (identity transfers)
/// and the hierarchy stops once no block can halve any axis.

#include "device.hpp"
#include "mg_topology.hpp"
#include "mg_transfer.hpp"
#include "structured.hpp"
#include "structured_halo.hpp"
#include "sweep.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace egg
{

/// Smallest interior node count an axis needs to be worth halving.
inline constexpr std::size_t kMinCoarsenNodes = 5;

/// Per-node relaxability masks of one level, in node units over that level's
/// packed padded store (one byte per node, ghosts zero). `free` marks every
/// relaxable DOF (Free tag — includes block-interface face nodes);
/// `free_interior` the strictly-interior subset (used by the next level's
/// interior-candidate selection; coarse face DOFs are free but NOT
/// free_interior). Level 0's masks come from the host sweep context; each
/// coarse level's masks are derived by the builder.
struct MgMasks {
    std::vector<std::uint8_t> free;
    std::vector<std::uint8_t> free_interior;
};

/// One coarse level: layout + factors (host), packed buffers and the per-DOF
/// index arrays (device, all UsmBuffer-owned). The x/x_swap pair double-buffers
/// the coarse Jacobi smooths exactly like run_block_jacobi's fine pair.
template <int D> struct MgLevel {
    BlockLayout<D> layout;                 ///< coarse packing geometry (host)
    std::vector<CoarsenFactor<D>> factor;  ///< per-block factors vs the parent level
    MgMasks masks;                         ///< host copies (next-level build + tests)
    std::size_t num_nodes = 0;             ///< padded node count of this level
    std::size_t n_free = 0;                ///< relaxed coarse DOFs

    UsmBuffer<real> x, x_swap;          ///< packed positions (double-buffered)
    UsmBuffer<real> rx;                 ///< R x_parent snapshot (correction base)
    UsmBuffer<real> tau;                ///< [num_nodes * D] FAS linear shift
    UsmBuffer<real> grad, hess, e0;     ///< smoother metric scratch
    UsmBuffer<real> corr;               ///< child's prolonged correction (zero-init:
                                        ///< ghost slots are never written and must
                                        ///< stay 0 under masked_axpy)
    UsmBuffer<int> dof_idx;             ///< [n_free] packed coarse node index
    UsmBuffer<int> interior_block;      ///< [n_free] owning block
    UsmBuffer<int> interior_logical;    ///< [n_free * D] logical index
    UsmBuffer<int> block_off, nstride;  ///< node-unit layout (device copies)
    UsmBuffer<std::uint8_t> free_mask;  ///< per coarse node (padded store)
    BlockTopologyDevice<D> topo;        ///< face-ghost fills + owner→copy broadcasts
                                        ///< (empty when no interface DOF relaxes)

    /// Synth-only group view over the free coarse DOFs, consumable by
    /// jacobi_sweep_once: every DOF has interior_block ≥ 0, so the stored-array
    /// patch path (and with it the empty per-sample/partition members) is never
    /// reached; n_free == ndof, so the boundary kernels early-return.
    [[nodiscard]] GroupViewT<D> group() const
    {
        GroupViewT<D> gv;
        gv.ndof = n_free;
        gv.n_free = n_free;
        gv.dof_idx = View1<const int> {dof_idx.data(), n_free};
        gv.P_of = View1<const int> {nullptr, 0};
        gv.sample_offset = View1<const int> {nullptr, 0};
        gv.part_of = View1<const int> {nullptr, 0};
        gv.row_of = View1<const int> {nullptr, 0};
        gv.interior_block = interior_block.data();
        gv.interior_logical = interior_logical.data();
        gv.block_off = block_off.data();
        gv.nstride = nstride.data();
        return gv;
    }
};

namespace detail
{

/// Per-block coarsening factors and coarse interior shapes; false only when
/// NO block halves any axis (the ladder has converged). A block with no
/// halvable axis is CARRIED unchanged (factor 1 everywhere): its coarse image
/// is the same size and its transfers degenerate to identities, so it keeps
/// relaxing at its own resolution while the rest of the mesh coarsens —
/// without this, one stubborn block (e.g. a 3^D H-grid corner, 27 nodes)
/// vetoes multigrid for the whole mesh. Cost note: a LARGE uncoarsenable
/// block (even node counts) is carried at full size through every level, so
/// its coarse sweeps cost fine-sweep prices — prefer odd node counts.
template <int D>
bool coarsen_shapes(const BlockLayout<D>& fine,
                    std::vector<CoarsenFactor<D>>& factor,
                    std::vector<typename BlockLayout<D>::Shape>& coarse_shapes)
{
    const std::size_t nb = fine.num_blocks();
    factor.assign(nb, CoarsenFactor<D> {});
    coarse_shapes.assign(nb, typename BlockLayout<D>::Shape {});
    bool any = false;
    for (std::size_t b = 0; b < nb; ++b) {
        const auto& shape = fine.interior_shape(b);
        for (int k = 0; k < D; ++k) {
            const std::size_t n = shape[static_cast<std::size_t>(k)];
            const bool halve = (n % 2 == 1) && n >= kMinCoarsenNodes;
            factor[b][k] = halve ? 2 : 1;
            coarse_shapes[b][static_cast<std::size_t>(k)] = halve ? ((n + 1) / 2) : n;
            any = any || halve;
        }
    }
    return any;
}

/// Flat node-unit index of an interior node (logical index) of block @p b.
template <int D>
std::size_t node_id(const BlockLayout<D>& layout,
                    std::size_t b,
                    const typename BlockLayout<D>::Shape& logical)
{ return layout.interior_node_offset(b, logical) / D; }

}  // namespace detail

/// Build the coarse levels below a fine level described by @p fine_layout and
/// @p fine_masks (level 0 masks are assembled from the host sweep context by
/// the executor). Returns coarse levels ordered fine→coarsest; empty when no
/// level can be built (plain Jacobi is the fallback). All device state lives
/// on @p q.
///
/// An INTERIOR coarse node (b, I) is free iff its fine image (b, I ⊙ factor)
/// is free-interior AND every fine node in the full-weighting stencil around
/// the image is free — the latter guarantees restrict_full_weight only reads
/// fine gradient slots the metric kernels wrote (internal constrained nodes
/// freeze their coarse neighbourhood).
///
/// An INTERFACE coarse node relaxes too when @p fine_halo is
/// supplied: candidates are the owner copies of shared face nodes surviving
/// the table coarsening (mg_topology.hpp), accepted iff (i) they sit strictly
/// inside a single block face, (ii) the coarse ghost plane behind their full
/// tangential neighbourhood was derivable, and (iii) every parent-level slot
/// their transfers read holds a relaxed DOF's gradient — directly free,
/// broadcast from a free owner, or halo-filled from a free neighbour node.
/// Anything else keeps the Dirichlet freeze.
template <int D>
std::vector<MgLevel<D>> build_mg_hierarchy(sycl::queue q,
                                           const BlockLayout<D>& fine_layout,
                                           const MgMasks& fine_masks,
                                           int max_levels,
                                           const HaloTablesHost<D>& fine_halo = {})
{
    using Shape = typename BlockLayout<D>::Shape;
    std::vector<MgLevel<D>> levels;

    const BlockLayout<D>* parent_layout = &fine_layout;
    const MgMasks* parent_masks = &fine_masks;
    HaloTablesHost<D> parent_tables = fine_halo;

    for (int lvl = 1; lvl < max_levels; ++lvl) {
        std::vector<CoarsenFactor<D>> factor;
        std::vector<Shape> coarse_shapes;
        if (!detail::coarsen_shapes<D>(*parent_layout, factor, coarse_shapes)) { break; }

        MgLevel<D> level;
        level.layout = BlockLayout<D> {coarse_shapes};
        level.factor = std::move(factor);
        level.num_nodes = level.layout.total_reals() / D;

        const std::size_t nb = level.layout.num_blocks();
        level.masks.free.assign(level.num_nodes, 0);
        level.masks.free_interior.assign(level.num_nodes, 0);

        std::vector<int> dof_idx, iblock, ilogical, block_off, nstride;
        for (std::size_t b = 0; b < nb; ++b) {
            block_off.push_back(static_cast<int>(level.layout.block_offset(b) / D));
            const Shape nstr = level.layout.node_strides(b);
            for (int k = 0; k < D; ++k) {
                nstride.push_back(static_cast<int>(nstr[static_cast<std::size_t>(k)] / D));
            }

            const Shape cshape = level.layout.interior_shape(b);
            std::size_t count = 1;
            for (int k = 0; k < D; ++k) { count *= cshape[static_cast<std::size_t>(k)]; }
            const std::array<std::size_t, D> cs = cshape;

            for (std::size_t i = 0; i < count; ++i) {
                const std::array<std::size_t, D> I = delinearize<D>(i, cs);
                Shape F {};
                for (int k = 0; k < D; ++k) {
                    F[static_cast<std::size_t>(k)] =
                      I[static_cast<std::size_t>(k)] * static_cast<std::size_t>(level.factor[b][k]);
                }
                if (parent_masks->free_interior[detail::node_id<D>(*parent_layout, b, F)] == 0) {
                    continue;
                }
                // Stencil-of-free check: every parent node the full-weighting
                // stencil reads must be a relaxed DOF (its gradient slot is
                // written each sweep). Offsets on factor-1 axes are identity.
                bool ok = true;
                constexpr int kOffsets = [] {
                    int p = 1;
                    for (int k = 0; k < D; ++k) { p *= 3; }
                    return p;
                }();
                for (int e = 0; e < kOffsets && ok; ++e) {
                    int rem = e;
                    Shape P = F;
                    bool valid = true;
                    for (int k = 0; k < D; ++k) {
                        const int o = (rem % 3) - 1;
                        rem /= 3;
                        if (level.factor[b][k] == 1) {
                            if (o != 0) {
                                valid = false;
                                break;
                            }
                        } else {
                            // F_k ≥ 2 for a free-interior image, so no underflow.
                            P[static_cast<std::size_t>(k)] =
                              F[static_cast<std::size_t>(k)] + static_cast<std::size_t>(o);
                        }
                    }
                    if (!valid) { continue; }
                    ok = parent_masks->free[detail::node_id<D>(*parent_layout, b, P)] != 0;
                }
                if (!ok) { continue; }

                const std::size_t cid = detail::node_id<D>(level.layout, b, I);
                level.masks.free[cid] = 1;
                level.masks.free_interior[cid] = 1;
                dof_idx.push_back(static_cast<int>(cid));
                iblock.push_back(static_cast<int>(b));
                for (int k = 0; k < D; ++k) {
                    ilogical.push_back(static_cast<int>(I[static_cast<std::size_t>(k)]));
                }
            }
        }

        // ---- Coarse interface relaxation --------------------------------------
        // Derive this level's cross-block tables from the parent's, then admit
        // the guarded owner face nodes as free DOFs (free, NOT free_interior).
        HaloTablesHost<D> child_tables =
          coarsen_halo_tables<D>(parent_tables, *parent_layout, level.layout, level.factor);
        if (!child_tables.share_src_off.empty()) {
            // Parent-level lookups: ghost dst → entry, copy → owner, and the
            // validity of one parent gradient slot (free, or a broadcast copy
            // of a free owner).
            std::unordered_map<std::size_t, std::size_t> parent_ghost;
            parent_ghost.reserve(parent_tables.src_block.size());
            for (std::size_t e = 0; e < parent_tables.src_block.size(); ++e) {
                parent_ghost.emplace(
                  detail::padded_node_id<D>(*parent_layout,
                                            static_cast<std::size_t>(parent_tables.dst_block[e]),
                                            parent_tables.dst_padded[e]),
                  e);
            }
            std::unordered_map<std::size_t, std::size_t> copy_owner;
            copy_owner.reserve(parent_tables.share_src_off.size());
            for (std::size_t p = 0; p < parent_tables.share_src_off.size(); ++p) {
                copy_owner.emplace(parent_tables.share_dst_off[p] / D,
                                   parent_tables.share_src_off[p] / D);
            }
            const auto read_ok = [&](std::size_t nid) {
                if (parent_masks->free[nid] != 0) { return true; }
                const auto it = copy_owner.find(nid);
                return it != copy_owner.end() && parent_masks->free[it->second] != 0;
            };
            std::unordered_set<std::size_t> child_ghost;
            child_ghost.reserve(child_tables.src_block.size());
            for (std::size_t e = 0; e < child_tables.src_block.size(); ++e) {
                child_ghost.insert(
                  detail::padded_node_id<D>(level.layout,
                                            static_cast<std::size_t>(child_tables.dst_block[e]),
                                            child_tables.dst_padded[e]));
            }

            constexpr int kOffsets = [] {
                int p = 1;
                for (int k = 0; k < D; ++k) { p *= 3; }
                return p;
            }();
            std::unordered_set<std::size_t> seen;
            for (const std::size_t owner_off : child_tables.share_src_off) {
                const std::size_t node = owner_off / D;
                if (!seen.insert(node).second) { continue; }
                const auto [b, padded] = detail::locate_node<D>(level.layout, node);
                const Shape cn = level.layout.interior_shape(b);
                const Shape fn = parent_layout->interior_shape(b);
                Shape I {};
                bool interior = true;
                for (std::size_t k = 0; k < D; ++k) {
                    if (padded[k] == 0) { interior = false; }
                    I[k] = padded[k] - 1;
                }
                if (!interior) { continue; }

                // (i) exactly one face axis, strictly inside that face.
                int axis = -1;
                int side = 0;
                bool ok = true;
                for (int k = 0; k < D && ok; ++k) {
                    const auto ku = static_cast<std::size_t>(k);
                    const bool lo = I[ku] == 0;
                    const bool hi = I[ku] == cn[ku] - 1;
                    if (lo || hi) {
                        if (axis >= 0) { ok = false; }
                        axis = k;
                        side = hi ? 1 : 0;
                    }
                }
                if (!ok || axis < 0) { continue; }
                for (int k = 0; k < D && ok; ++k) {
                    const auto ku = static_cast<std::size_t>(k);
                    if (k != axis && (I[ku] < 1 || I[ku] > cn[ku] - 2)) { ok = false; }
                }
                if (!ok) { continue; }

                // (ii) + (iii): walk the 3^D neighbourhood of the fine image.
                // In-block slots must be readable gradients; the slot behind
                // the face must be halo-filled from a free neighbour node on
                // BOTH levels (parent for the restriction, child for the
                // patch's ghost plane).
                for (int e = 0; e < kOffsets && ok; ++e) {
                    int rem = e;
                    std::array<long, D> N {};
                    std::array<long, D> C {};
                    for (int k = 0; k < D; ++k) {
                        const auto ku = static_cast<std::size_t>(k);
                        const int o = (rem % 3) - 1;
                        rem /= 3;
                        N[ku] = static_cast<long>(I[ku] * static_cast<std::size_t>(level.factor[b][k])) + o;
                        C[ku] = static_cast<long>(I[ku]) + o;
                    }
                    // Only the face axis can leave the block (tangentials are
                    // ≥ 1 from every other face by guard (i)).
                    if (N[static_cast<std::size_t>(axis)] < 0 ||
                        N[static_cast<std::size_t>(axis)] >= static_cast<long>(fn[static_cast<std::size_t>(axis)])) {
                        Shape gp {};
                        Shape cgp {};
                        for (int k = 0; k < D; ++k) {
                            const auto ku = static_cast<std::size_t>(k);
                            if (k == axis) {
                                gp[ku] = (side == 0) ? 0 : fn[ku] + 1;
                                cgp[ku] = (side == 0) ? 0 : cn[ku] + 1;
                            } else {
                                gp[ku] = static_cast<std::size_t>(N[ku]) + 1;
                                cgp[ku] = static_cast<std::size_t>(C[ku]) + 1;
                            }
                        }
                        const auto it =
                          parent_ghost.find(detail::padded_node_id<D>(*parent_layout, b, gp));
                        ok = it != parent_ghost.end() &&
                             parent_masks->free[detail::padded_node_id<D>(
                               *parent_layout,
                               static_cast<std::size_t>(parent_tables.src_block[it->second]),
                               parent_tables.src_padded[it->second])] != 0 &&
                             child_ghost.contains(
                               detail::padded_node_id<D>(level.layout, b, cgp));
                    } else {
                        Shape P {};
                        for (int k = 0; k < D; ++k) {
                            P[static_cast<std::size_t>(k)] =
                              static_cast<std::size_t>(N[static_cast<std::size_t>(k)]);
                        }
                        ok = read_ok(detail::node_id<D>(*parent_layout, b, P));
                    }
                }
                if (!ok) { continue; }

                level.masks.free[node] = 1;  // free, but NOT free_interior
                dof_idx.push_back(static_cast<int>(node));
                iblock.push_back(static_cast<int>(b));
                for (std::size_t k = 0; k < D; ++k) {
                    ilogical.push_back(static_cast<int>(I[k]));
                }
            }
        }

        level.n_free = dof_idx.size();
        if (level.n_free == 0) { break; }  // nothing to relax — deeper levels are pointless

        const std::size_t nd = level.num_nodes * static_cast<std::size_t>(D);
        level.x = UsmBuffer<real> {q, nd};
        level.x_swap = UsmBuffer<real> {q, nd};
        level.rx = UsmBuffer<real> {q, nd};
        level.tau = UsmBuffer<real> {q, nd};
        level.grad = UsmBuffer<real> {q, nd};
        level.hess = UsmBuffer<real> {q, level.num_nodes * static_cast<std::size_t>(D * D)};
        level.corr = UsmBuffer<real> {q, std::vector<real>(nd, 0.0_r)};
        level.e0 = UsmBuffer<real> {q, level.num_nodes};
        level.dof_idx = UsmBuffer<int> {q, dof_idx};
        level.interior_block = UsmBuffer<int> {q, iblock};
        level.interior_logical = UsmBuffer<int> {q, ilogical};
        level.block_off = UsmBuffer<int> {q, block_off};
        level.nstride = UsmBuffer<int> {q, nstride};
        level.free_mask = UsmBuffer<std::uint8_t> {q, level.masks.free};
        if (!child_tables.empty()) {
            // Needed whenever any interface DOF relaxes on this level (its
            // patch reads this level's ghost plane) — and, one level down,
            // whenever the CHILD has interface DOFs (their restriction reads
            // this level's gradient ghosts through the same tables).
            level.topo = BlockTopologyDevice<D> {q,
                                                 level.layout,
                                                 child_tables.src_block,
                                                 child_tables.src_padded,
                                                 child_tables.dst_block,
                                                 child_tables.dst_padded,
                                                 {},
                                                 {},
                                                 child_tables.share_src_off,
                                                 child_tables.share_dst_off};
        }

        levels.push_back(std::move(level));
        parent_layout = &levels.back().layout;
        parent_masks = &levels.back().masks;
        parent_tables = std::move(child_tables);
    }
    return levels;
}

}  // namespace egg
