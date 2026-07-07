// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

#pragma once

/// @file mg_topology.hpp
/// Host-side cross-block copy tables of one multigrid level, and the
/// derivation of a coarse level's tables from its parent's.
///
/// The fine level's tables arrive from the Python builder (bindings.cpp)
/// as conforming face-ghost fills — one (src interior, dst ghost) pair per
/// (interface node, neighbour) — plus owner→copy broadcast offsets for the
/// duplicated interface nodes. Coarse tables are derived here with NO
/// orientation logic of their own: injection places coarse node I at parent
/// node I ⊙ factor, so the parent entry behind the matching parent ghost
/// already names the twin block and tangential position; dividing the twin's
/// coordinates by ITS block's factors gives the coarse source. Odd interior
/// node counts (the coarsening rule, mg_level.hpp) make that division exact
/// even across reversed/permuted interfaces — parity is preserved — and the
/// division is asserted, which doubles as the interface-conformity check.
///
/// Derivation is per face and deliberately drop-not-throw: a face whose
/// entries cannot be classified (singular neighbourhood, non-conforming
/// remnant) simply produces no coarse entries, and the hierarchy builder
/// then keeps the affected coarse face nodes frozen — correctness never
/// depends on a face being derivable.

#include "mg_transfer.hpp"
#include "structured.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

namespace egg
{

/// Cross-block copy tables of one level, in host form (the buildable inputs
/// of BlockTopologyDevice, structured_halo.hpp). Ghost entries are the
/// conforming face fills only (singular fans live on the fine level and are
/// never coarsened); share entries are owner→copy element (double) offsets.
template <int D> struct HaloTablesHost {
    using Index = std::array<std::size_t, D>;
    std::vector<int> src_block;      ///< (E,) block of each source interior node
    std::vector<Index> src_padded;   ///< (E,) padded index of each source
    std::vector<int> dst_block;      ///< (E,) block of each destination ghost
    std::vector<Index> dst_padded;   ///< (E,) padded index of each destination
    std::vector<std::size_t> share_src_off;  ///< (K,) owner interior real offsets
    std::vector<std::size_t> share_dst_off;  ///< (K,) non-owner copy real offsets

    [[nodiscard]] bool empty() const { return src_block.empty() && share_src_off.empty(); }
};

namespace detail
{

/// (block, PADDED index) of the node with packed node id @p node — the
/// inverse of padded_node_offset / D. Linear over blocks (block counts are
/// small) then a stride divmod.
template <int D>
inline std::pair<std::size_t, typename BlockLayout<D>::Shape>
  locate_node(const BlockLayout<D>& layout, std::size_t node)
{
    const std::size_t off = node * static_cast<std::size_t>(D);
    std::size_t b = 0;
    while (b + 1 < layout.num_blocks() && layout.block_offset(b + 1) <= off) { ++b; }
    std::size_t rem = off - layout.block_offset(b);
    const auto strides = layout.node_strides(b);
    typename BlockLayout<D>::Shape padded {};
    for (std::size_t k = 0; k < D; ++k) {
        padded[k] = rem / strides[k];
        rem %= strides[k];
    }
    return {b, padded};
}

/// Packed node id of a padded index (padded_node_offset in node units).
template <int D>
inline std::size_t padded_node_id(const BlockLayout<D>& layout,
                                  std::size_t b,
                                  const typename BlockLayout<D>::Shape& padded)
{ return layout.padded_node_offset(b, padded) / static_cast<std::size_t>(D); }

/// The single face a ghost padded index sits behind: (axis, side), or
/// axis = −1 when the index is not a face ghost (corner/edge ghost slot —
/// the Python builder never emits those as ghost-fill destinations).
template <int D>
inline std::pair<int, int> ghost_face(const typename BlockLayout<D>::Shape& padded,
                                      const typename BlockLayout<D>::Shape& interior)
{
    int axis = -1;
    int side = 0;
    for (int k = 0; k < D; ++k) {
        const auto ku = static_cast<std::size_t>(k);
        const bool lo = padded[ku] == 0;
        const bool hi = padded[ku] == interior[ku] + 1;
        if (lo || hi) {
            if (axis >= 0) { return {-1, 0}; }
            axis = k;
            side = hi ? 1 : 0;
        }
    }
    return {axis, side};
}

}  // namespace detail

/// Derive the child level's ghost + share tables from the parent's. See the
/// file header for the method; @p factor is the child's per-block per-axis
/// coarsening factor vs the parent. Faces or entries that cannot be
/// classified/coarsened are dropped (their coarse nodes stay frozen).
template <int D>
HaloTablesHost<D> coarsen_halo_tables(const HaloTablesHost<D>& parent,
                                      const BlockLayout<D>& parent_layout,
                                      const BlockLayout<D>& child_layout,
                                      const std::vector<CoarsenFactor<D>>& factor)
{
    using Shape = typename BlockLayout<D>::Shape;
    HaloTablesHost<D> child;
    if (parent.empty()) { return child; }

    // dst ghost node id → parent entry, for the tangential lookups below.
    std::unordered_map<std::size_t, std::size_t> by_dst;
    by_dst.reserve(parent.src_block.size());
    for (std::size_t e = 0; e < parent.src_block.size(); ++e) {
        by_dst.emplace(detail::padded_node_id<D>(parent_layout,
                                                 static_cast<std::size_t>(parent.dst_block[e]),
                                                 parent.dst_padded[e]),
                       e);
    }

    // Classify each (receiving face, source block) SUBGROUP once: which of the
    // source block's axes is the interface normal (the coordinate constant
    // across the subgroup at distance 1 from a source-block face) and on which
    // side. Per-source-block classification handles "patchwork" contacts where
    // one face receives from several neighbour blocks (e.g. an O-shell face
    // over a lattice of core blocks); ambiguity invalidates only the subgroup.
    struct FaceInfo {
        Shape first {};                    // first entry's src logical
        std::array<bool, D> same {};       // per-axis constancy of src logical
        bool seen = false, valid = true;
        int normal = -1, side = 0;         // filled by the resolve pass
    };
    std::map<std::array<std::size_t, 4>, FaceInfo> faces;  // (dst blk, axis, side, src blk)

    for (std::size_t e = 0; e < parent.src_block.size(); ++e) {
        const auto db = static_cast<std::size_t>(parent.dst_block[e]);
        const auto [axis, side] = detail::ghost_face<D>(parent.dst_padded[e],
                                                        parent_layout.interior_shape(db));
        if (axis < 0) { continue; }
        FaceInfo& fi = faces[{db,
                              static_cast<std::size_t>(axis),
                              static_cast<std::size_t>(side),
                              static_cast<std::size_t>(parent.src_block[e])}];
        Shape logical = parent.src_padded[e];
        for (std::size_t k = 0; k < D; ++k) { logical[k] -= 1; }
        if (!fi.seen) {
            fi.seen = true;
            fi.first = logical;
            fi.same.fill(true);
        } else {
            for (std::size_t k = 0; k < D; ++k) {
                if (logical[k] != fi.first[k]) { fi.same[k] = false; }
            }
        }
    }
    for (auto& [key, fi] : faces) {
        const Shape& sn = parent_layout.interior_shape(key[3]);
        for (int k = 0; k < D; ++k) {
            const auto ku = static_cast<std::size_t>(k);
            if (!fi.same[ku]) { continue; }
            const bool lo = fi.first[ku] == 1;
            const bool hi = fi.first[ku] == sn[ku] - 2;
            if (!lo && !hi) { continue; }
            if (fi.normal >= 0) {  // more than one candidate — ambiguous
                fi.valid = false;
                break;
            }
            fi.normal = k;
            fi.side = hi ? 1 : 0;
        }
        if (fi.normal < 0) { fi.valid = false; }
    }
    std::map<std::array<std::size_t, 3>, bool> face_set;  // faces with any subgroup
    for (const auto& [key, fi] : faces) { face_set[{key[0], key[1], key[2]}] = true; }

    // Emit child ghost entries: for every child face-ghost slot, look up the
    // parent entry behind the child node's parent image, classify it through
    // its own (face, source block) subgroup, and divide the twin's coordinates
    // by the twin block's factors.
    for (const auto& fentry : face_set) {
        const auto& fkey = fentry.first;
        const std::size_t db = fkey[0];
        const auto axis = static_cast<int>(fkey[1]);
        const auto side = static_cast<int>(fkey[2]);
        const Shape cn = child_layout.interior_shape(db);
        const Shape fn = parent_layout.interior_shape(db);

        // Row-major enumeration of the child face's tangential positions.
        std::size_t count = 1;
        for (int k = 0; k < D; ++k) {
            if (k != axis) { count *= cn[static_cast<std::size_t>(k)]; }
        }
        std::array<std::size_t, D> tshape {};
        for (int k = 0; k < D; ++k) {
            tshape[static_cast<std::size_t>(k)] = (k == axis) ? 1 : cn[static_cast<std::size_t>(k)];
        }
        for (std::size_t i = 0; i < count; ++i) {
            const std::array<std::size_t, D> T = delinearize<D>(i, tshape);
            Shape fine_ghost {};
            Shape child_ghost {};
            for (int k = 0; k < D; ++k) {
                const auto ku = static_cast<std::size_t>(k);
                if (k == axis) {
                    fine_ghost[ku] = (side == 0) ? 0 : fn[ku] + 1;
                    child_ghost[ku] = (side == 0) ? 0 : cn[ku] + 1;
                } else {
                    fine_ghost[ku] = (T[ku] * static_cast<std::size_t>(factor[db][k])) + 1;
                    child_ghost[ku] = T[ku] + 1;
                }
            }
            const auto it = by_dst.find(detail::padded_node_id<D>(parent_layout, db, fine_ghost));
            if (it == by_dst.end()) { continue; }
            const std::size_t e = it->second;
            const auto sb = static_cast<std::size_t>(parent.src_block[e]);
            const auto fit = faces.find({db, fkey[1], fkey[2], sb});
            if (fit == faces.end() || !fit->second.valid) { continue; }
            const FaceInfo& fi = fit->second;
            const Shape sn_c = child_layout.interior_shape(sb);
            const Shape sn_f = parent_layout.interior_shape(sb);

            Shape q1 = parent.src_padded[e];
            for (std::size_t k = 0; k < D; ++k) { q1[k] -= 1; }
            if (q1[static_cast<std::size_t>(fi.normal)] !=
                ((fi.side == 0) ? 1 : sn_f[static_cast<std::size_t>(fi.normal)] - 2)) {
                continue;  // entry disagrees with the subgroup classification
            }
            Shape c {};
            bool ok = true;
            for (int k = 0; k < D && ok; ++k) {
                const auto ku = static_cast<std::size_t>(k);
                if (k == fi.normal) {
                    c[ku] = (fi.side == 0) ? 1 : sn_c[ku] - 2;
                    continue;
                }
                const auto f = static_cast<std::size_t>(factor[sb][k]);
                if (q1[ku] % f != 0) {  // parity/conformity violated — drop
                    ok = false;
                    break;
                }
                c[ku] = q1[ku] / f;
            }
            if (!ok) { continue; }
            for (std::size_t k = 0; k < D; ++k) { c[k] += 1; }  // logical → padded
            child.src_block.push_back(static_cast<int>(sb));
            child.src_padded.push_back(c);
            child.dst_block.push_back(static_cast<int>(db));
            child.dst_padded.push_back(child_ghost);
        }
    }

    // Share pairs subset-map: an owner→copy pair survives iff both nodes are
    // parent images of child nodes (every coordinate divisible by the owning
    // block's factor — parity makes owner/copy agree; disagreement means a
    // non-conforming remnant and the pair is dropped).
    const auto to_child = [&](std::size_t elem_off, std::size_t& child_off) {
        const auto [b, padded] = detail::locate_node<D>(parent_layout,
                                                        elem_off / static_cast<std::size_t>(D));
        Shape logical {};
        for (std::size_t k = 0; k < D; ++k) {
            if (padded[k] == 0) { return false; }  // ghost — share is interior-only
            logical[k] = padded[k] - 1;
        }
        Shape c {};
        for (int k = 0; k < D; ++k) {
            const auto ku = static_cast<std::size_t>(k);
            const auto f = static_cast<std::size_t>(factor[b][k]);
            if (logical[ku] % f != 0) { return false; }
            c[ku] = logical[ku] / f;
            if (c[ku] >= child_layout.interior_shape(b)[ku]) { return false; }
        }
        child_off = child_layout.interior_node_offset(b, c);
        return true;
    };
    for (std::size_t p = 0; p < parent.share_src_off.size(); ++p) {
        std::size_t so = 0;
        std::size_t co = 0;
        if (to_child(parent.share_src_off[p], so) && to_child(parent.share_dst_off[p], co)) {
            child.share_src_off.push_back(so);
            child.share_dst_off.push_back(co);
        }
    }
    return child;
}

}  // namespace egg
