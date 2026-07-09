// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

/// @file sweep.hpp
/// Device-resident block-Jacobi barrier/untangle sweep.
#pragma once

#include "curvature.hpp"
#include "device.hpp"
#include "geometry.hpp"
#include "metric.hpp"
#include "patch.hpp"
#include "structured_patch.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

namespace egg
{

// The kernels build the per-corner stencil through PatchViewT<D> and take the
// D×D Newton step; the A→T→detA math is single-sourced in
// patch.hpp::assemble_vecT. Everything is generic over D.

/// Backtracking budget of every per-DOF line search (α halvings before the
/// step is dropped); also the FasOptions::max_backtracks default.
inline constexpr int kMaxBacktracks = 10;

/// Per-partition typed SoA host data (one per entity type in the group).
/// Populated by `unpack_context` from the Python `entities` sub-dicts; the
/// SweepDeviceContextT ctor uploads to USM and builds PartitionView's
/// soa_view + seg views. Type-erased at host level (interpretation is per-E
/// in `load`); `seg` carries CSR payloads, empty for fixed-size types.
struct SoAHostRecord {
    /// One segmented (CSR) field: flat data + offset table.
    struct SegmentedField {
        std::vector<real> data;
        std::vector<int> off;  ///< Length count+1, off[0]==0.
    };
    EntityTag tag;
    int k_fields;
    std::vector<real> records;  // [count * k_fields] packed
    std::size_t count;
    std::vector<SegmentedField> seg;  // CSR fields (empty for fixed-size)
    std::vector<int> dof_local;       // [count] group-local DOF indices of this partition
};

template <int D> struct SweepGroupHostT {
    std::size_t ndof;                // DOFs in this group
    std::size_t total_samples;       // Σ P_of[d]
    std::vector<int> gc;             // [total_samples]
    std::vector<int> gn[D];          // [total_samples] per axis (was gn0, gn1)
    std::vector<real> s[D];          // [total_samples] per axis (was s0, s1)
    std::vector<real> W_inv;         // [total_samples * dim::wInv(D)], or one row if w_uniform
    bool w_uniform = false;          // W_inv is a single shared row (uniform/identity target)
    std::vector<real> weight;        // [total_samples] energy weight, or one row if wt_uniform
    bool wt_uniform = false;         // weight is a single shared value (e.g. all 1.0)
    std::vector<int> role;           // [total_samples]
    std::vector<real> J;             // [total_samples * dim::jSize(D)]
    std::vector<int> dof_idx;        // [ndof]
    std::vector<int> P_of;           // [ndof]  patch size per DOF
    std::vector<int> sample_offset;  // [ndof]  exclusive prefix sum of P_of
    std::vector<SoAHostRecord> soa;  // typed per-EntityTag SoA stores (the entity data)

    // Structured interior fast path: for a DOF whose entire metric patch lies
    // inside one block's interior, interior_block[d] names that block and
    // interior_logical[d*D + k] its logical index, so a consumer can synthesize
    // the patch indices from the block layout instead of reading this DOF's
    // stored gc/gn/role/s. interior_block[d] == -1 means "read the stored
    // arrays" (boundary/interface DOFs); both stay empty on the unstructured
    // path, where the fast path never applies.
    std::vector<int> interior_block;    // [ndof] or empty; -1 = use stored arrays
    std::vector<int> interior_logical;  // [ndof * D] or empty

    // Block-interface C2 curvature term (2D): per-DOF fixed-K window table. Each
    // free DOF lists the curvature windows it participates in — four node ids per
    // window (global; remapped to structured owner slots), the DOF's slot 0..3,
    // and the window weight. curv_k == 0 means no curvature term. Both consumers
    // (metric_kernel accumulation, interior_update_kernel trial energy) read it.
    int curv_k = 0;
    std::vector<int> curv_nodes;    // [ndof * curv_k * 4]
    std::vector<int> curv_slot;     // [ndof * curv_k]  (-1 = pad)
    std::vector<real> curv_weight;  // [ndof * curv_k]
};

template <int D> struct EnergyStencilHostT {
    std::size_t num_samples;
    std::vector<int> gc;  // [num_samples]
    std::vector<int> gn[D];
    std::vector<real> s[D];
    std::vector<real> W_inv;   // [num_samples * dim::wInv(D)]
    std::vector<real> weight;  // [num_samples] energy weight, or one row if uniform
};

template <int D> struct SweepContextHostT {
    std::vector<SweepGroupHostT<D>> groups;  // one group of all free DOFs
    EnergyStencilHostT<D> energy_stencil;
    std::size_t num_nodes;
    std::vector<real> X;  // [num_nodes * D] initial positions

    // Structured block layout in NODE units, for synthesizing interior DOF
    // patches on the device (empty on the unstructured path): block_off[b] is
    // block b's first node index, nstride[b*D + k] the padded node stride on
    // axis k. Pairs with SweepGroupHostT::interior_block/interior_logical.
    std::vector<int> block_off;  // [num_blocks] or empty
    std::vector<int> nstride;    // [num_blocks * D] or empty
};

/// One EntityTag partition of a group: a contiguous list of group-local DOF
/// indices sharing an entity type. Trivially copyable (captured by value into
/// the kernel); dof_list points at device USM. The kernel calls
/// tie_view(soa_view, seg) + load to build the concrete entity — no per-DOF
/// std::visit, no const void* erasure. seg views are null for fixed-size
/// types.
struct PartitionView {
    /// Max segmented slots (matches EntitySoA<E>::View).
    static constexpr int kMaxSeg = kMaxSoASeg;
    EntityTag tag;                                 ///< Entity type of this partition.
    std::size_t ndof;                              ///< Number of DOFs in this partition.
    const int* dof_list;                           ///< Group-local DOF indices, length @c ndof.
    SoAView<const real> soa_view {nullptr, 0, 0};  ///< Packed records (ndof, kFields).
    SegmentedView<real> seg[kMaxSeg] {};           ///< CSR fields (null for fixed-size).
};

template <int D> struct GroupViewT {
    std::size_t ndof;
    std::size_t n_free = 0;  // DOFs are Free-first reordered; [0, n_free) are Free
    // Per-occurrence (length total_samples, sliced per DOF by sample_offset):
    View1<const int> sample_id, role;
    // Shared deduplicated metric table (base pointers, indexed by sample_id[p],
    // shared by every group/the merged view — sized num table samples, NOT
    // total_samples). gc/gn/s/W_inv below are these bases; J is recomputed.
    View1<const int> gc;                            // [n_table]
    View1<const int> gn[D];                         // [n_table]
    View1<const std::int8_t> s[D];                  // [n_table] per-axis sign ±1
    View1<const real> W_inv;                        // [n_table * wInv], or one row if uniform
    int w_stride = dim::wInv(D);                    // 0 when W_inv is a uniform shared row
    View1<const real> weight;                       // [n_table] weight, or one row if uniform
    int wt_stride = 1;                              // 0 when weight is a uniform shared value
    View1<const int> dof_idx, P_of, sample_offset;  // [ndof]
    View1<const int> part_of, row_of;               // [ndof] DOF -> (partition, row)
    const PartitionView* partitions = nullptr;      // [num_partitions]
    std::size_t num_partitions = 0;                 // per-EntityTag splits

    // Structured interior fast path (null on the unstructured path): interior_block
    // [d] (or -1) names the block whose layout synthesizes DOF d's patch, and
    // interior_logical[d*D + k] its logical index; block_off/nstride are the shared
    // node-unit layout the synthesis indexes. See interior() and patch_eval_synth.
    const int* interior_block = nullptr;    // [ndof]
    const int* interior_logical = nullptr;  // [ndof * D]
    const int* block_off = nullptr;         // [num_blocks] (shared)
    const int* nstride = nullptr;           // [num_blocks * D] (shared)

    // Block-interface C2 curvature term (2D; null when absent): per-DOF fixed-K
    // window table (see SweepGroupHostT). curv_nodes holds structured node ids.
    int curv_k = 0;
    const int* curv_nodes = nullptr;    // [ndof * curv_k * 4]
    const int* curv_slot = nullptr;     // [ndof * curv_k]
    const real* curv_weight = nullptr;  // [ndof * curv_k]

    // True when DOF d's patch can be synthesized from the block layout (its whole
    // patch lies inside one block's interior); fills the block + logical index.
    bool interior(std::size_t d, int& block, int (&logical)[D]) const
    {
        if (interior_block == nullptr || interior_block[d] < 0) { return false; }
        block = interior_block[d];
        for (int k = 0; k < D; ++k) {
            logical[k] = interior_logical[(d * D) + static_cast<std::size_t>(k)];
        }
        return true;
    }

    // PatchViewT over DOF d's contiguous ragged slice
    // [sample_offset[d], sample_offset[d] + P_of[d]).
    PatchViewT<D> patch(std::size_t d) const
    {
        const auto off = static_cast<std::size_t>(sample_offset[d]);
        PatchViewT<D> pv;
        pv.P = P_of[d];
        // Per-occurrence arrays slice to this DOF; the metric table bases are
        // shared (global), indexed by sample_id[p] inside patch_eval.
        pv.sample_id = sample_id.data_handle() + off;
        pv.role = role.data_handle() + off;
        pv.gc = gc.data_handle();
        for (int k = 0; k < D; ++k) {
            pv.gn[k] = gn[k].data_handle();
            pv.s[k] = s[k].data_handle();
        }
        pv.W_inv = W_inv.data_handle();
        pv.w_stride = w_stride;
        pv.weight = weight.data_handle();
        pv.wt_stride = wt_stride;
        return pv;
    }

    // A view restricted to the per-DOF sub-range [begin, begin+count). Only the
    // per-DOF arrays (dof_idx, P_of, sample_offset, part_of, row_of) are sliced;
    // the per-sample arrays (gc/gn/s/W_inv/role/J) and the partition list are
    // shared unchanged — sample_offset still indexes the full per-sample tables.
    // Used by the Plan #1 interior/boundary split to launch one kernel over each
    // contiguous DOF class of a Free-first-reordered view.
    GroupViewT<D> dof_subrange(std::size_t begin, std::size_t count) const
    {
        GroupViewT<D> sub = *this;
        sub.ndof = count;
        sub.dof_idx = View1<const int> {dof_idx.data_handle() + begin, count};
        sub.P_of = View1<const int> {P_of.data_handle() + begin, count};
        sub.sample_offset = View1<const int> {sample_offset.data_handle() + begin, count};
        sub.part_of = View1<const int> {part_of.data_handle() + begin, count};
        sub.row_of = View1<const int> {row_of.data_handle() + begin, count};
        if (interior_block != nullptr) {
            sub.interior_block = interior_block + begin;
            sub.interior_logical = interior_logical + (begin * D);
        }
        if (curv_nodes != nullptr) {
            sub.curv_nodes = curv_nodes + (begin * curv_k * 4);
            sub.curv_slot = curv_slot + (begin * curv_k);
            sub.curv_weight = curv_weight + (begin * curv_k);
        }
        return sub;
    }
};

template <int D> struct StencilViewT {
    std::size_t num_samples;
    View1<const int> gc;
    View1<const int> gn[D];
    View1<const real> s[D];
    View1<const real> W_inv;  // [num_samples * wInv], or one row when uniform
    // W_inv row stride: dim::wInv(D), or 0 when every sample shares one row (a
    // uniform target), in which case W_inv holds a single dim::wInv(D)-wide row.
    int w_stride = dim::wInv(D);
    View1<const real> weight;  // [num_samples] energy weight, or one row when uniform
    int wt_stride = 1;         // 0 when every sample shares one weight value
};

// Base pointers of the shared deduplicated metric table, indexed by a patch
// occurrence's sample_id. One table is shared by every group and the merged view.
// J is not stored — its role block is recomputed in-kernel from s + W_inv.
template <int D> struct MetricTableViewT {
    View1<const int> gc;            // [n_table]
    View1<const int> gn[D];         // [n_table]
    View1<const std::int8_t> s[D];  // [n_table] per-axis sign ±1, widened on read
    View1<const real> W_inv;        // [n_table * dim::wInv(D)], or one row when uniform
    // W_inv row stride: dim::wInv(D), or 0 when every sample shares one row (a
    // uniform target), in which case W_inv holds a single dim::wInv(D)-wide row.
    int w_stride = dim::wInv(D);
    View1<const real> weight;  // [n_table] energy weight, or one row when uniform
    int wt_stride = 1;         // 0 when every sample shares one weight value
};

template <int D> class SweepDeviceContextT
{
  public:
    // Per-node warm-start parameter cache for iterative projections: stride
    // (D-1), one slot per global node, initialised non-finite (cold) so the
    // first sweep seeds via the coarse grid and subsequent sweeps warm-start.
    static constexpr std::size_t kSeedStride = (D > 1) ? (D - 1) : 1;

    SweepDeviceContextT(sycl::queue q, const SweepContextHostT<D>& host) :
        q_(q), num_nodes_(host.num_nodes), X_(q, host.X),
        seeds_(
          q,
          std::vector<real>(host.num_nodes * kSeedStride, std::numeric_limits<real>::quiet_NaN()))
    {
        // Deduplicate the per-sample payload (gc/gn/s/W_inv/J) across every group
        // into one shared table; each group keeps only a per-occurrence sample_id.
        const std::vector<std::vector<int>> group_sid = build_metric_table(host);

        groups_.reserve(host.groups.size());
        for (std::size_t gi = 0; gi < host.groups.size(); ++gi) {
            const auto& g = host.groups[gi];
            DeviceGroup dg;
            dg.ndof = g.ndof;
            dg.total_samples = g.total_samples;
            dg.sample_id = {q, group_sid[gi]};
            dg.role = {q, g.role};

            // Each SoAHostRecord is one EntityTag partition: dof_local is the
            // group-local DOF list, records/seg the already-typed SoA (built in
            // Python by group_entities_by_type or the C++ golden helper). The
            // upload is a direct copy — no per-DOF blob decode, no tag scan.
            dg.partitions.reserve(g.soa.size());
            for (const auto& rec : g.soa) {
                DevicePartition dp;
                dp.tag = rec.tag;
                dp.dof_list = {q, rec.dof_local};
                dispatch_entity_type<D>(rec.tag, [&]<class E>() {
                    if constexpr (HasEntitySoA<E>) {
                        using SoA = EntitySoA<E>;
                        dp.soa_records = UsmBuffer<real> {q, rec.records};
                        for (int j = 0; j < SoA::kSeg; ++j) {
                            if (j < static_cast<int>(rec.seg.size())) {
                                dp.seg_data[j] = UsmBuffer<real> {q, rec.seg[j].data};
                                dp.seg_off[j] = UsmBuffer<int> {q, rec.seg[j].off};
                            }
                        }
                    }
                });
                dg.partitions.push_back(std::move(dp));
            }

            // Invert the per-partition dof_local lists into per-DOF maps so the
            // single per-group kernel can find each DOF's entity: part_of[d]
            // indexes dg.partitions, row_of[d] is the DOF's row within that
            // partition's SoA records (g.soa[p].dof_local[r] == d).
            std::vector<int> part_of(g.ndof, -1), row_of(g.ndof, -1);
            for (std::size_t p = 0; p < g.soa.size(); ++p) {
                const auto& dl = g.soa[p].dof_local;
                for (std::size_t r = 0; r < dl.size(); ++r) {
                    part_of[static_cast<std::size_t>(dl[r])] = static_cast<int>(p);
                    row_of[static_cast<std::size_t>(dl[r])] = static_cast<int>(r);
                }
            }

            // Free-first reorder: per-DOF arrays permuted so Free (interior)
            // DOFs precede non-Free (boundary) ones — two contiguous
            // sub-ranges of one view. Per-sample arrays untouched
            // (sample_offset still indexes them). Free iff the owning
            // partition's tag is Free.
            const auto is_free = [&](std::size_t d) {
                const int p = part_of[d];
                return p >= 0 && g.soa[static_cast<std::size_t>(p)].tag == EntityTag::Free;
            };
            std::vector<int> perm;
            perm.reserve(g.ndof);
            for (std::size_t d = 0; d < g.ndof; ++d) {
                if (is_free(d)) { perm.push_back(static_cast<int>(d)); }
            }
            dg.n_free = perm.size();
            for (std::size_t d = 0; d < g.ndof; ++d) {
                if (!is_free(d)) { perm.push_back(static_cast<int>(d)); }
            }
            const auto gather = [&](const std::vector<int>& src) {
                std::vector<int> dst(src.size());
                for (std::size_t i = 0; i < perm.size(); ++i) {
                    dst[i] = src[static_cast<std::size_t>(perm[i])];
                }
                return dst;
            };
            dg.dof_idx = {q, gather(g.dof_idx)};
            dg.P_of = {q, gather(g.P_of)};
            dg.sample_offset = {q, gather(g.sample_offset)};
            dg.part_of = {q, gather(part_of)};
            dg.row_of = {q, gather(row_of)};

            // Interior fast-path descriptors (structured only): reorder in lockstep
            // with the per-DOF arrays. interior_logical is D-wide per DOF.
            if (!g.interior_block.empty()) {
                dg.interior_block = {q, gather(g.interior_block)};
                std::vector<int> ilog(g.ndof * static_cast<std::size_t>(D));
                for (std::size_t i = 0; i < perm.size(); ++i) {
                    const auto src = static_cast<std::size_t>(perm[i]);
                    for (int k = 0; k < D; ++k) {
                        ilog[(i * D) + static_cast<std::size_t>(k)] =
                          g.interior_logical[(src * D) + static_cast<std::size_t>(k)];
                    }
                }
                dg.interior_logical = {q, ilog};
            }

            // Curvature window table: K-wide per DOF, reordered in lockstep.
            if (g.curv_k > 0) {
                const int K = g.curv_k;
                dg.curv_k = K;
                std::vector<int> cn(g.ndof * static_cast<std::size_t>(K) * 4);
                std::vector<int> cs(g.ndof * static_cast<std::size_t>(K));
                std::vector<real> cw(g.ndof * static_cast<std::size_t>(K));
                for (std::size_t i = 0; i < perm.size(); ++i) {
                    const auto src = static_cast<std::size_t>(perm[i]);
                    for (int j = 0; j < K; ++j) {
                        const std::size_t dst_e = (i * static_cast<std::size_t>(K)) + j;
                        const std::size_t src_e = (src * static_cast<std::size_t>(K)) + j;
                        cs[dst_e] = g.curv_slot[src_e];
                        cw[dst_e] = g.curv_weight[src_e];
                        for (int m = 0; m < 4; ++m) {
                            cn[(dst_e * 4) + m] = g.curv_nodes[(src_e * 4) + m];
                        }
                    }
                }
                dg.curv_nodes = {q, cn};
                dg.curv_slot = {q, cs};
                dg.curv_weight = {q, cw};
            }

            groups_.push_back(std::move(dg));
        }

        // Upload the shared node-unit block layout once (empty unless structured).
        if (!host.block_off.empty()) {
            block_off_ = {q, host.block_off};
            nstride_ = {q, host.nstride};
        }
        const auto& es = host.energy_stencil;
        stencil_.num_samples = es.num_samples;
        stencil_.gc = {q, es.gc};
        for (int k = 0; k < D; ++k) {
            stencil_.gn[k] = {q, es.gn[k]};
            stencil_.s[k] = {q, es.s[k]};
        }
        stencil_.W_inv = {q, es.W_inv};
        // Energy weight per stencil sample: default to one shared 1.0 when the
        // wire omits it, so the reduction reads it with stride 0.
        stencil_.weight = {q, es.weight.empty() ? std::vector<real> {1.0_r} : es.weight};

        // Build the per-group PartitionView arrays (host copies), retained for
        // the merged block-Jacobi view (which uploads its own concatenated copy).
        host_pvs_.reserve(groups_.size());
        for (std::size_t gi = 0; gi < groups_.size(); ++gi) {
            const auto& dg = groups_[gi];
            std::vector<PartitionView> pvs;
            pvs.reserve(dg.partitions.size());
            for (const auto& dp : dg.partitions) {
                const std::size_t ndof = dp.dof_list.size();
                const real* soa_ptr = dp.soa_records.data();
                std::size_t kfields = 0;
                dispatch_entity_type<D>(dp.tag, [&]<class E>() {
                    if constexpr (HasEntitySoA<E>) {
                        kfields = static_cast<std::size_t>(EntitySoA<E>::kFields);
                    }
                });
                PartitionView pv {.tag = dp.tag,
                                  .ndof = ndof,
                                  .dof_list = dp.dof_list.data(),
                                  .soa_view = SoAView<const real> {soa_ptr, ndof, kfields}};
                // Fill segmented views from uploaded USM (null for fixed-size).
                for (int j = 0; j < PartitionView::kMaxSeg; ++j) {
                    pv.seg[j] = SegmentedView<real> {dp.seg_data[j].data(), dp.seg_off[j].data()};
                }
                pvs.push_back(pv);
            }
            host_pvs_.push_back(std::move(pvs));  // for the merged Jacobi view
        }
        stencil_view_ = stencil_.view();
    }

    const StencilViewT<D>& stencil_view() const { return stencil_view_; }
    real* X() const { return X_.data(); }
    std::size_t x_size() const { return X_.size(); }
    std::size_t num_nodes() const { return num_nodes_; }
    /// Per-node warm-start seed cache (stride @ref kSeedStride), persisted
    /// across sweeps; nullptr disables warm starts.
    real* seeds() { return seeds_.data(); }

    /// Single GroupViewT over all free DOFs for the block-Jacobi launch (under
    /// frozen halos every free DOF updates from the same snapshot). Lazily
    /// built and cached; the Free-first reorder + boundary bucketing here
    /// drive the interior/boundary split.
    const GroupViewT<D>& merged_group_view()
    {
        if (!merged_built_) {
            build_merged_view();
            merged_built_ = true;
        }
        return merged_view_;
    }

    /// One monomorphic boundary launch descriptor: a single entity-type
    /// partition's list of Free-first-reordered DOF positions into the merged
    /// view. Launching over positions[0..count) with the entity type fixed at
    /// launch collapses dispatch_entity_type to one arm (no control-flow
    /// union).
    struct BoundaryLaunch {
        EntityTag tag;         ///< Entity type of this partition (drives monomorphization).
        PartitionView pv;      ///< Entity SoA view (records + seg) for this partition.
        std::size_t count;     ///< Number of boundary DOFs in this partition.
        const int* positions;  ///< Device USM: reordered positions into the merged view.
    };

    /// Per-partition boundary launch descriptors, built lazily with the merged
    /// view; one entry per boundary partition present. Their positions exactly
    /// tile [merged.n_free, merged.ndof).
    const std::vector<BoundaryLaunch>& boundary_launches()
    {
        merged_group_view();  // ensure built
        return boundary_launches_;
    }

    void upload_X(const std::vector<real>& host_X)
    {
        X_.upload(host_X);
        // Positions reset → invalidate the warm-start cache (re-seed cold).
        seeds_.upload(std::vector<real>(seeds_.size(), std::numeric_limits<real>::quiet_NaN()));
    }
    void download_X(real* host_X) { X_.download(host_X); }

  private:
    struct DevicePartition {
        EntityTag tag;
        UsmBuffer<int> dof_list;      // group-local DOF indices, length = ndof of partition
        UsmBuffer<real> soa_records;  // packed SoA records, length = ndof * kFields (0 if none)
        UsmBuffer<real> seg_data[kMaxSoASeg];  // CSR data per segmented slot (empty if kSeg==0)
        UsmBuffer<int> seg_off[kMaxSoASeg];    // CSR offsets per segmented slot (empty if kSeg==0)
    };

    struct DeviceGroup {
        std::size_t ndof;
        std::size_t n_free = 0;  // count of Free DOFs (the [0, n_free) prefix)
        std::size_t total_samples;
        // Per-occurrence only (the bulk gc/gn/s/W_inv/J payload is deduplicated
        // into the context-level shared MetricTable, referenced by sample_id).
        UsmBuffer<int> sample_id, role;
        UsmBuffer<int> dof_idx, P_of, sample_offset;
        UsmBuffer<int> part_of, row_of;                   // [ndof] DOF -> (partition, row)
        UsmBuffer<int> interior_block, interior_logical;  // [ndof], [ndof*D]; empty if unstructured
        int curv_k = 0;                                   // curvature windows per DOF (0 = none)
        UsmBuffer<int> curv_nodes, curv_slot;             // [ndof*curv_k*4], [ndof*curv_k]
        UsmBuffer<real> curv_weight;                      // [ndof*curv_k]
        std::vector<DevicePartition> partitions;

        GroupViewT<D> view(const PartitionView* pvs,
                           std::size_t np,
                           const MetricTableViewT<D>& tbl,
                           const int* block_off,
                           const int* nstride) const
        {
            const std::size_t n = total_samples;
            GroupViewT<D> gv;
            gv.ndof = ndof;
            gv.n_free = n_free;
            gv.sample_id = View1<const int> {sample_id.data(), n};
            gv.role = View1<const int> {role.data(), n};
            gv.gc = tbl.gc;
            for (int k = 0; k < D; ++k) {
                gv.gn[k] = tbl.gn[k];
                gv.s[k] = tbl.s[k];
            }
            gv.W_inv = tbl.W_inv;
            gv.w_stride = tbl.w_stride;
            gv.weight = tbl.weight;
            gv.wt_stride = tbl.wt_stride;
            gv.dof_idx = View1<const int> {dof_idx.data(), ndof};
            gv.P_of = View1<const int> {P_of.data(), ndof};
            gv.sample_offset = View1<const int> {sample_offset.data(), ndof};
            gv.part_of = View1<const int> {part_of.data(), ndof};
            gv.row_of = View1<const int> {row_of.data(), ndof};
            gv.partitions = pvs;
            gv.num_partitions = np;
            if (interior_block.size() > 0) {
                gv.interior_block = interior_block.data();
                gv.interior_logical = interior_logical.data();
                gv.block_off = block_off;
                gv.nstride = nstride;
            }
            if (curv_nodes.size() > 0) {
                gv.curv_k = curv_k;
                gv.curv_nodes = curv_nodes.data();
                gv.curv_slot = curv_slot.data();
                gv.curv_weight = curv_weight.data();
            }
            return gv;
        }
    };

    struct DeviceStencil {
        std::size_t num_samples;
        UsmBuffer<int> gc;
        UsmBuffer<int> gn[D];
        UsmBuffer<real> s[D];
        UsmBuffer<real> W_inv;
        UsmBuffer<real> weight;

        StencilViewT<D> view() const
        {
            StencilViewT<D> sv;
            sv.num_samples = num_samples;
            sv.gc = View1<const int> {gc.data(), num_samples};
            for (int k = 0; k < D; ++k) {
                sv.gn[k] = View1<const int> {gn[k].data(), num_samples};
                sv.s[k] = View1<const real> {s[k].data(), num_samples};
            }
            // One shared row (size == wInv) => stride 0; otherwise per-sample rows.
            sv.W_inv = View1<const real> {W_inv.data(), W_inv.size()};
            sv.w_stride = (W_inv.size() == dim::wInv(D)) ? 0 : dim::wInv(D);
            // One shared value => stride 0; otherwise one weight per sample.
            sv.weight = View1<const real> {weight.data(), weight.size()};
            sv.wt_stride = (weight.size() == 1) ? 0 : 1;
            return sv;
        }
    };

    /// Concatenate every group's per-occurrence arrays into one GroupViewT
    /// (cached in the m_* buffers + merged_view_). Only the small per-occurrence
    /// sample_id/role and the per-DOF index arrays are joined — the bulk metric
    /// payload lives once in the shared table_ (referenced unchanged via
    /// sample_id, which is already a global table index). The per-DOF index arrays
    /// are rebuilt with the running sample/partition offsets; the partition list is
    /// the per-group PartitionView arrays concatenated (entity SoA pointers unchanged).
    void build_merged_view()
    {
        const std::size_t ng = groups_.size();
        std::size_t total_ndof = 0, total_samples = 0, total_parts = 0;
        for (std::size_t gi = 0; gi < ng; ++gi) {
            total_ndof += groups_[gi].ndof;
            total_samples += groups_[gi].total_samples;
            total_parts += host_pvs_[gi].size();
        }

        m_sample_id_ = UsmBuffer<int> {q_, total_samples};
        m_role_ = UsmBuffer<int> {q_, total_samples};

        std::vector<int> dof_idx(total_ndof), P_of(total_ndof), sample_offset(total_ndof);
        std::vector<int> part_of(total_ndof), row_of(total_ndof);
        // Interior fast-path descriptors carried through the merge (structured only;
        // -1 / 0 elsewhere, so the merged view's interior() simply never fires).
        const bool structured = !groups_.empty() && groups_[0].interior_block.size() > 0;
        std::vector<int> interior_block(total_ndof, -1);
        std::vector<int> interior_logical(total_ndof * static_cast<std::size_t>(D), 0);
        // Curvature window table carried through the merge (0-wide when absent).
        const bool has_curv = !groups_.empty() && groups_[0].curv_k > 0;
        const int curv_k = has_curv ? groups_[0].curv_k : 0;
        const std::size_t ck = static_cast<std::size_t>(curv_k);
        std::vector<int> curv_nodes(total_ndof * ck * 4, -1);
        std::vector<int> curv_slot(total_ndof * ck, -1);
        std::vector<real> curv_weight(total_ndof * ck, 0.0_r);
        std::vector<PartitionView> merged_pvs;
        merged_pvs.reserve(total_parts);

        std::size_t sbase = 0, dbase = 0, pbase = 0;
        for (std::size_t gi = 0; gi < ng; ++gi) {
            auto& dg = groups_[gi];
            const std::size_t gs = dg.total_samples, gd = dg.ndof;
            // sample_id already indexes the shared table_, so concatenation keeps
            // each occurrence's table index intact (no per-group rebasing).
            q_.memcpy(m_sample_id_.data() + sbase, dg.sample_id.data(), gs * sizeof(int));
            q_.memcpy(m_role_.data() + sbase, dg.role.data(), gs * sizeof(int));

            std::vector<int> g_dof_idx(gd), g_P_of(gd), g_so(gd), g_part(gd), g_row(gd);
            dg.dof_idx.download(g_dof_idx.data());
            dg.P_of.download(g_P_of.data());
            dg.sample_offset.download(g_so.data());
            dg.part_of.download(g_part.data());
            dg.row_of.download(g_row.data());
            for (std::size_t r = 0; r < gd; ++r) {
                dof_idx[dbase + r] = g_dof_idx[r];
                P_of[dbase + r] = g_P_of[r];
                sample_offset[dbase + r] = g_so[r] + static_cast<int>(sbase);
                part_of[dbase + r] = g_part[r] + static_cast<int>(pbase);
                row_of[dbase + r] = g_row[r];
            }
            if (structured) {
                std::vector<int> g_ib(gd), g_il(gd * static_cast<std::size_t>(D));
                dg.interior_block.download(g_ib.data());
                dg.interior_logical.download(g_il.data());
                for (std::size_t r = 0; r < gd; ++r) {
                    interior_block[dbase + r] = g_ib[r];
                    for (int k = 0; k < D; ++k) {
                        interior_logical[((dbase + r) * D) + static_cast<std::size_t>(k)] =
                          g_il[(r * D) + static_cast<std::size_t>(k)];
                    }
                }
            }
            if (has_curv && dg.curv_k > 0) {
                std::vector<int> g_cn(gd * ck * 4), g_cs(gd * ck);
                std::vector<real> g_cw(gd * ck);
                dg.curv_nodes.download(g_cn.data());
                dg.curv_slot.download(g_cs.data());
                dg.curv_weight.download(g_cw.data());
                for (std::size_t r = 0; r < gd; ++r) {
                    for (std::size_t j = 0; j < ck; ++j) {
                        const std::size_t d_e = ((dbase + r) * ck) + j;
                        const std::size_t s_e = (r * ck) + j;
                        curv_slot[d_e] = g_cs[s_e];
                        curv_weight[d_e] = g_cw[s_e];
                        for (int m = 0; m < 4; ++m) {
                            curv_nodes[(d_e * 4) + m] = g_cn[(s_e * 4) + m];
                        }
                    }
                }
            }
            for (const auto& pv : host_pvs_[gi]) { merged_pvs.push_back(pv); }

            sbase += gs;
            dbase += gd;
            pbase += host_pvs_[gi].size();
        }
        q_.wait();

        // Global Free-first reorder across merged DOFs (each group is already
        // Free-first internally; concatenation interleaves the classes). The
        // launch is split at n_free into interior and boundary kernels.
        const auto merged_is_free = [&](std::size_t i) {
            const int p = part_of[i];
            return p >= 0 && merged_pvs[static_cast<std::size_t>(p)].tag == EntityTag::Free;
        };
        std::vector<int> perm;
        perm.reserve(total_ndof);
        for (std::size_t i = 0; i < total_ndof; ++i) {
            if (merged_is_free(i)) { perm.push_back(static_cast<int>(i)); }
        }
        const std::size_t merged_n_free = perm.size();
        for (std::size_t i = 0; i < total_ndof; ++i) {
            if (!merged_is_free(i)) { perm.push_back(static_cast<int>(i)); }
        }
        const auto gather = [&](const std::vector<int>& src) {
            std::vector<int> dst(src.size());
            for (std::size_t i = 0; i < perm.size(); ++i) {
                dst[i] = src[static_cast<std::size_t>(perm[i])];
            }
            return dst;
        };

        const std::vector<int> r_part_of = gather(part_of);
        m_dof_idx_ = UsmBuffer<int> {q_, gather(dof_idx)};
        m_P_of_ = UsmBuffer<int> {q_, gather(P_of)};
        m_sample_offset_ = UsmBuffer<int> {q_, gather(sample_offset)};
        m_part_of_ = UsmBuffer<int> {q_, r_part_of};
        m_row_of_ = UsmBuffer<int> {q_, gather(row_of)};
        m_partitions_ = UsmBuffer<PartitionView> {q_, merged_pvs};
        if (structured) {
            m_interior_block_ = UsmBuffer<int> {q_, gather(interior_block)};
            std::vector<int> ril(interior_logical.size());
            for (std::size_t i = 0; i < perm.size(); ++i) {
                const auto src = static_cast<std::size_t>(perm[i]);
                for (int k = 0; k < D; ++k) {
                    ril[(i * D) + static_cast<std::size_t>(k)] =
                      interior_logical[(src * D) + static_cast<std::size_t>(k)];
                }
            }
            m_interior_logical_ = UsmBuffer<int> {q_, ril};
        }
        if (has_curv) {
            m_curv_k_ = curv_k;
            std::vector<int> rcn(total_ndof * ck * 4), rcs(total_ndof * ck);
            std::vector<real> rcw(total_ndof * ck);
            for (std::size_t i = 0; i < perm.size(); ++i) {
                const auto src = static_cast<std::size_t>(perm[i]);
                for (std::size_t j = 0; j < ck; ++j) {
                    const std::size_t d_e = (i * ck) + j;
                    const std::size_t s_e = (src * ck) + j;
                    rcs[d_e] = curv_slot[s_e];
                    rcw[d_e] = curv_weight[s_e];
                    for (int m = 0; m < 4; ++m) { rcn[(d_e * 4) + m] = curv_nodes[(s_e * 4) + m]; }
                }
            }
            m_curv_nodes_ = UsmBuffer<int> {q_, rcn};
            m_curv_slot_ = UsmBuffer<int> {q_, rcs};
            m_curv_weight_ = UsmBuffer<real> {q_, rcw};
        }

        // Per-partition reordered-position lists for the boundary tail: bucket
        // every boundary position by merged partition; each non-empty bucket is
        // one monomorphic launch. The interior prefix stays one entity-agnostic
        // launch. Reserve up front so the position UsmBuffers never reallocate
        // (BoundaryLaunch::positions holds raw pointers into them).
        std::vector<std::vector<int>> buckets(total_parts);
        for (std::size_t i = merged_n_free; i < total_ndof; ++i) {
            buckets[static_cast<std::size_t>(r_part_of[i])].push_back(static_cast<int>(i));
        }
        std::size_t n_bnd_parts = 0;
        for (const auto& b : buckets) {
            if (!b.empty()) { ++n_bnd_parts; }
        }
        boundary_launches_.clear();
        boundary_launches_.reserve(n_bnd_parts);
        m_part_positions_.clear();
        m_part_positions_.reserve(n_bnd_parts);
        std::size_t bnd_total = 0;
        for (std::size_t p = 0; p < total_parts; ++p) {
            if (buckets[p].empty()) { continue; }
            // Boundary positions must belong to non-Free partitions (Free DOFs
            // are exactly the [0, merged_n_free) prefix).
            assert(merged_pvs[p].tag != EntityTag::Free);
            bnd_total += buckets[p].size();
            m_part_positions_.emplace_back(q_, buckets[p]);
            boundary_launches_.push_back(
              BoundaryLaunch {.tag = merged_pvs[p].tag,
                              .pv = merged_pvs[p],
                              .count = buckets[p].size(),
                              .positions = m_part_positions_.back().data()});
        }
        // The boundary partitions must exactly tile the boundary tail.
        assert(bnd_total == total_ndof - merged_n_free);

        GroupViewT<D> gv;
        gv.ndof = total_ndof;
        gv.n_free = merged_n_free;
        gv.sample_id = View1<const int> {m_sample_id_.data(), total_samples};
        gv.role = View1<const int> {m_role_.data(), total_samples};
        // Bulk payload is the shared table — same bases as every per-group view.
        const MetricTableViewT<D> tbl = table_view();
        gv.gc = tbl.gc;
        for (int k = 0; k < D; ++k) {
            gv.gn[k] = tbl.gn[k];
            gv.s[k] = tbl.s[k];
        }
        gv.W_inv = tbl.W_inv;
        gv.w_stride = tbl.w_stride;
        gv.weight = tbl.weight;
        gv.wt_stride = tbl.wt_stride;
        gv.dof_idx = View1<const int> {m_dof_idx_.data(), total_ndof};
        gv.P_of = View1<const int> {m_P_of_.data(), total_ndof};
        gv.sample_offset = View1<const int> {m_sample_offset_.data(), total_ndof};
        gv.part_of = View1<const int> {m_part_of_.data(), total_ndof};
        gv.row_of = View1<const int> {m_row_of_.data(), total_ndof};
        gv.partitions = m_partitions_.data();
        gv.num_partitions = total_parts;
        if (structured) {
            gv.interior_block = m_interior_block_.data();
            gv.interior_logical = m_interior_logical_.data();
            gv.block_off = block_off_.data();
            gv.nstride = nstride_.data();
        }
        if (has_curv) {
            gv.curv_k = m_curv_k_;
            gv.curv_nodes = m_curv_nodes_.data();
            gv.curv_slot = m_curv_slot_.data();
            gv.curv_weight = m_curv_weight_.data();
        }
        merged_view_ = gv;

        // The merged view now owns its own concatenated per-occurrence arrays
        // (m_sample_id_/m_role_) and per-DOF index arrays. The per-group
        // `groups_[*]` copies they were built from are dead duplicates from here on;
        // free them. NOT freed: the shared `table_` (live payload for the merged
        // view), `partitions` (entity SoA, aliased by the merged PartitionViews),
        // and `stencil_`/`stencil_view_` (read by the energy reduction).
        for (auto& dg : groups_) {
            dg.sample_id = UsmBuffer<int> {};
            dg.role = UsmBuffer<int> {};
            dg.dof_idx = UsmBuffer<int> {};
            dg.P_of = UsmBuffer<int> {};
            dg.sample_offset = UsmBuffer<int> {};
            dg.part_of = UsmBuffer<int> {};
            dg.row_of = UsmBuffer<int> {};
            dg.interior_block = UsmBuffer<int> {};
            dg.interior_logical = UsmBuffer<int> {};
        }
    }

    sycl::queue q_;
    std::size_t num_nodes_;
    UsmBuffer<real> X_;
    UsmBuffer<real> seeds_;  // [num_nodes * kSeedStride] warm-start param cache
    // Shared node-unit block layout for interior patch synthesis (empty on the
    // unstructured path). Referenced by every group view + the merged view.
    UsmBuffer<int> block_off_, nstride_;
    std::vector<DeviceGroup> groups_;
    DeviceStencil stencil_;
    StencilViewT<D> stencil_view_ {};

    // Shared deduplicated metric table (built once in the ctor): every group view
    // and the merged view reference these bases via a per-occurrence sample_id, so
    // the bulk gc/gn/s/W_inv payload is stored once rather than per occurrence.
    // J is not stored — its role block is recomputed in-kernel from s + W_inv.
    struct MetricTable {
        std::size_t n = 0;
        UsmBuffer<int> gc;
        UsmBuffer<int> gn[D];
        UsmBuffer<std::int8_t> s[D];  // per-axis sign ±1 (was real; widened on read)
        UsmBuffer<real> W_inv;
        bool w_uniform = false;  // every sample shares one W_inv row → store one row
        UsmBuffer<real> weight;  // [n] energy weight, or one row when uniform
        bool wt_uniform = false;
    } table_;

    MetricTableViewT<D> table_view() const
    {
        MetricTableViewT<D> t;
        t.gc = View1<const int> {table_.gc.data(), table_.n};
        for (int k = 0; k < D; ++k) {
            t.gn[k] = View1<const int> {table_.gn[k].data(), table_.n};
            t.s[k] = View1<const std::int8_t> {table_.s[k].data(), table_.n};
        }
        const std::size_t wInv = dim::wInv(D);
        t.w_stride = table_.w_uniform ? 0 : static_cast<int>(wInv);
        t.W_inv =
          View1<const real> {table_.W_inv.data(), table_.w_uniform ? wInv : (table_.n * wInv)};
        t.wt_stride = table_.wt_uniform ? 0 : 1;
        t.weight = View1<const real> {table_.weight.data(), table_.wt_uniform ? 1 : table_.n};
        return t;
    }

    // Build the shared metric table from the per-group host payloads, returning
    // each group's per-occurrence sample_id (index into the table). Identical
    // (gc,gn,s,W_inv) occurrences — the same physical (cell,corner) sample shared
    // by a cell's corner DOFs — fold to one entry. The host J payload is ignored:
    // the role-selected block is recomputed in-kernel from s + W_inv (role_Jb).
    // The energy stencil is a separate, already-deduplicated table, left untouched.
    std::vector<std::vector<int>> build_metric_table(const SweepContextHostT<D>& host)
    {
        constexpr std::size_t wInv = dim::wInv(D);
        std::vector<int> t_gc;
        std::vector<int> t_gn[D];
        std::vector<std::int8_t>
          t_s[D];  // s is ±1; store as int8 (key still hashes the real bytes)
        std::vector<real> t_W;
        std::vector<real> t_wt;  // per-table-sample energy weight (parallel to t_W)
        std::vector<std::vector<int>> sid(host.groups.size());

        // Dedup key: the raw bytes of one occurrence's (gc, gn[], s[], W_inv).
        // Equal keys mean an exact byte match, which for the shared (cell,corner)
        // samples holds bit-for-bit; distinct cells that happen to match also fold
        // harmlessly (same payload → same read).
        std::unordered_map<std::string, int> seen;
        std::size_t total = 0;
        for (const auto& g : host.groups) { total += g.total_samples; }
        seen.reserve((total / dim::corners(D)) + 1);
        std::string key;

        const auto emit = [&](const SweepGroupHostT<D>& g, std::size_t p) {
            key.clear();
            const auto put = [&](const void* ptr, std::size_t n) {
                key.append(static_cast<const char*>(ptr), n);
            };
            put(&g.gc[p], sizeof(int));
            for (int k = 0; k < D; ++k) { put(&g.gn[k][p], sizeof(int)); }
            for (int k = 0; k < D; ++k) { put(&g.s[k][p], sizeof(real)); }
            // A uniform group carries one shared W_inv row; read row 0 for every
            // sample. The deduped table then collapses to a single row (below).
            const std::size_t wrow = g.w_uniform ? 0 : (p * wInv);
            put(&g.W_inv[wrow], wInv * sizeof(real));
            // Weight is part of the dedup identity: two otherwise-identical samples
            // with different weights (e.g. a base cell and its interface-orthogonality
            // twin) must not fold together.
            const real wv = g.weight.empty() ? 1.0_r : g.weight[g.wt_uniform ? 0 : p];
            put(&wv, sizeof(real));

            const auto [it, fresh] = seen.try_emplace(key, static_cast<int>(t_gc.size()));
            if (!fresh) { return it->second; }

            t_gc.push_back(g.gc[p]);
            for (int k = 0; k < D; ++k) {
                t_gn[k].push_back(g.gn[k][p]);
                t_s[k].push_back(static_cast<std::int8_t>(g.s[k][p]));
            }
            for (std::size_t e = 0; e < wInv; ++e) { t_W.push_back(g.W_inv[wrow + e]); }
            t_wt.push_back(wv);
            return it->second;
        };

        for (std::size_t gi = 0; gi < host.groups.size(); ++gi) {
            const auto& g = host.groups[gi];
            sid[gi].resize(g.total_samples);
            for (std::size_t p = 0; p < g.total_samples; ++p) { sid[gi][p] = emit(g, p); }
        }

        // Detect a uniform W_inv table (every row byte-identical to the first —
        // the common case of a fixed/identity target). When uniform, store a
        // single row and read it with stride 0; W_inv is the largest per-sample
        // field, so this drops the bulk of the table for those workloads.
        bool uniform = !t_W.empty();
        for (std::size_t r = 1; r < t_gc.size() && uniform; ++r) {
            for (std::size_t e = 0; e < wInv; ++e) {
                if (t_W[(r * wInv) + e] != t_W[e]) {
                    uniform = false;
                    break;
                }
            }
        }
        table_.w_uniform = uniform;
        if (uniform) { t_W.resize(wInv); }

        // Collapse an all-equal weight column to one shared value (the common
        // no-interface case, all 1.0), read with stride 0.
        bool wt_uniform = !t_wt.empty();
        for (std::size_t r = 1; r < t_wt.size() && wt_uniform; ++r) {
            if (t_wt[r] != t_wt[0]) { wt_uniform = false; }
        }
        table_.wt_uniform = wt_uniform;
        if (wt_uniform) { t_wt.resize(1); }

        table_.n = t_gc.size();
        table_.gc = {q_, t_gc};
        for (int k = 0; k < D; ++k) {
            table_.gn[k] = {q_, t_gn[k]};
            table_.s[k] = {q_, t_s[k]};
        }
        table_.W_inv = {q_, t_W};
        table_.weight = {q_, t_wt};
        return sid;
    }

    // Merged single-launch view for block-Jacobi (lazily built, then cached).
    std::vector<std::vector<PartitionView>> host_pvs_;  // per-group PartitionView (host copies)
    bool merged_built_ = false;
    GroupViewT<D> merged_view_ {};
    UsmBuffer<int> m_sample_id_, m_role_;
    UsmBuffer<int> m_dof_idx_, m_P_of_, m_sample_offset_, m_part_of_, m_row_of_;
    UsmBuffer<int> m_interior_block_, m_interior_logical_;  // structured only; empty otherwise
    int m_curv_k_ = 0;                                      // curvature windows per DOF (0 = none)
    UsmBuffer<int> m_curv_nodes_, m_curv_slot_;             // curvature table; empty otherwise
    UsmBuffer<real> m_curv_weight_;
    UsmBuffer<PartitionView> m_partitions_;

    // Per-partition boundary launch descriptors (§4.7 / S1). m_part_positions_
    // owns the device position arrays; boundary_launches_ holds raw pointers into
    // them (kept stable by the reserve in build_merged_view).
    std::vector<UsmBuffer<int>> m_part_positions_;
    std::vector<BoundaryLaunch> boundary_launches_;
};

/// Project @p p onto @p ent, warm-starting from this DOF's parameter cache
/// when the entity supports it (the iterative B-spline curve/surface).
/// @p seed_slot points at the (D-1)-real per-node seed (non-finite = cold);
/// on return it holds the converged foot parameter so the next sweep skips
/// the coarse-grid seed. Closed-form entities project cold (seed untouched).
template <int D, bool Warm, class E>
inline PtN<D> project_with_seed(const E& ent, const PtN<D>& p, real* seed_slot)
{
    constexpr int K = E::tdim;
    if constexpr (requires(Param<K> s) { ent.project_seeded(p, s, true); }) {
        Param<K> seed {};
        const bool has = (seed_slot != nullptr) && sycl::isfinite(seed_slot[0]);
        if (has) {
            for (int a = 0; a < K; ++a) { seed[a] = seed_slot[a]; }
        }
        const PtN<D> out = ent.template project_seeded<Warm>(p, seed, has);
        if (seed_slot != nullptr) {
            for (int a = 0; a < K; ++a) { seed_slot[a] = seed[a]; }
        }
        return out;
    } else {
        return ent.project(p);
    }
}

/// Tangent-reduced Newton step warm-starting the tangent basis from the seed
/// cache when the entity supports it (cf. @ref project_with_seed). Closed-form
/// / Line3 entities fall back to the cold @ref newton_delta; the interior
/// (k==D) case never reaches here.
template <int D, bool Warm, class E>
inline VecN<D> newton_delta_with_seed(
  const VecN<D>& g, const MatN<D>& H, const PtN<D>& pos, const E& ent, real* seed_slot)
{
    constexpr int K = E::tdim;
    if constexpr (requires(Param<K> s) { ent.tangent_basis_seeded(pos, s, true); }) {
        Param<K> seed {};
        const bool has = (seed_slot != nullptr) && sycl::isfinite(seed_slot[0]);
        if (has) {
            for (int a = 0; a < K; ++a) { seed[a] = seed_slot[a]; }
        }
        const std::array<PtN<D>, K> B = ent.template tangent_basis_seeded<Warm>(pos, seed, has);
        if (seed_slot != nullptr) {
            for (int a = 0; a < K; ++a) { seed_slot[a] = seed[a]; }
        }
        return newton_step_from_basis<D, K>(g, H, B);
    } else {
        return newton_delta<D>(g, H, pos, ent);
    }
}

/// Cold boundary sweep kernel (block-Jacobi, SoA-backed): one launch over the
/// constrained (non-Free) DOFs, projecting each trial onto its entity. The
/// heavy body (patch eval, backtracking, energy/min-det) is entity-agnostic
/// and compiled once; the entity-specific Newton delta + per-trial projection
/// are isolated behind @ref with_entity — a cheap tag switch running a small
/// callable, so entity variants are never all inlined (no std::visit
/// occupancy collapse). Double-buffered: reads use @p X (frozen snapshot),
/// the scatter writes @p X_out (null = in place), race-free since distinct
/// DOFs are distinct nodes. @p omega scales the Newton step BEFORE the
/// projected line search (each trial is projected onto the entity), so
/// damping never pulls a constrained DOF off its geometry — a post-hoc
/// world-space blend would. @p seeds is the optional per-node warm-start
/// cache (stride D−1, global node id; null = cold projection), persisted by
/// the owning context so iterative projections converge in 1–2 Newton steps.
/// Interior DOFs are relaxed by the fissioned metric/interior kernels; the
/// warm boundary path is @ref sweep_boundary_paramls_kernel.
template <int D, class Obj>
inline void sweep_kernel(sycl::queue& q,
                         const GroupViewT<D>& g,
                         real* X,
                         Obj objective,
                         real* X_out = nullptr,
                         real omega = 1.0_r,
                         real* seeds = nullptr)
{
    if (g.ndof == 0) { return; }
    constexpr std::size_t kSeedStride = (D > 1) ? (D - 1) : 1;
    const PartitionView* parts = g.partitions;
    real* out = X_out ? X_out : X;
    q.parallel_for(sycl::range<1>(g.ndof), [=](sycl::id<1> idx) {
        const std::size_t d = idx[0];
        const PatchViewT<D> pv = g.patch(d);
        const int dof = g.dof_idx[d];

        // 1. patch_eval → (grad, hess, e0).
        const PatchResultT<D> r = patch_eval<D>(pv, X, objective);
        const PtN<D> pos = load_pt<D>(X, dof);

        // 2. Tangent-reduced Newton step via the scoped tag switch (the only place
        //    the Newton math touches entity geometry). The DOF's entity lives in
        //    partition part_of[d] at SoA row row_of[d]; warm-start the tangent from
        //    this DOF's seed cache (the same slot the line-search projection uses).
        VecN<D> delta {};
        {
            const PartitionView& part = parts[g.part_of[d]];
            const auto row = static_cast<std::size_t>(g.row_of[d]);
            real* seed_slot = (seeds != nullptr)
                                ? (seeds + (static_cast<std::size_t>(dof) * kSeedStride))
                                : nullptr;
            with_entity<D>(part.tag, part.soa_view, part.seg, row, [&](const auto& ent) {
                delta = newton_delta_with_seed<D, false>(r.grad, r.hess, pos, ent, seed_slot);
            });
        }
        // Jacobi damping on the step itself: trials stay projected, so the
        // accepted point is exactly what gets stored.
        for (int k = 0; k < D; ++k) { delta[k] *= omega; }

        // 3. Backtracking: halve alpha up to kMaxBacktracks×; accept on
        //    finite(e) && e <= e0 + tol::energy && objective.accept_mindet(mindet).
        real alpha = 1.0_r;
        PtN<D> cur = pos;
        bool accepted = false;
        for (int it = 0; it < kMaxBacktracks && !accepted; ++it) {
            PtN<D> raw;
            for (int k = 0; k < D; ++k) { raw[k] = pos[k] + alpha * delta[k]; }
            // Constrained DOFs project the trial onto their entity via the same
            // scoped tag switch as the Newton step above.
            PtN<D> trial = raw;
            {
                const PartitionView& part = parts[g.part_of[d]];
                const EntityTag tag = part.tag;
                const auto row = static_cast<std::size_t>(g.row_of[d]);
                if (tag != EntityTag::Free) {
                    real* seed_slot = (seeds != nullptr)
                                        ? (seeds + (static_cast<std::size_t>(dof) * kSeedStride))
                                        : nullptr;
                    with_entity<D>(tag, part.soa_view, part.seg, row, [&](const auto& ent) {
                        trial = project_with_seed<D, false>(ent, raw, seed_slot);
                    });
                }
            }

            real e_new;
            real mdet;
            trial_energy_mindet<D>(pv, X, dof, trial, objective, e_new, mdet);

            const bool ok = sycl::isfinite(e_new) && (e_new <= r.energy + tol::energy) &&
                            objective.accept_mindet(mdet);
            if (ok) {
                cur = trial;
                accepted = true;
            } else {
                alpha *= 0.5_r;
            }
        }

        // 4. Scatter into the double-buffered X_out snapshot. Race-free: distinct
        //    DOFs are distinct nodes, so the scatter never collides.
        store_pt<D>(out, dof, cur);
    });
}

/// Metric-eval kernel: per DOF, patch_eval (or its synthesized twin for
/// interior-eligible DOFs) scattered into per-node buffers keyed by global
/// node id, so the same buffers serve the interior subrange and the boundary
/// launches. The metric half of @ref sweep_kernel hoisted out — no Newton, no
/// projection, no with_entity — so the metric grad+jhj working set lives in
/// its own kernel (interior 144 VGPR → ~88 here + ~80 in the update kernel,
/// both under the gfx1030 128-VGPR 2-wave tier).
///
/// @param q         SYCL queue.
/// @param g         Group view (DOF lists, patches, block layout).
/// @param X         Node coordinates (read).
/// @param grad_buf  [num_nodes * D]    dE/dpos per node.
/// @param hess_buf  [num_nodes * D*D]  contracted Hessian per node (row-major).
/// @param e0_buf    [num_nodes]        baseline patch energy per node.
/// @param objective Metric objective functor.
/// @tparam HasTau   FAS coherence term: minimize E(x) − Σ_d τ_d·x_d instead of
///                  E — the stored gradient is shifted by −τ_d (fas.hpp). A
///                  compile-time branch so the fine-level (HasTau = false)
///                  instantiation is bit-identical to the pre-FAS kernel.
/// @param tau       [num_nodes * D] per-node linear shift (HasTau only).
template <int D, class Obj, bool HasTau = false>
inline void metric_kernel(sycl::queue& q,
                          const GroupViewT<D>& g,
                          const real* X,
                          real* grad_buf,
                          real* hess_buf,
                          real* e0_buf,
                          Obj objective,
                          const real* tau = nullptr)
{
    if (g.ndof == 0) { return; }
    q.parallel_for(sycl::range<1>(g.ndof), [=](sycl::id<1> idx) {
        const std::size_t d = idx[0];
        const int dof = g.dof_idx[d];
        // Interior-eligible DOFs synthesize their patch from the block layout with
        // an identity target (no stored gc/gn/s/W_inv reads); every other DOF reads
        // its stored occurrences. Mirrors the trial-energy branch in Kernel C.
        int block;
        int logical[D];
        const bool synth = g.interior(d, block, logical);
        const PatchResultT<D> r =
          synth ? patch_eval_synth<D>(g.block_off, g.nstride, block, logical, X, objective)
                : patch_eval<D>(g.patch(d), X, objective);
        const std::size_t base = static_cast<std::size_t>(dof);
        real* gslot = grad_buf + (base * D);
        for (int k = 0; k < D; ++k) {
            if constexpr (HasTau) {
                gslot[k] = r.grad[k] - tau[(base * D) + static_cast<std::size_t>(k)];
            } else {
                gslot[k] = r.grad[k];
            }
        }
        real* hslot = hess_buf + (base * D * D);
        for (int k = 0; k < D * D; ++k) { hslot[k] = r.hess[k]; }
        e0_buf[base] = r.energy;

        // Block-interface C2 curvature term (2D): add the DOF's window
        // contributions on top of the shape metric. The det barrier is
        // untouched, so validity is still owned by the shape/mindet gate.
        if constexpr (D == 2) {
            if (g.curv_nodes != nullptr) {
                const int K = g.curv_k;
                const std::size_t off = d * static_cast<std::size_t>(K);
                VecN<2> cg {};
                MatN<2> ch {};
                real ce = 0.0_r;
                curv_dof_accumulate(K,
                                    g.curv_nodes + (off * 4),
                                    g.curv_slot + off,
                                    g.curv_weight + off,
                                    X,
                                    cg,
                                    ch,
                                    ce);
                gslot[0] += cg[0];
                gslot[1] += cg[1];
                hslot[0] += ch[0];
                hslot[1] += ch[1];
                hslot[2] += ch[2];
                hslot[3] += ch[3];
                e0_buf[base] += ce;
            }
        }
    });
}

/// Interior update kernel: Newton solve + backtracking + scatter, reading the
/// metric (grad, hess, e0) precomputed by @ref metric_kernel. No entity, no
/// projection — the step is plain solveNxN<D> and the trial is the raw step.
/// Trial energy reads neighbours from @p X, the frozen block-Jacobi snapshot.
///
/// @param q         SYCL queue.
/// @param g         Group view (DOF lists, patches, block layout).
/// @param X         Read buffer (frozen snapshot).
/// @param X_out     Write buffer (null = in place).
/// @param omega     Relaxation weight (1.0 = undamped).
/// @param grad_buf,hess_buf,e0_buf Metric buffers from @ref metric_kernel.
/// @param objective Metric objective functor.
/// @tparam HasTau   FAS coherence term (see @ref metric_kernel): the gradient
///                  in grad_buf is already shifted; here the line-search
///                  energies get the matching shift −τ_d·x_d (the Hessian is
///                  untouched — the shift is linear — and accept_mindet stays
///                  on the raw geometric det).
/// @param tau       [num_nodes * D] per-node linear shift (HasTau only).
template <int D, class Obj, bool HasTau = false>
inline void interior_update_kernel(sycl::queue& q,
                                   const GroupViewT<D>& g,
                                   real* X,
                                   real* X_out,
                                   real omega,
                                   const real* grad_buf,
                                   const real* hess_buf,
                                   const real* e0_buf,
                                   Obj objective,
                                   const real* tau = nullptr)
{
    if (g.ndof == 0) { return; }
    real* out = X_out ? X_out : X;
    q.parallel_for(sycl::range<1>(g.ndof), [=](sycl::id<1> idx) {
        const std::size_t d = idx[0];
        const int dof = g.dof_idx[d];
        const PtN<D> pos = load_pt<D>(X, dof);
        const std::size_t base = static_cast<std::size_t>(dof);

        // Load the precomputed metric for this node.
        VecN<D> grad {};
        MatN<D> hess {};
        const real* gslot = grad_buf + (base * D);
        for (int k = 0; k < D; ++k) { grad[k] = gslot[k]; }
        const real* hslot = hess_buf + (base * D * D);
        for (int k = 0; k < D * D; ++k) { hess[k] = hslot[k]; }
        real e0 = e0_buf[base];

        // FAS shift of the baseline energy: E(x) − τ_d·x_d at the current pos.
        VecN<D> tau_d {};
        if constexpr (HasTau) {
            for (int k = 0; k < D; ++k) {
                tau_d[k] = tau[(base * D) + static_cast<std::size_t>(k)];
                e0 -= tau_d[k] * pos[k];
            }
        }

        // Interior DOF: plain d×d Newton solve (no entity geometry).
        const VecN<D> delta = solveNxN<D>(hess, grad);

        // The trial-energy recompute reads the patch the same way the metric did:
        // synthesized from the block layout for interior-eligible DOFs (no stored
        // gc/gn/s reads), or from the stored arrays otherwise.
        int block;
        int logical[D];
        const bool synth = g.interior(d, block, logical);
        const PatchViewT<D> pv = synth ? PatchViewT<D> {} : g.patch(d);

        // Backtracking: halve alpha up to kMaxBacktracks×; trial == raw (interior
        // projects to identity). Accept on finite(e) && e <= e0 + tol && accept_mindet.
        real alpha = 1.0_r;
        PtN<D> cur = pos;
        bool accepted = false;
        for (int it = 0; it < kMaxBacktracks && !accepted; ++it) {
            PtN<D> trial;
            for (int k = 0; k < D; ++k) { trial[k] = pos[k] + (alpha * delta[k]); }
            real e_new = 0.0_r;
            real mdet = std::numeric_limits<real>::infinity();
            if (synth) {
                synth_trial_energy_mindet<
                  D>(g.block_off, g.nstride, block, logical, X, dof, trial, objective, e_new, mdet);
            } else {
                trial_energy_mindet<D>(pv, X, dof, trial, objective, e_new, mdet);
            }
            // Block-interface C2 curvature: add the DOF's window energies with
            // its node moved to the trial (others frozen), matching the curv
            // energy folded into e0 by metric_kernel — so the accept rule gates
            // on the composed objective. mdet (the barrier) is untouched.
            if constexpr (D == 2) {
                if (g.curv_nodes != nullptr) {
                    const int K = g.curv_k;
                    const std::size_t off = d * static_cast<std::size_t>(K);
                    e_new += curv_dof_trial_energy(
                      K, g.curv_nodes + (off * 4), g.curv_slot + off, g.curv_weight + off, X, trial);
                }
            }
            // Shift the trial energy by the same linear term as e0 above, so
            // the acceptance compares the FAS objective on both sides.
            if constexpr (HasTau) {
                for (int k = 0; k < D; ++k) { e_new -= tau_d[k] * trial[k]; }
            }
            const bool ok =
              sycl::isfinite(e_new) && (e_new <= e0 + tol::energy) && objective.accept_mindet(mdet);
            if (ok) {
                cur = trial;
                accepted = true;
            } else {
                alpha *= 0.5_r;
            }
        }
        if (omega != 1.0_r) {
            for (int k = 0; k < D; ++k) { cur[k] = pos[k] + (omega * (cur[k] - pos[k])); }
        }
        store_pt<D>(out, dof, cur);
    });
}

/// Plain SPD solve M x = b (K = 1 or 2), no Newton negation/fallback. Used
/// for the param-space reduction Δq = (JᵀJ)⁻¹ Jᵀδ; a near-singular Gram
/// matrix (degenerate frame at a trim corner) returns 0, so the line search
/// simply does not move that sweep.
template <int K> inline VecN<K> spd_solve(const MatN<K>& M, const VecN<K>& b)
{
    if constexpr (K == 1) {
        return VecN<1> {(sycl::fabs(M[0]) > tol::tiny) ? (b[0] / M[0]) : 0.0_r};
    } else {
        const real det = (M[0] * M[3]) - (M[1] * M[2]);
        if (sycl::fabs(det) < tol::tiny) { return VecN<2> {0.0_r, 0.0_r}; }
        const real inv = 1.0_r / det;
        return VecN<2> {inv * ((M[3] * b[0]) - (M[1] * b[1])),
                        inv * ((-M[2] * b[0]) + (M[0] * b[1]))};
    }
}

/// Converged foot parameter q* of @p p on @p param — warm-started
/// invert_seeded when iterative, else closed-form invert. Mirrors
/// @ref project_with_seed but returns the parameter, which the param-space
/// line search backtracks in.
template <int K, class P>
inline Param<K>
  foot_param_seeded(const P& param, const PtN<P::edim>& p, const Param<K>& seed, bool has_seed)
{
    if constexpr (requires { param.template invert_seeded<true>(p, seed, has_seed); }) {
        return param.template invert_seeded<true>(p, seed, has_seed);
    } else {
        static_cast<void>(seed);
        static_cast<void>(has_seed);
        return param.invert(p);
    }
}

/// Converged foot q* AND the raw tangent frame J at q*. For an iterative
/// B-spline foot the frame is harvested from the last newton_foot ders(nd=1),
/// saving a redundant frame(q*) re-evaluation; closed-form entities fall back
/// to invert + frame(q*).
template <int K, int D, class P>
inline std::pair<Param<K>, std::array<VecN<D>, K>>
  foot_and_frame_seeded(const P& param, const PtN<P::edim>& p, const Param<K>& seed, bool has_seed)
{
    std::array<VecN<D>, K> J {};
    if constexpr (requires { param.template invert_seeded<true>(p, seed, has_seed, &J); }) {
        const Param<K> q = param.template invert_seeded<true>(p, seed, has_seed, &J);
        return {q, J};
    } else {
        const Param<K> q = foot_param_seeded<K>(param, p, seed, has_seed);
        return {q, param.frame(q)};
    }
}

/// Warm boundary sweep kernel with param-space line search. Replaces the
/// per-trial reprojection of @ref sweep_kernel's warm path with: project ONCE
/// to the foot q*, reduce the tangent-space Newton step δ_world = B y to
/// Δq = (JᵀJ)⁻¹ Jᵀ δ_world (J = raw frame at q*, B = orthonormalize(J)), then
/// backtrack q_trial = trim.clamp(q* + αΔq) with value-only param.eval
/// (de Boor nd=0) — the heavy nd=1 derivative rows run once instead of ≤10×
/// per DOF. The constraint is preserved exactly (every trial lands on the
/// geometry) and the Δq reduction is exact (δ_world ∈ span(J)); the only
/// difference vs world-space LS is curvature at finite α. @p omega scales
/// the step (Δq, or the world-space δ in the reprojection arm) BEFORE the
/// line search for the same reason as @ref sweep_kernel: damping must never
/// pull a constrained DOF off its geometry. Warm path only — the cold
/// sweep 0 stays the world-space @ref sweep_kernel that seeds q*.
template <int D, class Obj>
inline void sweep_boundary_paramls_kernel(sycl::queue& q,
                                          const GroupViewT<D>& g,
                                          real* X,
                                          Obj objective,
                                          real* X_out,
                                          real omega,
                                          real* seeds)
{
    if (g.ndof == 0) { return; }
    constexpr std::size_t kSeedStride = (D > 1) ? (D - 1) : 1;
    const PartitionView* parts = g.partitions;
    real* out = X_out ? X_out : X;
    q.parallel_for(sycl::range<1>(g.ndof), [=](sycl::id<1> idx) {
        const std::size_t d = idx[0];
        const PatchViewT<D> pv = g.patch(d);
        const int dof = g.dof_idx[d];
        const PatchResultT<D> r = patch_eval<D>(pv, X, objective);
        const PtN<D> pos = load_pt<D>(X, dof);
        const PartitionView& part = parts[g.part_of[d]];
        const auto row = static_cast<std::size_t>(g.row_of[d]);
        real* seed_slot =
          (seeds != nullptr) ? (seeds + (static_cast<std::size_t>(dof) * kSeedStride)) : nullptr;

        PtN<D> cur = pos;
        with_entity<D>(part.tag, part.soa_view, part.seg, row, [&](const auto& ent) {
            using E = std::decay_t<decltype(ent)>;
            if constexpr (requires {
                              ent.param;
                              ent.trim;
                              E::tdim;
                          }) {
                constexpr int K = E::tdim;
                // 1. Project once → foot q* AND the raw frame J at q*. The frame
                //    is harvested from the projection's last de Boor nd=1 (F2
                //    Lever C) instead of a redundant frame(q*) re-evaluation. q*
                //    is written back to the seed slot (foot of the *old* position,
                //    as today); the accepted trial is NOT written back.
                Param<K> seed {};
                const bool has = (seed_slot != nullptr) && sycl::isfinite(seed_slot[0]);
                if (has) {
                    for (int a = 0; a < K; ++a) { seed[a] = seed_slot[a]; }
                }
                const auto [qstar, J] = foot_and_frame_seeded<K, D>(ent.param, pos, seed, has);
                if (seed_slot != nullptr) {
                    for (int a = 0; a < K; ++a) { seed_slot[a] = qstar[a]; }
                }
                const std::array<VecN<D>, K> B = orthonormalize<D, K>(J);

                // 3. Tangent-reduced Newton step δ_world = B y.
                const VecN<D> delta = newton_step_from_basis<D, K>(r.grad, r.hess, B);

                // 4. Reduce to a parametric step Δq = (JᵀJ)⁻¹ Jᵀ δ_world (exact,
                //    since δ_world ∈ span(J)).
                MatN<K> JtJ {};
                VecN<K> Jtd {};
                for (int a = 0; a < K; ++a) {
                    for (int b = 0; b < K; ++b) {
                        real s = 0.0_r;
                        for (int i = 0; i < D; ++i) { s += J[a][i] * J[b][i]; }
                        JtJ[(a * K) + b] = s;
                    }
                    real s2 = 0.0_r;
                    for (int i = 0; i < D; ++i) { s2 += J[a][i] * delta[i]; }
                    Jtd[a] = s2;
                }
                VecN<K> dq = spd_solve<K>(JtJ, Jtd);
                for (int a = 0; a < K; ++a) { dq[a] *= omega; }

                // 5. Param-space backtracking: q_trial = clamp(q* + αΔq), trial =
                //    eval(q_trial) — a value-only (nd=0) de Boor, no reprojection.
                real alpha = 1.0_r;
                bool accepted = false;
                for (int it = 0; it < kMaxBacktracks && !accepted; ++it) {
                    Param<K> qt;
                    for (int a = 0; a < K; ++a) { qt[a] = qstar[a] + (alpha * dq[a]); }
                    qt = ent.trim.clamp(qt);
                    const PtN<D> trial = ent.param.eval(qt);
                    real e_new;
                    real mdet;
                    trial_energy_mindet<D>(pv, X, dof, trial, objective, e_new, mdet);
                    const bool ok = sycl::isfinite(e_new) && (e_new <= r.energy + tol::energy) &&
                                    objective.accept_mindet(mdet);
                    if (ok) {
                        cur = trial;
                        accepted = true;
                    } else {
                        alpha *= 0.5_r;
                    }
                }
            } else if constexpr (requires { ent.project(pos); }) {
                // Bespoke closed-form entities (circle / line / arc / ellipse) have
                // no param/trim parametric pipeline, so the param-space arm above
                // skips them — which would freeze these boundary DOFs. Relax them
                // with the world-space reprojection line search instead: a Newton
                // step along the tangent, backtracking in world space with a cheap
                // analytic reprojection per trial. (The free/interior entity's
                // identity `project` also matches here, but free DOFs never enter
                // this boundary group — they are relaxed by the dedicated kernel.)
                VecN<D> delta =
                  newton_delta_with_seed<D, true>(r.grad, r.hess, pos, ent, seed_slot);
                for (int k = 0; k < D; ++k) { delta[k] *= omega; }
                real alpha = 1.0_r;
                bool accepted = false;
                for (int it = 0; it < kMaxBacktracks && !accepted; ++it) {
                    PtN<D> raw;
                    for (int k = 0; k < D; ++k) { raw[k] = pos[k] + (alpha * delta[k]); }
                    const PtN<D> trial = project_with_seed<D, true>(ent, raw, seed_slot);
                    real e_new;
                    real mdet;
                    trial_energy_mindet<D>(pv, X, dof, trial, objective, e_new, mdet);
                    const bool ok = sycl::isfinite(e_new) && (e_new <= r.energy + tol::energy) &&
                                    objective.accept_mindet(mdet);
                    if (ok) {
                        cur = trial;
                        accepted = true;
                    } else {
                        alpha *= 0.5_r;
                    }
                }
            }
        });
        store_pt<D>(out, dof, cur);
    });
}

/// Per-sweep total energy (Σμ) and min det A over the energy stencil, written
/// to out_e/out_m (USM scalars). Reduction identities are supplied explicitly,
/// so the targets need no host pre-initialisation.
template <int D, class Obj>
inline void reduce_energy_mindet(
  sycl::queue& q, const StencilViewT<D>& es, const real* X, real* out_e, real* out_m, Obj objective)
{
    auto e_red = sycl::reduction(out_e,
                                 0.0_r,
                                 std::plus<>(),
                                 sycl::property::reduction::initialize_to_identity {});
    auto m_red = sycl::reduction(out_m,
                                 std::numeric_limits<real>::infinity(),
                                 sycl::minimum<real>(),
                                 sycl::property::reduction::initialize_to_identity {});

    q.parallel_for(sycl::range<1>(es.num_samples),
                   e_red,
                   m_red,
                   [=](sycl::id<1> idx, auto& e_sum, auto& m_min) {
                       const std::size_t i = idx[0];
                       // W_inv row i is contiguous; w_stride is 0 for a uniform
                       // target (one shared row) or wInv for a per-sample table.
                       const real* w = &es.W_inv[i * static_cast<std::size_t>(es.w_stride)];
                       PtN<D> corner = load_pt<D>(X, es.gc[i]);
                       std::array<PtN<D>, D> nbr;
                       std::array<real, D> sc;
                       for (int k = 0; k < D; ++k) {
                           nbr[k] = load_pt<D>(X, es.gn[k][i]);
                           sc[k] = es.s[k][i];
                       }
                       real detA;
                       const VecTN<D> t = assemble_vecT<D>(corner, nbr, sc, w, detA);
                       // Weight view may be empty on coarse levels that never carry
                       // one — read unit weight then (stride 0 reads the shared row).
                       const real wt = es.weight.extent(0) > 0
                                         ? es.weight[i * static_cast<std::size_t>(es.wt_stride)]
                                         : 1.0_r;
                       e_sum += wt * objective.value(t);
                       m_min.combine(detA);
                   });
}

/// One double-buffered block-Jacobi sweep: pre-sweep hook on the read buffer,
/// interior metric/update kernel pair over the Free prefix, then the boundary
/// kernel over the non-Free tail (cold world-space on the first sweep to seed
/// the warm-start cache, warm param-LS after). Reads @p x_in (frozen
/// snapshot), writes @p x_out; the caller swaps. Shared by @ref
/// run_block_jacobi and the FAS pre/post-smooths — a coarse level passes a
/// no-op hook, a view with n_free == ndof (no boundary tail, both boundary
/// kernels early-return), and null @p seeds.
///
/// @param mg        merged group view (Free-first reordered).
/// @param cold_boundary  true on the first sweep of a run (cold projection).
/// @param seeds     per-node warm-start cache (null = cold projections).
/// @param grad_buf,hess_buf,e0_buf  per-node metric scratch, keyed by global
///                  node id (written by metric_kernel, read by the update).
template <int D, class Obj, class PreSweep>
inline void jacobi_sweep_once(sycl::queue& q,
                              const GroupViewT<D>& mg,
                              Obj objective,
                              PreSweep& before_sweep,
                              real* x_in,
                              real* x_out,
                              real omega,
                              bool cold_boundary,
                              real* seeds,
                              real* grad_buf,
                              real* hess_buf,
                              real* e0_buf)
{
    before_sweep(q, x_in);  // refresh ghosts + shared-node copies on the read buffer
    // Interior metric/update split: metric_kernel writes (grad, hess, e0)
    // per node; interior_update_kernel reads them back and does the Newton
    // solve + backtracking. Both run over [0, n_free) on the frozen x_in;
    // the update writes disjoint x_out slots, so block-Jacobi correctness
    // holds. Separate kernels keep the working sets apart (144 → ~88 + ~80
    // VGPR).
    const GroupViewT<D> interior = mg.dof_subrange(0, mg.n_free);
    metric_kernel<D, Obj>(q, interior, x_in, grad_buf, hess_buf, e0_buf, objective);
    interior_update_kernel<D, Obj>(q,
                                   interior,
                                   x_in,
                                   x_out,
                                   omega,
                                   grad_buf,
                                   hess_buf,
                                   e0_buf,
                                   objective);
    // Boundary cold/warm split: the first sweep runs the robust cold kernel
    // (8×8 grid + exact Newton nd=2) to project topologically-placed nodes
    // and populate the warm-seed cache; later sweeps run the lean warm
    // kernel (GN nd=1 — the de Boor second-order rows stay out of the hot
    // kernel). NB: per-entity monomorphization of the boundary kernel was
    // measured to REGRESS (BSpline arm alone 208 VGPR vs the union's 184,
    // plus a launch explosion), so the boundary stays one union kernel.
    const GroupViewT<D> bndry = mg.dof_subrange(mg.n_free, mg.ndof - mg.n_free);
    if (cold_boundary) {
        sweep_kernel<D, Obj>(q, bndry, x_in, objective, x_out, omega, seeds);
    } else {
        // Warm path: param-space line search (see kernel doc).
        sweep_boundary_paramls_kernel<D, Obj>(q, bndry, x_in, objective, x_out, omega, seeds);
    }
}

/// Reusable device scratch of the smoother drivers, owned by the resident
/// session (StructuredExecutorT) so per-chunk run/run_fas calls do not
/// re-allocate — USM malloc/free can synchronise the device, which is pure
/// dead time in the launch-bound small-mesh regime. Lazily (re)sized; no
/// state is carried between calls (each driver initialises what it reads:
/// x_new is copied from X, the metric buffers are written before read, corr
/// is memset per run_fas call).
template <int D> struct SweepScratchT {
    UsmBuffer<real> x_new;           ///< [x_size] double-buffer partner of X
    UsmBuffer<real> grad, hess, e0;  ///< [nn*D], [nn*D*D], [nn] metric scratch
    UsmBuffer<real> corr;            ///< [x_size] FAS prolonged correction
    UsmBuffer<real> ls;              ///< [4] FAS line-search readback

    /// Size the Jacobi buffers (x_new + metric) for nn nodes / x_size reals.
    void ensure_jacobi(sycl::queue& q, std::size_t nn, std::size_t x_size)
    {
        if (x_new.size() != x_size) { x_new = UsmBuffer<real> {q, x_size}; }
        if (e0.size() != nn) {
            grad = UsmBuffer<real> {q, nn * static_cast<std::size_t>(D)};
            hess = UsmBuffer<real> {q, nn * static_cast<std::size_t>(D * D)};
            e0 = UsmBuffer<real> {q, nn};
        }
    }
    /// The Jacobi buffers plus the FAS-only correction/readback slots.
    void ensure_fas(sycl::queue& q, std::size_t nn, std::size_t x_size)
    {
        ensure_jacobi(q, nn, x_size);
        if (corr.size() != x_size) { corr = UsmBuffer<real> {q, x_size}; }
        if (ls.size() != 4) { ls = UsmBuffer<real> {q, 4}; }
    }
};

/// Double-buffered block-Jacobi driver: n_sweeps of (pre-sweep hook → merged
/// Jacobi launch reading X_in, writing X_new → energy/min-det reduction →
/// swap). Under the frozen-halo cadence every free DOF reads the previous
/// snapshot, so there is no intra-sweep dependency. @p before_sweep is the
/// per-sweep halo hook; @p omega the SOR weight; @p scratch the reusable
/// device buffers (lazily sized here). X_new is copied from X each call;
/// each sweep writes every free DOF and ghosts are re-exchanged, so no
/// per-sweep full copy is needed. The canonical ctx buffer is restored if an
/// odd number of swaps left the result in X_new.
template <int D, class Obj, class PreSweep>
inline std::pair<std::vector<real>, std::vector<real>> run_block_jacobi(sycl::queue& q,
                                                                        SweepDeviceContextT<D>& ctx,
                                                                        SweepScratchT<D>& scratch,
                                                                        Obj objective,
                                                                        int n_sweeps,
                                                                        PreSweep before_sweep,
                                                                        real omega = 1.0_r,
                                                                        int report_every = 1)
{
    const int k = (report_every <= 0) ? n_sweeps : report_every;
    const std::size_t report_count = static_cast<std::size_t>((n_sweeps + k - 1) / k);
    UsmBuffer<real> d_e(q, report_count);
    UsmBuffer<real> d_m(q, report_count);
    const std::size_t nbuf = ctx.x_size();

    // Per-node metric scratch for the interior split, keyed by global node id
    // so the same buffers serve the boundary launches. Written by the metric
    // kernel before the update kernel reads them each sweep — stale slots are
    // never read, so no per-sweep memset.
    const std::size_t nn = ctx.num_nodes();
    scratch.ensure_jacobi(q, nn, nbuf);
    UsmBuffer<real>& x_new_buf = scratch.x_new;
    UsmBuffer<real>& grad_buf = scratch.grad;
    UsmBuffer<real>& hess_buf = scratch.hess;
    UsmBuffer<real>& e0_buf = scratch.e0;

    real* canonical = ctx.X();
    real* x_in = canonical;
    real* x_out = x_new_buf.data();
    q.memcpy(x_out, x_in, nbuf * sizeof(real)).wait();  // X_new := copy of X

    const GroupViewT<D>& mg = ctx.merged_group_view();

    constexpr int kSyncEvery = 32;
    std::size_t report_idx = 0;
    for (int s = 0; s < n_sweeps; ++s) {
        jacobi_sweep_once<D, Obj>(q,
                                  mg,
                                  objective,
                                  before_sweep,
                                  x_in,
                                  x_out,
                                  omega,
                                  /*cold_boundary=*/s == 0,
                                  ctx.seeds(),
                                  grad_buf.data(),
                                  hess_buf.data(),
                                  e0_buf.data());
        if ((s + 1) % k == 0 || s == n_sweeps - 1) {
            reduce_energy_mindet<D>(q,
                                    ctx.stencil_view(),
                                    x_out,
                                    d_e.data() + report_idx,
                                    d_m.data() + report_idx,
                                    objective);
            ++report_idx;
        }
        std::swap(x_in, x_out);
        if ((s + 1) % kSyncEvery == 0) { q.wait(); }
    }
    q.wait();
    assert(report_idx == report_count);

    // After the loop the freshest values are in x_in (post-swap). Make the ctx's
    // canonical buffer hold them so downloads / subsequent runs see the result.
    if (x_in != canonical) { q.memcpy(canonical, x_in, nbuf * sizeof(real)).wait(); }

    std::vector<real> energies(report_count), mindets(report_count);
    d_e.download(energies.data());
    d_m.download(mindets.data());
    return {energies, mindets};
}

/// Convenience overload with one-shot scratch (tests / single calls); resident
/// sessions pass their own SweepScratchT to reuse the buffers across calls.
template <int D, class Obj, class PreSweep>
inline std::pair<std::vector<real>, std::vector<real>> run_block_jacobi(sycl::queue& q,
                                                                        SweepDeviceContextT<D>& ctx,
                                                                        Obj objective,
                                                                        int n_sweeps,
                                                                        PreSweep before_sweep,
                                                                        real omega = 1.0_r,
                                                                        int report_every = 1)
{
    SweepScratchT<D> scratch;
    return run_block_jacobi<D>(q,
                               ctx,
                               scratch,
                               objective,
                               n_sweeps,
                               before_sweep,
                               omega,
                               report_every);
}

// D=2 legacy aliases for the binding's oracle surface and existing call sites.
using SweepGroupHost = SweepGroupHostT<2>;
using EnergyStencilHost = EnergyStencilHostT<2>;
using SweepContextHost = SweepContextHostT<2>;
using GroupView = GroupViewT<2>;
using StencilView = StencilViewT<2>;
using SweepDeviceContext = SweepDeviceContextT<2>;

}  // namespace egg
