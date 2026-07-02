// sweep.hpp — device-resident colored Gauss-Seidel barrier/untangle sweep.
#pragma once

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

// The colored Gauss-Seidel kernels build the per-corner stencil through the
// D-neighbour PatchViewT<D> and take the D×D Newton step. The A→T→detA math is
// single-sourced in patch.hpp::assemble_vecT; everything here is generic over D,
// and only the dimensions whose math exists (D=2 in Phase 1) are instantiated.

/// @brief Per-partition typed SoA host data (one per entity type present in the
///        group). Populated by `unpack_context` from the Python `entities`
///        sub-dicts (Phase 1b-B); the `SweepDeviceContextT` ctor uploads these
///        to USM and constructs the `PartitionView::soa_view` + `seg` views.
///        All SoA views share the one `SoAView<const real>` type for records,
///        so this struct is type-erased at the host level (the interpretation is
///        per-`E` in `load`). The `seg` vector carries variable-length CSR
///        payloads (B-spline knots/ctrl) — empty for fixed-size types.
struct SoAHostRecord {
    /// @brief One segmented (CSR) field: flat data + offset table.
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
    std::size_t ndof;                // DOFs in this colour
    std::size_t total_samples;       // Σ P_of[d]
    std::vector<int> gc;             // [total_samples]
    std::vector<int> gn[D];          // [total_samples] per axis (was gn0, gn1)
    std::vector<real> s[D];        // [total_samples] per axis (was s0, s1)
    std::vector<real> W_inv;       // [total_samples * dim::wInv(D)]
    std::vector<int> role;           // [total_samples]
    std::vector<real> J;           // [total_samples * dim::jSize(D)]
    std::vector<int> dof_idx;        // [ndof]
    std::vector<int> P_of;           // [ndof]  patch size per DOF
    std::vector<int> sample_offset;  // [ndof]  exclusive prefix sum of P_of
    std::vector<SoAHostRecord> soa;  // typed per-(colour,EntityTag) SoA stores (the entity data)

    // Structured interior fast path: for a DOF whose entire metric patch lies
    // inside one block's interior, interior_block[d] names that block and
    // interior_logical[d*D + k] its logical index, so a consumer can synthesize
    // the patch indices from the block layout instead of reading this DOF's
    // stored gc/gn/role/s. interior_block[d] == -1 means "read the stored
    // arrays" (boundary/interface DOFs); both stay empty on the unstructured
    // path, where the fast path never applies.
    std::vector<int> interior_block;    // [ndof] or empty; -1 = use stored arrays
    std::vector<int> interior_logical;  // [ndof * D] or empty
};

template <int D> struct EnergyStencilHostT {
    std::size_t num_samples;
    std::vector<int> gc;  // [num_samples]
    std::vector<int> gn[D];
    std::vector<real> s[D];
    std::vector<real> W_inv;  // [num_samples * dim::wInv(D)]
};

template <int D> struct SweepContextHostT {
    std::vector<SweepGroupHostT<D>> groups;  // colour-ordered
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

/// @brief One (colour, EntityTag) partition of a colour group: a contiguous
///        list of group-local DOF indices sharing the same entity type.
///
/// Trivially copyable (captured by value into the SYCL kernel lambda). The
/// `dof_list` points at device USM; `ndof` is the partition's DOF count. The
/// sweep instantiates one monomorphic kernel per partition, indexing the
/// group's typed SoA via `dof_list[i]` — no per-DOF `std::visit`.
///
/// `soa_view` is the packed-record SoA view (extents `(ndof, kFields)`); `seg`
/// carries the segmented (CSR) views for variable-length fields (B-spline
/// knots/ctrl, composite records) — null for fixed-size types. The kernel calls
/// `tie_view(soa_view, seg)` + `load` to build the concrete entity (every 2D
/// type has an @ref egg::HasEntitySoA specialization since Phase 4 retired the
/// positional blob). All SoA views share the one typed `SoAView<const real>`
/// — no `const void*`, no type erasure, no cast.
struct PartitionView {
    static constexpr int kMaxSeg = kMaxSoASeg;  ///< Max segmented slots (matches EntitySoA<E>::View).
    EntityTag tag;                 ///< Entity type of this partition.
    std::size_t ndof;              ///< Number of DOFs in this partition.
    const int* dof_list;           ///< Group-local DOF indices, length @c ndof.
    SoAView<const real> soa_view{nullptr, 0, 0};  ///< Packed records (ndof, kFields).
    SegmentedView<real> seg[kMaxSeg]{};  ///< CSR fields (null for fixed-size).
};

template <int D> struct GroupViewT {
    std::size_t ndof;
    std::size_t n_free = 0;  // DOFs are Free-first reordered; [0, n_free) are Free
    // Per-occurrence (length total_samples, sliced per DOF by sample_offset):
    View1<const int> sample_id, role;
    // Shared deduplicated metric table (base pointers, indexed by sample_id[p],
    // shared by every group/the merged view — sized num table samples, NOT
    // total_samples). gc/gn/s/W_inv below are these bases; J is recomputed.
    View1<const int> gc;                                 // [n_table]
    View1<const int> gn[D];                              // [n_table]
    View1<const std::int8_t> s[D];                       // [n_table] per-axis sign ±1
    View1<const real> W_inv;                           // [n_table * wInv], or one row if uniform
    int w_stride = dim::wInv(D);                          // 0 when W_inv is a uniform shared row
    View1<const int> dof_idx, P_of, sample_offset;       // [ndof]
    View1<const int> part_of, row_of;                    // [ndof] DOF -> (partition, row)
    const PartitionView* partitions = nullptr;           // [num_partitions]
    std::size_t num_partitions = 0;                      // per-(colour,EntityTag) splits

    // Structured interior fast path (null on the unstructured path): interior_block
    // [d] (or -1) names the block whose layout synthesizes DOF d's patch, and
    // interior_logical[d*D + k] its logical index; block_off/nstride are the shared
    // node-unit layout the synthesis indexes. See interior() and patch_eval_synth.
    const int* interior_block = nullptr;    // [ndof]
    const int* interior_logical = nullptr;  // [ndof * D]
    const int* block_off = nullptr;         // [num_blocks] (shared)
    const int* nstride = nullptr;           // [num_blocks * D] (shared)

    // True when DOF d's patch can be synthesized from the block layout (its whole
    // patch lies inside one block's interior); fills the block + logical index.
    bool interior(std::size_t d, int& block, int (&logical)[D]) const
    {
        if (interior_block == nullptr || interior_block[d] < 0) { return false; }
        block = interior_block[d];
        for (int k = 0; k < D; ++k) { logical[k] = interior_logical[(d * D) + static_cast<std::size_t>(k)]; }
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
        return sub;
    }
};

template <int D> struct StencilViewT {
    std::size_t num_samples;
    View1<const int> gc;
    View1<const int> gn[D];
    View1<const real> s[D];
    View2<const real> W_inv;  // [num_samples][wInv]
};

// Base pointers of the shared deduplicated metric table, indexed by a patch
// occurrence's sample_id. One table is shared by every group and the merged view.
// J is not stored — its role block is recomputed in-kernel from s + W_inv.
template <int D> struct MetricTableViewT {
    View1<const int> gc;       // [n_table]
    View1<const int> gn[D];    // [n_table]
    View1<const std::int8_t> s[D];  // [n_table] per-axis sign ±1, widened on read
    View1<const real> W_inv; // [n_table * dim::wInv(D)], or one row when uniform
    // W_inv row stride: dim::wInv(D), or 0 when every sample shares one row (a
    // uniform target), in which case W_inv holds a single dim::wInv(D)-wide row.
    int w_stride = dim::wInv(D);
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
        seeds_(q,
               std::vector<real>(host.num_nodes * kSeedStride,
                                   std::numeric_limits<real>::quiet_NaN()))
    {
        // Deduplicate the per-sample payload (gc/gn/s/W_inv/J) across every group
        // into one shared table; each group keeps only a per-occurrence sample_id.
        const std::vector<std::vector<int>> group_sid = build_metric_table(host);

        groups_.reserve(host.groups.size());
        group_partitions_.reserve(host.groups.size());
        for (std::size_t gi = 0; gi < host.groups.size(); ++gi) {
            const auto& g = host.groups[gi];
            DeviceGroup dg;
            dg.ndof = g.ndof;
            dg.total_samples = g.total_samples;
            dg.sample_id = {q, group_sid[gi]};
            dg.role = {q, g.role};

            // Each SoAHostRecord in g.soa is one (colour, EntityTag) partition:
            // its dof_local is the group-local DOF list, its records/seg the
            // already-typed SoA (built in Python by group_entities_by_type, or by
            // the C++ golden test's blob→SoA helper). The upload is a direct copy
            // — no per-DOF blob decode, no tag scan (Phase 4 retired the blob).
            // The monomorphic kernel iterates one partition per launch, indexing
            // its typed SoA via dof_list.
            dg.partitions.reserve(g.soa.size());
            for (const auto& rec : g.soa) {
                DevicePartition dp;
                dp.tag = rec.tag;
                dp.dof_list = {q, rec.dof_local};
                dispatch_entity_type<D>(rec.tag, [&]<class E>() {
                    if constexpr (HasEntitySoA<E>) {
                        using SoA = EntitySoA<E>;
                        dp.soa_records = UsmBuffer<real>{q, rec.records};
                        for (int j = 0; j < SoA::kSeg; ++j) {
                            if (j < static_cast<int>(rec.seg.size())) {
                                dp.seg_data[j] = UsmBuffer<real>{q, rec.seg[j].data};
                                dp.seg_off[j] = UsmBuffer<int>{q, rec.seg[j].off};
                            }
                        }
                    }
                });
                dg.partitions.push_back(std::move(dp));
            }

            // Invert the per-partition dof_local lists into per-DOF maps so the
            // single per-colour kernel can find each DOF's entity: part_of[d]
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

            // Free-first reorder: permute the per-DOF arrays so every Free
            // (interior) DOF precedes the non-Free (boundary) ones, making the
            // two classes contiguous sub-ranges of one view. The per-sample
            // arrays are untouched — sample_offset still indexes them. A DOF is
            // Free iff its owning partition's tag is Free. Plan #1 step 2.
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

        // Build the stable PartitionView arrays (raw device pointers) and the
        // GroupViewT views in lockstep; both index groups_ by position.
        group_views_.reserve(groups_.size());
        group_partitions_.reserve(groups_.size());
        for (std::size_t gi = 0; gi < groups_.size(); ++gi) {
            auto& dg = groups_[gi];
            std::vector<PartitionView> pvs;
            pvs.reserve(dg.partitions.size());
            for (const auto& dp : dg.partitions) {
                const std::size_t ndof = dp.dof_list.size();
                const real* soa_ptr = dp.soa_records.data();
                std::size_t kfields = 0;
                dispatch_entity_type<D>(dp.tag, [&]<class E>() {
                    if constexpr (HasEntitySoA<E>) { kfields = static_cast<std::size_t>(EntitySoA<E>::kFields); }
                });
                PartitionView pv {
                  .tag = dp.tag, .ndof = ndof, .dof_list = dp.dof_list.data(),
                  .soa_view = SoAView<const real> {soa_ptr, ndof, kfields}};
                // Fill segmented views from uploaded USM (null for fixed-size).
                for (int j = 0; j < PartitionView::kMaxSeg; ++j) {
                    pv.seg[j] = SegmentedView<real>{
                      dp.seg_data[j].data(), dp.seg_off[j].data()};
                }
                pvs.push_back(pv);
            }
            // The per-colour kernel indexes this array ON the device (via
            // part_of[d]), so it must live in USM — a host std::vector pointer
            // would fault on the GPU. Upload once; the device pointer is stable.
            group_partitions_.emplace_back(q, pvs);
            group_views_.push_back(
              dg.view(group_partitions_.back().data(), pvs.size(), table_view(),
                      block_off_.data(), nstride_.data()));
            host_pvs_.push_back(std::move(pvs));  // retained for the merged Jacobi view
        }
        stencil_view_ = stencil_.view();
    }

    const std::vector<GroupViewT<D>>& group_views() const { return group_views_; }
    /// @brief True once the merged block-Jacobi view has been built and the
    ///        per-group per-sample buffers backing @ref group_views() were freed.
    ///        Colored-GS is unavailable on this instance thereafter.
    bool per_group_released() const { return per_group_released_; }
    const StencilViewT<D>& stencil_view() const { return stencil_view_; }
    real* X() const { return X_.data(); }
    std::size_t x_size() const { return X_.size(); }
    std::size_t num_nodes() const { return num_nodes_; }
    /// @brief Per-node warm-start seed cache (stride @ref kSeedStride), persisted
    ///        across sweeps. Forwarded to the block-Jacobi kernel; null-equivalent
    ///        cold path is used by colored-GS (passes nullptr).
    real* seeds() { return seeds_.data(); }

    /// @brief A single GroupViewT spanning *all* colour groups, for block-Jacobi.
    ///
    /// Under frozen halos every free DOF updates from the same snapshot, so the
    /// colour ordering is irrelevant and all DOFs can run in **one launch**. The
    /// in-order queue serialises per-group launches, so a merged view (not merely
    /// dropping the dependency) is what recovers full occupancy. Built lazily on
    /// first use and cached: the bulk per-sample arrays are concatenated by
    /// device-to-device copy; the small per-DOF index arrays are rebuilt with the
    /// running sample/partition offsets; the partition list (entity SoA pointers)
    /// is concatenated unchanged. The per-colour @ref group_views() stay intact
    /// for colored-GS.
    const GroupViewT<D>& merged_group_view()
    {
        if (!merged_built_) {
            build_merged_view();
            merged_built_ = true;
        }
        return merged_view_;
    }

    /// @brief One monomorphic boundary launch descriptor: a single (entity-type)
    ///        partition's list of **Free-first-reordered DOF positions** into the
    ///        merged view (§4.7). Kernels A/C launch over `positions[0..count)`;
    ///        for each position `d` they recover the patch (`merged.patch(d)`), the
    ///        global node (`merged.dof_idx[d]`), and the SoA row (`merged.row_of[d]`)
    ///        exactly as the subrange kernel does — but with the entity type fixed
    ///        at launch (`tag`/`pv.soa_view`), so `dispatch_entity_type` collapses
    ///        to one arm and the +144 control-flow union is gone. Block-Jacobi only
    ///        (the merged view is a single "colour"); never used by colored-GS.
    struct BoundaryLaunch {
        EntityTag tag;          ///< Entity type of this partition (drives monomorphization).
        PartitionView pv;       ///< Entity SoA view (records + seg) for this partition.
        std::size_t count;      ///< Number of boundary DOFs in this partition.
        const int* positions;   ///< Device USM: reordered positions into the merged view.
    };

    /// @brief The per-partition boundary launch descriptors (§4.7 / S1). Built
    ///        lazily alongside the merged view; one entry per boundary partition
    ///        actually present (~1–2 for sphere_in_cube). The union of every
    ///        entry's positions is exactly `[merged.n_free, merged.ndof)`.
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
        UsmBuffer<int> dof_list;         // group-local DOF indices, length = ndof of partition
        UsmBuffer<real> soa_records;   // packed SoA records, length = ndof * kFields (0 if none)
        UsmBuffer<real> seg_data[kMaxSoASeg];  // CSR data per segmented slot (empty if kSeg==0)
        UsmBuffer<int> seg_off[kMaxSoASeg];      // CSR offsets per segmented slot (empty if kSeg==0)
    };

    struct DeviceGroup {
        std::size_t ndof;
        std::size_t n_free = 0;  // count of Free DOFs (the [0, n_free) prefix)
        std::size_t total_samples;
        // Per-occurrence only (the bulk gc/gn/s/W_inv/J payload is deduplicated
        // into the context-level shared MetricTable, referenced by sample_id).
        UsmBuffer<int> sample_id, role;
        UsmBuffer<int> dof_idx, P_of, sample_offset;
        UsmBuffer<int> part_of, row_of;  // [ndof] DOF -> (partition, row)
        UsmBuffer<int> interior_block, interior_logical;  // [ndof], [ndof*D]; empty if unstructured
        std::vector<DevicePartition> partitions;

        GroupViewT<D> view(const PartitionView* pvs, std::size_t np, const MetricTableViewT<D>& tbl,
                           const int* block_off, const int* nstride) const
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
            return gv;
        }
    };

    struct DeviceStencil {
        std::size_t num_samples;
        UsmBuffer<int> gc;
        UsmBuffer<int> gn[D];
        UsmBuffer<real> s[D];
        UsmBuffer<real> W_inv;

        StencilViewT<D> view() const
        {
            StencilViewT<D> sv;
            sv.num_samples = num_samples;
            sv.gc = View1<const int> {gc.data(), num_samples};
            for (int k = 0; k < D; ++k) {
                sv.gn[k] = View1<const int> {gn[k].data(), num_samples};
                sv.s[k] = View1<const real> {s[k].data(), num_samples};
            }
            sv.W_inv = View2<const real> {W_inv.data(), num_samples, dim::wInv(D)};
            return sv;
        }
    };

    /// Concatenate every colour group's per-occurrence arrays into one GroupViewT
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
            for (const auto& pv : host_pvs_[gi]) { merged_pvs.push_back(pv); }

            sbase += gs;
            dbase += gd;
            pbase += host_pvs_[gi].size();
        }
        q_.wait();

        // Global Free-first reorder across all merged DOFs (each group is already
        // Free-first internally, but the concatenation interleaves the classes).
        // The single block-Jacobi launch is then split at n_free into the lean
        // interior kernel and the full boundary kernel. Plan #1 step 2.
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

        // Per-partition reordered-position lists for the boundary tail (§4.7 / S1).
        // Bucket every boundary position [merged_n_free, total_ndof) by its merged
        // partition; each non-empty bucket becomes one monomorphic launch. The
        // interior prefix [0, merged_n_free) stays a single entity-agnostic launch
        // and is intentionally not bucketed. Reserve up front so the position
        // UsmBuffers never reallocate (BoundaryLaunch::positions holds raw pointers
        // into them; a vector reallocation would move-construct each buffer — the
        // device pointer survives a move, but reserving keeps the invariant simple).
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
            // Every boundary position must belong to a non-Free partition (the
            // Free DOFs are exactly the [0, merged_n_free) prefix). This is the S1
            // unit-check that the reordered positions ↔ partition mapping holds.
            assert(merged_pvs[p].tag != EntityTag::Free);
            bnd_total += buckets[p].size();
            m_part_positions_.emplace_back(q_, buckets[p]);
            boundary_launches_.push_back(BoundaryLaunch {
              .tag = merged_pvs[p].tag,
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
        merged_view_ = gv;

        // The merged view now owns its own concatenated per-occurrence arrays
        // (m_sample_id_/m_role_) and per-DOF index arrays. The per-group
        // `groups_[*]` copies they were built from are dead duplicates from here on.
        // Free them. (The shared table_ is NOT freed — it is the live payload for
        // both the merged view and any per-group view.)
        //
        // This commits the instance to the block-Jacobi path: colored-GS reads
        // `group_views()`, whose View1s alias exactly these per-group buffers, so
        // it must not run afterward. `group_views_` is cleared (so the dangling
        // views are unreachable rather than silently reading freed memory) and the
        // release is flagged; `run_colored_gs` asserts the flag is unset. In
        // practice each smoother gets a fresh executor (see bindings.cpp), so the
        // two never share an instance. NOT freed: `partitions` (entity SoA, aliased
        // by the merged PartitionViews) and `stencil_`/`stencil_view_` (read by the
        // energy reduction on both paths).
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
        group_views_.clear();
        per_group_released_ = true;
    }

    sycl::queue q_;
    std::size_t num_nodes_;
    UsmBuffer<real> X_;
    UsmBuffer<real> seeds_;  // [num_nodes * kSeedStride] warm-start param cache
    // Shared node-unit block layout for interior patch synthesis (empty on the
    // unstructured path). Referenced by every group view + the merged view.
    UsmBuffer<int> block_off_, nstride_;
    std::vector<DeviceGroup> groups_;
    std::vector<UsmBuffer<PartitionView>> group_partitions_;  // device-resident PartitionView arrays
    DeviceStencil stencil_;
    std::vector<GroupViewT<D>> group_views_;
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
        t.W_inv = View1<const real> {table_.W_inv.data(),
                                     table_.w_uniform ? wInv : (table_.n * wInv)};
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
        std::vector<std::int8_t> t_s[D];  // s is ±1; store as int8 (key still hashes the real bytes)
        std::vector<real> t_W;
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
            put(&g.W_inv[p * wInv], wInv * sizeof(real));

            const auto [it, fresh] = seen.try_emplace(key, static_cast<int>(t_gc.size()));
            if (!fresh) { return it->second; }

            t_gc.push_back(g.gc[p]);
            for (int k = 0; k < D; ++k) {
                t_gn[k].push_back(g.gn[k][p]);
                t_s[k].push_back(static_cast<std::int8_t>(g.s[k][p]));
            }
            for (std::size_t e = 0; e < wInv; ++e) { t_W.push_back(g.W_inv[(p * wInv) + e]); }
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

        table_.n = t_gc.size();
        table_.gc = {q_, t_gc};
        for (int k = 0; k < D; ++k) {
            table_.gn[k] = {q_, t_gn[k]};
            table_.s[k] = {q_, t_s[k]};
        }
        table_.W_inv = {q_, t_W};
        return sid;
    }

    // Merged single-launch view for block-Jacobi (lazily built, then cached).
    std::vector<std::vector<PartitionView>> host_pvs_;  // per-group PartitionView (host copies)
    bool merged_built_ = false;
    bool per_group_released_ = false;  // per-group per-occurrence buffers freed after the merge
    GroupViewT<D> merged_view_ {};
    UsmBuffer<int> m_sample_id_, m_role_;
    UsmBuffer<int> m_dof_idx_, m_P_of_, m_sample_offset_, m_part_of_, m_row_of_;
    UsmBuffer<int> m_interior_block_, m_interior_logical_;  // structured only; empty otherwise
    UsmBuffer<PartitionView> m_partitions_;

    // Per-partition boundary launch descriptors (§4.7 / S1). m_part_positions_
    // owns the device position arrays; boundary_launches_ holds raw pointers into
    // them (kept stable by the reserve in build_merged_view).
    std::vector<UsmBuffer<int>> m_part_positions_;
    std::vector<BoundaryLaunch> boundary_launches_;
};

/// @brief Single per-colour sweep kernel (backup architecture, SoA-backed).
///
/// One kernel per colour over the whole group's @c ndof — not one per
/// (colour, EntityTag) partition. The heavy body (patch eval, backtracking,
/// energy/min-det assembly) is **entity-agnostic** and compiled exactly once;
/// the only entity-specific steps — the tangent-reduced Newton delta and the
/// per-trial projection — are isolated behind @ref with_entity, a single cheap
/// `switch` over the DOF's tag that loads the concrete entity from its SoA slot
/// and runs a *small* callable. This keeps launches ∝ colours (no per-partition
/// launch explosion) while never inlining all entity variants into the body
/// (no `std::visit` occupancy collapse). Race-free within a colour: distinct
/// DOFs are distinct nodes, so the scatter never collides.
/// The kernel supports both in-place colored-GS and double-buffered block-Jacobi
/// via @p X_out: reads (patch eval, trial-loop node substitution, current
/// position) always use @p X; the final scatter writes @p X_out. When @p X_out is
/// null the write is in-place (X_out == X) — the colored-GS behaviour. For
/// block-Jacobi @p X is the frozen read buffer and @p X_out the write buffer, so
/// every free DOF updates from the same snapshot (no intra-sweep dependency). The
/// relaxation weight @p omega damps the simultaneous Jacobi update
/// (`cur = pos + ω·(cur − pos)`); ω=1 is the undamped step (and exact for
/// colored-GS).
///
/// @tparam D Embedding dimension.
/// @tparam Obj Objective type.
/// @param q SYCL queue.
/// @param g The colour group view (per-DOF patch arrays + partition list + map).
/// @param X Global node positions to read (device USM).
/// @param objective The objective functor.
/// @param X_out Buffer to scatter into (null → in-place, X_out == X).
/// @param omega Relaxation weight (1.0 = undamped).
/// @brief Project @p p onto @p ent, warm-starting from this DOF's parameter
///        cache when the entity supports it (the iterative B-spline curve /
///        surface). @p seed_slot points at the (D-1)-real per-node seed
///        (non-finite ⇒ cold); on return it holds the converged foot parameter
///        so the next sweep skips the expensive coarse-grid seed. Closed-form
///        entities have no @c project_seeded and project cold (seed untouched).
template <int D, bool Warm, class E>
inline PtN<D> project_with_seed(const E& ent, const PtN<D>& p, real* seed_slot)
{
    constexpr int K = E::tdim;
    if constexpr (requires(Param<K> s) { ent.project_seeded(p, s, true); }) {
        Param<K> seed {};
        const bool has = (seed_slot != nullptr) && std::isfinite(seed_slot[0]);
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

/// @brief Tangent-reduced Newton step that warm-starts the tangent basis from
///        this DOF's parameter cache when the entity supports it (the iterative
///        B-spline curve/surface) — so the Newton step's projection skips the
///        cold coarse-grid solve every sweep, just like @ref project_with_seed
///        does for the line-search projection. Closed-form / Line3 entities (no
///        @c tangent_basis_seeded) fall back to the cold @ref newton_delta; the
///        interior (k==D) case never reaches here (handled by the FreeOnly path).
template <int D, bool Warm, class E>
inline VecN<D> newton_delta_with_seed(
  const VecN<D>& g, const MatN<D>& H, const PtN<D>& pos, const E& ent, real* seed_slot)
{
    constexpr int K = E::tdim;
    if constexpr (requires(Param<K> s) { ent.tangent_basis_seeded(pos, s, true); }) {
        Param<K> seed {};
        const bool has = (seed_slot != nullptr) && std::isfinite(seed_slot[0]);
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

/// @param seeds Optional per-node warm-start parameter cache, stride (D-1),
///        indexed by global node id. Null disables warm starting (cold
///        projection, e.g. the colored-GS path). Persisted across sweeps by the
///        owning context so the iterative projections converge in 1–2 Newton
///        steps from the previous foot instead of the coarse-grid seed.
///
/// @tparam FreeOnly When true the launched view is guaranteed to contain only
///        `EntityTag::Free` DOFs (interior), so all entity/projection/tangent
///        dispatch is statically elided: the Newton step is the plain `solveNxN`
///        and the line-search trial is the raw step (no projection). This is the
///        lean interior kernel of the Plan #1 interior/boundary split — it omits
///        the `with_entity` call graph entirely, removing the geometry spills.
///        `false` (default) is the full geometry-aware kernel (every entity type,
///        correctness preserved), launched over the boundary DOFs.
template <int D, class Obj, bool FreeOnly = false, bool Warm = false>
inline void sweep_colour_kernel(sycl::queue& q, const GroupViewT<D>& g, real* X, Obj objective,
                                real* X_out = nullptr, real omega = 1.0_r, real* seeds = nullptr)
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

        // 2. Newton step. Interior (FreeOnly) DOFs take the plain d×d solve — no
        //    entity geometry, so the whole `with_entity` dispatch is elided.
        //    Boundary DOFs take the tangent-reduced step via the scoped tag
        //    switch (the only place the Newton math touches entity geometry).
        VecN<D> delta {};
        if constexpr (FreeOnly) {
            delta = solveNxN<D>(r.hess, r.grad);
        } else {
            // The DOF's entity lives in partition part_of[d] at SoA row row_of[d].
            // Warm-start the Newton-step tangent from this DOF's seed cache (same
            // slot the line-search projection uses below), so the iterative
            // B-spline tangent skips the cold coarse-grid solve every sweep.
            const PartitionView& part = parts[g.part_of[d]];
            const auto row = static_cast<std::size_t>(g.row_of[d]);
            real* seed_slot =
              (seeds != nullptr) ? (seeds + (static_cast<std::size_t>(dof) * kSeedStride)) : nullptr;
            with_entity<D>(part.tag, part.soa_view, part.seg, row, [&](const auto& ent) {
                delta = newton_delta_with_seed<D, Warm>(r.grad, r.hess, pos, ent, seed_slot);
            });
        }

        // 3. Backtracking: halve alpha up to 10×; accept on
        //    finite(e) && e <= e0 + 1e-12 && objective.accept_mindet(mindet).
        real alpha = 1.0_r;
        PtN<D> cur = pos;
        bool accepted = false;
        for (int it = 0; it < 10 && !accepted; ++it) {
            PtN<D> raw;
            for (int k = 0; k < D; ++k) { raw[k] = pos[k] + alpha * delta[k]; }
            // Interior (FreeOnly) DOFs project to identity → trial = raw, no
            // dispatch. Constrained DOFs project via the same scoped tag switch
            // as the Newton step above.
            PtN<D> trial = raw;
            if constexpr (!FreeOnly) {
                const PartitionView& part = parts[g.part_of[d]];
                const EntityTag tag = part.tag;
                const auto row = static_cast<std::size_t>(g.row_of[d]);
                if (tag != EntityTag::Free) {
                    real* seed_slot =
                      (seeds != nullptr) ? (seeds + (static_cast<std::size_t>(dof) * kSeedStride))
                                         : nullptr;
                    with_entity<D>(tag, part.soa_view, part.seg, row, [&](const auto& ent) {
                        trial = project_with_seed<D, Warm>(ent, raw, seed_slot);
                    });
                }
            }

            real e_new = 0.0_r;
            real mdet = std::numeric_limits<real>::infinity();
            for (int p = 0; p < pv.P; ++p) {
                // Substitute the trial position for the moving DOF; all other
                // nodes load from X. The A→T→detA math is the single source in
                // patch.hpp::assemble_vecT.
                auto node = [&](int ni) -> PtN<D> {
                    return ni == dof ? trial : load_pt<D>(X, ni);
                };
                const int sid = pv.sample_id[p];  // shared-table index of occurrence p
                PtN<D> corner = node(pv.gc[sid]);
                std::array<PtN<D>, D> nbr;
                std::array<real, D> sc;
                for (int k = 0; k < D; ++k) {
                    nbr[k] = node(pv.gn[k][sid]);
                    sc[k] = pv.s[k][sid];
                }
                real detA;
                const VecTN<D> t =
                  assemble_vecT<D>(corner, nbr, sc, &pv.W_inv[pv.w_stride * sid], detA);
                e_new += objective.value(t);
                mdet = std::min(detA, mdet);
            }

            const bool ok = std::isfinite(e_new) && (e_new <= r.energy + tol::energy) &&
                            objective.accept_mindet(mdet);
            if (ok) {
                cur = trial;
                accepted = true;
            } else {
                alpha *= 0.5_r;
            }
        }

        // 4. Optional Jacobi relaxation weighting: cur = pos + ω·(cur − pos).
        //    ω=1 leaves cur untouched (colored-GS / undamped Jacobi).
        if (omega != 1.0_r) {
            for (int k = 0; k < D; ++k) { cur[k] = pos[k] + (omega * (cur[k] - pos[k])); }
        }

        // 5. Scatter. In-place (colored-GS) writes X; double-buffered Jacobi
        //    writes the separate X_out snapshot. Race-free: distinct DOFs are
        //    distinct nodes, so the scatter never collides.
        store_pt<D>(out, dof, cur);
    });
}

/// @brief Kernel B — metric eval (§4.3). For each DOF in @p g, run `patch_eval`
///        and scatter (grad, hess, energy) into the per-node USM buffers, keyed by
///        **global node id** (`dof_idx[d]`, §5) so the same buffers serve both the
///        interior subrange launch and the per-partition boundary launches.
///
/// This is exactly the metric half of @ref sweep_colour_kernel hoisted into its
/// own kernel: bit-identical `patch_eval`, no Newton step, no projection, no
/// `with_entity`. Splitting it out lets the heavy metric grad+jhj working set live
/// in a kernel of its own (interior 144 → ~88 metric here + ~80 solve in Kernel C,
/// both under the gfx1030 128-VGPR 2-wave tier). Entity-agnostic: one launch over
/// the whole subrange regardless of entity type.
///
/// @param grad_buf  [num_nodes * D]     dE/dpos per node.
/// @param hess_buf  [num_nodes * D*D]   contracted Hessian per node (row-major).
/// @param e0_buf    [num_nodes]         baseline patch energy per node.
template <int D, class Obj>
inline void metric_kernel(sycl::queue& q, const GroupViewT<D>& g, const real* X, real* grad_buf,
                          real* hess_buf, real* e0_buf, Obj objective)
{
    if (g.ndof == 0) { return; }
    q.parallel_for(sycl::range<1>(g.ndof), [=](sycl::id<1> idx) {
        const std::size_t d = idx[0];
        const PatchViewT<D> pv = g.patch(d);
        const int dof = g.dof_idx[d];
        const PatchResultT<D> r = patch_eval<D>(pv, X, objective);
        const std::size_t base = static_cast<std::size_t>(dof);
        real* gslot = grad_buf + (base * D);
        for (int k = 0; k < D; ++k) { gslot[k] = r.grad[k]; }
        real* hslot = hess_buf + (base * D * D);
        for (int k = 0; k < D * D; ++k) { hslot[k] = r.hess[k]; }
        e0_buf[base] = r.energy;
    });
}

/// @brief Kernel C (interior / `FreeOnly`) — Newton solve + backtracking + scatter
///        (§4.4, §4.5), reading the metric (grad, hess, e0) precomputed by
///        @ref metric_kernel from the per-node buffers. No entity, no projection:
///        the interior Newton step is the plain `solveNxN<D>` and the line-search
///        trial is the raw step — structurally the existing interior kernel with
///        the metric hoisted out. The trial energy recompute reads neighbour
///        positions from @p X (the frozen snapshot for block-Jacobi).
///
/// @param X      Read buffer (block-Jacobi: frozen x_in; colored-GS: in-place X).
/// @param X_out  Write buffer (null → in-place, X_out == X).
/// @param omega  Relaxation weight (1.0 = undamped).
template <int D, class Obj>
inline void interior_update_kernel(sycl::queue& q, const GroupViewT<D>& g, real* X, real* X_out,
                                   real omega, const real* grad_buf, const real* hess_buf,
                                   const real* e0_buf, Obj objective)
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
        const real e0 = e0_buf[base];

        // Interior DOF: plain d×d Newton solve (no entity geometry).
        const VecN<D> delta = solveNxN<D>(hess, grad);

        // The trial-energy recompute reads the patch the same way the metric did:
        // synthesized from the block layout for interior-eligible DOFs (no stored
        // gc/gn/s reads), or from the stored arrays otherwise.
        int block;
        int logical[D];
        const bool synth = g.interior(d, block, logical);
        const PatchViewT<D> pv = synth ? PatchViewT<D> {} : g.patch(d);

        // Backtracking: halve alpha up to 10×; trial == raw (interior projects to
        // identity). Accept on finite(e) && e <= e0 + tol && accept_mindet.
        real alpha = 1.0_r;
        PtN<D> cur = pos;
        bool accepted = false;
        for (int it = 0; it < 10 && !accepted; ++it) {
            PtN<D> trial;
            for (int k = 0; k < D; ++k) { trial[k] = pos[k] + (alpha * delta[k]); }
            real e_new = 0.0_r;
            real mdet = std::numeric_limits<real>::infinity();
            if (synth) {
                synth_trial_energy_mindet<D>(g.block_off, g.nstride, block, logical, X, dof, trial,
                                             objective, e_new, mdet);
            } else {
                for (int p = 0; p < pv.P; ++p) {
                    auto node = [&](int ni) -> PtN<D> {
                        return ni == dof ? trial : load_pt<D>(X, ni);
                    };
                    const int sid = pv.sample_id[p];  // shared-table index of occurrence p
                    PtN<D> corner = node(pv.gc[sid]);
                    std::array<PtN<D>, D> nbr;
                    std::array<real, D> sc;
                    for (int k = 0; k < D; ++k) {
                        nbr[k] = node(pv.gn[k][sid]);
                        sc[k] = pv.s[k][sid];
                    }
                    real detA;
                    const VecTN<D> t =
                      assemble_vecT<D>(corner, nbr, sc, &pv.W_inv[pv.w_stride * sid], detA);
                    e_new += objective.value(t);
                    mdet = std::min(detA, mdet);
                }
            }
            const bool ok = std::isfinite(e_new) && (e_new <= e0 + tol::energy) &&
                            objective.accept_mindet(mdet);
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

/// @brief Plain SPD solve `M x = b` (K = 1 or 2), no Newton negation/fallback.
///        Used for the param-space reduction `Δq = (JᵀJ)⁻¹ Jᵀδ` (§4.1), where
///        `JᵀJ` is SPD; a near-singular Gram matrix (degenerate frame at a trim
///        corner) returns 0 so the line search simply does not move that sweep.
template <int K> inline VecN<K> spd_solve(const MatN<K>& M, const VecN<K>& b)
{
    if constexpr (K == 1) {
        return VecN<1> {(std::fabs(M[0]) > tol::tiny) ? (b[0] / M[0]) : 0.0_r};
    } else {
        const real det = (M[0] * M[3]) - (M[1] * M[2]);
        if (std::fabs(det) < tol::tiny) { return VecN<2> {0.0_r, 0.0_r}; }
        const real inv = 1.0_r / det;
        return VecN<2> {inv * ((M[3] * b[0]) - (M[1] * b[1])),
                        inv * ((-M[2] * b[0]) + (M[0] * b[1]))};
    }
}

/// @brief The converged foot parameter `q*` of @p p on @p param, warm-started from
///        @p seed when the parametrization is iterative (B-spline `invert_seeded`),
///        else the closed-form `invert`. Mirrors @ref project_with_seed but returns
///        the **parameter** (not the projected point) — the param-space LS needs
///        `q*` itself to backtrack in (u,v).
template <int K, class P>
inline Param<K> foot_param_seeded(const P& param, const PtN<P::edim>& p, const Param<K>& seed,
                                  bool has_seed)
{
    if constexpr (requires { param.template invert_seeded<true>(p, seed, has_seed); }) {
        return param.template invert_seeded<true>(p, seed, has_seed);
    } else {
        static_cast<void>(seed);
        static_cast<void>(has_seed);
        return param.invert(p);
    }
}

/// @brief The converged foot `q*` **and** the raw tangent frame `J` at `q*`.
///
/// For an iterative B-spline foot, the frame is harvested from the last
/// `newton_foot` `ders(nd=1)` (F2 Lever C) — saving the redundant `frame(q*)`
/// re-evaluation in the param-LS kernel. For closed-form entities (no warm
/// `invert_seeded` frame out-param) it falls back to `invert` + `frame(q*)`.
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

/// @brief Warm boundary sweep kernel with **param-space line search** (§4.1 / S3).
///
/// Replaces the per-trial 3D reprojection of @ref sweep_colour_kernel's warm
/// boundary path with: project ONCE to the foot parameter `q*`, reduce the
/// tangent-space Newton step `δ_world = B y` to a parametric step
/// `Δq = (JᵀJ)⁻¹ Jᵀ δ_world` (J = raw frame at `q*`, B = orthonormalize(J)), then
/// backtrack `q_trial = trim.clamp(q* + αΔq)` with a **value-only** `param.eval`
/// (de Boor nd=0). The heavy de Boor **derivative** rows (nd=1) are evaluated once
/// (in `frame`/`invert_seeded`) instead of ≤10× per DOF in the backtracking loop.
///
/// Numerically this preserves the constraint exactly (every trial lands on the
/// geometry via `eval`) and the Δq reduction is exact (`δ_world ∈ span(J)`); the
/// only difference vs world-space LS is curvature at finite α — gated by the
/// energy/sph_dev parity check (§7 step 3). Block-Jacobi warm path only (the cold
/// s==0 pass stays the world-space @ref sweep_colour_kernel that seeds `q*`).
template <int D, class Obj>
inline void sweep_boundary_paramls_kernel(sycl::queue& q, const GroupViewT<D>& g, real* X,
                                          Obj objective, real* X_out, real omega, real* seeds)
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
                const bool has = (seed_slot != nullptr) && std::isfinite(seed_slot[0]);
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
                const VecN<K> dq = spd_solve<K>(JtJ, Jtd);

                // 5. Param-space backtracking: q_trial = clamp(q* + αΔq), trial =
                //    eval(q_trial) — a value-only (nd=0) de Boor, no reprojection.
                real alpha = 1.0_r;
                bool accepted = false;
                for (int it = 0; it < 10 && !accepted; ++it) {
                    Param<K> qt;
                    for (int a = 0; a < K; ++a) { qt[a] = qstar[a] + (alpha * dq[a]); }
                    qt = ent.trim.clamp(qt);
                    const PtN<D> trial = ent.param.eval(qt);
                    real e_new = 0.0_r;
                    real mdet = std::numeric_limits<real>::infinity();
                    for (int p = 0; p < pv.P; ++p) {
                        auto node = [&](int ni) -> PtN<D> {
                            return ni == dof ? trial : load_pt<D>(X, ni);
                        };
                        const int sid = pv.sample_id[p];  // shared-table index
                        PtN<D> corner = node(pv.gc[sid]);
                        std::array<PtN<D>, D> nbr;
                        std::array<real, D> sc;
                        for (int k = 0; k < D; ++k) {
                            nbr[k] = node(pv.gn[k][sid]);
                            sc[k] = pv.s[k][sid];
                        }
                        real detA;
                        const VecTN<D> t =
                          assemble_vecT<D>(corner, nbr, sc, &pv.W_inv[pv.w_stride * sid], detA);
                        e_new += objective.value(t);
                        mdet = std::min(detA, mdet);
                    }
                    const bool ok = std::isfinite(e_new) && (e_new <= r.energy + tol::energy) &&
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
                const VecN<D> delta =
                  newton_delta_with_seed<D, true>(r.grad, r.hess, pos, ent, seed_slot);
                real alpha = 1.0_r;
                bool accepted = false;
                for (int it = 0; it < 10 && !accepted; ++it) {
                    PtN<D> raw;
                    for (int k = 0; k < D; ++k) { raw[k] = pos[k] + (alpha * delta[k]); }
                    const PtN<D> trial = project_with_seed<D, true>(ent, raw, seed_slot);
                    real e_new = 0.0_r;
                    real mdet = std::numeric_limits<real>::infinity();
                    for (int p = 0; p < pv.P; ++p) {
                        auto node = [&](int ni) -> PtN<D> {
                            return ni == dof ? trial : load_pt<D>(X, ni);
                        };
                        const int sid = pv.sample_id[p];  // shared-table index
                        PtN<D> corner = node(pv.gc[sid]);
                        std::array<PtN<D>, D> nbr;
                        std::array<real, D> sc;
                        for (int k = 0; k < D; ++k) {
                            nbr[k] = node(pv.gn[k][sid]);
                            sc[k] = pv.s[k][sid];
                        }
                        real detA;
                        const VecTN<D> t = assemble_vecT<D>(
                          corner, nbr, sc, &pv.W_inv[pv.w_stride * sid], detA);
                        e_new += objective.value(t);
                        mdet = std::min(detA, mdet);
                    }
                    const bool ok = std::isfinite(e_new) && (e_new <= r.energy + tol::energy) &&
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
        if (omega != 1.0_r) {
            for (int k = 0; k < D; ++k) { cur[k] = pos[k] + (omega * (cur[k] - pos[k])); }
        }
        store_pt<D>(out, dof, cur);
    });
}

// Per-sweep total energy (Σ μ) and min det A over the energy stencil, written to
// out_e/out_m (USM scalars). Reduction identities are supplied explicitly, so no
// host pre-initialisation of the targets is needed.
template <int D, class Obj>
inline void reduce_energy_mindet(sycl::queue& q,
                                 const StencilViewT<D>& es,
                                 const real* X,
                                 real* out_e,
                                 real* out_m,
                                 Obj objective)
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
                       // W_inv is [num_samples][wInv] row-major; row i is contiguous.
                       const real* w = es.W_inv.data_handle() + i * dim::wInv(D);
                       PtN<D> corner = load_pt<D>(X, es.gc[i]);
                       std::array<PtN<D>, D> nbr;
                       std::array<real, D> sc;
                       for (int k = 0; k < D; ++k) {
                           nbr[k] = load_pt<D>(X, es.gn[k][i]);
                           sc[k] = es.s[k][i];
                       }
                       real detA;
                       const VecTN<D> t = assemble_vecT<D>(corner, nbr, sc, w, detA);
                       e_sum += objective.value(t);
                       m_min.combine(detA);
                   });
}

/// @brief Shared colored Gauss-Seidel driver: @c n_sweeps of (pre-sweep hook →
///        per-colour kernels → energy/min-det reduction) on the in-order queue.
///
/// Both the unstructured @ref ExecutorT and the structured @ref
/// StructuredExecutorT compose this; the only difference between them is the
/// @c before_sweep hook, called as `before_sweep(q, X)` before each sweep's
/// colour kernels. The unstructured path passes a no-op; the structured path
/// passes a per-sweep @ref halo_exchange (cadence 1.4b). Colours are serialized
/// by the in-order queue, and the race-free colouring guarantees no DOF reads a
/// same-colour mate's X within a sweep.
///
/// @p report_every controls how often the energy/min-det reduction is run:
///   - `1` (default, legacy)  : every sweep  — returns `n_sweeps` pairs.
///   - `k > 1`                : every `k`-th sweep, plus always the final
///                                sweep of the run               — returns `ceil(n/k)` pairs.
///   - `<= 0`                 : only the final sweep of this run  — returns `1` pair.
/// The returned vectors carry exactly the reported reductions (length matches
/// the number of reductions actually performed, not `n_sweeps`).
template <int D, class Obj, class PreSweep>
inline std::pair<std::vector<real>, std::vector<real>>
  run_colored_gs(sycl::queue& q, SweepDeviceContextT<D>& ctx, Obj objective, int n_sweeps,
                 PreSweep before_sweep, int report_every = 1)
{
    const int k = (report_every <= 0) ? n_sweeps : report_every;
    const std::size_t report_count =
      static_cast<std::size_t>((n_sweeps + k - 1) / k);
    UsmBuffer<real> d_e(q, report_count);
    UsmBuffer<real> d_m(q, report_count);
    // Colored-GS reads ctx.group_views(), which alias the per-group per-sample
    // buffers. Building the merged block-Jacobi view frees those, so colored-GS
    // must not follow block-Jacobi on the same executor instance.
    assert(!ctx.per_group_released() &&
           "colored-GS cannot run after block-Jacobi on the same executor (group_views_ was freed)");
    real* X = ctx.X();

    // Per-node metric scratch for the interior B/C split (§4.5), keyed by global
    // node id and reused across colours/sweeps (Kernel B overwrites before Kernel
    // C reads). Only the interior split is mirrored to colored-GS — the boundary
    // monomorphization stays block-Jacobi-only (§6: per-colour×per-type launches
    // are the documented regression).
    const std::size_t nn = ctx.num_nodes();
    UsmBuffer<real> grad_buf(q, nn * D);
    UsmBuffer<real> hess_buf(q, nn * D * D);
    UsmBuffer<real> e0_buf(q, nn);

    constexpr int kSyncEvery = 32;
    std::size_t report_idx = 0;
    for (int s = 0; s < n_sweeps; ++s) {
        before_sweep(q, X);
        // Per colour: interior B/C split over [0, n_free) then the full boundary
        // launch over the tail. Within a colour all DOFs are distinct nodes, so
        // the launches over disjoint subsets are race-free (serialized by the
        // in-order queue). Plan #1 step 4 + fission §4.5.
        for (const auto& gv : ctx.group_views()) {
            const GroupViewT<D> interior = gv.dof_subrange(0, gv.n_free);
            metric_kernel<D, Obj>(q, interior, X, grad_buf.data(), hess_buf.data(), e0_buf.data(),
                                  objective);
            interior_update_kernel<D, Obj>(q, interior, X, nullptr, 1.0_r, grad_buf.data(),
                                           hess_buf.data(), e0_buf.data(), objective);
            sweep_colour_kernel<D, Obj, false>(
              q, gv.dof_subrange(gv.n_free, gv.ndof - gv.n_free), X, objective);
        }
        if ((s + 1) % k == 0 || s == n_sweeps - 1) {
            reduce_energy_mindet<D>(q, ctx.stencil_view(), X,
                                    d_e.data() + report_idx, d_m.data() + report_idx, objective);
            ++report_idx;
        }
        if ((s + 1) % kSyncEvery == 0) { q.wait(); }
    }
    q.wait();
    assert(report_idx == report_count);

    std::vector<real> energies(report_count), mindets(report_count);
    d_e.download(energies.data());
    d_m.download(mindets.data());
    return {energies, mindets};
}

/// @brief Double-buffered block-Jacobi driver: @c n_sweeps of (pre-sweep hook →
///        one merged Jacobi launch reading X_in, writing X_new → energy/min-det
///        reduction on X_new → swap).
///
/// The structured analogue of @ref run_colored_gs, but instead of the ~14-deep
/// per-colour dependency chain it issues a **single** launch over the merged
/// group view (@ref SweepDeviceContextT::merged_group_view): under the frozen-halo
/// cadence every free DOF reads the previous snapshot, so there is no colour
/// ordering and no intra-sweep dependency. @p before_sweep is the same per-sweep
/// halo hook the structured colored-GS path uses (halo_exchange + broadcast on the
/// read buffer). @p omega is the SOR/damping weight forwarded to the kernel.
///
/// A second USM buffer (X_new) is allocated once and initialised to a copy of X;
/// each sweep writes every free DOF, fixed nodes stay constant, and ghosts are
/// re-exchanged, so no per-sweep full copy is needed. After the loop the canonical
/// ctx buffer is restored if an odd number of swaps left the result in X_new.
template <int D, class Obj, class PreSweep>
inline std::pair<std::vector<real>, std::vector<real>>
  run_block_jacobi(sycl::queue& q, SweepDeviceContextT<D>& ctx, Obj objective, int n_sweeps,
                   PreSweep before_sweep, real omega = 1.0_r, int report_every = 1)
{
    const int k = (report_every <= 0) ? n_sweeps : report_every;
    const std::size_t report_count =
      static_cast<std::size_t>((n_sweeps + k - 1) / k);
    UsmBuffer<real> d_e(q, report_count);
    UsmBuffer<real> d_m(q, report_count);
    const std::size_t nbuf = ctx.x_size();
    UsmBuffer<real> x_new_buf(q, nbuf);

    // Per-node metric scratch for the interior B/C split (§5). Sized to num_nodes
    // and keyed by global node id so the same buffers serve the boundary launches
    // too; written by Kernel B before Kernel C reads them every sweep, so no
    // per-sweep memset is needed (stale slots are never read).
    const std::size_t nn = ctx.num_nodes();
    UsmBuffer<real> grad_buf(q, nn * D);
    UsmBuffer<real> hess_buf(q, nn * D * D);
    UsmBuffer<real> e0_buf(q, nn);

    real* canonical = ctx.X();
    real* x_in = canonical;
    real* x_out = x_new_buf.data();
    q.memcpy(x_out, x_in, nbuf * sizeof(real)).wait();  // X_new := copy of X

    const GroupViewT<D>& mg = ctx.merged_group_view();

    constexpr int kSyncEvery = 32;
    std::size_t report_idx = 0;
    for (int s = 0; s < n_sweeps; ++s) {
        before_sweep(q, x_in);  // refresh ghosts + shared-node copies on the read buffer
        // Interior B/C split (§4.5): Kernel B writes (grad, hess, e0) per node,
        // Kernel C reads them back and does the Newton solve + backtracking. Both
        // run over [0, n_free) and read the frozen x_in; C writes disjoint x_out
        // slots, so block-Jacobi correctness is preserved. The metric and solve
        // working sets now live in separate kernels (144 → ~88 + ~80 VGPR).
        const GroupViewT<D> interior = mg.dof_subrange(0, mg.n_free);
        metric_kernel<D, Obj>(q, interior, x_in, grad_buf.data(), hess_buf.data(), e0_buf.data(),
                              objective);
        interior_update_kernel<D, Obj>(q, interior, x_in, x_out, omega, grad_buf.data(),
                                       hess_buf.data(), e0_buf.data(), objective);
        // Boundary cold/warm split: the first sweep runs the robust cold kernel
        // (8×8 grid + exact Newton nd=2) to project the topologically-placed nodes
        // onto the geometry and populate the warm-seed cache; every subsequent
        // sweep runs the lean warm kernel (GN nd=1, no grid / no nd=2 → the heavy
        // de Boor second-order rows stay out of the hot kernel). NB: per-entity
        // monomorphization of this boundary kernel was measured to REGRESS it (the
        // BSpline arm alone is 208 VGPR vs the union's 184 — the runtime switch is
        // actually more register-efficient than the isolated heavy arm — plus a
        // per-colour launch explosion), so the boundary stays one union kernel.
        const GroupViewT<D> bndry = mg.dof_subrange(mg.n_free, mg.ndof - mg.n_free);
        if (s == 0) {
            sweep_colour_kernel<D, Obj, false, false>(q, bndry, x_in, objective, x_out, omega,
                                                      ctx.seeds());
        } else {
            // Warm path: param-space line search (§4.1 / S3) — project once to the
            // foot q*, backtrack in (u,v) with value-only de Boor instead of
            // reprojecting every trial.
            sweep_boundary_paramls_kernel<D, Obj>(q, bndry, x_in, objective, x_out, omega,
                                                  ctx.seeds());
        }
        if ((s + 1) % k == 0 || s == n_sweeps - 1) {
            reduce_energy_mindet<D>(q, ctx.stencil_view(), x_out,
                                    d_e.data() + report_idx, d_m.data() + report_idx,
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

template <int D> class ExecutorT
{
  public:
    ExecutorT(const sycl::queue& queue, const SweepContextHostT<D>& host) :
        q_(sycl::queue(queue.get_context(), queue.get_device(),
          {sycl::property::queue::in_order(),
           sycl::property::queue::AdaptiveCpp_coarse_grained_events{}})),
        ctx_(q_, host)
    {
    }

    SweepDeviceContextT<D>& ctx() { return ctx_; }

// Run n_sweeps, dispatching the objective kind once via std::visit.
    // @p report_every throttles the energy/min-det reduction cadence
    // (see @ref run_colored_gs); default preserves the legacy per-sweep contract.
    std::pair<std::vector<real>, std::vector<real>>
      run_sweeps(int n_sweeps, const ObjectiveKindT<D>& kind = ShapeObjectiveT<D> {},
                 int report_every = 1)
    {
        return std::visit(
          [&](auto objective) {
              // Unstructured path: no halo exchange (one global X, not per-block).
              return run_colored_gs<D>(q_, ctx_, objective, n_sweeps,
                                        [](sycl::queue&, real*) {}, report_every);
          },
          kind);
    }

  private:
    sycl::queue q_;
    SweepDeviceContextT<D> ctx_;
};

// D=2 legacy aliases for the binding's oracle surface and existing call sites.
using SweepGroupHost = SweepGroupHostT<2>;
using EnergyStencilHost = EnergyStencilHostT<2>;
using SweepContextHost = SweepContextHostT<2>;
using GroupView = GroupViewT<2>;
using StencilView = StencilViewT<2>;
using SweepDeviceContext = SweepDeviceContextT<2>;
using Executor = ExecutorT<2>;

}  // namespace egg
