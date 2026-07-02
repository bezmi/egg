// sweep.hpp — device-resident colored Gauss-Seidel barrier/untangle sweep.
#pragma once

#include "device.hpp"
#include "geometry.hpp"
#include "metric.hpp"
#include "patch.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <sycl/sycl.hpp>
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
///        All SoA views share the one `SoAView<const double>` type for records,
///        so this struct is type-erased at the host level (the interpretation is
///        per-`E` in `load`). The `seg` vector carries variable-length CSR
///        payloads (B-spline knots/ctrl) — empty for fixed-size types.
struct SoAHostRecord {
    /// @brief One segmented (CSR) field: flat data + offset table.
    struct SegmentedField {
        std::vector<double> data;
        std::vector<int> off;  ///< Length count+1, off[0]==0.
    };
    EntityTag tag;
    int k_fields;
    std::vector<double> records;  // [count * k_fields] packed
    std::size_t count;
    std::vector<SegmentedField> seg;  // CSR fields (empty for fixed-size)
    std::vector<int> dof_local;       // [count] group-local DOF indices of this partition
};

template <int D> struct SweepGroupHostT {
    std::size_t ndof;                // DOFs in this colour
    std::size_t total_samples;       // Σ P_of[d]
    std::vector<int> gc;             // [total_samples]
    std::vector<int> gn[D];          // [total_samples] per axis (was gn0, gn1)
    std::vector<double> s[D];        // [total_samples] per axis (was s0, s1)
    std::vector<double> W_inv;       // [total_samples * dim::wInv(D)]
    std::vector<int> role;           // [total_samples]
    std::vector<double> J;           // [total_samples * dim::jSize(D)]
    std::vector<int> dof_idx;        // [ndof]
    std::vector<int> P_of;           // [ndof]  patch size per DOF
    std::vector<int> sample_offset;  // [ndof]  exclusive prefix sum of P_of
    std::vector<SoAHostRecord> soa;  // typed per-(colour,EntityTag) SoA stores (the entity data)
};

template <int D> struct EnergyStencilHostT {
    std::size_t num_samples;
    std::vector<int> gc;  // [num_samples]
    std::vector<int> gn[D];
    std::vector<double> s[D];
    std::vector<double> W_inv;  // [num_samples * dim::wInv(D)]
};

template <int D> struct SweepContextHostT {
    std::vector<SweepGroupHostT<D>> groups;  // colour-ordered
    EnergyStencilHostT<D> energy_stencil;
    std::size_t num_nodes;
    std::vector<double> X;  // [num_nodes * D] initial positions
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
/// positional blob). All SoA views share the one typed `SoAView<const double>`
/// — no `const void*`, no type erasure, no cast.
struct PartitionView {
    static constexpr int kMaxSeg = kMaxSoASeg;  ///< Max segmented slots (matches EntitySoA<E>::View).
    EntityTag tag;                 ///< Entity type of this partition.
    std::size_t ndof;              ///< Number of DOFs in this partition.
    const int* dof_list;           ///< Group-local DOF indices, length @c ndof.
    SoAView<const double> soa_view{nullptr, 0, 0};  ///< Packed records (ndof, kFields).
    SegmentedView<double> seg[kMaxSeg]{};  ///< CSR fields (null for fixed-size).
};

template <int D> struct GroupViewT {
    std::size_t ndof;
    std::size_t n_free = 0;  // DOFs are Free-first reordered; [0, n_free) are Free
    View1<const int> gc, role;                           // [total_samples]
    View1<const int> gn[D];                              // [total_samples]
    View1<const double> s[D];                            // [total_samples]
    View1<const double> W_inv;                           // [total_samples * wInv]
    View1<const double> J;                               // [total_samples * jSize]
    View1<const int> dof_idx, P_of, sample_offset;       // [ndof]
    View1<const int> part_of, row_of;                    // [ndof] DOF -> (partition, row)
    const PartitionView* partitions = nullptr;           // [num_partitions]
    std::size_t num_partitions = 0;                      // per-(colour,EntityTag) splits

    // PatchViewT over DOF d's contiguous ragged slice
    // [sample_offset[d], sample_offset[d] + P_of[d]).
    PatchViewT<D> patch(std::size_t d) const
    {
        const auto off = static_cast<std::size_t>(sample_offset[d]);
        PatchViewT<D> pv;
        pv.P = P_of[d];
        pv.gc = gc.data_handle() + off;
        for (int k = 0; k < D; ++k) {
            pv.gn[k] = gn[k].data_handle() + off;
            pv.s[k] = s[k].data_handle() + off;
        }
        pv.W_inv = W_inv.data_handle() + (off * dim::wInv(D));
        pv.role = role.data_handle() + off;
        pv.J = J.data_handle() + (off * dim::jSize(D));
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
        return sub;
    }
};

template <int D> struct StencilViewT {
    std::size_t num_samples;
    View1<const int> gc;
    View1<const int> gn[D];
    View1<const double> s[D];
    View2<const double> W_inv;  // [num_samples][wInv]
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
               std::vector<double>(host.num_nodes * kSeedStride,
                                   std::numeric_limits<double>::quiet_NaN()))
    {
        groups_.reserve(host.groups.size());
        group_partitions_.reserve(host.groups.size());
        for (const auto& g : host.groups) {
            DeviceGroup dg;
            dg.ndof = g.ndof;
            dg.total_samples = g.total_samples;
            dg.gc = {q, g.gc};
            dg.role = {q, g.role};
            for (int k = 0; k < D; ++k) {
                dg.gn[k] = {q, g.gn[k]};
                dg.s[k] = {q, g.s[k]};
            }
            dg.W_inv = {q, g.W_inv};
            dg.J = {q, g.J};

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
                        dp.soa_records = UsmBuffer<double>{q, rec.records};
                        for (int j = 0; j < SoA::kSeg; ++j) {
                            if (j < static_cast<int>(rec.seg.size())) {
                                dp.seg_data[j] = UsmBuffer<double>{q, rec.seg[j].data};
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

            groups_.push_back(std::move(dg));
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
                const double* soa_ptr = dp.soa_records.data();
                std::size_t kfields = 0;
                dispatch_entity_type<D>(dp.tag, [&]<class E>() {
                    if constexpr (HasEntitySoA<E>) { kfields = static_cast<std::size_t>(EntitySoA<E>::kFields); }
                });
                PartitionView pv {
                  .tag = dp.tag, .ndof = ndof, .dof_list = dp.dof_list.data(),
                  .soa_view = SoAView<const double> {soa_ptr, ndof, kfields}};
                // Fill segmented views from uploaded USM (null for fixed-size).
                for (int j = 0; j < PartitionView::kMaxSeg; ++j) {
                    pv.seg[j] = SegmentedView<double>{
                      dp.seg_data[j].data(), dp.seg_off[j].data()};
                }
                pvs.push_back(pv);
            }
            // The per-colour kernel indexes this array ON the device (via
            // part_of[d]), so it must live in USM — a host std::vector pointer
            // would fault on the GPU. Upload once; the device pointer is stable.
            group_partitions_.emplace_back(q, pvs);
            group_views_.push_back(dg.view(group_partitions_.back().data(), pvs.size()));
            host_pvs_.push_back(std::move(pvs));  // retained for the merged Jacobi view
        }
        stencil_view_ = stencil_.view();
    }

    const std::vector<GroupViewT<D>>& group_views() const { return group_views_; }
    const StencilViewT<D>& stencil_view() const { return stencil_view_; }
    double* X() const { return X_.data(); }
    std::size_t x_size() const { return X_.size(); }
    std::size_t num_nodes() const { return num_nodes_; }
    /// @brief Per-node warm-start seed cache (stride @ref kSeedStride), persisted
    ///        across sweeps. Forwarded to the block-Jacobi kernel; null-equivalent
    ///        cold path is used by colored-GS (passes nullptr).
    double* seeds() { return seeds_.data(); }

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

    void upload_X(const std::vector<double>& host_X)
    {
        X_.upload(host_X);
        // Positions reset → invalidate the warm-start cache (re-seed cold).
        seeds_.upload(std::vector<double>(seeds_.size(), std::numeric_limits<double>::quiet_NaN()));
    }
    void download_X(double* host_X) { X_.download(host_X); }

  private:
    struct DevicePartition {
        EntityTag tag;
        UsmBuffer<int> dof_list;         // group-local DOF indices, length = ndof of partition
        UsmBuffer<double> soa_records;   // packed SoA records, length = ndof * kFields (0 if none)
        UsmBuffer<double> seg_data[kMaxSoASeg];  // CSR data per segmented slot (empty if kSeg==0)
        UsmBuffer<int> seg_off[kMaxSoASeg];      // CSR offsets per segmented slot (empty if kSeg==0)
    };

    struct DeviceGroup {
        std::size_t ndof;
        std::size_t n_free = 0;  // count of Free DOFs (the [0, n_free) prefix)
        std::size_t total_samples;
        UsmBuffer<int> gc, role;
        UsmBuffer<int> gn[D];
        UsmBuffer<double> s[D];
        UsmBuffer<double> W_inv, J;
        UsmBuffer<int> dof_idx, P_of, sample_offset;
        UsmBuffer<int> part_of, row_of;  // [ndof] DOF -> (partition, row)
        std::vector<DevicePartition> partitions;

        GroupViewT<D> view(const PartitionView* pvs, std::size_t np) const
        {
            const std::size_t n = total_samples;
            GroupViewT<D> gv;
            gv.ndof = ndof;
            gv.n_free = n_free;
            gv.gc = View1<const int> {gc.data(), n};
            gv.role = View1<const int> {role.data(), n};
            for (int k = 0; k < D; ++k) {
                gv.gn[k] = View1<const int> {gn[k].data(), n};
                gv.s[k] = View1<const double> {s[k].data(), n};
            }
            gv.W_inv = View1<const double> {W_inv.data(), n * dim::wInv(D)};
            gv.J = View1<const double> {J.data(), n * dim::jSize(D)};
            gv.dof_idx = View1<const int> {dof_idx.data(), ndof};
            gv.P_of = View1<const int> {P_of.data(), ndof};
            gv.sample_offset = View1<const int> {sample_offset.data(), ndof};
            gv.part_of = View1<const int> {part_of.data(), ndof};
            gv.row_of = View1<const int> {row_of.data(), ndof};
            gv.partitions = pvs;
            gv.num_partitions = np;
            return gv;
        }
    };

    struct DeviceStencil {
        std::size_t num_samples;
        UsmBuffer<int> gc;
        UsmBuffer<int> gn[D];
        UsmBuffer<double> s[D];
        UsmBuffer<double> W_inv;

        StencilViewT<D> view() const
        {
            StencilViewT<D> sv;
            sv.num_samples = num_samples;
            sv.gc = View1<const int> {gc.data(), num_samples};
            for (int k = 0; k < D; ++k) {
                sv.gn[k] = View1<const int> {gn[k].data(), num_samples};
                sv.s[k] = View1<const double> {s[k].data(), num_samples};
            }
            sv.W_inv = View2<const double> {W_inv.data(), num_samples, dim::wInv(D)};
            return sv;
        }
    };

    /// Concatenate every colour group's tables into one GroupViewT (cached in the
    /// m_* buffers + merged_view_). Bulk per-sample arrays are joined by
    /// device-to-device copy; the small per-DOF index arrays are rebuilt on host
    /// with the running sample/partition offsets; the partition list is the
    /// per-group PartitionView arrays concatenated (entity SoA pointers unchanged).
    void build_merged_view()
    {
        const std::size_t ng = groups_.size();
        std::size_t total_ndof = 0, total_samples = 0, total_parts = 0;
        for (std::size_t gi = 0; gi < ng; ++gi) {
            total_ndof += groups_[gi].ndof;
            total_samples += groups_[gi].total_samples;
            total_parts += host_pvs_[gi].size();
        }
        const std::size_t wInv = dim::wInv(D), jSize = dim::jSize(D);

        m_gc_ = UsmBuffer<int> {q_, total_samples};
        m_role_ = UsmBuffer<int> {q_, total_samples};
        for (int k = 0; k < D; ++k) {
            m_gn_[k] = UsmBuffer<int> {q_, total_samples};
            m_s_[k] = UsmBuffer<double> {q_, total_samples};
        }
        m_W_inv_ = UsmBuffer<double> {q_, total_samples * wInv};
        m_J_ = UsmBuffer<double> {q_, total_samples * jSize};

        std::vector<int> dof_idx(total_ndof), P_of(total_ndof), sample_offset(total_ndof);
        std::vector<int> part_of(total_ndof), row_of(total_ndof);
        std::vector<PartitionView> merged_pvs;
        merged_pvs.reserve(total_parts);

        std::size_t sbase = 0, dbase = 0, pbase = 0;
        for (std::size_t gi = 0; gi < ng; ++gi) {
            auto& dg = groups_[gi];
            const std::size_t gs = dg.total_samples, gd = dg.ndof;
            q_.memcpy(m_gc_.data() + sbase, dg.gc.data(), gs * sizeof(int));
            q_.memcpy(m_role_.data() + sbase, dg.role.data(), gs * sizeof(int));
            for (int k = 0; k < D; ++k) {
                q_.memcpy(m_gn_[k].data() + sbase, dg.gn[k].data(), gs * sizeof(int));
                q_.memcpy(m_s_[k].data() + sbase, dg.s[k].data(), gs * sizeof(double));
            }
            q_.memcpy(m_W_inv_.data() + sbase * wInv, dg.W_inv.data(), gs * wInv * sizeof(double));
            q_.memcpy(m_J_.data() + sbase * jSize, dg.J.data(), gs * jSize * sizeof(double));

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

        m_dof_idx_ = UsmBuffer<int> {q_, gather(dof_idx)};
        m_P_of_ = UsmBuffer<int> {q_, gather(P_of)};
        m_sample_offset_ = UsmBuffer<int> {q_, gather(sample_offset)};
        m_part_of_ = UsmBuffer<int> {q_, gather(part_of)};
        m_row_of_ = UsmBuffer<int> {q_, gather(row_of)};
        m_partitions_ = UsmBuffer<PartitionView> {q_, merged_pvs};

        GroupViewT<D> gv;
        gv.ndof = total_ndof;
        gv.n_free = merged_n_free;
        gv.gc = View1<const int> {m_gc_.data(), total_samples};
        gv.role = View1<const int> {m_role_.data(), total_samples};
        for (int k = 0; k < D; ++k) {
            gv.gn[k] = View1<const int> {m_gn_[k].data(), total_samples};
            gv.s[k] = View1<const double> {m_s_[k].data(), total_samples};
        }
        gv.W_inv = View1<const double> {m_W_inv_.data(), total_samples * wInv};
        gv.J = View1<const double> {m_J_.data(), total_samples * jSize};
        gv.dof_idx = View1<const int> {m_dof_idx_.data(), total_ndof};
        gv.P_of = View1<const int> {m_P_of_.data(), total_ndof};
        gv.sample_offset = View1<const int> {m_sample_offset_.data(), total_ndof};
        gv.part_of = View1<const int> {m_part_of_.data(), total_ndof};
        gv.row_of = View1<const int> {m_row_of_.data(), total_ndof};
        gv.partitions = m_partitions_.data();
        gv.num_partitions = total_parts;
        merged_view_ = gv;
    }

    sycl::queue q_;
    std::size_t num_nodes_;
    UsmBuffer<double> X_;
    UsmBuffer<double> seeds_;  // [num_nodes * kSeedStride] warm-start param cache
    std::vector<DeviceGroup> groups_;
    std::vector<UsmBuffer<PartitionView>> group_partitions_;  // device-resident PartitionView arrays
    DeviceStencil stencil_;
    std::vector<GroupViewT<D>> group_views_;
    StencilViewT<D> stencil_view_ {};

    // Merged single-launch view for block-Jacobi (lazily built, then cached).
    std::vector<std::vector<PartitionView>> host_pvs_;  // per-group PartitionView (host copies)
    bool merged_built_ = false;
    GroupViewT<D> merged_view_ {};
    UsmBuffer<int> m_gc_, m_role_;
    UsmBuffer<int> m_gn_[D];
    UsmBuffer<double> m_s_[D];
    UsmBuffer<double> m_W_inv_, m_J_;
    UsmBuffer<int> m_dof_idx_, m_P_of_, m_sample_offset_, m_part_of_, m_row_of_;
    UsmBuffer<PartitionView> m_partitions_;
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
///        surface). @p seed_slot points at the (D-1)-double per-node seed
///        (non-finite ⇒ cold); on return it holds the converged foot parameter
///        so the next sweep skips the expensive coarse-grid seed. Closed-form
///        entities have no @c project_seeded and project cold (seed untouched).
template <int D, class E>
inline PtN<D> project_with_seed(const E& ent, const PtN<D>& p, double* seed_slot)
{
    constexpr int K = E::tdim;
    if constexpr (requires(Param<K> s) { ent.project_seeded(p, s, true); }) {
        Param<K> seed {};
        const bool has = (seed_slot != nullptr) && std::isfinite(seed_slot[0]);
        if (has) {
            for (int a = 0; a < K; ++a) { seed[a] = seed_slot[a]; }
        }
        const PtN<D> out = ent.project_seeded(p, seed, has);
        if (seed_slot != nullptr) {
            for (int a = 0; a < K; ++a) { seed_slot[a] = seed[a]; }
        }
        return out;
    } else {
        return ent.project(p);
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
template <int D, class Obj, bool FreeOnly = false>
inline void sweep_colour_kernel(sycl::queue& q, const GroupViewT<D>& g, double* X, Obj objective,
                                double* X_out = nullptr, double omega = 1.0, double* seeds = nullptr)
{
    if (g.ndof == 0) { return; }
    constexpr std::size_t kSeedStride = (D > 1) ? (D - 1) : 1;
    const PartitionView* parts = g.partitions;
    double* out = X_out ? X_out : X;
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
            const PartitionView& part = parts[g.part_of[d]];
            const auto row = static_cast<std::size_t>(g.row_of[d]);
            with_entity<D>(part.tag, part.soa_view, part.seg, row,
                           [&](const auto& ent) { delta = newton_delta<D>(r.grad, r.hess, pos, ent); });
        }

        // 3. Backtracking: halve alpha up to 10×; accept on
        //    finite(e) && e <= e0 + 1e-12 && objective.accept_mindet(mindet).
        double alpha = 1.0;
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
                    double* seed_slot =
                      (seeds != nullptr) ? (seeds + (static_cast<std::size_t>(dof) * kSeedStride))
                                         : nullptr;
                    with_entity<D>(tag, part.soa_view, part.seg, row,
                                   [&](const auto& ent) { trial = project_with_seed<D>(ent, raw, seed_slot); });
                }
            }

            double e_new = 0.0;
            double mdet = std::numeric_limits<double>::infinity();
            for (int p = 0; p < pv.P; ++p) {
                // Substitute the trial position for the moving DOF; all other
                // nodes load from X. The A→T→detA math is the single source in
                // patch.hpp::assemble_vecT.
                auto node = [&](int ni) -> PtN<D> {
                    return ni == dof ? trial : load_pt<D>(X, ni);
                };
                PtN<D> corner = node(pv.gc[p]);
                std::array<PtN<D>, D> nbr;
                std::array<double, D> sc;
                for (int k = 0; k < D; ++k) {
                    nbr[k] = node(pv.gn[k][p]);
                    sc[k] = pv.s[k][p];
                }
                double detA;
                const VecTN<D> t =
                  assemble_vecT<D>(corner, nbr, sc, &pv.W_inv[dim::wInv(D) * p], detA);
                e_new += objective.value(t);
                mdet = std::min(detA, mdet);
            }

            const bool ok = std::isfinite(e_new) && (e_new <= r.energy + 1e-12) &&
                            objective.accept_mindet(mdet);
            if (ok) {
                cur = trial;
                accepted = true;
            } else {
                alpha *= 0.5;
            }
        }

        // 4. Optional Jacobi relaxation weighting: cur = pos + ω·(cur − pos).
        //    ω=1 leaves cur untouched (colored-GS / undamped Jacobi).
        if (omega != 1.0) {
            for (int k = 0; k < D; ++k) { cur[k] = pos[k] + (omega * (cur[k] - pos[k])); }
        }

        // 5. Scatter. In-place (colored-GS) writes X; double-buffered Jacobi
        //    writes the separate X_out snapshot. Race-free: distinct DOFs are
        //    distinct nodes, so the scatter never collides.
        store_pt<D>(out, dof, cur);
    });
}

// Per-sweep total energy (Σ μ) and min det A over the energy stencil, written to
// out_e/out_m (USM scalars). Reduction identities are supplied explicitly, so no
// host pre-initialisation of the targets is needed.
template <int D, class Obj>
inline void reduce_energy_mindet(sycl::queue& q,
                                 const StencilViewT<D>& es,
                                 const double* X,
                                 double* out_e,
                                 double* out_m,
                                 Obj objective)
{
    auto e_red = sycl::reduction(out_e,
                                 0.0,
                                 std::plus<>(),
                                 sycl::property::reduction::initialize_to_identity {});
    auto m_red = sycl::reduction(out_m,
                                 std::numeric_limits<double>::infinity(),
                                 sycl::minimum<double>(),
                                 sycl::property::reduction::initialize_to_identity {});

    q.parallel_for(sycl::range<1>(es.num_samples),
                   e_red,
                   m_red,
                   [=](sycl::id<1> idx, auto& e_sum, auto& m_min) {
                       const std::size_t i = idx[0];
                       // W_inv is [num_samples][wInv] row-major; row i is contiguous.
                       const double* w = es.W_inv.data_handle() + i * dim::wInv(D);
                       PtN<D> corner = load_pt<D>(X, es.gc[i]);
                       std::array<PtN<D>, D> nbr;
                       std::array<double, D> sc;
                       for (int k = 0; k < D; ++k) {
                           nbr[k] = load_pt<D>(X, es.gn[k][i]);
                           sc[k] = es.s[k][i];
                       }
                       double detA;
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
inline std::pair<std::vector<double>, std::vector<double>>
  run_colored_gs(sycl::queue& q, SweepDeviceContextT<D>& ctx, Obj objective, int n_sweeps,
                 PreSweep before_sweep, int report_every = 1)
{
    const int k = (report_every <= 0) ? n_sweeps : report_every;
    const std::size_t report_count =
      static_cast<std::size_t>((n_sweeps + k - 1) / k);
    UsmBuffer<double> d_e(q, report_count);
    UsmBuffer<double> d_m(q, report_count);
    double* X = ctx.X();

    constexpr int kSyncEvery = 32;
    std::size_t report_idx = 0;
    for (int s = 0; s < n_sweeps; ++s) {
        before_sweep(q, X);
        // Per colour: lean interior launch over [0, n_free) then the full
        // boundary launch over the tail. Within a colour all DOFs are distinct
        // nodes, so the two launches over disjoint subsets are race-free
        // (serialized by the in-order queue). Plan #1 step 4.
        for (const auto& gv : ctx.group_views()) {
            sweep_colour_kernel<D, Obj, true>(q, gv.dof_subrange(0, gv.n_free), X, objective);
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

    std::vector<double> energies(report_count), mindets(report_count);
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
inline std::pair<std::vector<double>, std::vector<double>>
  run_block_jacobi(sycl::queue& q, SweepDeviceContextT<D>& ctx, Obj objective, int n_sweeps,
                   PreSweep before_sweep, double omega = 1.0, int report_every = 1)
{
    const int k = (report_every <= 0) ? n_sweeps : report_every;
    const std::size_t report_count =
      static_cast<std::size_t>((n_sweeps + k - 1) / k);
    UsmBuffer<double> d_e(q, report_count);
    UsmBuffer<double> d_m(q, report_count);
    const std::size_t nbuf = ctx.x_size();
    UsmBuffer<double> x_new_buf(q, nbuf);

    double* canonical = ctx.X();
    double* x_in = canonical;
    double* x_out = x_new_buf.data();
    q.memcpy(x_out, x_in, nbuf * sizeof(double)).wait();  // X_new := copy of X

    const GroupViewT<D>& mg = ctx.merged_group_view();

    constexpr int kSyncEvery = 32;
    std::size_t report_idx = 0;
    for (int s = 0; s < n_sweeps; ++s) {
        before_sweep(q, x_in);  // refresh ghosts + shared-node copies on the read buffer
        // Interior/boundary split: the lean FreeOnly kernel over [0, n_free)
        // (no projection, no seeds) and the full geometry kernel over the
        // boundary tail. Both read the frozen x_in and write disjoint x_out
        // slots, so block-Jacobi correctness is preserved. Plan #1 step 4.
        sweep_colour_kernel<D, Obj, true>(
          q, mg.dof_subrange(0, mg.n_free), x_in, objective, x_out, omega, nullptr);
        sweep_colour_kernel<D, Obj, false>(
          q, mg.dof_subrange(mg.n_free, mg.ndof - mg.n_free), x_in, objective, x_out, omega,
          ctx.seeds());
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
    if (x_in != canonical) { q.memcpy(canonical, x_in, nbuf * sizeof(double)).wait(); }

    std::vector<double> energies(report_count), mindets(report_count);
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
    std::pair<std::vector<double>, std::vector<double>>
      run_sweeps(int n_sweeps, const ObjectiveKindT<D>& kind = ShapeObjectiveT<D> {},
                 int report_every = 1)
    {
        return std::visit(
          [&](auto objective) {
              // Unstructured path: no halo exchange (one global X, not per-block).
              return run_colored_gs<D>(q_, ctx_, objective, n_sweeps,
                                        [](sycl::queue&, double*) {}, report_every);
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
