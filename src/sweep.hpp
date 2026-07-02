// sweep.hpp — device-resident colored Gauss-Seidel barrier/untangle sweep.
#pragma once

#include "device.hpp"
#include "geometry.hpp"
#include "metric.hpp"
#include "patch.hpp"

#include <algorithm>
#include <array>
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
    View1<const int> gc, role;                           // [total_samples]
    View1<const int> gn[D];                              // [total_samples]
    View1<const double> s[D];                            // [total_samples]
    View1<const double> W_inv;                           // [total_samples * wInv]
    View1<const double> J;                               // [total_samples * jSize]
    View1<const int> dof_idx, P_of, sample_offset;       // [ndof]
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
    SweepDeviceContextT(sycl::queue q, const SweepContextHostT<D>& host) :
        q_(q), num_nodes_(host.num_nodes), X_(q, host.X)
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
            dg.dof_idx = {q, g.dof_idx};
            dg.P_of = {q, g.P_of};
            dg.sample_offset = {q, g.sample_offset};

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
        group_partitions_.resize(groups_.size());
        for (std::size_t gi = 0; gi < groups_.size(); ++gi) {
            auto& dg = groups_[gi];
            auto& pvs = group_partitions_[gi];
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
            group_views_.push_back(dg.view(pvs.data(), pvs.size()));
        }
        stencil_view_ = stencil_.view();
    }

    const std::vector<GroupViewT<D>>& group_views() const { return group_views_; }
    const StencilViewT<D>& stencil_view() const { return stencil_view_; }
    double* X() const { return X_.data(); }
    std::size_t num_nodes() const { return num_nodes_; }

    void upload_X(const std::vector<double>& host_X) { X_.upload(host_X); }
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
        std::size_t total_samples;
        UsmBuffer<int> gc, role;
        UsmBuffer<int> gn[D];
        UsmBuffer<double> s[D];
        UsmBuffer<double> W_inv, J;
        UsmBuffer<int> dof_idx, P_of, sample_offset;
        std::vector<DevicePartition> partitions;

        GroupViewT<D> view(const PartitionView* pvs, std::size_t np) const
        {
            const std::size_t n = total_samples;
            GroupViewT<D> gv;
            gv.ndof = ndof;
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

    sycl::queue q_;
    std::size_t num_nodes_;
    UsmBuffer<double> X_;
    std::vector<DeviceGroup> groups_;
    std::vector<std::vector<PartitionView>> group_partitions_;  // stable backing for views
    DeviceStencil stencil_;
    std::vector<GroupViewT<D>> group_views_;
    StencilViewT<D> stencil_view_ {};
};

/// @brief Monomorphic per-partition sweep kernel (Phase 1a).
///
/// One kernel per (colour, EntityTag) partition: each work item loads its
/// group-local DOF @c d from @p part.dof_list, builds the concrete entity @c E
/// via @ref EntitySoA "EntitySoA<E>::load" from the partition's typed SoA (no
/// `std::visit`, no `make_entity`, no per-lane dispatch), and runs the unchanged
/// Newton/backtracking body on a fully concrete entity. The entity type @p E is
/// a compile-time template parameter, so the Newton step, per-trial projection,
/// and energy reduction are all monomorphic — no variant alternatives are
/// inlined into the kernel.
/// @tparam D Embedding dimension.
/// @tparam E Entity type (satisfies @ref GeometryEntity).
/// @tparam Obj Objective type.
/// @param q SYCL queue.
/// @param g The colour group view (per-DOF patch arrays + partition list).
/// @param part The partition to iterate (entity tag @c E, group-local DOF list).
/// @param X Global node positions (device USM).
/// @param objective The objective functor.
template <int D, class E, class Obj>
inline void
  sweep_partition_kernel(sycl::queue& q, const GroupViewT<D>& g, const PartitionView& part, double* X, Obj objective)
{
    if (part.ndof == 0) { return; }
    q.parallel_for(sycl::range<1>(part.ndof), [=](sycl::id<1> idx) {
        const std::size_t d = static_cast<std::size_t>(part.dof_list[idx[0]]);
        const PatchViewT<D> pv = g.patch(d);
        const int dof = g.dof_idx[d];

        // 1. patch_eval → (grad, hess, e0).
        const PatchResultT<D> r = patch_eval<D>(pv, X, objective);
        const PtN<D> pos = load_pt<D>(X, dof);

        // Build the concrete entity ONCE per DOF from the packed SoA records via
        // tie_view + load (coalesced, no blob stride, no std::visit, no variant):
        // the Newton step, per-trial projection, and energy reduction below all
        // run on a monomorphic, fully-concrete E. Every 2D entity type has an
        // EntitySoA<E> specialization (Phase 4 retired the positional blob), so
        // this is the single load path.
        const E ent = EntitySoA<E>::load(EntitySoA<E>::tie_view(part.soa_view, part.seg), idx[0]);

        // 2. Newton step (tangent-reduced if constrained).
        const VecN<D> delta = newton_delta<D>(r.grad, r.hess, pos, ent);

        // 3. Backtracking: halve alpha up to 10×; accept on
        //    finite(e) && e <= e0 + 1e-12 && objective.accept_mindet(mindet).
        double alpha = 1.0;
        PtN<D> cur = pos;
        bool accepted = false;
        for (int it = 0; it < 10 && !accepted; ++it) {
            PtN<D> raw;
            for (int k = 0; k < D; ++k) { raw[k] = pos[k] + alpha * delta[k]; }
            PtN<D> trial;
            if constexpr (std::is_same_v<E, Free<D>>) {
                trial = raw;
            } else {
                trial = ent.project(raw);
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

        // 4. Scatter (race-free within colour).
        store_pt<D>(X, dof, cur);
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

template <int D> class ExecutorT
{
  public:
    ExecutorT(const sycl::queue& queue, const SweepContextHostT<D>& host) :
        q_(sycl::queue(
          queue.get_context(), queue.get_device(), {sycl::property::queue::in_order()})),
        ctx_(q_, host)
    {
    }

    SweepDeviceContextT<D>& ctx() { return ctx_; }

    // Run n_sweeps, dispatching the objective kind once via std::visit.
    std::pair<std::vector<double>, std::vector<double>>
      run_sweeps(int n_sweeps, const ObjectiveKindT<D>& kind = ShapeObjectiveT<D> {})
    {
        return std::visit([&](auto objective) { return run_impl(objective, n_sweeps); }, kind);
    }

  private:
    template <class Obj>
    std::pair<std::vector<double>, std::vector<double>> run_impl(Obj objective, int n_sweeps)
    {
        UsmBuffer<double> d_e(q_, static_cast<std::size_t>(n_sweeps));
        UsmBuffer<double> d_m(q_, static_cast<std::size_t>(n_sweeps));
        double* X = ctx_.X();

        constexpr int kSyncEvery = 32;
        for (int s = 0; s < n_sweeps; ++s) {
            for (const auto& gv : ctx_.group_views()) {
                // Launch one monomorphic kernel per (colour, EntityTag) partition
                // via dispatch_entity_type — no in-kernel std::visit, no per-lane
                // dispatch. Partitions are serialized by the in-order queue; the
                // race-free colouring guarantees no cross-partition dependency
                // within a colour (no DOF reads a same-colour mate's X).
                for (std::size_t p = 0; p < gv.num_partitions; ++p) {
                    const auto& part = gv.partitions[p];
                    dispatch_entity_type<D>(part.tag, [&]<class E>() {
                        sweep_partition_kernel<D, E>(q_, gv, part, X, objective);
                    });
                }
            }
            reduce_energy_mindet<D>(q_,
                                    ctx_.stencil_view(),
                                    X,
                                    d_e.data() + s,
                                    d_m.data() + s,
                                    objective);
            if ((s + 1) % kSyncEvery == 0) { q_.wait(); }
        }
        q_.wait();

        std::vector<double> energies(n_sweeps), mindets(n_sweeps);
        d_e.download(energies.data());
        d_m.download(mindets.data());
        return {energies, mindets};
    }

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
