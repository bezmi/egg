// sweep.hpp — device-resident colored Gauss-Seidel barrier/untangle sweep.
#pragma once

#include "device.hpp"
#include "geometry.hpp"
#include "metric.hpp"
#include "patch.hpp"

#include <cstddef>
#include <functional>
#include <limits>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

namespace egg
{

struct SweepGroupHost {
    std::size_t D;                   // DOFs in this colour
    std::size_t total_samples;       // Σ P_of[d]
    std::vector<int> gc;             // [total_samples]
    std::vector<int> gn0;            // [total_samples]
    std::vector<int> gn1;            // [total_samples]
    std::vector<double> s0;          // [total_samples]
    std::vector<double> s1;          // [total_samples]
    std::vector<double> W_inv;       // [total_samples*4]
    std::vector<int> role;           // [total_samples]
    std::vector<double> J;           // [total_samples*24]
    std::vector<int> dof_idx;        // [D]
    std::vector<int> tag;            // [D]
    std::vector<int> P_of;           // [D]  patch size per DOF
    std::vector<int> sample_offset;  // [D]  exclusive prefix sum of P_of
    std::vector<double> params;      // [D*12]
};

struct EnergyStencilHost {
    std::size_t num_samples;
    std::vector<int> gc;  // [num_samples]
    std::vector<int> gn0;
    std::vector<int> gn1;
    std::vector<double> s0;
    std::vector<double> s1;
    std::vector<double> W_inv;  // [num_samples*4]
};

struct SweepContextHost {
    std::vector<SweepGroupHost> groups;  // colour-ordered
    EnergyStencilHost energy_stencil;
    std::size_t num_nodes;
    std::vector<double> X;  // [num_nodes*2] initial positions
};

struct GroupView {
    std::size_t D;
    View1<const int> gc, gn0, gn1, role;                 // [total_samples]
    View1<const double> s0, s1;                          // [total_samples]
    View1<const double> W_inv;                           // [total_samples*4]
    View1<const double> J;                               // [total_samples*24]
    View1<const int> dof_idx, tag, P_of, sample_offset;  // [D]
    View1<const double> params;                          // [D*12]

    // PatchView over DOF d's contiguous ragged slice
    // [sample_offset[d], sample_offset[d] + P_of[d]).
    PatchView patch(std::size_t d) const
    {
        const std::size_t off = static_cast<std::size_t>(sample_offset[d]);
        return PatchView {P_of[d],
                          gc.data_handle() + off,
                          gn0.data_handle() + off,
                          gn1.data_handle() + off,
                          s0.data_handle() + off,
                          s1.data_handle() + off,
                          W_inv.data_handle() + off * 4,
                          role.data_handle() + off,
                          J.data_handle() + off * 24};
    }
};

struct StencilView {
    std::size_t num_samples;
    View1<const int> gc, gn0, gn1;
    View1<const double> s0, s1;
    View2<const double> W_inv;  // [num_samples][4]
};

class SweepDeviceContext
{
  public:
    SweepDeviceContext(sycl::queue q, const SweepContextHost& host) :
        q_(q), num_nodes_(host.num_nodes), X_(q, host.X)
    {
        groups_.reserve(host.groups.size());
        for (const auto& g : host.groups) {
            groups_.push_back(DeviceGroup {g.D,
                                           g.total_samples,
                                           {q, g.gc},
                                           {q, g.gn0},
                                           {q, g.gn1},
                                           {q, g.role},
                                           {q, g.s0},
                                           {q, g.s1},
                                           {q, g.W_inv},
                                           {q, g.J},
                                           {q, g.dof_idx},
                                           {q, g.tag},
                                           {q, g.P_of},
                                           {q, g.sample_offset},
                                           {q, g.params}});
        }
        const auto& es = host.energy_stencil;
        stencil_ = DeviceStencil {es.num_samples,
                                  {q, es.gc},
                                  {q, es.gn0},
                                  {q, es.gn1},
                                  {q, es.s0},
                                  {q, es.s1},
                                  {q, es.W_inv}};

        group_views_.reserve(groups_.size());
        for (const auto& dg : groups_) group_views_.push_back(dg.view());
        stencil_view_ = stencil_.view();
    }

    const std::vector<GroupView>& group_views() const { return group_views_; }
    const StencilView& stencil_view() const { return stencil_view_; }
    double* X() const { return X_.data(); }
    std::size_t num_nodes() const { return num_nodes_; }

    void upload_X(const std::vector<double>& host_X) { X_.upload(host_X); }
    void download_X(double* host_X) { X_.download(host_X); }

  private:
    struct DeviceGroup {
        std::size_t D;
        std::size_t total_samples;
        UsmBuffer<int> gc, gn0, gn1, role;
        UsmBuffer<double> s0, s1, W_inv, J;
        UsmBuffer<int> dof_idx, tag, P_of, sample_offset;
        UsmBuffer<double> params;

        GroupView view() const
        {
            const std::size_t n = total_samples;
            return GroupView {D,
                              View1<const int> {gc.data(), n},
                              View1<const int> {gn0.data(), n},
                              View1<const int> {gn1.data(), n},
                              View1<const int> {role.data(), n},
                              View1<const double> {s0.data(), n},
                              View1<const double> {s1.data(), n},
                              View1<const double> {W_inv.data(), n * 4},
                              View1<const double> {J.data(), n * 24},
                              View1<const int> {dof_idx.data(), D},
                              View1<const int> {tag.data(), D},
                              View1<const int> {P_of.data(), D},
                              View1<const int> {sample_offset.data(), D},
                              View1<const double> {params.data(), D * 12}};
        }
    };

    struct DeviceStencil {
        std::size_t num_samples;
        UsmBuffer<int> gc, gn0, gn1;
        UsmBuffer<double> s0, s1, W_inv;

        StencilView view() const
        {
            return StencilView {num_samples,
                                View1<const int> {gc.data(), num_samples},
                                View1<const int> {gn0.data(), num_samples},
                                View1<const int> {gn1.data(), num_samples},
                                View1<const double> {s0.data(), num_samples},
                                View1<const double> {s1.data(), num_samples},
                                View2<const double> {W_inv.data(), num_samples, 4}};
        }
    };

    sycl::queue q_;
    std::size_t num_nodes_;
    UsmBuffer<double> X_;
    std::vector<DeviceGroup> groups_;
    DeviceStencil stencil_;
    std::vector<GroupView> group_views_;
    StencilView stencil_view_ {};
};

inline void
  sweep_group_kernel(sycl::queue& q, const GroupView& g, double* X, Objective auto objective)
{
    if (g.D == 0) return;
    q.parallel_for(sycl::range<1>(g.D), [=](sycl::id<1> idx) {
        const std::size_t d = idx[0];
        const PatchView pv = g.patch(d);
        const int dof = g.dof_idx[d];
        const int tag = g.tag[d];
        const double* params = g.params.data_handle() + d * 12;

        // 1. patch_eval → (grad, hess, e0); 2. Newton step (tangent-reduced if
        //    constrained).
        const PatchResult r = patch_eval(pv, X, objective);
        const Pt pos {X[2 * dof], X[2 * dof + 1]};
        const Vec2 delta = newton_delta(r.grad, r.hess, pos, tag, params);

        // 3. Backtracking: halve alpha up to 10×; accept on
        //    finite(e) && e <= e0 + 1e-12 && objective.accept_mindet(mindet).
        double alpha = 1.0;
        Pt cur = pos;
        bool accepted = false;
        for (int it = 0; it < 10 && !accepted; ++it) {
            const Pt raw {pos[0] + alpha * delta[0], pos[1] + alpha * delta[1]};
            const Pt trial = (tag == TAG_FREE) ? raw : project(raw, tag, params);

            double e_new = 0.0;
            double mdet = std::numeric_limits<double>::infinity();
            for (int p = 0; p < pv.P; ++p) {
                auto node = [&](int ni) -> Pt {
                    return ni == dof ? trial : Pt {X[2 * ni], X[2 * ni + 1]};
                };
                const Pt corner = node(pv.gc[p]);
                const Pt nbr0 = node(pv.gn0[p]);
                const Pt nbr1 = node(pv.gn1[p]);
                const double a00 = pv.s0[p] * (nbr0[0] - corner[0]);
                const double a10 = pv.s0[p] * (nbr0[1] - corner[1]);
                const double a01 = pv.s1[p] * (nbr1[0] - corner[0]);
                const double a11 = pv.s1[p] * (nbr1[1] - corner[1]);
                const double detA = a00 * a11 - a01 * a10;
                const double* w = &pv.W_inv[4 * p];
                const VecT t {a00 * w[0] + a01 * w[2],
                              a00 * w[1] + a01 * w[3],
                              a10 * w[0] + a11 * w[2],
                              a10 * w[1] + a11 * w[3]};
                e_new += objective.value(t);
                if (detA < mdet) mdet = detA;
            }

            const bool ok =
              std::isfinite(e_new) && (e_new <= r.energy + 1e-12) && objective.accept_mindet(mdet);
            if (ok) {
                cur = trial;
                accepted = true;
            } else {
                alpha *= 0.5;
            }
        }

        // 4. Scatter (race-free within colour).
        X[2 * dof + 0] = cur[0];
        X[2 * dof + 1] = cur[1];
    });
}

// Per-sweep total energy (Σ μ) and min det A over the energy stencil, written to
// out_e/out_m (USM scalars). Reduction identities are supplied explicitly, so no
// host pre-initialisation of the targets is needed.
inline void reduce_energy_mindet(sycl::queue& q,
                                 const StencilView& es,
                                 const double* X,
                                 double* out_e,
                                 double* out_m,
                                 Objective auto objective)
{
    auto e_red = sycl::reduction(out_e,
                                 0.0,
                                 std::plus<double>(),
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
                       const int gc = es.gc[i], gn0 = es.gn0[i], gn1 = es.gn1[i];
                       const double cx = X[2 * gc], cy = X[2 * gc + 1];
                       const double n0x = X[2 * gn0], n0y = X[2 * gn0 + 1];
                       const double n1x = X[2 * gn1], n1y = X[2 * gn1 + 1];
                       const double a00 = es.s0[i] * (n0x - cx);
                       const double a10 = es.s0[i] * (n0y - cy);
                       const double a01 = es.s1[i] * (n1x - cx);
                       const double a11 = es.s1[i] * (n1y - cy);
                       const double detA = a00 * a11 - a01 * a10;
                       const VecT t {a00 * es.W_inv[i, 0] + a01 * es.W_inv[i, 2],
                                     a00 * es.W_inv[i, 1] + a01 * es.W_inv[i, 3],
                                     a10 * es.W_inv[i, 0] + a11 * es.W_inv[i, 2],
                                     a10 * es.W_inv[i, 1] + a11 * es.W_inv[i, 3]};
                       e_sum += objective.value(t);
                       m_min.combine(detA);
                   });
}

class Executor
{
  public:
    Executor(const sycl::queue& queue, const SweepContextHost& host) :
        q_(sycl::queue(
          queue.get_context(), queue.get_device(), {sycl::property::queue::in_order()})),
        ctx_(q_, host)
    {
    }

    SweepDeviceContext& ctx() { return ctx_; }

    // Run n_sweeps, dispatching the objective kind once via std::visit.
    std::pair<std::vector<double>, std::vector<double>>
      run_sweeps(int n_sweeps, const ObjectiveKind& kind = ShapeObjective {})
    {
        return std::visit([&](auto objective) { return run_impl(objective, n_sweeps); }, kind);
    }

  private:
    std::pair<std::vector<double>, std::vector<double>> run_impl(Objective auto objective,
                                                                 int n_sweeps)
    {
        UsmBuffer<double> d_e(q_, static_cast<std::size_t>(n_sweeps));
        UsmBuffer<double> d_m(q_, static_cast<std::size_t>(n_sweeps));
        double* X = ctx_.X();

        constexpr int kSyncEvery = 32;
        for (int s = 0; s < n_sweeps; ++s) {
            for (const auto& gv : ctx_.group_views()) sweep_group_kernel(q_, gv, X, objective);
            reduce_energy_mindet(q_,
                                 ctx_.stencil_view(),
                                 X,
                                 d_e.data() + s,
                                 d_m.data() + s,
                                 objective);
            if ((s + 1) % kSyncEvery == 0) q_.wait();
        }
        q_.wait();

        std::vector<double> energies(n_sweeps), mindets(n_sweeps);
        d_e.download(energies.data());
        d_m.download(mindets.data());
        return {energies, mindets};
    }

    sycl::queue q_;
    SweepDeviceContext ctx_;
};

}  // namespace egg
