// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

/// @file directional.hpp
/// Directional soft-energy terms (topology-aware constraints) over small node
/// tuples with frozen reference vectors — D-generic. Mirrors
/// egg.smoothing.directional exactly (the NumPy oracle). Kinds:
///  - parallel (nodes i, j):  E = w·((û·n̂)²), û = ε-unit of X[j]−X[i], n̂ the
///    frozen boundary normal — a marked chain segment stays wall-parallel,
///    its length free.
///  - line (nodes a, c, b):   E = w·|q̂1+q̂2|², q̂k the ε-unit directions away
///    from c — a framed fan's through legs stay opposed.
///  - stem (nodes n1, c):     E = w·((q̂·â)²), â the frozen through axis — the
///    normal leg stays perpendicular to the through rail.
/// All normalisations are ε-regularised (x/sqrt(|x|²+ε²)), so energies and
/// gradients stay finite through degenerate states. Gradients are closed-form;
/// the per-node Hessian is the Gauss-Newton (PSD) block plus a block-Jacobi
/// coupling stabiliser on the diagonal (same rationale as curvature.hpp). The
/// det barrier is untouched: the terms only add energy, validity stays owned by
/// the shape/mindet gate. The table layout (4 node slots, kind, D-component
/// frozen ref and projector axis, weight) is one schema for every kind,
/// including the 3D-only kinds (sheet-normal, cross-sectional line/stem along
/// singular edges) that land later — unknown kinds contribute zero here.
#pragma once

#include "core.hpp"

#include <cmath>

namespace egg
{

inline constexpr int kDirParallel = 0;
inline constexpr int kDirLine = 1;
inline constexpr int kDirStem = 2;

/// Whole-sample directional table view (energy reductions). Node ids index the
/// same X buffer as the energy stencil (structured or global, per context).
template <int D> struct DirTableViewT {
    std::size_t num = 0;
    const int* nodes = nullptr;    ///< [num * 4], -1 pads
    const int* kind = nullptr;     ///< [num]
    const real* ref = nullptr;     ///< [num * D] frozen reference vector
    const real* axis = nullptr;    ///< [num * D] frozen projector direction (3D kinds)
    const real* weight = nullptr;  ///< [num]
    real eps = 0.0_r;
};

/// r = (u·ref)/s with s = sqrt(|u|²+ε²); fills dr = ∂r/∂u. The shared residual
/// of the parallel and stem kinds (E = w·r²).
template <int D>
inline real dir_dot_residual(const VecN<D>& u, const real* ref, real eps, VecN<D>& dr)
{
    real s2 = eps * eps;
    for (int k = 0; k < D; ++k) { s2 += u[k] * u[k]; }
    const real s = std::sqrt(s2);
    real c = 0.0_r;
    for (int k = 0; k < D; ++k) { c += u[k] * ref[k]; }
    c /= s;
    const real inv_s = 1.0_r / s;
    const real inv_s2 = 1.0_r / s2;
    for (int k = 0; k < D; ++k) { dr[k] = (ref[k] * inv_s) - (c * u[k] * inv_s2); }
    return c;
}

/// ε-unit direction q̂ = q/sqrt(|q|²+ε²) and its Jacobian J = ∂q̂/∂q
/// (row-major, symmetric): J = (I − q qᵀ/s²)/s.
template <int D> inline void dir_unit_jac(const VecN<D>& v, real eps, VecN<D>& vh, MatN<D>& J)
{
    real s2 = eps * eps;
    for (int k = 0; k < D; ++k) { s2 += v[k] * v[k]; }
    const real s = std::sqrt(s2);
    const real inv_s = 1.0_r / s;
    const real inv_s3 = inv_s / s2;
    for (int k = 0; k < D; ++k) { vh[k] = v[k] * inv_s; }
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            J[(i * D) + j] = ((i == j) ? inv_s : 0.0_r) - (v[i] * v[j] * inv_s3);
        }
    }
}

/// Energy of one directional sample (full sample, unshared). @p p holds the
/// sample's node positions in slot order (arity 2 or 3 by kind). Unknown
/// (reserved) kinds contribute zero.
template <int D>
inline real dir_window_energy(int kind, const PtN<D>* p, const real* ref, real w, real eps)
{
    if (kind == kDirParallel || kind == kDirStem) {
        const int head = (kind == kDirParallel) ? 1 : 0;
        VecN<D> u;
        for (int k = 0; k < D; ++k) { u[k] = p[head][k] - p[1 - head][k]; }
        VecN<D> dr;
        const real c = dir_dot_residual<D>(u, ref, eps, dr);
        return w * c * c;
    }
    if (kind == kDirLine) {
        VecN<D> q1, q2, h1, h2;
        for (int k = 0; k < D; ++k) {
            q1[k] = p[0][k] - p[1][k];
            q2[k] = p[2][k] - p[1][k];
        }
        MatN<D> J1, J2;
        dir_unit_jac<D>(q1, eps, h1, J1);
        dir_unit_jac<D>(q2, eps, h2, J2);
        real e = 0.0_r;
        for (int k = 0; k < D; ++k) {
            const real r = h1[k] + h2[k];
            e += r * r;
        }
        return w * e;
    }
    return 0.0_r;
}

/// Gradient, Gauss-Newton Hessian block, and energy that one directional
/// sample contributes to the node at @p slot; adds into (grad, hess, e0). The
/// diagonal also gets the coupling-row-sum stabiliser so the per-DOF Newton
/// step stays contractive under frozen neighbours (cf. curvature.hpp).
template <int D>
inline void dir_window_accumulate(int kind,
                                  const PtN<D>* p,
                                  const real* ref,
                                  real w,
                                  real eps,
                                  int slot,
                                  VecN<D>& grad,
                                  MatN<D>& hess,
                                  real& e0)
{
    if (kind == kDirParallel || kind == kDirStem) {
        const int head = (kind == kDirParallel) ? 1 : 0;
        VecN<D> u;
        for (int k = 0; k < D; ++k) { u[k] = p[head][k] - p[1 - head][k]; }
        VecN<D> dr;
        const real c = dir_dot_residual<D>(u, ref, eps, dr);
        e0 += w * c * c;
        const real sgn = (slot == head) ? 1.0_r : -1.0_r;
        const real gc = 2.0_r * w * c * sgn;
        const real hc = 2.0_r * w;
        real dd = 0.0_r;
        for (int k = 0; k < D; ++k) {
            grad[k] += gc * dr[k];
            dd += dr[k] * dr[k];
        }
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) { hess[(i * D) + j] += hc * dr[i] * dr[j]; }
            // The other endpoint's residual gradient is −dr: coupling |dr|².
            hess[(i * D) + i] += hc * dd;
        }
        return;
    }
    if (kind == kDirLine) {
        VecN<D> q1, q2, h1, h2;
        for (int k = 0; k < D; ++k) {
            q1[k] = p[0][k] - p[1][k];
            q2[k] = p[2][k] - p[1][k];
        }
        MatN<D> J1, J2;
        dir_unit_jac<D>(q1, eps, h1, J1);
        dir_unit_jac<D>(q2, eps, h2, J2);
        VecN<D> r;
        real e = 0.0_r;
        for (int k = 0; k < D; ++k) {
            r[k] = h1[k] + h2[k];
            e += r[k] * r[k];
        }
        e0 += w * e;
        // ∂r/∂p_slot: J1 (slot 0 = a), J2 (slot 2 = b), −(J1+J2) (slot 1 = c);
        // all symmetric.
        MatN<D> Js;
        for (int k = 0; k < D * D; ++k) {
            Js[k] = (slot == 0) ? J1[k] : ((slot == 2) ? J2[k] : -(J1[k] + J2[k]));
        }
        const real gc = 2.0_r * w;
        for (int i = 0; i < D; ++i) {
            real gi = 0.0_r;
            for (int j = 0; j < D; ++j) { gi += Js[(i * D) + j] * r[j]; }
            grad[i] += gc * gi;
        }
        // GN block 2w·Js·Jsᵀ (= Js² by symmetry).
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                real h = 0.0_r;
                for (int m = 0; m < D; ++m) { h += Js[(i * D) + m] * Js[(j * D) + m]; }
                hess[(i * D) + j] += gc * h;
            }
        }
        // Coupling stabiliser via Frobenius norms of the three slot Jacobians.
        real f1 = 0.0_r, f2 = 0.0_r, fc = 0.0_r;
        for (int k = 0; k < D * D; ++k) {
            f1 += J1[k] * J1[k];
            f2 += J2[k] * J2[k];
            const real jc = J1[k] + J2[k];
            fc += jc * jc;
        }
        f1 = std::sqrt(f1);
        f2 = std::sqrt(f2);
        fc = std::sqrt(fc);
        const real fs = (slot == 0) ? f1 : ((slot == 2) ? f2 : fc);
        const real fo = f1 + f2 + fc - fs;
        const real stab = gc * fs * fo;
        for (int i = 0; i < D; ++i) { hess[(i * D) + i] += stab; }
        return;
    }
}

/// Arity (used node slots) of a directional kind; 0 for reserved kinds.
inline int dir_kind_arity(int kind)
{
    if (kind == kDirLine) { return 3; }
    if (kind == kDirParallel || kind == kDirStem) { return 2; }
    return 0;
}

/// Accumulate a free DOF's @p K directional samples into (grad, hess, e0),
/// reading node positions from @p X. Layout mirrors the curvature table:
/// nodes [K*4], slot/kind/weight [K], ref/axis [K*D]; slot −1 pads.
template <int D>
inline void dir_dof_accumulate(int K,
                               const int* nodes,
                               const int* slot,
                               const int* kind,
                               const real* ref,
                               const real* axis,
                               const real* weight,
                               real eps,
                               const real* X,
                               VecN<D>& grad,
                               MatN<D>& hess,
                               real& e0)
{
    static_cast<void>(axis);  // projector direction: 3D-only kinds, not yet consumed
    for (int j = 0; j < K; ++j) {
        const int s = slot[j];
        if (s < 0) { continue; }
        const int arity = dir_kind_arity(kind[j]);
        if (arity == 0) { continue; }
        const int* nd = nodes + (j * 4);
        PtN<D> p[3];
        for (int m = 0; m < arity; ++m) { p[m] = load_pt<D>(X, nd[m]); }
        dir_window_accumulate<D>(kind[j], p, ref + (j * D), weight[j], eps, s, grad, hess, e0);
    }
}

/// A free DOF's directional energy with its own node moved to @p trial (the
/// other sample nodes stay at the frozen @p X snapshot) — the line-search
/// increment matching the e0 folded in by dir_dof_accumulate.
template <int D>
inline real dir_dof_trial_energy(int K,
                                 const int* nodes,
                                 const int* slot,
                                 const int* kind,
                                 const real* ref,
                                 const real* axis,
                                 const real* weight,
                                 real eps,
                                 const real* X,
                                 const PtN<D>& trial)
{
    static_cast<void>(axis);
    real e = 0.0_r;
    for (int j = 0; j < K; ++j) {
        const int s = slot[j];
        if (s < 0) { continue; }
        const int arity = dir_kind_arity(kind[j]);
        if (arity == 0) { continue; }
        const int* nd = nodes + (j * 4);
        PtN<D> p[3];
        for (int m = 0; m < arity; ++m) { p[m] = (m == s) ? trial : load_pt<D>(X, nd[m]); }
        e += dir_window_energy<D>(kind[j], p, ref + (j * D), weight[j], eps);
    }
    return e;
}

/// Total directional energy of @p dv at @p X, ACCUMULATED into @p out_e (no
/// initialize_to_identity: runs after reduce_energy_mindet has written the
/// shape total, so the reported energy is the composed objective).
template <int D>
inline void
  reduce_directional_energy(sycl::queue& q, const DirTableViewT<D>& dv, const real* X, real* out_e)
{
    if (dv.num == 0) { return; }
    auto e_red = sycl::reduction(out_e, std::plus<>());
    q.parallel_for(sycl::range<1>(dv.num), e_red, [=](sycl::id<1> idx, auto& e_sum) {
        const std::size_t i = idx[0];
        const int arity = dir_kind_arity(dv.kind[i]);
        if (arity == 0) { return; }
        const int* nd = dv.nodes + (i * 4);
        PtN<D> p[3];
        for (int m = 0; m < arity; ++m) { p[m] = load_pt<D>(X, nd[m]); }
        e_sum += dir_window_energy<D>(dv.kind[i], p, dv.ref + (i * D), dv.weight[i], dv.eps);
    });
}

/// Directional energies of the two line-search trials x + a·corr, a ∈ {a0, a1},
/// ACCUMULATED into out[0] / out[2] after reduce_linesearch_pair wrote the
/// shape totals — the FAS safeguard then gates on the composed objective.
template <int D>
inline void reduce_directional_linesearch_pair(sycl::queue& q,
                                               const DirTableViewT<D>& dv,
                                               const real* X,
                                               const real* corr,
                                               real a0,
                                               real a1,
                                               real* out)
{
    if (dv.num == 0) { return; }
    auto e0_red = sycl::reduction(out + 0, std::plus<>());
    auto e1_red = sycl::reduction(out + 2, std::plus<>());
    q.parallel_for(sycl::range<1>(dv.num),
                   e0_red,
                   e1_red,
                   [=](sycl::id<1> idx, auto& e0_sum, auto& e1_sum) {
                       const std::size_t i = idx[0];
                       const int arity = dir_kind_arity(dv.kind[i]);
                       if (arity == 0) { return; }
                       const int* nd = dv.nodes + (i * 4);
                       PtN<D> px[3], pc[3];
                       for (int m = 0; m < arity; ++m) {
                           px[m] = load_pt<D>(X, nd[m]);
                           pc[m] = load_pt<D>(corr, nd[m]);
                       }
                       const real* ref = dv.ref + (i * D);
                       const real w = dv.weight[i];
                       const auto eval = [&](real a, auto& e_sum) {
                           PtN<D> p[3];
                           for (int m = 0; m < arity; ++m) {
                               for (int k = 0; k < D; ++k) { p[m][k] = px[m][k] + (a * pc[m][k]); }
                           }
                           e_sum += dir_window_energy<D>(dv.kind[i], p, ref, w, dv.eps);
                       };
                       eval(a0, e0_sum);
                       eval(a1, e1_sum);
                   });
}

/// Scatter the directional term's nodal gradient into the fine gradient
/// buffer @p y ([num_nodes * D], pre-filled by the caller). Atomic adds:
/// samples share nodes, and the caller's gather kernels have already run on
/// the same in-order queue. Used by the control solve, whose reduced gradient
/// pulls the fine nodal gradient back into the control space — the accept
/// energies compose the term via the reductions above, so the gradient must
/// see it too or the line search stalls against the model.
template <int D>
inline void
  scatter_directional_grad(sycl::queue& q, const DirTableViewT<D>& dv, const real* X, real* y)
{
    if (dv.num == 0) { return; }
    q.parallel_for(sycl::range<1>(dv.num), [=](sycl::id<1> idx) {
        const std::size_t i = idx[0];
        const int arity = dir_kind_arity(dv.kind[i]);
        if (arity == 0) { return; }
        const int* nd = dv.nodes + (i * 4);
        PtN<D> p[3];
        for (int m = 0; m < arity; ++m) { p[m] = load_pt<D>(X, nd[m]); }
        const real* ref = dv.ref + (i * D);
        for (int m = 0; m < arity; ++m) {
            VecN<D> g {};
            MatN<D> h {};
            real e0 = 0.0_r;
            dir_window_accumulate<D>(dv.kind[i], p, ref, dv.weight[i], dv.eps, m, g, h, e0);
            real* row = y + (static_cast<std::size_t>(nd[m]) * D);
            for (int k = 0; k < D; ++k) {
                sycl::atomic_ref<real,
                                 sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                  slot {row[k]};
                slot.fetch_add(g[k]);
            }
        }
    });
}

}  // namespace egg
