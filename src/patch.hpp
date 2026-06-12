// patch.hpp — per-DOF patch evaluation: the batched closed-form shape_2d
// grad/Hess assembled through the chain Jacobian J, plus patch energy and
// min det A, for one moving DOF over its P corner samples.
//
// Mirrors egg.smoothing.batch.patch_eval / energy_and_mindet and their JAX
// twins (batch_jax.patch_eval_jax / energy_and_mindet_jax) exactly. The layout
// matches batch.py: A[:, k] = s_k·(nbr_k − corner), T = A·W_inv, and
// vec(T) = [T00, T01, T10, T11] (row-major) — the convention metric.hpp uses.
//
// patch_eval returns the raw (grad, hess) in the full 2D space (as the oracle
// does); the tangent reduction for constrained DOFs is newton_delta (mirroring
// batch_jax._newton_delta_one). patch_energy_mindet is the cheaper energy-only
// form that backs the line-search trials. Everything is inline, allocation-free,
// device-callable (reuses metric.hpp / solve.hpp / geometry.hpp).
#pragma once

#include "geometry.hpp"
#include "metric.hpp"
#include "solve.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <variant>

namespace egg
{

// Non-owning view of one DOF's stencil, generalised to D axis-neighbours. The
// per-corner stencil couples a corner to its D axis-neighbours (gn[k]/s[k]); for
// D=2 gn[0]/gn[1] are the old gn0/gn1, so the indices and math are bit-identical.
// W_inv is P×(d×d) row-major; J is P×(jRows×jCols) row-major (make_chain_J).
// role ∈ {0=corner, 1..D=neighbour axis (role−1), -1=absent}.
template <int D> struct PatchViewT {
    int P;
    const int* gc;
    const int* gn[D];     // was gn0, gn1
    const double* s[D];   // was s0, s1
    const double* W_inv;  // [P * dim::wInv(D)]
    const int* role;
    const double* J;  // [P * dim::jSize(D)]
};

template <int D> struct PatchResultT {
    VecN<D> grad;  // dE/dpos (D,)
    MatN<D> hess;  // d²E/dpos² (D×D) row-major
    double energy;
    double mindet;
};

// Minimal energy-only view: just the sample arrays needed for vec(T) and det(A),
// with no role/J. The energy/min-det path takes this so "doesn't use role/J" is
// encoded in the type, not in a comment (#2). PatchViewT is a structural superset,
// so the shared sample_vecT / patch_energy_mindet templates accept either.
template <int D> struct StencilSampleViewT {
    int P;
    const int* gc;
    const int* gn[D];
    const double* s[D];
    const double* W_inv;  // [P * dim::wInv(D)]
};

// det(A): generic *structure*, specialized *arithmetic* where a closed form
// exists, so D=2 stays bit-identical (a generic LU/cofactor is not). A is row-major.
template <int D> inline double det(const MatN<D>& A);
template <> inline double det<2>(const MatN<2>& A) { return A[0] * A[3] - A[1] * A[2]; }

// The single A→T→detA site: vec(T) and det(A) from a corner + its D axis-neighbours,
// the per-axis scales s[k], and W_inv (row-major). A[:,k] = s[k]·(nbr[k]−corner),
// T = A·W_inv; returns vec(T) row-major and writes det(A). For D=2 the matmul
// accumulation order (k outer, j inner) reproduces the old unrolled T00.. exactly.
template <int D>
inline VecTN<D> assemble_vecT(const PtN<D>& corner,
                              const std::array<PtN<D>, D>& nbr,
                              const std::array<double, D>& s,
                              const double* w,
                              double& detA)
{
    MatN<D> A;  // row-major A[i*D + k]; A[:,k] = s[k]·(nbr[k] − corner)
    for (int i = 0; i < D; ++i)
        for (int k = 0; k < D; ++k) A[i * D + k] = s[k] * (nbr[k][i] - corner[i]);
    detA = det<D>(A);
    // T[i,k] = sum_j A[i,j] w[j,k]; keep (i,k) outer, j inner.
    VecTN<D> T;
    for (int i = 0; i < D; ++i)
        for (int k = 0; k < D; ++k) {
            double acc = 0.0;
            for (int j = 0; j < D; ++j) acc += A[i * D + j] * w[j * D + k];
            T[i * D + k] = acc;
        }
    return T;
}

// vec(T) and det(A) for sample p, read from the flat node array via the view's
// gc/gn[] indices. Generic over any view exposing gc/gn[]/s[]/W_inv.
template <int D, class V> inline VecTN<D> sample_vecT(const V& sv, const double* X, int p, double& detA)
{
    const PtN<D> corner = load_pt<D>(X, sv.gc[p]);
    std::array<PtN<D>, D> nbr;
    std::array<double, D> s;
    for (int k = 0; k < D; ++k) {
        nbr[k] = load_pt<D>(X, sv.gn[k][p]);
        s[k] = sv.s[k][p];
    }
    return assemble_vecT<D>(corner, nbr, s, &sv.W_inv[dim::wInv(D) * p], detA);
}

// Patch energy (sum μ) and min det(A) — the cheap trial path. Mirrors
// batch.energy_and_mindet. Templated on the dimension and the Objective.
template <int D, class V, ObjectiveD<D> M = ShapeObjectiveT<D>>
inline void patch_energy_mindet(
  const V& sv, const double* X, double& energy, double& mindet, M objective = {})
{
    energy = 0.0;
    mindet = std::numeric_limits<double>::infinity();
    for (int p = 0; p < sv.P; ++p) {
        double detA;
        const VecTN<D> t = sample_vecT<D>(sv, X, p, detA);
        energy += objective.value(t);
        if (detA < mindet) mindet = detA;
    }
}

// Full patch evaluation: (grad, hess, energy, mindet) in one pass. Mirrors
// batch.patch_eval (numerically identical to the JAX patch_eval_jax). For D=2 the
// accumulation orders match the original unrolled code, so it is bit-identical.
template <int D, ObjectiveD<D> M = ShapeObjectiveT<D>>
inline PatchResultT<D> patch_eval(const PatchViewT<D>& sv, const double* X, M objective = {})
{
    constexpr int kVT = dim::vecT(D);
    constexpr int kJC = dim::jCols(D);
    PatchResultT<D> r {};
    r.grad = VecN<D> {};
    r.hess = MatN<D> {};
    r.energy = 0.0;
    r.mindet = std::numeric_limits<double>::infinity();

    for (int p = 0; p < sv.P; ++p) {
        double detA;
        const VecTN<D> t = sample_vecT<D>(sv, X, p, detA);

        // --- energy & mindet ---
        r.energy += objective.value(t);
        if (detA < r.mindet) r.mindet = detA;

        const double* w = &sv.W_inv[dim::wInv(D) * p];  // row-major d×d

        // --- gradient ---
        // dmu_dT (vec order, row-major) then dmu_dA[i,k] = sum_j dmu_dT[i,j]·w[k,j].
        const GradN<D> g = objective.grad(t);
        double dA[D][D];
        for (int i = 0; i < D; ++i)
            for (int k = 0; k < D; ++k) {
                double acc = 0.0;
                for (int j = 0; j < D; ++j) acc += g[i * D + j] * w[k * D + j];
                dA[i][k] = acc;
            }

        const int role = sv.role[p];
        double c[D];
        for (int i = 0; i < D; ++i) c[i] = 0.0;
        if (role == 0) {  // corner: c[i] = -sum_k s[k]·dA[i][k]
            for (int i = 0; i < D; ++i) {
                double acc = 0.0;
                for (int k = 0; k < D; ++k) acc += sv.s[k][p] * dA[i][k];
                c[i] = -acc;
            }
        } else if (role >= 1) {  // neighbour on axis (role−1)
            const int ax = role - 1;
            for (int i = 0; i < D; ++i) c[i] = sv.s[ax][p] * dA[i][ax];
        }  // role == -1: absent, contributes nothing
        for (int i = 0; i < D; ++i) r.grad[i] += c[i];

        // --- hessian ---
        // Select the D columns of J for this role (cols D·role + {0..D-1}); zero
        // the whole block when the DOF is absent (role < 0). Jb is kVT×D.
        if (role >= 0) {
            const HessN<D> H = objective.hess(t);  // (d²)×(d²) row-major
            const double* Jp = &sv.J[dim::jSize(D) * p];  // jRows×jCols row-major
            const int c0 = D * role;                      // first selected column
            double Jb[kVT][D];
            for (int a = 0; a < kVT; ++a)
                for (int k = 0; k < D; ++k) Jb[a][k] = Jp[a * kJC + c0 + k];
            // HJb[a,j] = sum_b H[a,b] Jb[b,j].
            double HJb[kVT][D];
            for (int a = 0; a < kVT; ++a)
                for (int j = 0; j < D; ++j) {
                    double acc = 0.0;
                    for (int b = 0; b < kVT; ++b) acc += H[a * kVT + b] * Jb[b][j];
                    HJb[a][j] = acc;
                }
            for (int i = 0; i < D; ++i)
                for (int j = 0; j < D; ++j) {
                    double acc = 0.0;
                    for (int a = 0; a < kVT; ++a) acc += Jb[a][i] * HJb[a][j];
                    r.hess[i * D + j] += acc;
                }
        }
    }
    return r;
}

// Newton step δ (D,) for one DOF, dispatched on a concrete (monomorphic) entity.
// Free uses the full D×D Hessian; constrained entities reduce onto the entity
// tangent basis (single column at D=2). Mirrors batch_jax._newton_delta_one.
template <int D, GeometryEntity E>
inline VecN<D> newton_delta(const VecN<D>& g, const MatN<D>& H, const PtN<D>& pos, const E& entity)
{
    if constexpr (std::is_same_v<E, Free>) {
        static_cast<void>(pos);
        static_cast<void>(entity);
        return solveNxN<D>(H, g);
    } else {
        const PtN<D> b = entity.tangent(pos);  // (d, 1) column
        // A_mat = bᵀ H b (scalar); rhs = bᵀ g (scalar).
        double Hb[D];
        for (int i = 0; i < D; ++i) {
            double acc = 0.0;
            for (int j = 0; j < D; ++j) acc += H[i * D + j] * b[j];
            Hb[i] = acc;
        }
        double A = 0.0, rhs = 0.0;
        for (int i = 0; i < D; ++i) {
            A += b[i] * Hb[i];
            rhs += b[i] * g[i];
        }
        const double step = solve1x1(A, rhs);  // -rhs/A with fallback
        VecN<D> d;
        for (int i = 0; i < D; ++i) d[i] = b[i] * step;
        return d;
    }
}

// (tag, params) convenience overload for host-side / oracle callers (e.g. the
// newton_step binding). Visits once; not on the device hot path.
template <int D = kDefaultDim>
inline VecN<D> newton_delta(const VecN<D>& g, const MatN<D>& H, const PtN<D>& pos, Tag tag, const double* params)
{
    return std::visit([&](const auto& e) { return newton_delta<D>(g, H, pos, e); },
                      make_entity<D>(tag, params));
}

// D=2 legacy aliases for the oracle surface and existing call sites.
using PatchView = PatchViewT<2>;
using StencilSampleView = StencilSampleViewT<2>;
using PatchResult = PatchResultT<2>;

}  // namespace egg
