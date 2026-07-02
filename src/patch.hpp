#pragma once

#include "geometry.hpp"
#include "metric.hpp"
#include "solve.hpp"

#include <algorithm>
#include <array>
#include <limits>

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
    const int* gn[D];
    const real* s[D];
    const real* W_inv;  // [P * dim::wInv(D)]
    const int* role;
    const real* J;  // [P * dim::jSize(D)]
};

template <int D> struct PatchResultT {
    VecN<D> grad;  // dE/dpos (D,)
    MatN<D> hess;  // d²E/dpos² (D×D) row-major
    real energy;
    real mindet;
};

template <int D> struct StencilSampleViewT {
    int P;
    const int* gc;
    const int* gn[D];
    const real* s[D];
    const real* W_inv;  // [P * dim::wInv(D)]
};

// det(A): generic *structure*, specialized *arithmetic* where a closed form
// exists, so D=2 stays bit-identical (a generic LU/cofactor is not). A is row-major.
template <int D> inline real det(const MatN<D>& A);
template <> inline real det<2>(const MatN<2>& A) { return (A[0] * A[3]) - (A[1] * A[2]); }
template <> inline real det<3>(const MatN<3>& A)
{
    return (A[0] * ((A[4] * A[8]) - (A[5] * A[7]))) - (A[1] * ((A[3] * A[8]) - (A[5] * A[6]))) +
           (A[2] * ((A[3] * A[7]) - (A[4] * A[6])));
}

// The single A→T→detA site: vec(T) and det(A) from a corner + its D axis-neighbours,
// the per-axis scales s[k], and W_inv (row-major). A[:,k] = s[k]·(nbr[k]−corner),
// T = A·W_inv; returns vec(T) row-major and writes det(A). For D=2 the matmul
// accumulation order (k outer, j inner) reproduces the old unrolled T00.. exactly.
template <int D>
inline VecTN<D> assemble_vecT(const PtN<D>& corner,
                              const std::array<PtN<D>, D>& nbr,
                              const std::array<real, D>& s,
                              const real* w,
                              real& detA)
{
    MatN<D> A;  // row-major A[i*D + k]; A[:,k] = s[k]·(nbr[k] − corner)
    for (int i = 0; i < D; ++i) {
        for (int k = 0; k < D; ++k) { A[(i * D) + k] = s[k] * (nbr[k][i] - corner[i]); }
    }
    detA = det<D>(A);
    // T[i,k] = sum_j A[i,j] w[j,k]; keep (i,k) outer, j inner.
    VecTN<D> T;
    for (int i = 0; i < D; ++i) {
        for (int k = 0; k < D; ++k) {
            real acc = 0.0_r;
            for (int j = 0; j < D; ++j) { acc += A[(i * D) + j] * w[(j * D) + k]; }
            T[(D * i) + k] = acc;
        }
    }
    return T;
}

// vec(T) and det(A) for sample p, read from the flat node array via the view's
// gc/gn[] indices. Generic over any view exposing gc/gn[]/s[]/W_inv.
template <int D, class V>
inline VecTN<D> sample_vecT(const V& sv, const real* X, int p, real& detA)
{
    const PtN<D> corner = load_pt<D>(X, sv.gc[p]);
    std::array<PtN<D>, D> nbr;
    std::array<real, D> s;
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
  const V& sv, const real* X, real& energy, real& mindet, M objective = {})
{
    energy = 0.0_r;
    mindet = std::numeric_limits<real>::infinity();
    for (int p = 0; p < sv.P; ++p) {
        real detA;
        const VecTN<D> t = sample_vecT<D>(sv, X, p, detA);
        energy += objective.value(t);
        mindet = std::min(detA, mindet);
    }
}

// Full patch evaluation: (grad, hess, energy, mindet) in one pass. Mirrors
// batch.patch_eval (numerically identical to the JAX patch_eval_jax). For D=2 the
// accumulation orders match the original unrolled code, so it is bit-identical.
template <int D, ObjectiveD<D> M = ShapeObjectiveT<D>>
__attribute__((noinline)) inline PatchResultT<D>
  patch_eval(const PatchViewT<D>& sv, const real* X, M objective = {})
{
    constexpr int kVT = dim::vecT(D);
    constexpr int kJC = dim::jCols(D);
    // Fused path when the objective offers eval_jhj (ShapeObjectiveT<3>): one CSE
    // pass yields value + metric gradient + the contracted Hessian Jbᵀ H Jb,
    // without ever materialising the 81-entry metric Hessian (the live state that
    // pins the 3D sweep kernel at 256 VGPR). Otherwise the separate
    // value/grad/hess path (D=2 closed form, or the AD untangle objective) runs.
    constexpr bool kUseJhj = requires(M m, VecTN<D> tt, real* d) { m.eval_jhj(tt, d, d, d, d); };
    PatchResultT<D> r {};
    r.grad = VecN<D> {};
    r.hess = MatN<D> {};
    r.energy = 0.0_r;
    r.mindet = std::numeric_limits<real>::infinity();

    for (int p = 0; p < sv.P; ++p) {
        real detA;
        const VecTN<D> t = sample_vecT<D>(sv, X, p, detA);
        if (detA < r.mindet) { r.mindet = detA; }
        const real* w = &sv.W_inv[dim::wInv(D) * p];  // row-major d×d
        const int role = sv.role[p];

        // Metric gradient g (vec order, row-major). The fused path also folds in
        // the energy and the contracted-Hessian contribution here.
        GradN<D> g;
        if constexpr (kUseJhj) {
            // Role-selected Jb (kVT×D, row-major); zero when the DOF is absent so
            // the returned jhj is itself zero (the Hessian block is role-gated).
            real Jb[kVT * D];
            for (int e = 0; e < kVT * D; ++e) { Jb[e] = 0.0_r; }
            if (role >= 0) {
                const real* Jp = &sv.J[dim::jSize(D) * p];
                const int c0 = D * role;
                for (int a = 0; a < kVT; ++a) {
                    for (int k = 0; k < D; ++k) { Jb[(a * D) + k] = Jp[(a * kJC) + c0 + k]; }
                }
            }
            real val = 0.0_r;
            real gg[kVT];
            real jhj[D * D];
            objective.eval_jhj(t, Jb, &val, gg, jhj);
            r.energy += val;
            for (int i = 0; i < kVT; ++i) { g[i] = gg[i]; }
            for (int i = 0; i < D * D; ++i) { r.hess[i] += jhj[i]; }
        } else {
            r.energy += objective.value(t);
            g = objective.grad(t);
        }

        // --- gradient contraction (common) ---
        // dmu_dA[i,k] = sum_j dmu_dT[i,j]·w[k,j].
        real dA[D][D];
        for (int i = 0; i < D; ++i) {
            for (int k = 0; k < D; ++k) {
                real acc = 0.0_r;
                for (int j = 0; j < D; ++j) { acc += g[(i * D) + j] * w[(k * D) + j]; }
                dA[i][k] = acc;
            }
        }
        real c[D];
        for (int i = 0; i < D; ++i) { c[i] = 0.0_r; }
        if (role == 0) {  // corner: c[i] = -sum_k s[k]·dA[i][k]
            for (int i = 0; i < D; ++i) {
                real acc = 0.0_r;
                for (int k = 0; k < D; ++k) { acc += sv.s[k][p] * dA[i][k]; }
                c[i] = -acc;
            }
        } else if (role >= 1) {  // neighbour on axis (role−1)
            const int ax = role - 1;
            for (int i = 0; i < D; ++i) { c[i] = sv.s[ax][p] * dA[i][ax]; }
        }  // role == -1: absent, contributes nothing
        for (int i = 0; i < D; ++i) { r.grad[i] += c[i]; }

        // --- hessian (non-fused path) ---
        // Select the D columns of J for this role (cols D·role + {0..D-1}); zero
        // the whole block when the DOF is absent (role < 0). Jb is kVT×D.
        if constexpr (!kUseJhj) {
            if (role >= 0) {
                const HessN<D> H = objective.hess(t);         // (d²)×(d²) row-major
                const real* Jp = &sv.J[dim::jSize(D) * p];  // jRows×jCols row-major
                const int c0 = D * role;                      // first selected column
                real Jb[kVT][D];
                for (int a = 0; a < kVT; ++a) {
                    for (int k = 0; k < D; ++k) { Jb[a][k] = Jp[(a * kJC) + c0 + k]; }
                }
                // HJb[a,j] = sum_b H[a,b] Jb[b,j].
                real HJb[kVT][D];
                for (int a = 0; a < kVT; ++a) {
                    for (int j = 0; j < D; ++j) {
                        real acc = 0.0_r;
                        for (int b = 0; b < kVT; ++b) { acc += H[(a * kVT) + b] * Jb[b][j]; }
                        HJb[a][j] = acc;
                    }
                }
                for (int i = 0; i < D; ++i) {
                    for (int j = 0; j < D; ++j) {
                        real acc = 0.0_r;
                        for (int a = 0; a < kVT; ++a) { acc += Jb[a][i] * HJb[a][j]; }
                        r.hess[(i * D) + j] += acc;
                    }
                }
            }
        }
    }
    return r;
}

/// @brief Newton step @f$ \delta @f$ (D,) for one DOF on a concrete (monomorphic) entity.
///
/// Interior DOFs (@c k==D) use the full @f$ D \times D @f$ Hessian; constrained
/// entities reduce onto the entity tangent basis @f$ B @f$ (@f$ D \times k @f$):
/// @f$ M = B^\top H B @f$, @f$ r = B^\top g @f$, solve @f$ M y = -r @f$, then
/// @f$ \delta = B y @f$. The @c k==1 curve case is special-cased to `solve1x1`,
/// reproducing the legacy single-column reduction bit-identically; the @c k==2
/// surface arm (@c k==2) compiles now but is only exercised once 3D surface
/// entities exist.
/// @tparam D Embedding dimension.
/// @tparam E Entity type (satisfies @ref GeometryEntity); supplies the tangent basis.
/// @param g Gradient @f$ g @f$ at the DOF.
/// @param H Hessian @f$ H @f$ at the DOF.
/// @param pos Current node position (where the tangent basis is evaluated).
/// @param entity The boundary entity constraining the DOF.
/// @return The Newton step @f$ \delta @f$.
/// @brief The tangent-reduced Newton step given a precomputed tangent basis @p B.
///
/// Reduce onto the k tangent columns B (D×k): M = BᵀHB (k×k), r = Bᵀg (k); solve
/// M y = −r, then δ = B y. For k==1 this is exactly the legacy
/// b·solve1x1(bᵀHb, bᵀg) path (bit-identical single-column reduction). Split out
/// so the cold (@ref newton_delta) and warm-seeded (sweep kernel) paths share the
/// reduction and differ only in how B is obtained.
template <int D, int k>
inline VecN<D>
  newton_step_from_basis(const VecN<D>& g, const MatN<D>& H, const std::array<PtN<D>, k>& B)
{
    MatN<k> M {};
    VecN<k> r {};
    for (int a = 0; a < k; ++a) {
        real Hb[D];
        for (int i = 0; i < D; ++i) {
            real s = 0.0_r;
            for (int j = 0; j < D; ++j) { s += H[(i * D) + j] * B[a][j]; }
            Hb[i] = s;
        }
        for (int bcol = 0; bcol < k; ++bcol) {
            real s = 0.0_r;
            for (int i = 0; i < D; ++i) { s += B[bcol][i] * Hb[i]; }
            M[(a * k) + bcol] = s;
        }
        real s = 0.0_r;
        for (int i = 0; i < D; ++i) { s += B[a][i] * g[i]; }
        r[a] = s;
    }
    VecN<k> y;
    if constexpr (k == 1) {
        y = VecN<1> {solve1x1(M[0], r[0])};  // -r/M with fallback (legacy path)
    } else {
        y = solveNxN<k>(M, r);
    }
    VecN<D> d {};
    for (int i = 0; i < D; ++i) {
        real s = 0.0_r;
        for (int a = 0; a < k; ++a) { s += B[a][i] * y[a]; }
        d[i] = s;
    }
    return d;
}

template <int D, GeometryEntity E>
inline VecN<D> newton_delta(const VecN<D>& g, const MatN<D>& H, const PtN<D>& pos, const E& entity)
{
    constexpr int k = E::tdim;  // Free: k==D; surface: k==D-1; curve: k==1
    if constexpr (k == D) {
        static_cast<void>(pos);
        static_cast<void>(entity);
        return solveNxN<D>(H, g);  // interior DOF, full d×d solve
    } else {
        // Cold path: the tangent basis comes from a fresh (coarse-grid) projection.
        return newton_step_from_basis<D, k>(g, H, entity.tangent_basis(pos));
    }
}

// (tag, params) convenience overload for host-side / oracle callers (e.g. the
// newton_step binding). Selects the concrete entity type once via
// dispatch_entity_type + decode_entity<E> (no std::visit, no variant); not on
// the device hot path. Fixed-size entities only (no arena).
template <int D = kDefaultDim>
inline VecN<D>
  newton_delta(const VecN<D>& g, const MatN<D>& H, const PtN<D>& pos, Tag tag, const real* params)
{
    VecN<D> out {};
    dispatch_entity_type<D>(static_cast<EntityTag>(tag), [&]<class E>() {
        out = newton_delta<D>(g, H, pos, decode_entity<E>(params));
    });
    return out;
}

// D=2 legacy aliases for the oracle surface and existing call sites.
using PatchView = PatchViewT<2>;
using StencilSampleView = StencilSampleViewT<2>;
using PatchResult = PatchResultT<2>;

}  // namespace egg
