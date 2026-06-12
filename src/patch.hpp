#pragma once

#include "geometry.hpp"
#include "metric.hpp"
#include "solve.hpp"

#include <algorithm>
#include <array>
#include <limits>
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
    const int* gn[D];
    const double* s[D];
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
template <> inline double det<2>(const MatN<2>& A) { return (A[0] * A[3]) - (A[1] * A[2]); }

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
    for (int i = 0; i < D; ++i) {
        for (int k = 0; k < D; ++k) { A[(i * D) + k] = s[k] * (nbr[k][i] - corner[i]); }
    }
    detA = det<D>(A);
    // T[i,k] = sum_j A[i,j] w[j,k]; keep (i,k) outer, j inner.
    VecTN<D> T;
    for (int i = 0; i < D; ++i) {
        for (int k = 0; k < D; ++k) {
            double acc = 0.0;
            for (int j = 0; j < D; ++j) { acc += A[(i * D) + j] * w[(j * D) + k]; }
            T[(D * i) + k] = acc;
        }
    }
    return T;
}

// vec(T) and det(A) for sample p, read from the flat node array via the view's
// gc/gn[] indices. Generic over any view exposing gc/gn[]/s[]/W_inv.
template <int D, class V>
inline VecTN<D> sample_vecT(const V& sv, const double* X, int p, double& detA)
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
        mindet = std::min(detA, mindet);
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
        if (detA < r.mindet) { r.mindet = detA; }

        const double* w = &sv.W_inv[dim::wInv(D) * p];  // row-major d×d

        // --- gradient ---
        // dmu_dT (vec order, row-major) then dmu_dA[i,k] = sum_j dmu_dT[i,j]·w[k,j].
        const GradN<D> g = objective.grad(t);
        double dA[D][D];
        for (int i = 0; i < D; ++i) {
            for (int k = 0; k < D; ++k) {
                double acc = 0.0;
                for (int j = 0; j < D; ++j) { acc += g[(i * D) + j] * w[(k * D) + j]; }
                dA[i][k] = acc;
            }
        }

        const int role = sv.role[p];
        double c[D];
        for (int i = 0; i < D; ++i) { c[i] = 0.0; }
        if (role == 0) {  // corner: c[i] = -sum_k s[k]·dA[i][k]
            for (int i = 0; i < D; ++i) {
                double acc = 0.0;
                for (int k = 0; k < D; ++k) { acc += sv.s[k][p] * dA[i][k]; }
                c[i] = -acc;
            }
        } else if (role >= 1) {  // neighbour on axis (role−1)
            const int ax = role - 1;
            for (int i = 0; i < D; ++i) { c[i] = sv.s[ax][p] * dA[i][ax]; }
        }  // role == -1: absent, contributes nothing
        for (int i = 0; i < D; ++i) { r.grad[i] += c[i]; }

        // --- hessian ---
        // Select the D columns of J for this role (cols D·role + {0..D-1}); zero
        // the whole block when the DOF is absent (role < 0). Jb is kVT×D.
        if (role >= 0) {
            const HessN<D> H = objective.hess(t);         // (d²)×(d²) row-major
            const double* Jp = &sv.J[dim::jSize(D) * p];  // jRows×jCols row-major
            const int c0 = D * role;                      // first selected column
            double Jb[kVT][D];
            for (int a = 0; a < kVT; ++a) {
                for (int k = 0; k < D; ++k) { Jb[a][k] = Jp[(a * kJC) + c0 + k]; }
            }
            // HJb[a,j] = sum_b H[a,b] Jb[b,j].
            double HJb[kVT][D];
            for (int a = 0; a < kVT; ++a)
                for (int j = 0; j < D; ++j) {
                    double acc = 0.0;
                    for (int b = 0; b < kVT; ++b) { acc += H[(a * kVT) + b] * Jb[b][j]; }
                    HJb[a][j] = acc;
                }
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    double acc = 0.0;
                    for (int a = 0; a < kVT; ++a) { acc += Jb[a][i] * HJb[a][j]; }
                    r.hess[(i * D) + j] += acc;
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
template <int D, GeometryEntity E>
inline VecN<D> newton_delta(const VecN<D>& g, const MatN<D>& H, const PtN<D>& pos, const E& entity)
{
    constexpr int k = E::tdim;  // Free: k==D; surface: k==D-1; curve: k==1
    if constexpr (k == D) {
        static_cast<void>(pos);
        static_cast<void>(entity);
        return solveNxN<D>(H, g);  // interior DOF, full d×d solve
    } else {
        // Reduce onto the k tangent columns B (D×k): M = BᵀHB (k×k), r = Bᵀg (k).
        // Solve M y = −r, then δ = B y. For k==1 this is exactly the legacy
        // b·solve1x1(bᵀHb, bᵀg) path (bit-identical single-column reduction).
        const std::array<PtN<D>, k> B = entity.tangent_basis(pos);
        MatN<k> M {};
        VecN<k> r {};
        for (int a = 0; a < k; ++a) {
            double Hb[D];
            for (int i = 0; i < D; ++i) {
                double s = 0.0;
                for (int j = 0; j < D; ++j) { s += H[(i * D) + j] * B[a][j]; }
                Hb[i] = s;
            }
            for (int bcol = 0; bcol < k; ++bcol) {
                double s = 0.0;
                for (int i = 0; i < D; ++i) { s += B[bcol][i] * Hb[i]; }
                M[(a * k) + bcol] = s;
            }
            double s = 0.0;
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
            double s = 0.0;
            for (int a = 0; a < k; ++a) { s += B[a][i] * y[a]; }
            d[i] = s;
        }
        return d;
    }
}

// (tag, params) convenience overload for host-side / oracle callers (e.g. the
// newton_step binding). Visits once; not on the device hot path.
template <int D = kDefaultDim>
inline VecN<D>
  newton_delta(const VecN<D>& g, const MatN<D>& H, const PtN<D>& pos, Tag tag, const double* params)
{
    return std::visit([&](const auto& e) { return newton_delta<D>(g, H, pos, e); },
                      make_entity<D>(tag, params));
}

// D=2 legacy aliases for the oracle surface and existing call sites.
using PatchView = PatchViewT<2>;
using StencilSampleView = StencilSampleViewT<2>;
using PatchResult = PatchResultT<2>;

}  // namespace egg
