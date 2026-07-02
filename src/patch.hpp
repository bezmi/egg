#pragma once

#include "geometry.hpp"
#include "metric.hpp"
#include "solve.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>

namespace egg
{

// Non-owning view of one DOF's stencil, generalised to D axis-neighbours. The
// per-corner stencil couples a corner to its D axis-neighbours (gn[k]/s[k]); for
// D=2 gn[0]/gn[1] are the old gn0/gn1, so the indices and math are bit-identical.
//
// The bulk per-sample payload (gc/gn/s/W_inv) is deduplicated into one shared
// table — identical (cell,corner) samples appear in every corner DOF's patch but
// are stored once. So gc/gn/s/W_inv are the *shared-table base* pointers and are
// indexed by `sample_id[p]`, NOT by `p`. Only `sample_id` and `role` are
// per-occurrence (length P, sliced to this DOF). W_inv row stride is dim::wInv(D),
// keyed on the table index. The chain-Jacobian J is not stored — its role-selected
// block is recomputed in-kernel from s + W_inv (see role_Jb).
// role ∈ {0=corner, 1..D=neighbour axis (role−1), -1=absent}.
template <int D> struct PatchViewT {
    int P;
    const int* sample_id;  // [P] index of each occurrence into the shared table
    const int* role;       // [P]
    const int* gc;         // shared-table base; read gc[sample_id[p]]
    const int* gn[D];
    const std::int8_t* s[D];  // per-axis sign ±1; stored as int8, widened on read
    const real* W_inv;  // shared-table base; row sample_id[p] is dim::wInv(D) wide
    // Row stride into W_inv (dim::wInv(D) normally). A uniform W_inv table (every
    // sample shares one row, e.g. an identity target) is stored as a single row
    // with stride 0, so every sample reads row 0; the read site multiplies the
    // table index by this stride, so the kernel is unchanged otherwise.
    int w_stride = dim::wInv(D);
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

// Shared-table index of occurrence p: a PatchViewT carries a `sample_id`
// indirection (deduplicated payload), while a raw stencil view (StencilSampleViewT)
// stores its payload one-per-occurrence and indexes directly by p.
template <class V> inline int table_index(const V& sv, int p)
{
    if constexpr (requires { sv.sample_id; }) {
        return sv.sample_id[p];
    } else {
        return p;
    }
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
    // W_inv row stride is dim::wInv(D) for a per-sample table, or this view's
    // w_stride when it carries one (0 for a uniform table → always row 0).
    std::size_t woff = static_cast<std::size_t>(dim::wInv(D)) * static_cast<std::size_t>(p);
    if constexpr (requires { sv.w_stride; }) {
        woff = static_cast<std::size_t>(sv.w_stride) * static_cast<std::size_t>(p);
    }
    return assemble_vecT<D>(corner, nbr, s, &sv.W_inv[woff], detA);
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
        const VecTN<D> t = sample_vecT<D>(sv, X, table_index(sv, p), detA);
        energy += objective.value(t);
        mindet = std::min(detA, mindet);
    }
}

// Role-selected chain-Jacobian block Jb (kVT×D, row-major), recomputed from the
// per-axis signs @p s and W_inv @p w (row-major d×d) rather than read from a
// stored table. Jb is the D columns of the constant J = d·vec(T)/d·coords that
// belong to the DOF's node (role 0 = corner, role r≥1 = axis-(r−1) neighbour).
//
// J's closed form, J[i·d+c, coord] = Σ_k W_inv[k,c]·dA[i,k,coord] with
// dA[i,k,·] = −s_k on corner[i] and +s_k on nbr_k[i], makes the selected block
// sparse: for output row a (i = a/D, c = a%D) only column i is nonzero. So the
// corner block is Jb[a][i] = −Σ_k s_k·W_inv[k,c], and the axis-`ax` neighbour
// block is Jb[a][i] = s_ax·W_inv[ax,c]. role < 0 (absent) → all-zero block.
template <int D> inline void role_Jb(int role, const real* s, const real* w, real* Jb)
{
    constexpr int kVT = dim::vecT(D);
    for (int e = 0; e < kVT * D; ++e) { Jb[e] = 0.0_r; }
    if (role == 0) {
        for (int a = 0; a < kVT; ++a) {
            const int i = a / D, c = a % D;
            real acc = 0.0_r;
            for (int k = 0; k < D; ++k) { acc += s[k] * w[(k * D) + c]; }
            Jb[(a * D) + i] = -acc;
        }
    } else if (role >= 1) {
        const int ax = role - 1;
        for (int a = 0; a < kVT; ++a) {
            const int i = a / D, c = a % D;
            Jb[(a * D) + i] = s[ax] * w[(ax * D) + c];
        }
    }
}

// Accumulate one metric sample into the running patch result, given its vec(T)
// `t`, its row-major d×d `w` (= W_inv), the DOF's `role`, and the per-axis signs
// `svals`. Factored out of patch_eval so the stored-array path and the
// synthesized structured path share one audited math body; the caller supplies
// (t, w, role, svals) from whichever source and owns the mindet update. The
// fused eval_jhj branch (ShapeObjectiveT<3>) folds value + gradient + contracted
// Hessian without materialising the 81-entry metric Hessian.
template <int D, ObjectiveD<D> M>
inline void accumulate_sample(
  M& objective, const VecTN<D>& t, const real* w, int role, const real* svals, PatchResultT<D>& r)
{
    constexpr int kVT = dim::vecT(D);
    constexpr bool kUseJhj = requires(M m, VecTN<D> tt, real* d) { m.eval_jhj(tt, d, d, d, d); };

    // Metric gradient g (vec order, row-major). The fused path also folds in the
    // energy and the contracted-Hessian contribution here.
    GradN<D> g;
    if constexpr (kUseJhj) {
        // Role-selected Jb (kVT×D, row-major), recomputed from s + W_inv; zero
        // when the DOF is absent so the returned jhj is itself zero.
        real Jb[kVT * D];
        role_Jb<D>(role, svals, w, Jb);
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
            for (int k = 0; k < D; ++k) { acc += svals[k] * dA[i][k]; }
            c[i] = -acc;
        }
    } else if (role >= 1) {  // neighbour on axis (role−1)
        const int ax = role - 1;
        for (int i = 0; i < D; ++i) { c[i] = svals[ax] * dA[i][ax]; }
    }  // role == -1: absent, contributes nothing
    for (int i = 0; i < D; ++i) { r.grad[i] += c[i]; }

    // --- hessian (non-fused path) ---
    // Role-selected Jb (kVT×D, recomputed from s + W_inv); skip entirely when
    // the DOF is absent (role < 0) so the costly metric Hessian is not formed.
    if constexpr (!kUseJhj) {
        if (role >= 0) {
            const HessN<D> H = objective.hess(t);  // (d²)×(d²) row-major
            real Jbf[kVT * D];
            role_Jb<D>(role, svals, w, Jbf);
            const auto Jb = [&](int a, int k) -> real { return Jbf[(a * D) + k]; };
            // HJb[a,j] = sum_b H[a,b] Jb[b,j].
            real HJb[kVT][D];
            for (int a = 0; a < kVT; ++a) {
                for (int j = 0; j < D; ++j) {
                    real acc = 0.0_r;
                    for (int b = 0; b < kVT; ++b) { acc += H[(a * kVT) + b] * Jb(b, j); }
                    HJb[a][j] = acc;
                }
            }
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    real acc = 0.0_r;
                    for (int a = 0; a < kVT; ++a) { acc += Jb(a, i) * HJb[a][j]; }
                    r.hess[(i * D) + j] += acc;
                }
            }
        }
    }
}

// Full patch evaluation: (grad, hess, energy, mindet) in one pass. Mirrors
// batch.patch_eval (numerically identical to the JAX patch_eval_jax). For D=2 the
// accumulation orders match the original unrolled code, so it is bit-identical.
template <int D, ObjectiveD<D> M = ShapeObjectiveT<D>>
__attribute__((noinline)) inline PatchResultT<D>
  patch_eval(const PatchViewT<D>& sv, const real* X, M objective = {})
{
    PatchResultT<D> r {};
    r.grad = VecN<D> {};
    r.hess = MatN<D> {};
    r.energy = 0.0_r;
    r.mindet = std::numeric_limits<real>::infinity();

    for (int p = 0; p < sv.P; ++p) {
        const int sid = sv.sample_id[p];  // shared-table index of this occurrence
        real detA;
        const VecTN<D> t = sample_vecT<D>(sv, X, sid, detA);
        if (detA < r.mindet) { r.mindet = detA; }
        const real* w = &sv.W_inv[sv.w_stride * sid];  // row-major d×d (stride 0 ⇒ uniform)
        const int role = sv.role[p];
        real svals[D];
        for (int k = 0; k < D; ++k) { svals[k] = sv.s[k][sid]; }
        accumulate_sample<D>(objective, t, w, role, svals, r);
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
