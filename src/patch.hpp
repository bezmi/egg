// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

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

/// Non-owning view of one DOF's stencil, generalised to D axis-neighbours
/// (for D=2, gn[0]/gn[1] are the old gn0/gn1 — bit-identical).
///
/// The bulk per-sample payload (gc/gn/s/W_inv) is deduplicated into one shared
/// table: gc/gn/s/W_inv are the *shared-table base* pointers, indexed by
/// `sample_id[p]`, NOT by p. Only `sample_id` and `role` are per-occurrence
/// (length P, sliced to this DOF). The chain-Jacobian J is not stored — its
/// role-selected block is recomputed in-kernel from s + W_inv (see role_Jb).
/// role: 0 = corner, 1..D = neighbour on axis role−1, −1 = absent.
template <int D> struct PatchViewT {
    int P;
    const int* sample_id;  // [P] index of each occurrence into the shared table
    const int* role;       // [P]
    const int* gc;         // shared-table base; read gc[sample_id[p]]
    const int* gn[D];
    const std::int8_t* s[D];  // per-axis sign ±1; stored as int8, widened on read
    const real* W_inv;        // shared-table base; row sample_id[p] is dim::wInv(D) wide
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

/// det(A), row-major: generic structure, specialized arithmetic per D so the
/// D=2 build stays bit-identical (a generic LU/cofactor is not).
template <int D> inline real det(const MatN<D>& A);
template <> inline real det<2>(const MatN<2>& A) { return (A[0] * A[3]) - (A[1] * A[2]); }
template <> inline real det<3>(const MatN<3>& A)
{
    return (A[0] * ((A[4] * A[8]) - (A[5] * A[7]))) - (A[1] * ((A[3] * A[8]) - (A[5] * A[6]))) +
           (A[2] * ((A[3] * A[7]) - (A[4] * A[6])));
}

/// The single A→T→detA site: A[:,k] = s[k]·(nbr[k]−corner), T = A·W_inv;
/// returns vec(T) row-major and writes det(A). For D=2 the accumulation order
/// (k outer, j inner) reproduces the old unrolled T00.. exactly.
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

/// Shared-table index of occurrence p: `sample_id[p]` when the view carries
/// the indirection (PatchViewT), else p (StencilSampleViewT).
template <class V> inline int table_index(const V& sv, int p)
{
    if constexpr (requires { sv.sample_id; }) {
        return sv.sample_id[p];
    } else {
        return p;
    }
}

/// vec(T) and det(A) for sample p, read from the flat node array through the
/// view's gc/gn[] indices. Generic over any view exposing gc/gn[]/s[]/W_inv.
template <int D, class V> inline VecTN<D> sample_vecT(const V& sv, const real* X, int p, real& detA)
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

/// Patch energy (Σμ) and min det(A) — the cheap trial path (mirrors
/// batch.energy_and_mindet).
template <int D, class V, ObjectiveD<D> M = ShapeObjectiveT<D>>
inline void
  patch_energy_mindet(const V& sv, const real* X, real& energy, real& mindet, M objective = {})
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

/// Line-search trial energy + min det over one DOF's patch, substituting
/// @p trial for the moving node @p dof (every other node loads from @p X).
/// The stored-array twin of structured_patch.hpp::synth_trial_energy_mindet
/// and the single body behind every kernel backtracking loop, so the
/// acceptance inputs cannot drift between the interior/boundary paths. The
/// A→T→detA math is the single source in @ref assemble_vecT; statement order
/// matches the historical in-kernel copies (fp64 bit-identity). Plain inline:
/// the SSCP device pass inlines it into each kernel like the hand-written
/// bodies it replaces (inlining attributes are ignored there anyway).
template <int D, class Obj>
inline void trial_energy_mindet(const PatchViewT<D>& pv,
                                const real* X,
                                int dof,
                                const PtN<D>& trial,
                                Obj& objective,
                                real& e_new,
                                real& mdet)
{
    e_new = 0.0_r;
    mdet = std::numeric_limits<real>::infinity();
    for (int p = 0; p < pv.P; ++p) {
        auto node = [&](int ni) -> PtN<D> { return ni == dof ? trial : load_pt<D>(X, ni); };
        const int sid = pv.sample_id[p];  // shared-table index of occurrence p
        PtN<D> corner = node(pv.gc[sid]);
        std::array<PtN<D>, D> nbr;
        std::array<real, D> sc;
        for (int k = 0; k < D; ++k) {
            nbr[k] = node(pv.gn[k][sid]);
            sc[k] = pv.s[k][sid];
        }
        real detA;
        const VecTN<D> t = assemble_vecT<D>(corner, nbr, sc, &pv.W_inv[pv.w_stride * sid], detA);
        e_new += objective.value(t);
        mdet = std::min(detA, mdet);
    }
}

/// Role-selected chain-Jacobian block Jb (kVT×D, row-major), recomputed from
/// the per-axis signs @p s and W_inv @p w instead of a stored table. Jb is the
/// D columns of the constant J = d·vec(T)/d·coords belonging to the DOF's node.
/// From J's closed form (J[i·d+c, coord] = Σ_k W_inv[k,c]·dA[i,k,coord],
/// dA = ∓s_k on corner/neighbour) the block is sparse: for row a (i = a/D,
/// c = a%D) only column i is nonzero — corner: −Σ_k s_k·W_inv[k,c];
/// axis-ax neighbour: s_ax·W_inv[ax,c]; role < 0 → all-zero.
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

/// Accumulate one metric sample (t, w = W_inv, role, per-axis signs) into the
/// running patch result. Factored out of patch_eval so the stored-array and
/// synthesized structured paths share one audited math body; the caller owns
/// the mindet update. The fused eval_jhj branch (ShapeObjectiveT<3>) folds
/// value + gradient + contracted Hessian without materialising the 81-entry
/// metric Hessian.
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

/// Full patch evaluation: (grad, hess, energy, mindet) in one pass. Mirrors
/// batch.patch_eval (numerically identical to the JAX patch_eval_jax); D=2
/// accumulation order matches the original unrolled code bit-for-bit.
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

/// Tangent-reduced Newton step given a precomputed tangent basis B (D×k):
/// M = BᵀHB, r = Bᵀg, solve M y = −r, δ = B y. k==1 is exactly the legacy
/// b·solve1x1(bᵀHb, bᵀg) path (bit-identical). Split out so the cold
/// (@ref newton_delta) and warm-seeded (sweep kernel) paths share the
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

/// Newton step δ (D,) for one DOF on a concrete (monomorphic) entity.
/// Interior DOFs (k==D) use the full D×D solve; constrained entities reduce
/// onto the entity tangent basis via @ref newton_step_from_basis, evaluated at
/// @p pos (cold path: fresh coarse-grid projection). The k==2 surface arm
/// compiles now but is only exercised once 3D surface entities exist.
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

/// (tag, params) convenience overload for host-side / oracle callers. Selects
/// the concrete entity type once via dispatch_entity_type + decode_entity<E>
/// (no variant); not on the device hot path. Fixed-size entities only.
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
