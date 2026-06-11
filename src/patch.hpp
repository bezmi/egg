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

namespace egg
{

// Non-owning view of one DOF's stencil. Per-sample arrays are length P; W_inv is
// P×(2×2) row-major [w00,w01,w10,w11]; J is P×(4×6) row-major (make_chain_J).
// role ∈ {0=corner, 1=nbr0, 2=nbr1, -1=absent}. Index arrays gc/gn0/gn1 point
// into the flat node array X (node i at X[2i], X[2i+1]).
struct PatchView {
    int P;
    const int* gc;
    const int* gn0;
    const int* gn1;
    const double* s0;
    const double* s1;
    const double* W_inv;  // [P*4]
    const int* role;
    const double* J;  // [P*24]
};

struct PatchResult {
    Vec2 grad;  // dE/dpos (2,)
    Mat2 hess;  // d²E/dpos² (2×2) row-major [h00,h01,h10,h11]
    double energy;
    double mindet;
};

// vec(T) and det(A) for sample p: A col0 = s0·(nbr0−corner), col1 = s1·(nbr1−corner),
// T = A·W_inv. Returns vec(T)=[T00,T01,T10,T11]; writes det(A).
inline VecT sample_vecT(const PatchView& s, const double* X, int p, double& detA)
{
    const double cx = X[2 * s.gc[p]], cy = X[2 * s.gc[p] + 1];
    const double n0x = X[2 * s.gn0[p]], n0y = X[2 * s.gn0[p] + 1];
    const double n1x = X[2 * s.gn1[p]], n1y = X[2 * s.gn1[p] + 1];
    // A = [[a00, a01], [a10, a11]]; col0 = (a00, a10), col1 = (a01, a11).
    const double a00 = s.s0[p] * (n0x - cx);
    const double a10 = s.s0[p] * (n0y - cy);
    const double a01 = s.s1[p] * (n1x - cx);
    const double a11 = s.s1[p] * (n1y - cy);
    detA = a00 * a11 - a01 * a10;
    const double* w = &s.W_inv[4 * p];  // [w00, w01, w10, w11]
    // T[i,k] = sum_j A[i,j] w[j,k].
    const double T00 = a00 * w[0] + a01 * w[2];
    const double T01 = a00 * w[1] + a01 * w[3];
    const double T10 = a10 * w[0] + a11 * w[2];
    const double T11 = a10 * w[1] + a11 * w[3];
    return VecT {T00, T01, T10, T11};
}

// Patch energy (sum μ) and min det(A) — the cheap trial path. Mirrors
// batch.energy_and_mindet. Templated on the Objective (barrier / δ-untangle); the
// default keeps the barrier shape objective for existing call sites.
template <Objective M = ShapeObjective>
inline void patch_energy_mindet(
  const PatchView& s, const double* X, double& energy, double& mindet, M objective = {})
{
    energy = 0.0;
    mindet = std::numeric_limits<double>::infinity();
    for (int p = 0; p < s.P; ++p) {
        double detA;
        const VecT t = sample_vecT(s, X, p, detA);
        energy += objective.value(t);
        if (detA < mindet) mindet = detA;
    }
}

// Full patch evaluation: (grad, hess, energy, mindet) in one pass. Mirrors
// batch.patch_eval (numerically identical to the JAX patch_eval_jax). Templated
// on the Objective; the default keeps the barrier shape objective.
template <Objective M = ShapeObjective>
inline PatchResult patch_eval(const PatchView& s, const double* X, M objective = {})
{
    PatchResult r {};
    r.grad = Vec2 {0.0, 0.0};
    r.hess = Mat2 {0.0, 0.0, 0.0, 0.0};
    r.energy = 0.0;
    r.mindet = std::numeric_limits<double>::infinity();

    for (int p = 0; p < s.P; ++p) {
        double detA;
        const VecT t = sample_vecT(s, X, p, detA);

        // --- energy & mindet ---
        r.energy += objective.value(t);
        if (detA < r.mindet) r.mindet = detA;

        const double* w = &s.W_inv[4 * p];  // [w00, w01, w10, w11]

        // --- gradient ---
        // dmu_dT (2×2, vec order [g00,g01,g10,g11]) then dmu_dA = dmu_dT · W_inv^T.
        const Grad g = objective.grad(t);  // [g00, g01, g10, g11]
        // dmu_dA[i,k] = sum_j dmu_dT[i,j] · w[k,j].
        // col0 = dmu_dA[:,0] = (dmu_dA[0,0], dmu_dA[1,0]); col1 = dmu_dA[:,1].
        const double da00 = g[0] * w[0] + g[1] * w[1];  // i=0,k=0
        const double da01 = g[0] * w[2] + g[1] * w[3];  // i=0,k=1
        const double da10 = g[2] * w[0] + g[3] * w[1];  // i=1,k=0
        const double da11 = g[2] * w[2] + g[3] * w[3];  // i=1,k=1
        const double col0x = da00, col0y = da10;
        const double col1x = da01, col1y = da11;

        const int role = s.role[p];
        double cx = 0.0, cy = 0.0;  // this sample's contribution to grad
        if (role == 0) {            // corner
            cx = -(s.s0[p] * col0x + s.s1[p] * col1x);
            cy = -(s.s0[p] * col0y + s.s1[p] * col1y);
        } else if (role == 1) {  // nbr0
            cx = s.s0[p] * col0x;
            cy = s.s0[p] * col0y;
        } else if (role == 2) {  // nbr1
            cx = s.s1[p] * col1x;
            cy = s.s1[p] * col1y;
        }  // role == -1: absent, contributes nothing
        r.grad[0] += cx;
        r.grad[1] += cy;

        // --- hessian ---
        // Select the 2 columns of J for this role (cols 2*role_safe + {0,1}); zero
        // the whole block when the DOF is absent (role < 0). Jb is 4×2.
        if (role >= 0) {
            const Hess H =
              objective.hess(t);  // 4×4 row-major (barrier closed-form / untangle dual-AD)
            const double* Jp = &s.J[24 * p];  // 4×6 row-major
            const int c0 = 2 * role;          // first selected column
            double Jb[4][2];
            for (int a = 0; a < 4; ++a) {
                Jb[a][0] = Jp[a * 6 + c0];
                Jb[a][1] = Jp[a * 6 + c0 + 1];
            }
            // blk[i,j] = sum_{a,b} Jb[a,i] H[a,b] Jb[b,j]; accumulate into hess.
            // HJb[a,j] = sum_b H[a,b] Jb[b,j].
            double HJb[4][2];
            for (int a = 0; a < 4; ++a)
                for (int j = 0; j < 2; ++j) {
                    double acc = 0.0;
                    for (int b = 0; b < 4; ++b) acc += H[a * 4 + b] * Jb[b][j];
                    HJb[a][j] = acc;
                }
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) {
                    double acc = 0.0;
                    for (int a = 0; a < 4; ++a) acc += Jb[a][i] * HJb[a][j];
                    r.hess[i * 2 + j] += acc;
                }
        }
    }
    return r;
}

// Newton step δ (2,) for one DOF. Free (tag 0) uses the full 2×2 Hessian;
// constrained DOFs reduce onto the entity tangent basis (tag-dispatched). Mirrors
// batch_jax._newton_delta_one. `pos` is the DOF's current position.
inline Vec2 newton_delta(const Vec2& g, const Mat2& H, const Pt& pos, Tag tag, const double* params)
{
    if (tag == TAG_FREE) return solve2x2(H, g);
    const Pt b = tangent_space(pos, tag, params);  // (d, 1) column
    // A_mat = bᵀ H b (scalar); rhs = bᵀ g (scalar).
    const double Hb0 = H[0] * b[0] + H[1] * b[1];
    const double Hb1 = H[2] * b[0] + H[3] * b[1];
    const double A = b[0] * Hb0 + b[1] * Hb1;
    const double rhs = b[0] * g[0] + b[1] * g[1];
    const double step = solve1x1(A, rhs);  // -rhs/A with fallback
    return Vec2 {b[0] * step, b[1] * step};
}

}  // namespace egg
