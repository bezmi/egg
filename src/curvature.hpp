// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

/// @file curvature.hpp
/// Block-interface curvature-continuity (C2) functional over a four-node grid
/// line window (2D). Energy = weight·(theta1 − theta2)², theta1 = turn(p0,p1,p2),
/// theta2 = turn(p1,p2,p3), turn = signed angle between successive edges. Zero on
/// a straight line and on a constant-curvature arc; positive at a kink. Mirrors
/// egg.smoothing.interface_c2 exactly (the NumPy oracle): the gradient is the
/// closed form (perp(v)/|v|² is the gradient of a vector's angle) and the
/// per-node Hessian is the Gauss-Newton block 2·weight·(dr/dp)(dr/dp)ᵀ (PSD),
/// so the interior Newton solve stays well-posed. The det barrier is untouched,
/// so the term only adds energy and never inverts a cell.
#pragma once

#include "core.hpp"

#include <array>
#include <cmath>

namespace egg
{

/// Rotate a 2-vector by +90 degrees: perp((x, y)) = (−y, x).
inline VecN<2> curv_perp(const VecN<2>& v) { return VecN<2> {-v[1], v[0]}; }

/// Signed turning angle at b along a→b→c: angle(c−b) − angle(b−a).
inline real curv_turn(const PtN<2>& a, const PtN<2>& b, const PtN<2>& c)
{
    const VecN<2> e1 {b[0] - a[0], b[1] - a[1]};
    const VecN<2> e2 {c[0] - b[0], c[1] - b[1]};
    const real cross = (e1[0] * e2[1]) - (e1[1] * e2[0]);
    const real dt = (e1[0] * e2[0]) + (e1[1] * e2[1]);
    return std::atan2(cross, dt);
}

/// Curvature-continuity energy of one four-node window (unweighted).
inline real curv_window_energy(const PtN<2>& p0,
                               const PtN<2>& p1,
                               const PtN<2>& p2,
                               const PtN<2>& p3)
{
    const real r = curv_turn(p0, p1, p2) - curv_turn(p1, p2, p3);
    return r * r;
}

/// dr/dp for the four window slots, where r = turn(p0,p1,p2) − turn(p1,p2,p3).
/// The middle edge p2−p1 is shared by both angles, so g1b == g2a collapses the
/// slot-1/slot-2 rows (see egg.smoothing.interface_c2). Returns r; fills @p dr.
inline real curv_window_residual_grad(const PtN<2>& p0,
                                      const PtN<2>& p1,
                                      const PtN<2>& p2,
                                      const PtN<2>& p3,
                                      std::array<VecN<2>, 4>& dr)
{
    const VecN<2> e1a {p1[0] - p0[0], p1[1] - p0[1]};
    const VecN<2> e2a {p2[0] - p1[0], p2[1] - p1[1]};  // shared middle edge
    const VecN<2> e2b {p3[0] - p2[0], p3[1] - p2[1]};

    const real l1a = (e1a[0] * e1a[0]) + (e1a[1] * e1a[1]);
    const real l2a = (e2a[0] * e2a[0]) + (e2a[1] * e2a[1]);
    const real l2b = (e2b[0] * e2b[0]) + (e2b[1] * e2b[1]);
    const real i1a = l1a > 0.0_r ? 1.0_r / l1a : 0.0_r;
    const real i2a = l2a > 0.0_r ? 1.0_r / l2a : 0.0_r;
    const real i2b = l2b > 0.0_r ? 1.0_r / l2b : 0.0_r;

    const VecN<2> g1a {-e1a[1] * i1a, e1a[0] * i1a};  // perp(e1a)/|e1a|²
    const VecN<2> g2a {-e2a[1] * i2a, e2a[0] * i2a};  // perp(e2a)/|e2a|² (= g1b)
    const VecN<2> g2b {-e2b[1] * i2b, e2b[0] * i2b};  // perp(e2b)/|e2b|²

    dr[0] = g1a;  // dr/dp0
    dr[1] = VecN<2> {-g1a[0] - (2.0_r * g2a[0]), -g1a[1] - (2.0_r * g2a[1])};  // dr/dp1
    dr[2] = VecN<2> {(2.0_r * g2a[0]) + g2b[0], (2.0_r * g2a[1]) + g2b[1]};    // dr/dp2
    dr[3] = VecN<2> {-g2b[0], -g2b[1]};  // dr/dp3

    const real cross1 = (e1a[0] * e2a[1]) - (e1a[1] * e2a[0]);
    const real dot1 = (e1a[0] * e2a[0]) + (e1a[1] * e2a[1]);
    const real cross2 = (e2a[0] * e2b[1]) - (e2a[1] * e2b[0]);
    const real dot2 = (e2a[0] * e2b[0]) + (e2a[1] * e2b[1]);
    return std::atan2(cross1, dot1) - std::atan2(cross2, dot2);
}

/// Gradient (2-vec) and Gauss-Newton Hessian block (2×2) that window slot @p
/// slot (0..3) contributes to its node, scaled by @p weight. Adds into @p grad
/// and @p hess (row-major) so a node can accumulate every window it is in.
inline void curv_window_accumulate(const PtN<2>& p0,
                                   const PtN<2>& p1,
                                   const PtN<2>& p2,
                                   const PtN<2>& p3,
                                   int slot,
                                   real weight,
                                   VecN<2>& grad,
                                   MatN<2>& hess)
{
    std::array<VecN<2>, 4> dr {};
    const real r = curv_window_residual_grad(p0, p1, p2, p3, dr);
    const VecN<2>& d = dr[slot];
    const real gcoef = 2.0_r * weight * r;
    grad[0] += gcoef * d[0];
    grad[1] += gcoef * d[1];
    const real hcoef = 2.0_r * weight;
    hess[0] += hcoef * d[0] * d[0];
    hess[1] += hcoef * d[0] * d[1];
    hess[2] += hcoef * d[1] * d[0];
    hess[3] += hcoef * d[1] * d[1];
}

/// Accumulate a free DOF's @p K curvature windows into (grad, hess, e0), reading
/// node positions from @p X. @p nodes is [K*4] structured node ids, @p slot [K]
/// the DOF's slot in each window (−1 = pad), @p weight [K] the window weight.
inline void curv_dof_accumulate(int K,
                                const int* nodes,
                                const int* slot,
                                const real* weight,
                                const real* X,
                                VecN<2>& grad,
                                MatN<2>& hess,
                                real& e0)
{
    for (int j = 0; j < K; ++j) {
        const int s = slot[j];
        if (s < 0) { continue; }
        const int* wn = nodes + (j * 4);
        const PtN<2> p0 = load_pt<2>(X, wn[0]);
        const PtN<2> p1 = load_pt<2>(X, wn[1]);
        const PtN<2> p2 = load_pt<2>(X, wn[2]);
        const PtN<2> p3 = load_pt<2>(X, wn[3]);
        std::array<VecN<2>, 4> dr {};
        const real r = curv_window_residual_grad(p0, p1, p2, p3, dr);
        const real w = weight[j];
        // Gauss-Newton gradient/Hessian for this DOF's slot.
        const VecN<2>& d = dr[s];
        const real gcoef = 2.0_r * w * r;
        grad[0] += gcoef * d[0];
        grad[1] += gcoef * d[1];
        const real hc = 2.0_r * w;
        hess[0] += hc * d[0] * d[0];
        hess[1] += hc * d[0] * d[1];
        hess[2] += hc * d[1] * d[0];
        hess[3] += hc * d[1] * d[1];
        // Block-Jacobi stabiliser: this 4th-order (biharmonic) term couples the
        // DOF to the other three window nodes, frozen during the sweep. Add the
        // coupling row-sum |dr_s|·Σ|dr_m| to the diagonal so the per-DOF step is
        // contractive (diagonally dominant) — the minimiser is unchanged (grad
        // untouched), only the step is tempered, so no need to over-damp omega.
        const real ds = std::sqrt((d[0] * d[0]) + (d[1] * d[1]));
        real off = 0.0_r;
        for (int m = 0; m < 4; ++m) {
            if (m == s) { continue; }
            off += std::sqrt((dr[m][0] * dr[m][0]) + (dr[m][1] * dr[m][1]));
        }
        const real stab = hc * ds * off;
        hess[0] += stab;
        hess[3] += stab;
        e0 += w * r * r;
    }
}

/// A free DOF's curvature energy with its own node moved to @p trial (the other
/// window nodes stay at the frozen @p X snapshot) — the line-search increment.
inline real curv_dof_trial_energy(int K,
                                  const int* nodes,
                                  const int* slot,
                                  const real* weight,
                                  const real* X,
                                  const PtN<2>& trial)
{
    real e = 0.0_r;
    for (int j = 0; j < K; ++j) {
        const int s = slot[j];
        if (s < 0) { continue; }
        const int* wn = nodes + (j * 4);
        PtN<2> p[4];
        for (int m = 0; m < 4; ++m) { p[m] = (m == s) ? trial : load_pt<2>(X, wn[m]); }
        e += weight[j] * curv_window_energy(p[0], p[1], p[2], p[3]);
    }
    return e;
}

}  // namespace egg
