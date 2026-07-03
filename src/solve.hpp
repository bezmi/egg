// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

/// @file solve.hpp
/// Closed-form tiny linear solves for the Newton step, with the JAX fallback
/// (batch_jax._solve_with_fallback): step = -solve(H, g); non-finite ->
/// -0.1 g/|g|; |g| < tol::znorm -> 0. Allocation-free, non-throwing, no
/// decompositions: safe inside device kernels.
#pragma once

#include "core.hpp"

namespace egg
{

using Vec2 = VecN<2>;  ///< d-vector (D=2 alias)
using Mat2 = MatN<2>;  ///< d×d row-major [h00, h01, h10, h11]

inline bool finite2(const Vec2& v) { return sycl::isfinite(v[0]) && sycl::isfinite(v[1]); }

/// Solve H x = -g (full D×D free-DOF Newton step) with the JAX fallback.
/// Primary template intentionally undefined; each dimension supplies its
/// closed form as an explicit specialization.
template <int D> VecN<D> solveNxN(const MatN<D>& H, const VecN<D>& g);

/// D=2: exact 2×2 closed form (bit-identical to the original solve2x2).
template <> inline VecN<2> solveNxN<2>(const MatN<2>& H, const VecN<2>& g)
{
    const real gnorm = sycl::sqrt((g[0] * g[0]) + (g[1] * g[1]));
    if (gnorm < tol::znorm) { return Vec2 {0.0_r, 0.0_r}; }

    const real det = (H[0] * H[3]) - (H[1] * H[2]);
    // x = -H^{-1} g = -(1/det) adj(H) g, adj(H) = [[h11,-h01],[-h10,h00]]
    const real inv = 1.0_r / det;
    Vec2 x {-inv * ((H[3] * g[0]) - (H[1] * g[1])), -inv * ((-H[2] * g[0]) + (H[0] * g[1]))};
    if (!finite2(x)) {
        const real c = -0.1_r / gnorm;
        return Vec2 {c * g[0], c * g[1]};
    }
    return x;
}

inline bool finite3(const VecN<3>& v)
{ return sycl::isfinite(v[0]) && sycl::isfinite(v[1]) && sycl::isfinite(v[2]); }

/// D=3: 3×3 closed form via the adjugate, same fallback structure as D=2.
template <> inline VecN<3> solveNxN<3>(const MatN<3>& H, const VecN<3>& g)
{
    const real gnorm = sycl::sqrt((g[0] * g[0]) + (g[1] * g[1]) + (g[2] * g[2]));
    if (gnorm < tol::znorm) { return VecN<3> {0.0_r, 0.0_r, 0.0_r}; }

    // Cofactors c_ij of H (adj(H) = cof(H)^T); x = -(1/det) adj(H) g.
    const real c00 = (H[4] * H[8]) - (H[5] * H[7]);
    const real c01 = -((H[3] * H[8]) - (H[5] * H[6]));
    const real c02 = (H[3] * H[7]) - (H[4] * H[6]);
    const real det = (H[0] * c00) + (H[1] * c01) + (H[2] * c02);
    const real inv = 1.0_r / det;
    const real c10 = -((H[1] * H[8]) - (H[2] * H[7]));
    const real c11 = (H[0] * H[8]) - (H[2] * H[6]);
    const real c12 = -((H[0] * H[7]) - (H[1] * H[6]));
    const real c20 = (H[1] * H[5]) - (H[2] * H[4]);
    const real c21 = -((H[0] * H[5]) - (H[2] * H[3]));
    const real c22 = (H[0] * H[4]) - (H[1] * H[3]);
    const VecN<3> x {-inv * ((c00 * g[0]) + (c10 * g[1]) + (c20 * g[2])),
                     -inv * ((c01 * g[0]) + (c11 * g[1]) + (c21 * g[2])),
                     -inv * ((c02 * g[0]) + (c12 * g[1]) + (c22 * g[2]))};
    if (!finite3(x)) {
        const real c = -0.1_r / gnorm;
        return VecN<3> {c * g[0], c * g[1], c * g[2]};
    }
    return x;
}

/// Legacy name kept for the 2D call sites.
inline Vec2 solve2x2(const Mat2& H, const Vec2& g) { return solveNxN<2>(H, g); }

/// Tangent-reduced scalar Newton step: solve a*x = -r (constrained DOF, 1×1).
inline real solve1x1(real a, real res)
{
    const real rnorm = sycl::fabs(res);
    if (rnorm < tol::znorm) { return 0.0_r; }
    const real x = -res / a;
    if (!sycl::isfinite(x)) { return -0.1_r * res / rnorm; }
    return x;
}

}  // namespace egg
