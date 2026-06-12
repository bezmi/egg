// solve.hpp — closed-form tiny linear solves for the Newton step, with the same
// singular/non-finite fallback as the JAX path (batch_jax._solve_with_fallback):
//   step = -solve(H, g);
//   if step non-finite  -> -0.1 * g / |g|
//   if |g| < 1e-15      -> 0
// Allocation-free, no throwing, no decompositions: safe inside device kernels.
#pragma once

#include "core.hpp"

#include <array>
#include <cmath>

namespace egg
{

// The per-DOF Newton solve. solveNxN<D> is generic structure; the D==2 body is
// the exact closed-form 2×2 (kept bit-identical), and a 3D core adds the D==3
// specialization (e.g. Cholesky). Vec2/Mat2 are D=2 aliases for the call sites.
using Vec2 = VecN<2>;  // d-vector
using Mat2 = MatN<2>;  // d×d row-major [h00, h01, h10, h11]

inline bool finite2(const Vec2& v) { return std::isfinite(v[0]) && std::isfinite(v[1]); }

// Solve H x = -g for the full D×D (free DOF) Newton step, with the JAX
// singular/non-finite fallback. Primary template is intentionally undefined;
// every dimension provides an explicit specialization with its closed form.
template <int D> VecN<D> solveNxN(const MatN<D>& H, const VecN<D>& g);

// D=2: the exact 2×2 closed form (bit-identical to the original solve2x2).
template <> inline VecN<2> solveNxN<2>(const MatN<2>& H, const VecN<2>& g)
{
    const double gnorm = std::sqrt(g[0] * g[0] + g[1] * g[1]);
    if (gnorm < 1e-15) return Vec2 {0.0, 0.0};

    const double det = H[0] * H[3] - H[1] * H[2];
    // x = -H^{-1} g = -(1/det) adj(H) g, adj(H) = [[h11,-h01],[-h10,h00]]
    const double inv = 1.0 / det;
    Vec2 x {-inv * (H[3] * g[0] - H[1] * g[1]), -inv * (-H[2] * g[0] + H[0] * g[1])};
    if (!finite2(x)) {
        const double c = -0.1 / gnorm;
        return Vec2 {c * g[0], c * g[1]};
    }
    return x;
}

// Legacy name kept for the 2D call sites.
inline Vec2 solve2x2(const Mat2& H, const Vec2& g) { return solveNxN<2>(H, g); }

// Tangent-reduced scalar Newton step: solve a*x = -r (constrained DOF, 1x1).
inline double solve1x1(double a, double r)
{
    const double rnorm = std::abs(r);
    if (rnorm < 1e-15) return 0.0;
    const double x = -r / a;
    if (!std::isfinite(x)) return -0.1 * r / rnorm;
    return x;
}

}  // namespace egg
