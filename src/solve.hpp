// solve.hpp — closed-form tiny linear solves for the Newton step, with the same
// singular/non-finite fallback as the JAX path (batch_jax._solve_with_fallback):
//   step = -solve(H, g);
//   if step non-finite  -> -0.1 * g / |g|
//   if |g| < 1e-15      -> 0
// Allocation-free, no throwing, no decompositions: safe inside device kernels.
#pragma once

#include <array>
#include <cmath>

namespace egg
{

using Vec2 = std::array<double, 2>;
using Mat2 = std::array<double, 4>;  // [h00, h01, h10, h11] row-major

inline bool finite2(const Vec2& v) { return std::isfinite(v[0]) && std::isfinite(v[1]); }

// Solve H x = -g for the full 2x2 (free DOF) Newton step.
inline Vec2 solve2x2(const Mat2& H, const Vec2& g)
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
