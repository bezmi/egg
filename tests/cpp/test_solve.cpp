// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

// test_solve.cpp — closed-form Newton-step solves and the singular fallback
// (mirroring batch_jax._solve_with_fallback).
#include "metric.hpp"
#include "solve.hpp"
#include "real_tol.hpp"
#include "ut_cfg.hpp"

#include <cmath>

using namespace boost::ut;
using namespace egg;

namespace
{
bool close(double a, double b, double tol)
{
    tol = egg_test::real_tol(tol);
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}
}  // namespace

static const suite<"solve"> solve_suite = [] {
    "solve2x2 satisfies H x + g = 0"_test = [] {
        // SPD-ish matrices with a known solve; check the residual directly.
        const Mat2 cases[] = {{2.0, 0.0, 0.0, 3.0}, {4.0, 1.0, 1.0, 2.0}, {10.0, -2.0, -2.0, 5.0}};
        const Vec2 g {0.7, -1.3};
        for (const auto& H : cases) {
            const Vec2 x = solve2x2(H, g);
            const double r0 = H[0] * x[0] + H[1] * x[1] + g[0];
            const double r1 = H[2] * x[0] + H[3] * x[1] + g[1];
            expect(close(r0, 0.0, 1e-12));
            expect(close(r1, 0.0, 1e-12));
        }
    };

    "solve2x2 uses Hessian from the metric"_test = [] {
        // Use a real 2x2 sub-block of the shape_2d Hessian (free-DOF Newton step).
        const Hess H = mu_hess_closedform(VecT {1.2, 0.4, -0.1, 0.9});
        const Mat2 Hf {H[0], H[1], H[4], H[5]};
        const Vec2 g {0.05, -0.02};
        const Vec2 x = solve2x2(Hf, g);
        const double r0 = Hf[0] * x[0] + Hf[1] * x[1] + g[0];
        const double r1 = Hf[2] * x[0] + Hf[3] * x[1] + g[1];
        expect(close(r0, 0.0, 1e-10));
        expect(close(r1, 0.0, 1e-10));
    };

    "solve2x2 singular -> gradient-descent fallback"_test = [] {
        const Mat2 Hsing {1.0, 2.0, 2.0, 4.0};  // det = 0
        const Vec2 g {3.0, 4.0};                // |g| = 5
        const Vec2 x = solve2x2(Hsing, g);
        expect(close(x[0], -0.1 * g[0] / 5.0, 1e-14));
        expect(close(x[1], -0.1 * g[1] / 5.0, 1e-14));
    };

    "solve2x2 zero gradient -> zero step"_test = [] {
        const Mat2 H {2.0, 0.0, 0.0, 2.0};
        const Vec2 x = solve2x2(H, Vec2 {0.0, 0.0});
        expect(close(x[0], 0.0, 1e-300));
        expect(close(x[1], 0.0, 1e-300));
    };

    "solve1x1 scalar step and fallback"_test = [] {
        expect(close(solve1x1(4.0, -2.0), 0.5, 1e-14));  // -r/a
        expect(close(solve1x1(0.0, 3.0), -0.1, 1e-14));  // a=0 -> -0.1*r/|r|
        expect(close(solve1x1(5.0, 0.0), 0.0, 1e-300));  // r~0 -> 0
    };
};
