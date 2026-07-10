// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

// test_curvature.cpp — host-side parity of the block-interface C2 curvature
// functional (src/curvature.hpp) vs the NumPy oracle egg.smoothing.interface_c2.
// Golden values (energy, per-slot gradient, per-slot Gauss-Newton Hessian) were
// produced from curvature_energy_grad_hess on the window below with weight 1.7.
// Host-side (no kernel launch), so it rides in the cpp_tests runner.
#include "real_tol.hpp"
#include "curvature.hpp"
#include "ut_cfg.hpp"

#include <array>
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

// The golden window and weight (mirrors the Python oracle input).
const PtN<2> p0 {0.0_r, 0.0_r};
const PtN<2> p1 {1.0_r, 0.2_r};
const PtN<2> p2 {2.1_r, 0.1_r};
const PtN<2> p3 {3.0_r, -0.3_r};
constexpr real kWeight = 1.7_r;

constexpr double kEnergy = 0.0026536332100884148;
const std::array<std::array<double, 2>, 4> kGrad {{
  {-0.025832804637445694, 0.12916402318722844},
  {0.0038113974055247772, -0.37139950273835853},
  {0.077415462536959107, 0.366872103987466},
  {-0.055394055305038201, -0.12463662443633593},
}};
const std::array<std::array<double, 4>, 4> kHess {{
  {0.1257396449704142, -0.62869822485207094, -0.62869822485207094, 3.143491124260354},
  {0.0027371435750076774, -0.26671943503130341, -0.26671943503130341, 25.990327169161432},
  {1.1292355358357227, 5.3514505158666497, 5.3514505158666497, 25.360539688095326},
  {0.57816983738973349, 1.3008821341269001, 1.3008821341269001, 2.9269848017855247},
}};
}  // namespace

static const suite<"curvature"> curvature_suite = [] {
    "curv_energy_matches_oracle"_test = [] {
        const real e = kWeight * curv_window_energy(p0, p1, p2, p3);
        expect(close(e, kEnergy, 1e-12));
    };

    "curv_grad_hess_match_oracle"_test = [] {
        for (int slot = 0; slot < 4; ++slot) {
            VecN<2> grad {};
            MatN<2> hess {};
            curv_window_accumulate(p0, p1, p2, p3, slot, kWeight, grad, hess);
            expect(close(grad[0], kGrad[slot][0], 1e-11)) << "grad0 slot " << slot;
            expect(close(grad[1], kGrad[slot][1], 1e-11)) << "grad1 slot " << slot;
            for (int k = 0; k < 4; ++k) {
                expect(close(hess[k], kHess[slot][k], 1e-11)) << "hess" << k << " slot " << slot;
            }
        }
    };

    // A straight line has zero turning-angle change; a kink is positive.
    "curv_zero_on_straight_positive_on_kink"_test = [] {
        const PtN<2> a {0.0_r, 0.0_r}, b {1.0_r, 0.0_r}, c {2.5_r, 0.0_r}, d {4.0_r, 0.0_r};
        expect(close(curv_window_energy(a, b, c, d), 0.0, 1e-14));
        const PtN<2> ck {2.5_r, 0.5_r};
        expect(curv_window_energy(a, b, ck, d) > 1e-3_r);
    };
};
