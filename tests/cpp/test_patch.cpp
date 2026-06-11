// test_patch.cpp — host-side parity of per-DOF patch evaluation (src/patch.hpp)
// vs the JAX oracle (egg.smoothing.batch_jax): patch_eval (grad/hess/energy/
// mindet), patch_energy_mindet, and the tangent-reduced newton_delta. Golden
// values from tests/cpp/golden_patch.hpp (synthetic free + circle/lineseg-
// constrained patches).
//
// This suite runs under the default compiler in the `cpp_tests` executable; the
// SYCL device variant (CPU + GPU) lives in test_patch_device.cpp.
#include "golden_patch.hpp"
#include "patch.hpp"
#include "ut_cfg.hpp"

#include <cmath>
#include <format>

using namespace boost::ut;
using namespace egg;

namespace
{
bool close(double a, double b, double tol) { return std::abs(a - b) <= tol * (1.0 + std::abs(b)); }

// Build a PatchView pointing into one golden sample's (padded) arrays.
PatchView view_of(const golden::PatchSample& s)
{
    return PatchView {s.P,
                      s.gc.data(),
                      s.gn0.data(),
                      s.gn1.data(),
                      s.s0.data(),
                      s.s1.data(),
                      s.W_inv.data(),
                      s.role.data(),
                      s.J.data()};
}
}  // namespace

static const suite<"patch"> patch_suite = [] {
    const std::size_t n = golden::kPatchSamples.size();
    // grad/hess/delta: closed-form vs autodiff -> ~1e-9; energy/mindet: both
    // closed-form -> tighter.
    constexpr double kTolGH = 1e-9;
    constexpr double kTolE = 1e-10;

    "patch_eval grad/hess/energy/mindet match oracle"_test = [n] {
        boost::ut::log << std::format("  patch_eval over {} golden patches\n", n);
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kPatchSamples[k];
            const PatchResult r = patch_eval(view_of(s), s.X.data());
            for (int i = 0; i < 2; ++i)
                expect(close(r.grad[i], s.grad[i], kTolGH))
                  << std::format("patch {} grad[{}]: {} vs {}", k, i, r.grad[i], s.grad[i]);
            for (int i = 0; i < 4; ++i)
                expect(close(r.hess[i], s.hess[i], kTolGH))
                  << std::format("patch {} hess[{}]: {} vs {}", k, i, r.hess[i], s.hess[i]);
            expect(close(r.energy, s.energy, kTolE))
              << std::format("patch {} energy: {} vs {}", k, r.energy, s.energy);
            expect(close(r.mindet, s.mindet, kTolE))
              << std::format("patch {} mindet: {} vs {}", k, r.mindet, s.mindet);
        }
    };

    "patch_energy_mindet matches patch_eval energy/mindet"_test = [n] {
        boost::ut::log << std::format("  patch_energy_mindet over {} patches\n", n);
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kPatchSamples[k];
            double e, md;
            patch_energy_mindet(view_of(s), s.X.data(), e, md);
            expect(close(e, s.energy, kTolE))
              << std::format("patch {} energy: {} vs {}", k, e, s.energy);
            expect(close(md, s.mindet, kTolE))
              << std::format("patch {} mindet: {} vs {}", k, md, s.mindet);
        }
    };

    "newton_delta matches _newton_delta_one"_test = [n] {
        boost::ut::log << std::format("  newton_delta over {} patches\n", n);
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kPatchSamples[k];
            // Feed the oracle's exact grad/hess so the step is tested in isolation.
            const Vec2 g {s.grad[0], s.grad[1]};
            const Mat2 H {s.hess[0], s.hess[1], s.hess[2], s.hess[3]};
            const Pt pos {s.X[0], s.X[1]};  // moving DOF is node 0
            const Vec2 d = newton_delta(g, H, pos, s.tag, s.params.data());
            for (int i = 0; i < 2; ++i)
                expect(close(d[i], s.delta[i], kTolGH))
                  << std::format("patch {} (tag {}) delta[{}]: {} vs {}",
                                 k,
                                 s.tag,
                                 i,
                                 d[i],
                                 s.delta[i]);
        }
    };
};
