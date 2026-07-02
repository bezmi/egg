// test_patch.cpp — host-side parity of per-DOF patch evaluation (src/patch.hpp)
// vs the JAX oracle (egg.smoothing.batch_jax): patch_eval (grad/hess/energy/
// mindet), patch_energy_mindet, and the tangent-reduced newton_delta. Golden
// values from tests/cpp/golden_patch.hpp (synthetic free + circle/lineseg-
// constrained patches).
//
// This suite runs under the default compiler in the `cpp_tests` executable; the
// SYCL device variant (CPU + GPU) lives in test_patch_device.cpp.
#include "real_tol.hpp"
#include "golden_patch.hpp"
#include "patch.hpp"
#include "ut_cfg.hpp"

#include <cmath>
#include <format>

using namespace boost::ut;
using namespace egg;

namespace
{
bool close(double a, double b, double tol)
{
    tol = egg_test::real_tol(tol);
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

// Build a PatchView pointing into one golden sample's (padded) arrays.
// Owns egg::real copies of the golden double inputs and a PatchView pointing
// into them. Returned by value: std::vector's move preserves its heap buffer,
// so the view's pointers stay valid through the move. (No-op copy when
// egg::real == double.)
struct PatchInputs {
    std::vector<egg::real> s0, s1, W_inv, J, X;
    PatchView v;
};

PatchInputs view_of(const golden::PatchSample& s)
{
    PatchInputs in;
    in.s0 = egg_test::to_real(s.s0.data(), s.s0.size());
    in.s1 = egg_test::to_real(s.s1.data(), s.s1.size());
    in.W_inv = egg_test::to_real(s.W_inv.data(), s.W_inv.size());
    in.J = egg_test::to_real(s.J.data(), s.J.size());
    in.X = egg_test::to_real(s.X.data(), s.X.size());
    in.v = PatchView {s.P,         s.gc.data(),    s.gn0.data(),
                      s.gn1.data(), in.s0.data(),  in.s1.data(),
                      in.W_inv.data(), s.role.data(), in.J.data()};
    return in;
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
            const auto in = view_of(s);
            const PatchResult r = patch_eval(in.v, in.X.data());
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
            const auto in = view_of(s);
            egg::real e, md;
            patch_energy_mindet<2>(in.v, in.X.data(), e, md);
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
            const Vec2 g {static_cast<egg::real>(s.grad[0]), static_cast<egg::real>(s.grad[1])};
            const Mat2 H {static_cast<egg::real>(s.hess[0]),
                          static_cast<egg::real>(s.hess[1]),
                          static_cast<egg::real>(s.hess[2]),
                          static_cast<egg::real>(s.hess[3])};
            const Pt pos {static_cast<egg::real>(s.X[0]),
                          static_cast<egg::real>(s.X[1])};  // moving DOF is node 0
            const auto params = egg_test::to_real(s.params);
            const Vec2 d = newton_delta<2>(g, H, pos, s.tag, params.data());
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
