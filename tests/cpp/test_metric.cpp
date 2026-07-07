// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

// test_metric.cpp — shape_2d metric value/grad/Hessian parity.
//
// Asserts, for every curated golden sample (reference values from the NumPy
// oracle, egg.smoothing.metrics):
//   * mu_value             == golden mu
//   * mu_grad_closedform   == golden grad
//   * mu_hess_closedform   == golden hess (and is symmetric)
//   * mu_grad_dual         == closed-form grad and golden grad   (1st-order AD)
//   * mu_hess_dual         == closed-form hess and golden hess    (2nd-order AD)
#include "golden_metric.hpp"
#include "metric.hpp"
#include "real_tol.hpp"
#include "ut_cfg.hpp"

#include <cmath>
#include <format>

using namespace boost::ut;
using namespace egg;

namespace
{

constexpr double kTol = 1e-10;  // value / 1st-order
constexpr double kTol2 = 1e-9;  // 2nd-order AD (looser; near-singular case)

VecT to_vecT(const std::array<double, 4>& a)
{
    return VecT {static_cast<egg::real>(a[0]),
                 static_cast<egg::real>(a[1]),
                 static_cast<egg::real>(a[2]),
                 static_cast<egg::real>(a[3])};
}

// Absolute-or-relative closeness, robust to the larger Hessian magnitudes in
// the near-singular sample.
bool close(double a, double b, double tol)
{
    tol = egg_test::real_tol(tol);
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

}  // namespace

static const suite<"metric"> metric_suite = [] {
    const std::size_t n = golden::kSamples.size();

    "mu_value matches oracle"_test = [n] {
        boost::ut::log << std::format("  checking mu over {} golden samples\n", n);
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kSamples[k];
            expect(close(mu_value(to_vecT(s.t)), s.mu, 1e-12))
              << std::format("sample {}: mu={} expected={}", k, mu_value(to_vecT(s.t)), s.mu);
        }
    };

    "closed-form grad matches oracle"_test = [n] {
        boost::ut::log << std::format("  checking dmu/dvecT (4 comps x {} samples)\n", n);
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kSamples[k];
            const Grad g = mu_grad_closedform(to_vecT(s.t));
            for (int i = 0; i < 4; ++i)
                expect(close(g[i], s.grad[i], 1e-12))
                  << std::format("sample {} grad[{}]: {} vs {}", k, i, g[i], s.grad[i]);
        }
    };

    "closed-form hess matches oracle and is symmetric"_test = [n] {
        boost::ut::log << std::format("  checking 4x4 d2mu/dvecT2 + symmetry over {} samples\n", n);
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kSamples[k];
            const Hess H = mu_hess_closedform(to_vecT(s.t));
            for (int i = 0; i < 16; ++i)
                expect(close(H[i], s.hess[i], 1e-12))
                  << std::format("sample {} hess[{}]: {} vs {}", k, i, H[i], s.hess[i]);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    expect(close(H[i * 4 + j], H[j * 4 + i], 1e-12))
                      << std::format("sample {} asym H[{}][{}]", k, i, j);
        }
    };

    "dual-AD grad matches closed-form and oracle"_test = [n] {
        boost::ut::log << std::format("  forward-mode Dual<4> grad vs closed-form/oracle\n");
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kSamples[k];
            const VecT t = to_vecT(s.t);
            const Grad gd = mu_grad_dual(t);
            const Grad gc = mu_grad_closedform(t);
            for (int i = 0; i < 4; ++i) {
                expect(close(gd[i], gc[i], kTol))
                  << std::format("sample {} grad[{}]: AD {} vs closed {}", k, i, gd[i], gc[i]);
                expect(close(gd[i], s.grad[i], kTol))
                  << std::format("sample {} grad[{}]: AD {} vs oracle {}", k, i, gd[i], s.grad[i]);
            }
        }
    };

    "dual-AD hess matches closed-form and oracle"_test = [n] {
        boost::ut::log << std::format("  2nd-order Dual2<4> Hessian vs closed-form/oracle\n");
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kSamples[k];
            const VecT t = to_vecT(s.t);
            const Hess Hd = mu_hess_dual(t);
            const Hess Hc = mu_hess_closedform(t);
            for (int i = 0; i < 16; ++i) {
                expect(close(Hd[i], Hc[i], kTol2))
                  << std::format("sample {} hess[{}]: AD {} vs closed {}", k, i, Hd[i], Hc[i]);
                expect(close(Hd[i], s.hess[i], kTol2))
                  << std::format("sample {} hess[{}]: AD {} vs oracle {}", k, i, Hd[i], s.hess[i]);
            }
        }
    };

    // shape+size: the closed forms against dual-AD over the same generic
    // scalar mu_shape_size2d (cross-language parity vs the NumPy oracle is
    // covered by tests/smoothing/test_shape_size.py via cpp_core.metric_eval).
    "shape_size closed-form grad/hess match dual-AD"_test = [n] {
        boost::ut::log << std::format("  shape_size closed forms vs Dual/Dual2 AD\n");
        for (std::size_t k = 0; k < n; ++k) {
            const VecT t = to_vecT(golden::kSamples[k].t);
            const Dual<kVecT> rg = mu_shape_size2d(seed_dual<kVecT>(t[0], 0),
                                                   seed_dual<kVecT>(t[1], 1),
                                                   seed_dual<kVecT>(t[2], 2),
                                                   seed_dual<kVecT>(t[3], 3));
            expect(close(mu_ss_value(t), rg.v, kTol))
              << std::format("sample {} value: closed {} vs AD {}", k, mu_ss_value(t), rg.v);
            const Grad g = mu_ss_grad_closedform(t);
            for (int i = 0; i < 4; ++i)
                expect(close(g[i], rg.g[i], kTol))
                  << std::format("sample {} grad[{}]: closed {} vs AD {}", k, i, g[i], rg.g[i]);
            const Dual2<kVecT> rh = mu_shape_size2d(seed_dual2<kVecT>(t[0], 0),
                                                    seed_dual2<kVecT>(t[1], 1),
                                                    seed_dual2<kVecT>(t[2], 2),
                                                    seed_dual2<kVecT>(t[3], 3));
            const Hess H = mu_ss_hess_closedform(t);
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    expect(close(H[(i * 4) + j], rh.h[i][j], kTol2))
                      << std::format("sample {} hess[{}][{}]: closed {} vs AD {}",
                                     k,
                                     i,
                                     j,
                                     H[(i * 4) + j],
                                     rh.h[i][j]);
                    expect(close(H[(i * 4) + j], H[(j * 4) + i], kTol2))
                      << std::format("sample {} asym H[{}][{}]", k, i, j);
                }
            }
        }
    };
};
