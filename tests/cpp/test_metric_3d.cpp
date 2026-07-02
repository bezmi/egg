// test_metric_3d.cpp — mu_cond3 metric value/grad/Hessian parity (D=3).
//
// Asserts, for every curated golden sample (reference values from the
// closed-form C++ oracle cpp_core.metric_eval_3d, i.e. mu_cond3_eval generated
// by scripts/derive_cond3.py):
//   * mu_cond3 value         == golden mu              (template barrier)
//   * mu_cond3_eval.val      == golden mu              (closed-form production)
//   * mu_cond3_eval.grad     == golden grad            (closed-form production)
//   * mu_cond3_eval.hess     == golden hess (+ symmetric)
//   * Dual<9> AD grad        == closed-form and golden grad   (1st-order AD)
//   * Dual2<9> AD hess       == closed-form and golden hess    (2nd-order AD)
//
// This pins the GPU hot-path closed form (mu_cond3_eval) to the de-risking
// Dual2<9> AD reference (mu_cond3 over duals) — the parity the metric.hpp
// header comment promises (~1e-14).
#include "dual.hpp"
#include "golden_metric_3d.hpp"
#include "metric.hpp"
#include "ut_cfg.hpp"

#include <array>
#include <cmath>
#include <format>

using namespace boost::ut;
using namespace egg;

namespace
{

VecTN<3> to_vecT3(const std::array<double, 9>& a)
{
    VecTN<3> t;
    for (int i = 0; i < 9; ++i) { t[i] = a[i]; }
    return t;
}

// Absolute-or-relative closeness, robust to the larger Hessian magnitudes in
// the near-singular sample.
bool close(double a, double b, double tol) { return std::abs(a - b) <= tol * (1.0 + std::abs(b)); }

// Dual<9> first-order AD gradient of mu_cond3.
GradN<3> cond3_grad_dual(const VecTN<3>& t)
{
    std::array<Dual<9>, 9> td;
    for (int i = 0; i < 9; ++i) { td[i] = seed_dual<9>(t[i], i); }
    const Dual<9> r = mu_cond3(td);
    GradN<3> g;
    for (int i = 0; i < 9; ++i) { g[i] = r.g[i]; }
    return g;
}

// Dual2<9> second-order AD Hessian of mu_cond3.
HessN<3> cond3_hess_dual(const VecTN<3>& t)
{
    std::array<Dual2<9>, 9> td;
    for (int i = 0; i < 9; ++i) { td[i] = seed_dual2<9>(t[i], i); }
    const Dual2<9> r = mu_cond3(td);
    HessN<3> H {};
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) { H[(i * 9) + j] = r.h[i][j]; }
    }
    return H;
}

constexpr double kTol = 1e-10;   // 1st-order AD
constexpr double kTol2 = 1e-9;   // 2nd-order AD (looser; near-singular case)

}  // namespace

static const suite<"metric3d_parity"> metric3d_parity_suite = [] {
    const std::size_t n = golden::kSamples3D.size();

    "mu_cond3 value matches oracle"_test = [n] {
        boost::ut::log << std::format("  checking mu_cond3 over {} golden samples\n", n);
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kSamples3D[k];
            const VecTN<3> t = to_vecT3(s.t);
            expect(close(mu_cond3(t), s.mu, 1e-12))
              << std::format("sample {}: mu_cond3={} expected={}", k, mu_cond3(t), s.mu);
            const MuCond3Eval r = mu_cond3_eval(t.data());
            expect(close(r.val, s.mu, 1e-12))
              << std::format("sample {}: mu_cond3_eval.val={} expected={}", k, r.val, s.mu);
        }
    };

    "closed-form grad matches oracle"_test = [n] {
        boost::ut::log << std::format("  checking dmu/dvecT (9 comps x {} samples)\n", n);
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kSamples3D[k];
            const MuCond3Eval r = mu_cond3_eval(to_vecT3(s.t).data());
            for (int i = 0; i < 9; ++i)
                expect(close(r.grad[i], s.grad[i], 1e-11))
                  << std::format("sample {} grad[{}]: {} vs {}", k, i, r.grad[i], s.grad[i]);
        }
    };

    "closed-form hess matches oracle and is symmetric"_test = [n] {
        boost::ut::log << std::format("  checking 9x9 d2mu/dvecT2 + symmetry over {} samples\n", n);
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kSamples3D[k];
            const MuCond3Eval r = mu_cond3_eval(to_vecT3(s.t).data());
            for (int i = 0; i < 81; ++i)
                expect(close(r.hess[i], s.hess[i], 1e-10))
                  << std::format("sample {} hess[{}]: {} vs {}", k, i, r.hess[i], s.hess[i]);
            for (int i = 0; i < 9; ++i)
                for (int j = 0; j < 9; ++j)
                    expect(close(r.hess[(i * 9) + j], r.hess[(j * 9) + i], 1e-12))
                      << std::format("sample {} asym H[{}][{}]", k, i, j);
        }
    };

    "dual-AD grad matches closed-form and oracle"_test = [n] {
        boost::ut::log << std::format("  forward-mode Dual<9> grad vs closed-form/oracle\n");
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kSamples3D[k];
            const VecTN<3> t = to_vecT3(s.t);
            const GradN<3> gd = cond3_grad_dual(t);
            const MuCond3Eval r = mu_cond3_eval(t.data());
            for (int i = 0; i < 9; ++i) {
                expect(close(gd[i], r.grad[i], kTol))
                  << std::format("sample {} grad[{}]: AD {} vs closed {}", k, i, gd[i], r.grad[i]);
                expect(close(gd[i], s.grad[i], kTol))
                  << std::format("sample {} grad[{}]: AD {} vs oracle {}", k, i, gd[i], s.grad[i]);
            }
        }
    };

    "dual-AD hess matches closed-form and oracle"_test = [n] {
        boost::ut::log << std::format("  2nd-order Dual2<9> Hessian vs closed-form/oracle\n");
        for (std::size_t k = 0; k < n; ++k) {
            const auto& s = golden::kSamples3D[k];
            const VecTN<3> t = to_vecT3(s.t);
            const HessN<3> Hd = cond3_hess_dual(t);
            const MuCond3Eval r = mu_cond3_eval(t.data());
            for (int i = 0; i < 81; ++i) {
                expect(close(Hd[i], r.hess[i], kTol2))
                  << std::format("sample {} hess[{}]: AD {} vs closed {}", k, i, Hd[i], r.hess[i]);
                expect(close(Hd[i], s.hess[i], kTol2))
                  << std::format("sample {} hess[{}]: AD {} vs oracle {}", k, i, Hd[i], s.hess[i]);
            }
        }
    };

    // The fused mu_cond3_jhj (production D=3 patch path) must reproduce the value,
    // gradient, and the contracted Hessian jhj = Jbᵀ H Jb of the reference
    // mu_cond3_eval + an explicit contraction, over a few curated Jb (9×3).
    "fused mu_cond3_jhj matches eval + explicit JtHJ"_test = [n] {
        boost::ut::log << std::format("  mu_cond3_jhj vs mu_cond3_eval contraction\n");
        // Curated chain matrices Jb (9×3 row-major): identity-ish, skewed, dense.
        std::array<std::array<double, 27>, 3> jbs {};
        for (int a = 0; a < 9; ++a) {
            for (int k = 0; k < 3; ++k) {
                jbs[0][(a * 3) + k] = (a % 3 == k) ? 1.0 : 0.0;
                jbs[1][(a * 3) + k] = 0.1 * static_cast<double>(((a * 7) + (k * 3)) % 5) - 0.2;
                jbs[2][(a * 3) + k] = std::sin(0.3 * a + 0.7 * k) + 0.5;
            }
        }
        for (std::size_t kk = 0; kk < n; ++kk) {
            const auto& s = golden::kSamples3D[kk];
            const VecTN<3> t = to_vecT3(s.t);
            const MuCond3Eval ref = mu_cond3_eval(t.data());
            for (const auto& Jb : jbs) {
                double val = 0.0;
                std::array<double, 9> grad {};
                std::array<double, 9> jhj {};
                mu_cond3_jhj(t.data(), Jb.data(), &val, grad.data(), jhj.data());
                expect(close(val, ref.val, 1e-12)) << std::format("sample {} val", kk);
                for (int i = 0; i < 9; ++i) {
                    expect(close(grad[i], ref.grad[i], 1e-11))
                      << std::format("sample {} grad[{}]", kk, i);
                }
                // Explicit jhj = Jbᵀ H Jb from the full Hessian.
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        double acc = 0.0;
                        for (int a = 0; a < 9; ++a) {
                            for (int b = 0; b < 9; ++b) {
                                acc += Jb[(a * 3) + i] * ref.hess[(a * 9) + b] * Jb[(b * 3) + j];
                            }
                        }
                        expect(close(jhj[(i * 3) + j], acc, 1e-9))
                          << std::format("sample {} jhj[{},{}]: {} vs {}", kk, i, j,
                                         jhj[(i * 3) + j], acc);
                    }
                }
            }
        }
    };
};
