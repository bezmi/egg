// the 2D shape_2d metric mu(T) = |T|^2 / (2 det T) - 1, with both
// a closed-form value/grad/Hess (the in-kernel production path) and dual-AD
// value/grad/Hess (the de-risking parity path). Mirrors the NumPy oracle in
// egg.smoothing.metrics (_shape2d_value / _shape2d_value_grad /
// _shape2d_hess_T) exactly; vec(T) = [a, b, c, d] = [T00, T01, T10, T11].
#pragma once

#include "core.hpp"
#include "dual.hpp"

#include <array>
#include <cmath>
#include <concepts>
#include <string_view>
#include <variant>

namespace egg
{

// Flattened vec(T)-space shapes. The un-suffixed names are the D=2 aliases the
// closed-form 2D metric and the oracle surface use; VecTN/GradN/HessN<D> from
// core.hpp are the generic shapes the templated objectives are written over.
using VecT = VecTN<2>;  // [a, b, c, d] for d=2
using Grad = GradN<2>;  // dmu/dvec(T)
using Hess = HessN<2>;  // (d²)×(d²) row-major; 4×4 for d=2

// The shape_2d barrier μ = |T|²/(2 det T) − 1 and its closed-form grad/Hess are
// the 2D metric; a 3D core supplies a 3D barrier (e.g. the condition-number form)
// — most cleanly via dual.hpp AD over an N-generic μ.

// --- generic scalar metric, reused for double / Dual<4> / Dual2<4> ---
//
//   D  = a*d - b*c
//   s  = a^2 + b^2 + c^2 + d^2
//   mu = s / (2 D) - 1
template <typename T> constexpr T mu_shape2d(const T& a, const T& b, const T& c, const T& d)
{
    const T D = a * d - b * c;
    const T s = a * a + b * b + c * c + d * d;
    return s / (2.0 * D) - 1.0;
}

inline double mu_value(const VecT& t) { return mu_shape2d(t[0], t[1], t[2], t[3]); }

// --- closed-form gradient (matches _shape2d_value_grad) ---
inline Grad mu_grad_closedform(const VecT& t)
{
    const double a = t[0], b = t[1], c = t[2], d = t[3];
    const double D = (a * d) - (b * c);
    const double s = (a * a) + (b * b) + (c * c) + (d * d);
    const double coef = s / (2.0 * D * D);
    return Grad {(a / D) - (coef * d),
                 (b / D) + (coef * c),
                 (c / D) + (coef * b),
                 (d / D) - (coef * a)};
}

// d vec(C)/d vec(T) with cofactor vec(C) = [d, -c, -b, a].
// MIRROR: keep in sync with egg.smoothing.metrics._HESS_P and
// egg.smoothing.batch._HESS_P (Python). Cross-language, so unavoidable; any
// change must be applied to all three copies.
inline constexpr std::array<double, kVecT * kVecT> kHessP = {0.0,
                                                             0.0,
                                                             0.0,
                                                             1.0,  //
                                                             0.0,
                                                             0.0,
                                                             -1.0,
                                                             0.0,  //
                                                             0.0,
                                                             -1.0,
                                                             0.0,
                                                             0.0,  //
                                                             1.0,
                                                             0.0,
                                                             0.0,
                                                             0.0};

// --- closed-form Hessian (matches _shape2d_hess_T) ---
//   H = (1/D) I
//       - (1/D^2) (t cof^T + cof t^T)
//       + (s/D^3) cof cof^T
//       - (s/(2 D^2)) P
inline Hess mu_hess_closedform(const VecT& t)
{
    const double a = t[0], b = t[1], c = t[2], d = t[3];
    const double D = (a * d) - (b * c);
    const double s = (a * a) + (b * b) + (c * c) + (d * d);
    const std::array<double, kVecT> tv {a, b, c, d};
    const std::array<double, kVecT> cof {d, -c, -b, a};
    const double invD = 1.0 / D;
    const double invD2 = 1.0 / (D * D);
    const double sD3 = s / (D * D * D);
    const double s2D2 = s / (2.0 * D * D);
    Hess H {};
    for (int i = 0; i < kVecT; ++i) {
        for (int j = 0; j < kVecT; ++j) {
            const double eye = (i == j) ? 1.0 : 0.0;
            H[(i * kVecT) + j] = (invD * eye) - (invD2 * ((tv[i] * cof[j]) + (cof[i] * tv[j]))) +
                                 (cof[i] * sD3 * cof[j]) - (s2D2 * kHessP[(i * kVecT) + j]);
        }
    }
    return H;
}

// --- dual-AD gradient (forward mode, one pass with all 4 seeds) ---
inline Grad mu_grad_dual(const VecT& t)
{
    using Dv = Dual<kVecT>;  // AD over the kVecT components of vec(T)
    const Dv r = mu_shape2d(seed_dual<kVecT>(t[0], 0),
                            seed_dual<kVecT>(t[1], 1),
                            seed_dual<kVecT>(t[2], 2),
                            seed_dual<kVecT>(t[3], 3));
    return Grad {r.g[0], r.g[1], r.g[2], r.g[3]};
}

// --- dual-AD Hessian (second-order forward, one pass) ---
inline Hess mu_hess_dual(const VecT& t)
{
    using Dv = Dual2<kVecT>;
    const Dv r = mu_shape2d(seed_dual2<kVecT>(t[0], 0),
                            seed_dual2<kVecT>(t[1], 1),
                            seed_dual2<kVecT>(t[2], 2),
                            seed_dual2<kVecT>(t[3], 3));
    Hess H {};
    for (int i = 0; i < kVecT; ++i) {
        for (int j = 0; j < kVecT; ++j) { H[(i * kVecT) + j] = r.h[i][j]; }
    }
    return H;
}

// ---------------------------------------------------------------------------
// δ-continuation untangle surrogate (mirrors metrics._untangle_surrogate):
//   Dh(δ) = ½(D + √(D² + 4δ²))   (> 0 everywhere; → D as δ → 0 for D > 0)
//   μ_δ   = s / (2·Dh) − 1
// Generic over double / Dual<4> / Dual2<4> so value, grad, and Hessian all come
// from the one definition (closed form for value, dual-AD for the derivatives).
// ---------------------------------------------------------------------------
template <typename T>
inline T mu_untangle(const T& a, const T& b, const T& c, const T& d, double delta)
{
    using std::sqrt;  // egg::sqrt for Dual/Dual2 (ADL); std::sqrt for double
    const T D = a * d - b * c;
    const T s = a * a + b * b + c * c + d * d;
    const T disc = D * D + T(4.0 * delta * delta);
    const T Dh = 0.5 * (D + sqrt(disc));
    return s / (2.0 * Dh) - 1.0;
}

// ---------------------------------------------------------------------------
// 3D condition-number barrier (generic over double / Dual<9> / Dual2<9>):
//   μ_cond = |T|²·|T⁻¹|²/d² − 1 = (Σ tᵢ²)(Σ cof(T)ᵢ²) / (9 det²) − 1
// Rational in the entries of T (no fractional powers), so closed form and
// dual-AD are clean and device-friendly. vec(T) is row-major
// [t00 t01 t02 t10 ... t22]. Sanity: at d=2 the cofactor norm equals |T|², so
// the same formula collapses to s²/(4 det²) − 1 (the Python `_shape_value`).
// ---------------------------------------------------------------------------
template <typename T> constexpr T det3(const std::array<T, 9>& t)
{
    return (t[0] * ((t[4] * t[8]) - (t[5] * t[7]))) - (t[1] * ((t[3] * t[8]) - (t[5] * t[6]))) +
           (t[2] * ((t[3] * t[7]) - (t[4] * t[6])));
}

// The 9 cofactors of the 3×3 (row-major; adj = cof^T).
template <typename T> constexpr std::array<T, 9> cof3(const std::array<T, 9>& t)
{
    return {(t[4] * t[8]) - (t[5] * t[7]),
            -((t[3] * t[8]) - (t[5] * t[6])),
            (t[3] * t[7]) - (t[4] * t[6]),
            -((t[1] * t[8]) - (t[2] * t[7])),
            (t[0] * t[8]) - (t[2] * t[6]),
            -((t[0] * t[7]) - (t[1] * t[6])),
            (t[1] * t[5]) - (t[2] * t[4]),
            -((t[0] * t[5]) - (t[2] * t[3])),
            (t[0] * t[4]) - (t[1] * t[3])};
}

template <typename T> constexpr T mu_cond3(const std::array<T, 9>& t)
{
    const T D = det3(t);
    const std::array<T, 9> cof = cof3(t);
    T s = T(0.0);
    T q = T(0.0);
    for (int i = 0; i < 9; ++i) {
        s = s + (t[i] * t[i]);
        q = q + (cof[i] * cof[i]);
    }
    return (s * q) / (9.0 * D * D) - 1.0;
}

// δ-continuation untangle surrogate: det → Dh = ½(det + √(det² + 4δ²)) (> 0 on
// folded cells), mirroring the 2D mu_untangle substitution.
template <typename T> inline T mu_cond3_untangle(const std::array<T, 9>& t, double delta)
{
    using std::sqrt;  // egg::sqrt for Dual/Dual2 (ADL); std::sqrt for double
    const T D = det3(t);
    const T disc = (D * D) + T(4.0 * delta * delta);
    const T Dh = 0.5 * (D + sqrt(disc));
    const std::array<T, 9> cof = cof3(t);
    T s = T(0.0);
    T q = T(0.0);
    for (int i = 0; i < 9; ++i) {
        s = s + (t[i] * t[i]);
        q = q + (cof[i] * cof[i]);
    }
    return (s * q) / (9.0 * Dh * Dh) - 1.0;
}

// ---------------------------------------------------------------------------
// Objective concept + closed set of objective kinds (barrier / δ-untangle).
//
// An Objective supplies the per-cell μ value/grad/Hess w.r.t. vec(T) (μ is the
// objective integrand; the energy E = Σ μ is accumulated by the kernels) plus
// the line-search accept policy on min det A. The kinds are a closed set, so
// they live in a
// std::variant and are dispatched **once per run** with std::visit (outside the
// hot loop) to instantiate a monomorphic, fully-concrete kernel — no virtuals,
// no per-DOF type dispatch on device.
// ---------------------------------------------------------------------------
template <class M, int D>
concept ObjectiveD = requires(const M& m, const VecTN<D>& t, double det) {
    { m.value(t) } -> std::convertible_to<double>;
    { m.grad(t) } -> std::convertible_to<GradN<D>>;
    { m.hess(t) } -> std::convertible_to<HessN<D>>;
    // Line-search accept test on min det A for this objective's regime.
    { m.accept_mindet(det) } -> std::convertible_to<bool>;
};

// Barrier shape objective (minimises the shape-distortion metric μ). D=2 keeps
// the classic shape_2d closed forms (bit-identical production path); D=3 uses
// the condition-number barrier μ_cond3 with dual-AD derivatives.
template <int D> struct ShapeObjectiveT {
    static_assert(D == 2 || D == 3, "ShapeObjectiveT: unsupported dimension");
    static constexpr int kN = dim::vecT(D);

    [[nodiscard]] double value(const VecTN<D>& t) const
    {
        if constexpr (D == 2) {
            return mu_value(t);
        } else {
            return mu_cond3(t);
        }
    }
    [[nodiscard]] GradN<D> grad(const VecTN<D>& t) const
    {
        if constexpr (D == 2) {
            return mu_grad_closedform(t);
        } else {
            std::array<Dual<kN>, kN> td;
            for (int i = 0; i < kN; ++i) { td[i] = seed_dual<kN>(t[i], i); }
            const Dual<kN> r = mu_cond3(td);
            GradN<D> g;
            for (int i = 0; i < kN; ++i) { g[i] = r.g[i]; }
            return g;
        }
    }
    [[nodiscard]] HessN<D> hess(const VecTN<D>& t) const
    {
        if constexpr (D == 2) {
            return mu_hess_closedform(t);
        } else {
            std::array<Dual2<kN>, kN> td;
            for (int i = 0; i < kN; ++i) { td[i] = seed_dual2<kN>(t[i], i); }
            const Dual2<kN> r = mu_cond3(td);
            HessN<D> H {};
            for (int i = 0; i < kN; ++i) {
                for (int j = 0; j < kN; ++j) { H[(i * kN) + j] = r.h[i][j]; }
            }
            return H;
        }
    }
    // Barrier: a step is only valid if every cell stays positively oriented.
    [[nodiscard]] bool accept_mindet(double mindet) const { return mindet > 0.0; }
};

// δ-continuation untangle objective: surrogate value + dual-AD grad/Hess, with the
// relaxed accept rule (the surrogate is finite on folded cells, so min det A may
// be ≤ 0 during continuation). D=2 keeps the existing surrogate exactly; D=3
// substitutes Dh into the condition-number barrier.
template <int D> struct UntangleObjectiveT {
    static_assert(D == 2 || D == 3, "UntangleObjectiveT: unsupported dimension");
    static constexpr int kN = dim::vecT(D);
    double delta {0.0};

    [[nodiscard]] double value(const VecTN<D>& t) const
    {
        if constexpr (D == 2) {
            return mu_untangle(t[0], t[1], t[2], t[3], delta);
        } else {
            return mu_cond3_untangle(t, delta);
        }
    }
    [[nodiscard]] GradN<D> grad(const VecTN<D>& t) const
    {
        if constexpr (D == 2) {
            using Dv = Dual<kVecT>;
            const Dv r = mu_untangle(seed_dual<kVecT>(t[0], 0),
                                     seed_dual<kVecT>(t[1], 1),
                                     seed_dual<kVecT>(t[2], 2),
                                     seed_dual<kVecT>(t[3], 3),
                                     delta);
            return GradN<D> {r.g[0], r.g[1], r.g[2], r.g[3]};
        } else {
            std::array<Dual<kN>, kN> td;
            for (int i = 0; i < kN; ++i) { td[i] = seed_dual<kN>(t[i], i); }
            const Dual<kN> r = mu_cond3_untangle(td, delta);
            GradN<D> g;
            for (int i = 0; i < kN; ++i) { g[i] = r.g[i]; }
            return g;
        }
    }
    [[nodiscard]] HessN<D> hess(const VecTN<D>& t) const
    {
        if constexpr (D == 2) {
            using Dv = Dual2<kVecT>;
            const Dv r = mu_untangle(seed_dual2<kVecT>(t[0], 0),
                                     seed_dual2<kVecT>(t[1], 1),
                                     seed_dual2<kVecT>(t[2], 2),
                                     seed_dual2<kVecT>(t[3], 3),
                                     delta);
            HessN<D> H {};
            for (int i = 0; i < kVecT; ++i) {
                for (int j = 0; j < kVecT; ++j) { H[(i * kVecT) + j] = r.h[i][j]; }
            }
            return H;
        } else {
            std::array<Dual2<kN>, kN> td;
            for (int i = 0; i < kN; ++i) { td[i] = seed_dual2<kN>(t[i], i); }
            const Dual2<kN> r = mu_cond3_untangle(td, delta);
            HessN<D> H {};
            for (int i = 0; i < kN; ++i) {
                for (int j = 0; j < kN; ++j) { H[(i * kN) + j] = r.h[i][j]; }
            }
            return H;
        }
    }
    [[nodiscard]] bool accept_mindet(double) const { return true; }
};

// Closed set of objective kinds for std::visit dispatch.
template <int D> using ObjectiveKindT = std::variant<ShapeObjectiveT<D>, UntangleObjectiveT<D>>;

// Select the objective for a run. `phase == "untangle"` picks the δ-surrogate;
// anything else (default "barrier") picks the shape barrier.
template <int D = kDefaultDim>
inline ObjectiveKindT<D> make_objective(std::string_view phase, double delta)
{
    if (phase == "untangle") { return UntangleObjectiveT<D> {delta}; }
    return ShapeObjectiveT<D> {};
}

// D=2 legacy aliases for the oracle surface and existing call sites.
using ShapeObjective = ShapeObjectiveT<2>;
using UntangleObjective = UntangleObjectiveT<2>;
using ObjectiveKind = ObjectiveKindT<2>;

template <class M>
concept Objective = ObjectiveD<M, 2>;

static_assert(Objective<ShapeObjective>);
static_assert(Objective<UntangleObjective>);

}  // namespace egg
