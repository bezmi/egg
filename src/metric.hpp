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

// Barrier shape objective (minimises the shape-distortion metric μ). The closed
// form is the 2D arithmetic; D!=2 is gated until Phase 2 supplies the 3D barrier.
template <int D> struct ShapeObjectiveT {
    static_assert(D == 2, "ShapeObjectiveT: 3D shape barrier not yet implemented");
    [[nodiscard]] double value(const VecTN<D>& t) const { return mu_value(t); }
    [[nodiscard]] GradN<D> grad(const VecTN<D>& t) const { return mu_grad_closedform(t); }
    [[nodiscard]] HessN<D> hess(const VecTN<D>& t) const { return mu_hess_closedform(t); }
    // Barrier: a step is only valid if every cell stays positively oriented.
    [[nodiscard]] bool accept_mindet(double mindet) const { return mindet > 0.0; }
};

// δ-continuation untangle objective: surrogate value + dual-AD grad/Hess, with the
// relaxed accept rule (the surrogate is finite on folded cells, so min det A may
// be ≤ 0 during continuation). 2D closed/AD form; gated for D!=2 until Phase 2.
template <int D> struct UntangleObjectiveT {
    static_assert(D == 2, "UntangleObjectiveT: 3D untangle surrogate not yet implemented");
    double delta {0.0};

    [[nodiscard]] double value(const VecTN<D>& t) const
    { return mu_untangle(t[0], t[1], t[2], t[3], delta); }
    [[nodiscard]] GradN<D> grad(const VecTN<D>& t) const
    {
        using Dv = Dual<kVecT>;
        const Dv r = mu_untangle(seed_dual<kVecT>(t[0], 0),
                                 seed_dual<kVecT>(t[1], 1),
                                 seed_dual<kVecT>(t[2], 2),
                                 seed_dual<kVecT>(t[3], 3),
                                 delta);
        return GradN<D> {r.g[0], r.g[1], r.g[2], r.g[3]};
    }
    [[nodiscard]] HessN<D> hess(const VecTN<D>& t) const
    {
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
    if (phase == "untangle") return UntangleObjectiveT<D> {delta};
    return ShapeObjectiveT<D> {};
}

// D=2 legacy aliases for the oracle surface and existing call sites.
using ShapeObjective = ShapeObjectiveT<2>;
using UntangleObjective = UntangleObjectiveT<2>;
using ObjectiveKind = ObjectiveKindT<2>;

template <class M> concept Objective = ObjectiveD<M, 2>;

static_assert(Objective<ShapeObjective>);
static_assert(Objective<UntangleObjective>);

}  // namespace egg
