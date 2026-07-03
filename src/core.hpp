// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

#pragma once

#include <array>
#include <cmath>
#include <experimental/mdspan>
#include <limits>
#include <sycl/sycl.hpp>

namespace egg
{

/// Single value type of every numeric in src/. `double` unless
/// -DEGG_REAL_IS_FP32=1 selects `float`; the double build must stay
/// bit-identical to the pre-refactor code.
#if defined(EGG_REAL_IS_FP32) && EGG_REAL_IS_FP32
using real = float;
#else
using real = double;
#endif

/// `real` literal. A bare `2.0 * x` promotes the subexpression to `double`
/// and narrows back, silently defeating the fp32 build — write `2.0_r`.
consteval real operator""_r(long double v) { return static_cast<real>(v); }
/// `_r` as a named function.
constexpr real r(long double v) { return static_cast<real>(v); }

/// Precision-scaled tolerances. The double build keeps the exact legacy
/// literals (parity); the float build uses thresholds reachable in fp32.
namespace tol
{
inline constexpr real eps = std::numeric_limits<real>::epsilon();  ///< ~2.2e-16 / 1.2e-7
#if defined(EGG_REAL_IS_FP32) && EGG_REAL_IS_FP32
inline constexpr real tiny = 1e-20_r;   ///< det/denominator floor
inline constexpr real znorm = 1e-7_r;   ///< zero-vector guard
inline constexpr real newton = 1e-5_r;  ///< projection |dq| break
inline constexpr real energy = 1e-6_r;  ///< line-search accept slack
#else
inline constexpr real tiny = 1e-30_r;    ///< det/denominator floor
inline constexpr real znorm = 1e-15_r;   ///< zero-vector guard
inline constexpr real newton = 1e-9_r;   ///< projection |dq| break
inline constexpr real energy = 1e-12_r;  ///< line-search accept slack
#endif
}  // namespace tol

/// Per-sample array strides as constexpr functions of the dimension D, so any
/// `template<int D>` sizes its std::arrays without a global constant. The 2D
/// pipeline wires D = 2 throughout; sites whose math is still 2D-only carry a
/// `static_assert(kDim == 2, ...)`, so a 3D build yields a compile-time to-do
/// list instead of silent wrong results.
namespace dim
{
constexpr int vecT(int D) { return D * D; }                 ///< 4 / 9
constexpr int corners(int D) { return 1 << D; }             ///< 4 / 8
constexpr int nbrs(int D) { return D; }                     ///< 2 / 3
constexpr int wInv(int D) { return D * D; }                 ///< 4 / 9
constexpr int jRows(int D) { return D * D; }                ///< 4 / 9
constexpr int jCols(int D) { return D * (D + 1); }          ///< 6 / 12
constexpr int jSize(int D) { return jRows(D) * jCols(D); }  ///< 24 / 108
}  // namespace dim

// Size-algebra lock: catches a wrong formula before any 3D kernel exists.
static_assert(dim::jCols(2) == 6 && dim::jSize(2) == 24);
static_assert(dim::jCols(3) == 12 && dim::jSize(3) == 108);
static_assert(dim::corners(3) == 8 && dim::vecT(3) == 9);

/// @name Dimension-parameterised value types
/// PtN/VecN/MatN are per-axis shapes; VecTN/GradN/HessN the flattened
/// vec(T)-space shapes used by the metric.
/// @{
template <int N> using PtN = std::array<real, N>;
template <int N> using VecN = std::array<real, N>;
template <int N> using MatN = std::array<real, N * N>;  ///< row-major d×d

template <int D> using VecTN = std::array<real, dim::vecT(D)>;  ///< vec(T)
template <int D> using GradN = std::array<real, dim::vecT(D)>;  ///< dmu/dvec(T)
template <int D> using HessN = std::array<real, dim::vecT(D) * dim::vecT(D)>;
/// @}

/// Transition default: un-suffixed aliases below stay D=2 so the oracle
/// bindings and existing 2D call sites keep compiling; dimension itself flows
/// through `template<int D>`.
inline constexpr int kDefaultDim = 2;

using Pt = PtN<kDefaultDim>;    ///< point in R^d
using Vec = VecN<kDefaultDim>;  ///< d-vector (e.g. per-DOF gradient)
using Mat = MatN<kDefaultDim>;  ///< row-major d×d matrix (e.g. per-DOF Hessian)

/// @name Legacy D=2 size aliases (oracle surface / transition only)
/// @{
inline constexpr int kDim = kDefaultDim;
inline constexpr int kVecT = dim::vecT(kDefaultDim);
inline constexpr int kCornersPerCell = dim::corners(kDefaultDim);
inline constexpr int kNbrsPerCorner = dim::nbrs(kDefaultDim);
inline constexpr int kWInvSize = dim::wInv(kDefaultDim);
inline constexpr int kJRows = dim::jRows(kDefaultDim);
inline constexpr int kJCols = dim::jCols(kDefaultDim);
inline constexpr int kJSize = dim::jSize(kDefaultDim);
/// @}

/// @defgroup vecops Elementwise std::array vector ops
/// Used by the geometry charts (`eval`/`frame`) and `orthonormalize`. Defined
/// in namespace egg: found by unqualified lookup inside egg, not by ADL for
/// std::array elsewhere. Templated on std::size_t to match std::array — an
/// `int` size parameter is a non-deduced context.
/// @{

/// Elementwise sum.
template <std::size_t N>
inline std::array<real, N> operator+(const std::array<real, N>& a, const std::array<real, N>& b)
{
    std::array<real, N> out;
    for (std::size_t i = 0; i < N; ++i) { out[i] = a[i] + b[i]; }
    return out;
}
/// Elementwise difference.
template <std::size_t N>
inline std::array<real, N> operator-(const std::array<real, N>& a, const std::array<real, N>& b)
{
    std::array<real, N> out;
    for (std::size_t i = 0; i < N; ++i) { out[i] = a[i] - b[i]; }
    return out;
}
/// Scalar–vector product.
template <std::size_t N> inline std::array<real, N> operator*(real s, const std::array<real, N>& a)
{
    std::array<real, N> out;
    for (std::size_t i = 0; i < N; ++i) { out[i] = s * a[i]; }
    return out;
}
/// Euclidean inner product.
template <std::size_t N> inline real dot(const std::array<real, N>& a, const std::array<real, N>& b)
{
    real acc = 0.0_r;
    for (std::size_t i = 0; i < N; ++i) { acc += a[i] * b[i]; }
    return acc;
}
/// @f$ a/\lVert a\rVert @f$, or @f$ e_0 @f$ when @f$ \lVert a\rVert @f$ <
/// tol::znorm (matches the legacy tangent fallbacks).
template <std::size_t N> inline std::array<real, N> normalize(const std::array<real, N>& a)
{
    const real n = sycl::sqrt(dot(a, a));
    if (n < tol::znorm) {
        std::array<real, N> e {};
        e[0] = 1.0_r;
        return e;
    }
    return (1.0_r / n) * a;
}
/// @}

/// @name Flat coordinate access
/// Node i occupies X[D*i .. D*i+D-1]; looping over D keeps per-axis literals
/// out of the kernels. Default D keeps existing call sites unchanged.
/// @{
template <int D = kDefaultDim> inline PtN<D> load_pt(const real* X, int i)
{
    PtN<D> p;
    for (int k = 0; k < D; ++k) { p[k] = X[(D * i) + k]; }
    return p;
}

template <int D = kDefaultDim> inline void store_pt(real* X, int i, const PtN<D>& p)
{
    for (int k = 0; k < D; ++k) { X[(D * i) + k] = p[k]; }
}
/// @}

}  // namespace egg
