#pragma once

#include <array>
#include <experimental/mdspan>

namespace egg
{

// ---------------------------------------------------------------------------
// Spatial dimension of the core (the single knob). The 2D pipeline wires
// kDim = 2; the value types, kernel indexing, and device-buffer strides below
// are all expressed in terms of kDim and the derived sizes, so moving the core
// to 3D means supplying the dimension-specific *math* (the barrier metric, the
// d-neighbour corner stencil, the chain-Jacobian) — not rewriting the plumbing.
// Sites whose math is still 2D-only carry a `static_assert(kDim == 2, ...)`, so
// flipping kDim to 3 yields a precise compile-time to-do list rather than silent
// wrong results.
// ---------------------------------------------------------------------------
// Dimension size algebra. Every per-sample array stride is a pure constexpr
// function of the dimension D, so any `template<int D>` can size its std::arrays
// without a global constant. (Replaces the old module-global kVecT/kJSize/… —
// those survive below only as D=2 convenience aliases for the oracle surface.)
namespace dim
{
constexpr int vecT(int D) { return D * D; }                 // 4 / 9
constexpr int corners(int D) { return 1 << D; }             // 4 / 8
constexpr int nbrs(int D) { return D; }                     // 2 / 3
constexpr int wInv(int D) { return D * D; }                 // 4 / 9
constexpr int jRows(int D) { return D * D; }                // 4 / 9
constexpr int jCols(int D) { return D * (D + 1); }          // 6 / 12
constexpr int jSize(int D) { return jRows(D) * jCols(D); }  // 24 / 108
}  // namespace dim

// Size-algebra lock: catches a wrong size formula before any 3D kernel exists.
static_assert(dim::jCols(2) == 6 && dim::jSize(2) == 24);
static_assert(dim::jCols(3) == 12 && dim::jSize(3) == 108);
static_assert(dim::corners(3) == 8 && dim::vecT(3) == 9);

// Dimension-parameterised value types. PtN/VecN/MatN are the per-axis shapes;
// VecTN/GradN/HessN are the flattened vec(T)-space shapes used by the metric.
template <int N> using PtN = std::array<double, N>;
template <int N> using VecN = std::array<double, N>;
template <int N> using MatN = std::array<double, N * N>;  // row-major d×d

template <int D> using VecTN = std::array<double, dim::vecT(D)>;  // vec(T)
template <int D> using GradN = std::array<double, dim::vecT(D)>;  // dmu/dvec(T)
template <int D> using HessN = std::array<double, dim::vecT(D) * dim::vecT(D)>;

// Transition default: the only remaining mention of a "global" dim. The
// un-suffixed names survive as D=2 aliases so the oracle bindings and existing
// 2D call sites keep compiling; dimension itself flows through template<int D>.
inline constexpr int kDefaultDim = 2;

using Pt = PtN<kDefaultDim>;    // a point / coordinate vector in R^d
using Vec = VecN<kDefaultDim>;  // a d-vector (e.g. per-DOF gradient)
using Mat = MatN<kDefaultDim>;  // a d×d matrix (e.g. per-DOF Hessian), row-major

// Legacy size aliases (D=2). Kept for the oracle surface / transition only.
inline constexpr int kDim = kDefaultDim;
inline constexpr int kVecT = dim::vecT(kDefaultDim);
inline constexpr int kCornersPerCell = dim::corners(kDefaultDim);
inline constexpr int kNbrsPerCorner = dim::nbrs(kDefaultDim);
inline constexpr int kWInvSize = dim::wInv(kDefaultDim);
inline constexpr int kJRows = dim::jRows(kDefaultDim);
inline constexpr int kJCols = dim::jCols(kDefaultDim);
inline constexpr int kJSize = dim::jSize(kDefaultDim);

// Node i occupies X[D*i + 0 .. D*i + D-1] in the flat coordinate array.
// load_pt / store_pt loop over D, so the kernels never spell out per-axis
// literals. Default D = kDefaultDim keeps every existing call site unchanged.
template <int D = kDefaultDim> inline PtN<D> load_pt(const double* X, int i)
{
    PtN<D> p;
    for (int k = 0; k < D; ++k) { p[k] = X[(D * i) + k]; }
    return p;
}

template <int D = kDefaultDim> inline void store_pt(double* X, int i, const PtN<D>& p)
{
    for (int k = 0; k < D; ++k) { X[(D * i) + k] = p[k]; }
}

}  // namespace egg
