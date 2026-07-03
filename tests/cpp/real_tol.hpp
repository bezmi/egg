// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

// real_tol.hpp — precision-aware comparison tolerance for the C++ tests.
//
// The golden tables (golden_*.hpp) and the in-test reference math are computed
// in double. When the build selects egg::real = float (-DEGG_REAL_IS_FP32=ON),
// the device/library results only match those references to ~1e-4..1e-6, so the
// double-tuned literal tolerances (1e-9 .. 1e-12) would spuriously fail. real_tol
// floors a tolerance at the fp32 parity level in the float build and is the
// identity in the double build (so the double build keeps its exact thresholds).
#pragma once

#include "core.hpp"  // egg::real, EGG_REAL_IS_FP32

#include <array>
#include <cstddef>
#include <vector>

namespace egg_test
{

// True when the build selects egg::real = float.
#if defined(EGG_REAL_IS_FP32) && EGG_REAL_IS_FP32
inline constexpr bool kRealIsFloat = true;
#else
inline constexpr bool kRealIsFloat = false;
#endif

// Precision-scaled finite-difference step size. Under fp32 a step of 1e-6
// loses ~half its significant digits to cancellation; the float step is sized
// near sqrt(eps_fp32) ≈ 3.5e-4 to keep the FD reference meaningful.
inline constexpr double fd_step()
{
    if constexpr (kRealIsFloat) return 5e-4;
    else return 1e-6;
}

// Precision-scaled finite-difference tolerance floor. The cancellation floor
// for FD in fp32 (~ eps/h) is higher than the round-trip parity floor (5e-4),
// so FD-comparison tolerances get a separate, higher floor (2e-3). Identity
// in the double build.
inline constexpr double fd_tol(double tol_f64)
{
    if constexpr (kRealIsFloat) {
        constexpr double kFp32FdFloor = 2e-3;
        return tol_f64 < kFp32FdFloor ? kFp32FdFloor : tol_f64;
    } else {
        return tol_f64;
    }
}

// Narrow a double golden-input array to egg::real for passing into the
// real-typed library APIs (the golden tables stay double; inputs are real-typed
// in the float build). Identity copy when egg::real == double.
template <std::size_t N>
std::array<egg::real, N> to_real(const std::array<double, N>& a)
{
    std::array<egg::real, N> r;
    for (std::size_t i = 0; i < N; ++i) { r[i] = static_cast<egg::real>(a[i]); }
    return r;
}

// Same, for a flat double buffer of runtime length.
inline std::vector<egg::real> to_real(const double* p, std::size_t n)
{
    std::vector<egg::real> v(n);
    for (std::size_t i = 0; i < n; ++i) { v[i] = static_cast<egg::real>(p[i]); }
    return v;
}

// Floor a double-precision-tuned tolerance at the fp32 parity level when
// egg::real == float; identity otherwise.
inline constexpr double real_tol(double tol_f64)
{
#if defined(EGG_REAL_IS_FP32) && EGG_REAL_IS_FP32
    constexpr double kFp32Floor = 5e-4;  // relative parity vs the double reference
    return tol_f64 < kFp32Floor ? kFp32Floor : tol_f64;
#else
    return tol_f64;
#endif
}

}  // namespace egg_test
