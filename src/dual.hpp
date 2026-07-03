// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

/// @file dual.hpp
/// Forward-mode AD: `Dual<N>` (value + gradient) and `Dual2<N>` (forward-over-
/// forward "hyperdual": value + gradient + symmetric Hessian in one pass).
/// Seed with `seed_dual{,2}<N>(x, i)` (unit i-th partial), build the input with
/// each component seeded on its own direction, evaluate once, read `.g` / `.h`.
/// Used by the delta-continuation untangle; will matter for the 3D mu.
/// T defaults to egg::real so AD carries src/ precision without a
/// double<->real seam; T needs the usual arithmetic plus sycl::sqrt.
#pragma once

#include "core.hpp"

#include <array>
#include <cmath>

namespace egg
{

/// First-order forward-mode dual number.
template <int N, class T = real> struct Dual {
    T v {T(0)};
    std::array<T, N> g {};

    constexpr Dual() = default;
    constexpr Dual(T value) : v(value) { g.fill(T(0)); }  // NOLINT: implicit
    constexpr Dual(T value, std::array<T, N> grad) : v(value), g(grad) {}
};

/// @name Dual arithmetic (value + gradient chain rules)
/// @{
template <int N, class T> constexpr Dual<N, T> operator+(const Dual<N, T>& a, const Dual<N, T>& b)
{
    Dual<N, T> r;
    r.v = a.v + b.v;
    for (int i = 0; i < N; ++i) { r.g[i] = a.g[i] + b.g[i]; }
    return r;
}

template <int N, class T> constexpr Dual<N, T> operator-(const Dual<N, T>& a, const Dual<N, T>& b)
{
    Dual<N, T> r;
    r.v = a.v - b.v;
    for (int i = 0; i < N; ++i) { r.g[i] = a.g[i] - b.g[i]; }
    return r;
}

template <int N, class T> constexpr Dual<N, T> operator-(const Dual<N, T>& a)
{
    Dual<N, T> r;
    r.v = -a.v;
    for (int i = 0; i < N; ++i) { r.g[i] = -a.g[i]; }
    return r;
}

template <int N, class T> constexpr Dual<N, T> operator*(const Dual<N, T>& a, const Dual<N, T>& b)
{
    Dual<N, T> r;
    r.v = a.v * b.v;
    for (int i = 0; i < N; ++i) { r.g[i] = a.g[i] * b.v + a.v * b.g[i]; }
    return r;
}

template <int N, class T> constexpr Dual<N, T> operator/(const Dual<N, T>& a, const Dual<N, T>& b)
{
    Dual<N, T> r;
    const T inv = T(1) / b.v;
    r.v = a.v * inv;
    for (int i = 0; i < N; ++i) {
        r.g[i] = (a.g[i] - r.v * b.g[i]) * inv;  // (a' - (a/b) b') / b
    }
    return r;
}

// scalar mixed ops
template <int N, class T> constexpr Dual<N, T> operator*(T s, const Dual<N, T>& a)
{ return Dual<N, T>(s) * a; }
template <int N, class T> constexpr Dual<N, T> operator*(const Dual<N, T>& a, T s)
{ return a * Dual<N, T>(s); }
template <int N, class T> constexpr Dual<N, T> operator/(const Dual<N, T>& a, T s)
{ return a * (T(1) / s); }
template <int N, class T> constexpr Dual<N, T> operator-(const Dual<N, T>& a, T s)
{ return a - Dual<N, T>(s); }

/// f = sqrt(a): f' = a' / (2f).
template <int N, class T> inline Dual<N, T> sqrt(const Dual<N, T>& a)
{
    Dual<N, T> r;
    r.v = sycl::sqrt(a.v);
    const T coef = T(0.5) / r.v;
    for (int i = 0; i < N; ++i) { r.g[i] = coef * a.g[i]; }
    return r;
}
/// @}

/// Variable with unit i-th partial.
template <int N, class T = real> constexpr Dual<N, T> seed_dual(T x, int i)
{
    Dual<N, T> r(x);
    r.g[i] = T(1);
    return r;
}

/// Second-order forward-over-forward dual: one evaluation yields value,
/// gradient, and full symmetric Hessian.
template <int N, class T = real> struct Dual2 {
    T v {T(0)};
    std::array<T, N> g {};
    std::array<std::array<T, N>, N> h {};

    constexpr Dual2() = default;
    constexpr Dual2(T value) : v(value) {}  // NOLINT: implicit
};

/// @name Dual2 arithmetic (value + gradient + Hessian chain rules)
/// @{
template <int N, class T>
constexpr Dual2<N, T> operator+(const Dual2<N, T>& a, const Dual2<N, T>& b)
{
    Dual2<N, T> r;
    r.v = a.v + b.v;
    for (int i = 0; i < N; ++i) {
        r.g[i] = a.g[i] + b.g[i];
        for (int j = 0; j < N; ++j) { r.h[i][j] = a.h[i][j] + b.h[i][j]; }
    }
    return r;
}

template <int N, class T>
constexpr Dual2<N, T> operator-(const Dual2<N, T>& a, const Dual2<N, T>& b)
{
    Dual2<N, T> r;
    r.v = a.v - b.v;
    for (int i = 0; i < N; ++i) {
        r.g[i] = a.g[i] - b.g[i];
        for (int j = 0; j < N; ++j) { r.h[i][j] = a.h[i][j] - b.h[i][j]; }
    }
    return r;
}

template <int N, class T> constexpr Dual2<N, T> operator-(const Dual2<N, T>& a)
{
    Dual2<N, T> r;
    r.v = -a.v;
    for (int i = 0; i < N; ++i) {
        r.g[i] = -a.g[i];
        for (int j = 0; j < N; ++j) { r.h[i][j] = -a.h[i][j]; }
    }
    return r;
}

template <int N, class T>
constexpr Dual2<N, T> operator*(const Dual2<N, T>& a, const Dual2<N, T>& b)
{
    Dual2<N, T> r;
    r.v = a.v * b.v;
    for (int i = 0; i < N; ++i) { r.g[i] = a.g[i] * b.v + a.v * b.g[i]; }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            // (a b)'' = a'' b + a' b' + a' b' + a b''
            r.h[i][j] = a.h[i][j] * b.v + a.g[i] * b.g[j] + a.g[j] * b.g[i] + a.v * b.h[i][j];
        }
    }
    return r;
}

template <int N, class T>
constexpr Dual2<N, T> operator/(const Dual2<N, T>& a, const Dual2<N, T>& b)
{
    // q = a / b, from a = q b: q' = (a' - q b')/b; q'' = (a'' - 2 q' b' - q b'')/b
    Dual2<N, T> r;
    const T inv = T(1) / b.v;
    r.v = a.v * inv;
    for (int i = 0; i < N; ++i) { r.g[i] = (a.g[i] - r.v * b.g[i]) * inv; }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            r.h[i][j] = (a.h[i][j] - r.g[i] * b.g[j] - r.g[j] * b.g[i] - r.v * b.h[i][j]) * inv;
        }
    }
    return r;
}

// scalar mixed ops
template <int N, class T> constexpr Dual2<N, T> operator*(T s, const Dual2<N, T>& a)
{ return Dual2<N, T>(s) * a; }
template <int N, class T> constexpr Dual2<N, T> operator*(const Dual2<N, T>& a, T s)
{ return a * Dual2<N, T>(s); }
template <int N, class T> constexpr Dual2<N, T> operator/(const Dual2<N, T>& a, T s)
{ return a * (T(1) / s); }
template <int N, class T> constexpr Dual2<N, T> operator-(const Dual2<N, T>& a, T s)
{ return a - Dual2<N, T>(s); }

/// f = sqrt(a): f' = a'/(2f); f'' = (a'' - 2 f' f') / (2f).
template <int N, class T> inline Dual2<N, T> sqrt(const Dual2<N, T>& a)
{
    Dual2<N, T> r;
    r.v = sycl::sqrt(a.v);
    const T inv2f = T(0.5) / r.v;
    for (int i = 0; i < N; ++i) { r.g[i] = inv2f * a.g[i]; }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) { r.h[i][j] = inv2f * (a.h[i][j] - T(2) * r.g[i] * r.g[j]); }
    }
    return r;
}
/// @}

/// Variable with unit i-th partial.
template <int N, class T = real> constexpr Dual2<N, T> seed_dual2(T x, int i)
{
    Dual2<N, T> r(x);
    r.g[i] = T(1);
    return r;
}

}  // namespace egg
