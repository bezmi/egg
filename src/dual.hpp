//   Dual<N>   first-order forward mode: value + gradient[N].
//   Dual2<N>  second-order forward-over-forward ("hyperdual" / dual2nd):
//             value + gradient[N] + symmetric Hessian[N][N], all in one pass.
//
// Seeding convention: seed_dual<N>(x, i) makes a variable whose i-th partial is
// 1. To get a full gradient/Hessian, build the input vector with each component
// seeded on its own direction, evaluate the metric once, and read .g / .h.
//
// Not currently used for the shape mu, but is used for δ-continuation untangling.
// will be important for the 3D mu
//
// The scalar type is templated (`T = egg::real`) so the AD path carries the same
// precision as the rest of src/ without a double<->real seam; T must support the
// usual arithmetic and std::sqrt (real/float/double all do).
#pragma once

#include <array>
#include <cmath>

#include "core.hpp"

namespace egg
{

// First-order forward-mode dual.
template <int N, class T = real> struct Dual {
    T v {T(0)};
    std::array<T, N> g {};  // value-initialised to zeros

    constexpr Dual() = default;
    constexpr Dual(T value) : v(value) { g.fill(T(0)); }  // NOLINT: implicit
    constexpr Dual(T value, std::array<T, N> grad) : v(value), g(grad) {}
};

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

template <int N, class T> inline Dual<N, T> sqrt(const Dual<N, T>& a)
{
    Dual<N, T> r;
    r.v = std::sqrt(a.v);
    const T coef = T(0.5) / r.v;
    for (int i = 0; i < N; ++i) { r.g[i] = coef * a.g[i]; }
    return r;
}

template <int N, class T = real> constexpr Dual<N, T> seed_dual(T x, int i)
{
    Dual<N, T> r(x);
    r.g[i] = T(1);
    return r;
}

// Second-order forward-over-forward (hyperdual). Carries the full symmetric
// Hessian, so a single evaluation yields value, gradient, and Hessian.
template <int N, class T = real> struct Dual2 {
    T v {T(0)};
    std::array<T, N> g {};
    std::array<std::array<T, N>, N> h {};

    constexpr Dual2() = default;
    constexpr Dual2(T value) : v(value) {}  // NOLINT: implicit
};

template <int N, class T> constexpr Dual2<N, T> operator+(const Dual2<N, T>& a, const Dual2<N, T>& b)
{
    Dual2<N, T> r;
    r.v = a.v + b.v;
    for (int i = 0; i < N; ++i) {
        r.g[i] = a.g[i] + b.g[i];
        for (int j = 0; j < N; ++j) { r.h[i][j] = a.h[i][j] + b.h[i][j]; }
    }
    return r;
}

template <int N, class T> constexpr Dual2<N, T> operator-(const Dual2<N, T>& a, const Dual2<N, T>& b)
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

template <int N, class T> constexpr Dual2<N, T> operator*(const Dual2<N, T>& a, const Dual2<N, T>& b)
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

template <int N, class T> constexpr Dual2<N, T> operator/(const Dual2<N, T>& a, const Dual2<N, T>& b)
{
    // q = a / b. Differentiate a = q b twice:
    //   q'  = (a' - q b') / b
    //   q'' = (a'' - q' b' - q' b' - q b'') / b
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

template <int N, class T> inline Dual2<N, T> sqrt(const Dual2<N, T>& a)
{
    // f = sqrt(a): f' = a'/(2f); f'' = (a'' - 2 f' f') / (2f)
    Dual2<N, T> r;
    r.v = std::sqrt(a.v);
    const T inv2f = T(0.5) / r.v;
    for (int i = 0; i < N; ++i) { r.g[i] = inv2f * a.g[i]; }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) { r.h[i][j] = inv2f * (a.h[i][j] - T(2) * r.g[i] * r.g[j]); }
    }
    return r;
}

template <int N, class T = real> constexpr Dual2<N, T> seed_dual2(T x, int i)
{
    Dual2<N, T> r(x);
    r.g[i] = T(1);
    return r;
}

}  // namespace egg
