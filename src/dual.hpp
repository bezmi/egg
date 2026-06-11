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
#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace egg
{

// First-order forward-mode dual.
template <int N> struct Dual {
    double v {0.0};
    std::array<double, N> g {};  // value-initialised to zeros

    constexpr Dual() = default;
    constexpr Dual(double value) : v(value) { g.fill(0.0); }  // NOLINT: implicit
    constexpr Dual(double value, std::array<double, N> grad) : v(value), g(grad) {}
};

template <int N> constexpr Dual<N> operator+(const Dual<N>& a, const Dual<N>& b)
{
    Dual<N> r;
    r.v = a.v + b.v;
    for (int i = 0; i < N; ++i) r.g[i] = a.g[i] + b.g[i];
    return r;
}

template <int N> constexpr Dual<N> operator-(const Dual<N>& a, const Dual<N>& b)
{
    Dual<N> r;
    r.v = a.v - b.v;
    for (int i = 0; i < N; ++i) r.g[i] = a.g[i] - b.g[i];
    return r;
}

template <int N> constexpr Dual<N> operator-(const Dual<N>& a)
{
    Dual<N> r;
    r.v = -a.v;
    for (int i = 0; i < N; ++i) r.g[i] = -a.g[i];
    return r;
}

template <int N> constexpr Dual<N> operator*(const Dual<N>& a, const Dual<N>& b)
{
    Dual<N> r;
    r.v = a.v * b.v;
    for (int i = 0; i < N; ++i) r.g[i] = a.g[i] * b.v + a.v * b.g[i];
    return r;
}

template <int N> constexpr Dual<N> operator/(const Dual<N>& a, const Dual<N>& b)
{
    Dual<N> r;
    const double inv = 1.0 / b.v;
    r.v = a.v * inv;
    for (int i = 0; i < N; ++i) r.g[i] = (a.g[i] - r.v * b.g[i]) * inv;  // (a' - (a/b) b') / b
    return r;
}

// scalar mixed ops
template <int N> constexpr Dual<N> operator*(double s, const Dual<N>& a) { return Dual<N>(s) * a; }
template <int N> constexpr Dual<N> operator*(const Dual<N>& a, double s) { return a * Dual<N>(s); }
template <int N> constexpr Dual<N> operator/(const Dual<N>& a, double s) { return a * (1.0 / s); }
template <int N> constexpr Dual<N> operator-(const Dual<N>& a, double s) { return a - Dual<N>(s); }

template <int N> inline Dual<N> sqrt(const Dual<N>& a)
{
    Dual<N> r;
    r.v = std::sqrt(a.v);
    const double coef = 0.5 / r.v;
    for (int i = 0; i < N; ++i) r.g[i] = coef * a.g[i];
    return r;
}

template <int N> constexpr Dual<N> seed_dual(double x, int i)
{
    Dual<N> r(x);
    r.g[i] = 1.0;
    return r;
}

// Second-order forward-over-forward (hyperdual). Carries the full symmetric
// Hessian, so a single evaluation yields value, gradient, and Hessian.
template <int N> struct Dual2 {
    double v {0.0};
    std::array<double, N> g {};
    std::array<std::array<double, N>, N> h {};

    constexpr Dual2() = default;
    constexpr Dual2(double value) : v(value) {}  // NOLINT: implicit
};

template <int N> constexpr Dual2<N> operator+(const Dual2<N>& a, const Dual2<N>& b)
{
    Dual2<N> r;
    r.v = a.v + b.v;
    for (int i = 0; i < N; ++i) {
        r.g[i] = a.g[i] + b.g[i];
        for (int j = 0; j < N; ++j) r.h[i][j] = a.h[i][j] + b.h[i][j];
    }
    return r;
}

template <int N> constexpr Dual2<N> operator-(const Dual2<N>& a, const Dual2<N>& b)
{
    Dual2<N> r;
    r.v = a.v - b.v;
    for (int i = 0; i < N; ++i) {
        r.g[i] = a.g[i] - b.g[i];
        for (int j = 0; j < N; ++j) r.h[i][j] = a.h[i][j] - b.h[i][j];
    }
    return r;
}

template <int N> constexpr Dual2<N> operator-(const Dual2<N>& a)
{
    Dual2<N> r;
    r.v = -a.v;
    for (int i = 0; i < N; ++i) {
        r.g[i] = -a.g[i];
        for (int j = 0; j < N; ++j) r.h[i][j] = -a.h[i][j];
    }
    return r;
}

template <int N> constexpr Dual2<N> operator*(const Dual2<N>& a, const Dual2<N>& b)
{
    Dual2<N> r;
    r.v = a.v * b.v;
    for (int i = 0; i < N; ++i) r.g[i] = a.g[i] * b.v + a.v * b.g[i];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            // (a b)'' = a'' b + a' b' + a' b' + a b''
            r.h[i][j] = a.h[i][j] * b.v + a.g[i] * b.g[j] + a.g[j] * b.g[i] + a.v * b.h[i][j];
    return r;
}

template <int N> constexpr Dual2<N> operator/(const Dual2<N>& a, const Dual2<N>& b)
{
    // q = a / b. Differentiate a = q b twice:
    //   q'  = (a' - q b') / b
    //   q'' = (a'' - q' b' - q' b' - q b'') / b
    Dual2<N> r;
    const double inv = 1.0 / b.v;
    r.v = a.v * inv;
    for (int i = 0; i < N; ++i) r.g[i] = (a.g[i] - r.v * b.g[i]) * inv;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            r.h[i][j] = (a.h[i][j] - r.g[i] * b.g[j] - r.g[j] * b.g[i] - r.v * b.h[i][j]) * inv;
    return r;
}

// scalar mixed ops
template <int N> constexpr Dual2<N> operator*(double s, const Dual2<N>& a)
{ return Dual2<N>(s) * a; }
template <int N> constexpr Dual2<N> operator*(const Dual2<N>& a, double s)
{ return a * Dual2<N>(s); }
template <int N> constexpr Dual2<N> operator/(const Dual2<N>& a, double s) { return a * (1.0 / s); }
template <int N> constexpr Dual2<N> operator-(const Dual2<N>& a, double s)
{ return a - Dual2<N>(s); }

template <int N> inline Dual2<N> sqrt(const Dual2<N>& a)
{
    // f = sqrt(a): f' = a'/(2f); f'' = (a'' - 2 f' f') / (2f)
    Dual2<N> r;
    r.v = std::sqrt(a.v);
    const double inv2f = 0.5 / r.v;
    for (int i = 0; i < N; ++i) r.g[i] = inv2f * a.g[i];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) r.h[i][j] = inv2f * (a.h[i][j] - 2.0 * r.g[i] * r.g[j]);
    return r;
}

template <int N> constexpr Dual2<N> seed_dual2(double x, int i)
{
    Dual2<N> r(x);
    r.g[i] = 1.0;
    return r;
}

}  // namespace egg
