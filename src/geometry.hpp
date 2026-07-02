#pragma once

#include "core.hpp"
#include "entity_soa.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <numbers>
#include <span>
#include <type_traits>

namespace egg
{

using Tag = int;
inline constexpr Tag TAG_FREE = 0;
inline constexpr Tag TAG_LINESEG = 1;
inline constexpr Tag TAG_CIRCLE = 2;
inline constexpr Tag TAG_ELLIPSE = 3;
inline constexpr Tag TAG_SPHERE = 4;  // 3D surface; no 2D entity.
inline constexpr Tag TAG_PLANE = 5;   // 3D surface; no 2D entity.
inline constexpr Tag TAG_CIRCLEARC = 6;
inline constexpr Tag TAG_ELLIPSEARC = 7;
inline constexpr Tag TAG_QUADBEZIER = 8;
inline constexpr Tag TAG_CUBICBEZIER = 9;
inline constexpr Tag TAG_BSPLINE = 10;
inline constexpr Tag TAG_COMPOSITE = 11;
inline constexpr Tag TAG_CYLINDER = 12;     // 3D surface
inline constexpr Tag TAG_LINE3 = 13;        // 3D edge curve
inline constexpr Tag TAG_BSPLINESURF = 14;  // 3D tensor-product B-spline/NURBS surface

inline constexpr int kParamPad = 12;
/// @brief Arena record size of one composite-path segment: `[tag, params(kParamPad)]`.
inline constexpr int kCompositeRecSize = 1 + kParamPad;

// Pt is the D=2 coordinate type (PtN<2>). The 2D entity set is curves only
// (Free, line, circle, ellipse, arcs, Béziers); Sphere/Plane are genuine
// surfaces and live in the 3D entity set, not here.

// ===========================================================================
// Concept-modelled geometry entities (the single source of truth used by the
// kernels). Each entity is a trivially-copyable typed value type with its
// per-shape math inlined (bit-identical to the old raw-pointer free functions).
// The concrete type is selected per launch by `dispatch_entity_type<2>(tag, f)`
// and built with `EntitySoA<E>::load` (the device sweep) or `decode_entity<E>`
// (cold blob/oracle paths) — the kernel body is fully monomorphic in `E`, with
// no per-DOF `std::visit` and no value-level entity variant (retired in Phase 4).
// ===========================================================================

/// @brief Parameter-space coordinate: @f$ (t) @f$ for a curve, @f$ (u,v) @f$ for a surface.
/// @tparam K Intrinsic dimension of the parametrization.
template <int K> using Param = std::array<double, K>;

/// @brief The @p D identity columns @f$ e_0 \dots e_{D-1} @f$ (the free-DOF tangent basis).
/// @tparam D Embedding dimension.
/// @return The @f$ D @f$ unit basis vectors, one per column.
template <int D> inline std::array<VecN<D>, D> identity_columns()
{
    std::array<VecN<D>, D> b {};
    for (int i = 0; i < D; ++i) { b[i][i] = 1.0; }
    return b;
}

/// @brief Gram–Schmidt orthonormalization of @p K columns embedded in @p D.
/// @tparam D Embedding dimension.
/// @tparam K Number of columns (the intrinsic/tangent dimension); @c K==1 reduces
///           to a single `normalize`.
/// @param b The raw (non-orthonormal) tangent columns.
/// @return The orthonormalized columns.
template <int D, int K> inline std::array<VecN<D>, K> orthonormalize(std::array<VecN<D>, K> b)
{
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < i; ++j) { b[i] = b[i] - dot(b[i], b[j]) * b[j]; }
        b[i] = normalize(b[i]);
    }
    return b;
}

/// @brief Fused projection result returned by `project_frame`.
/// @tparam D Embedding dimension.
/// @tparam K Tangent dimension (number of basis columns).
template <int D, int K> struct Frame {
    PtN<D> pos;                    ///< The projected point.
    std::array<VecN<D>, K> basis;  ///< The orthonormal tangent columns.
    int eff_tdim;                  ///< Effective tdim, dropped by one when the
                                   ///< node lands on a trim boundary.
};

/// @brief A parametrization: an intrinsic-@c idim manifold embedded in @c edim,
///        given by the three maps `invert` (point → params), `eval`
///        (params → point), and `frame` (params → the @c idim raw tangent columns).
/// @tparam P The candidate parametrization type.
template <class P>
concept Parametrization = requires(const P param, PtN<P::edim> p, Param<P::idim> q) {
    { P::edim } -> std::convertible_to<int>;  // embedding dimension D
    { P::idim } -> std::convertible_to<int>;  // intrinsic dimension k (== tdim)
    { param.invert(p) } -> std::same_as<Param<P::idim>>;
    { param.eval(q) } -> std::same_as<PtN<P::edim>>;
    { param.frame(q) } -> std::same_as<std::array<VecN<P::edim>, P::idim>>;
};

/// @brief A boundary entity: what the constrained sweep actually calls.
///        `project_frame` is the fused (pos + orthonormal basis + effective tdim)
///        form; `project` and `tangent_basis` are projections of it used by the
///        existing call sites.
/// @tparam E The candidate entity type.
template <class E>
concept GeometryEntity = requires(const E& e, const typename E::Pt& p) {
    { E::tdim } -> std::convertible_to<int>;
    { e.project(p) } -> std::same_as<typename E::Pt>;
    { e.tangent_basis(p) } -> std::same_as<std::array<typename E::Vec, E::tdim>>;
    { e.project_frame(p) };
};

// --- Trim: a k-box / k-region, one specialization per intrinsic dim -----------

/// @brief Wrap @p x into the half-open interval @f$ [a, b) @f$ (for closed curves).
/// @param x Value to wrap.
/// @param a,b Interval bounds; if @f$ b \le a @f$, @p x is returned unchanged.
/// @return The wrapped value.
inline double wrap(double x, double a, double b)
{
    const double L = b - a;
    if (L <= 0.0) { return x; }
    double t = std::fmod(x - a, L);
    if (t < 0.0) { t += L; }
    return a + t;
}

/// @brief A k-box / k-region trim in parameter space.
/// @tparam K Intrinsic dimension; `Trim<1>` is a curve interval, `Trim<2>` a UV region.
template <int K> struct Trim;

/// @brief Curve trim: the parameter interval @f$ [t_0, t_1] @f$ (or a closed loop).
template <> struct Trim<1> {
    double t0, t1;        ///< Interval bounds.
    bool closed = false;  ///< If set, the curve is periodic and @c contains is always true.
    /// @brief Whether @p q lies within the interval.
    /// @param q Parameter to test.
    /// @return True if inside (always true when closed).
    [[nodiscard]] bool contains(Param<1> q) const { return closed || (q[0] >= t0 && q[0] <= t1); }
    /// @brief Clamp @p q onto the interval (wrapping when closed).
    /// @param q Parameter to clamp.
    /// @return The clamped (or wrapped) parameter.
    [[nodiscard]] Param<1> clamp(Param<1> q) const
    { return {closed ? wrap(q[0], t0, t1) : std::clamp(q[0], t0, t1)}; }
};

/// @brief Surface trim: a UV polygon (outer loop minus holes), arena-backed.
///        Empty spans mean "untrimmed" (the surface's full natural range).
template <> struct Trim<2> {
    std::span<const PtN<2>> verts;  ///< All loop vertices, concatenated.
    std::span<const int> loops;     ///< Offset table: [outer | hole0 | ... | end].
    /// @brief Even–odd inside test over all loops (outer minus holes).
    ///        Untrimmed (no loops) always contains.
    [[nodiscard]] bool contains(Param<2> uv) const
    {
        if (loops.size() < 2) { return true; }
        bool inside = false;
        for (std::size_t l = 0; l + 1 < loops.size(); ++l) {
            const int lo = loops[l], hi = loops[l + 1];
            for (int i = lo, j = hi - 1; i < hi; j = i++) {
                const PtN<2>&a = verts[i], &b = verts[j];
                // Even–odd ray cast in +u from uv.
                if ((a[1] > uv[1]) != (b[1] > uv[1]) &&
                    uv[0] < ((b[0] - a[0]) * (uv[1] - a[1]) / (b[1] - a[1])) + a[0]) {
                    inside = !inside;
                }
            }
        }
        return inside;
    }
    /// @brief Nearest point on the loop polylines (closed loops).
    [[nodiscard]] Param<2> clamp(Param<2> uv) const
    {
        Param<2> best = uv;
        double best_d = std::numeric_limits<double>::infinity();
        for (std::size_t l = 0; l + 1 < loops.size(); ++l) {
            const int lo = loops[l], hi = loops[l + 1];
            for (int i = lo, j = hi - 1; i < hi; j = i++) {
                const PtN<2>&a = verts[j], &b = verts[i];
                const VecN<2> ab = b - a;
                const double ab_sq = dot(ab, ab);
                double t = ab_sq > 1e-30 ? dot(PtN<2> {uv[0], uv[1]} - a, ab) / ab_sq : 0.0;
                t = std::clamp(t, 0.0, 1.0);
                const PtN<2> q = a + t * ab;
                const VecN<2> dq = q - PtN<2> {uv[0], uv[1]};
                const double dd = dot(dq, dq);
                if (dd < best_d) {
                    best_d = dd;
                    best = {q[0], q[1]};
                }
            }
        }
        return best;
    }
};

/// @brief The generic trimmed entity: any @ref Parametrization restricted to its k-region trim.
/// @tparam P The parametrization type (must satisfy @ref Parametrization).
template <Parametrization P> struct TrimmedEntity {
    using Pt = PtN<P::edim>;              ///< Point type in the embedding space.
    using Vec = VecN<P::edim>;            ///< Vector type in the embedding space.
    static constexpr int tdim = P::idim;  ///< Tangent dimension (== intrinsic dim).
    P param;                              ///< The underlying parametrization.
    Trim<P::idim> trim;                   ///< The parameter-space trim region.

    /// @brief Project @p p onto the trimmed parametrization, returning the fused frame.
    /// @param p Query point.
    /// @return The projected point, orthonormal tangent columns, and effective
    ///         tdim (one less than @c tdim when @p p lands on the trim boundary).
    [[nodiscard]] Frame<P::edim, tdim> project_frame(const Pt& p) const
    {
        auto q = param.invert(p);
        const bool inside = trim.contains(q);
        if (!inside) { q = trim.clamp(q); }
        return {.pos = param.eval(q),
                .basis = orthonormalize<P::edim, tdim>(param.frame(q)),
                .eff_tdim = inside ? tdim : tdim - 1};
    }
    /// @brief Project @p p onto the trimmed parametrization.
    /// @param p Query point.
    /// @return The projected point.
    [[nodiscard]] Pt project(const Pt& p) const { return project_frame(p).pos; }
    /// @brief Orthonormal tangent basis at the projection of @p p.
    /// @param p Query point.
    /// @return The @c tdim orthonormal tangent columns.
    [[nodiscard]] std::array<Vec, tdim> tangent_basis(const Pt& p) const
    { return project_frame(p).basis; }
};

/// @brief Interior (free) DOF: intrinsic dimension @c k==D, identity projection.
/// @tparam D Embedding dimension.
template <int D> struct Free {
    using Pt = PtN<D>;              ///< Point type.
    using Vec = VecN<D>;            ///< Vector type.
    static constexpr int tdim = D;  ///< Tangent dimension equals the embedding dimension.
    /// @brief Identity projection (a free node moves anywhere).
    [[nodiscard]] Pt project(const Pt& p) const { return p; }
    /// @brief The full identity tangent basis.
    [[nodiscard]] std::array<Vec, D> tangent_basis(const Pt&) const
    { return identity_columns<D>(); }
    /// @brief Fused frame: the point itself, identity basis, full @c tdim.
    [[nodiscard]] Frame<D, D> project_frame(const Pt& p) const
    { return {.pos = p, .basis = identity_columns<D>(), .eff_tdim = D}; }
};

// ---------------------------------------------------------------------------
// The existing analytic 2D entities (tdim==1, full-range / untrimmed). Their
// closed-form project/tangent are kept verbatim -- bit-identical to the legacy
// raw-pointer math -- and each is wrapped in the richer GeometryEntity interface
// (tangent_basis is the single tangent column; project_frame fuses the two).
// They are NOT routed through the generic invert->clamp->eval pipeline, because
// a trig reparametrization of the circle/ellipse would not be bit-exact.
// ---------------------------------------------------------------------------

/// @brief A line segment from @f$ (s_x, s_y) @f$ to @f$ (e_x, e_y) @f$, clamped to @f$ [0,1] @f$.
struct LineSeg {
    using Pt = PtN<2>;
    using Vec = VecN<2>;
    static constexpr int tdim = 1;
    double sx, sy, ex, ey;
    [[nodiscard]] Pt project(const Pt& p) const
    {
        const double abx = ex - sx, aby = ey - sy;
        const double ab_sq = (abx * abx) + (aby * aby);
        double t = (((p[0] - sx) * abx) + ((p[1] - sy) * aby)) / std::fmax(ab_sq, 1e-30);
        t = t < 0.0 ? 0.0 : t;
        t = t > 1.0 ? 1.0 : t;  // clip to [0, 1]
        return Pt {sx + (t * abx), sy + (t * aby)};
    }
    [[nodiscard]] Vec tangent(const Pt&) const
    {
        const double abx = ex - sx, aby = ey - sy;
        const double norm = std::sqrt((abx * abx) + (aby * aby));
        if (norm < 1e-15) {
            return Vec {1.0, 0.0};  // eye[:, 0]
        }
        return Vec {abx / norm, aby / norm};
    }
    [[nodiscard]] std::array<Vec, 1> tangent_basis(const Pt& p) const { return {tangent(p)}; }
    [[nodiscard]] Frame<2, 1> project_frame(const Pt& p) const
    { return {.pos = project(p), .basis = {tangent(p)}, .eff_tdim = 1}; }
};
/// @brief A circle of radius @p r centred at @f$ (c_x, c_y) @f$; radial projection.
struct Circle {
    using Pt = PtN<2>;
    using Vec = VecN<2>;
    static constexpr int tdim = 1;
    double cx, cy, r;
    [[nodiscard]] Pt project(const Pt& p) const
    {
        const double dx = p[0] - cx, dy = p[1] - cy;
        const double dist = std::sqrt((dx * dx) + (dy * dy));
        if (dist < 1e-15) {
            return Pt {cx + r, cy};  // arbitrary on-circle point
        }
        return Pt {cx + (r * dx / dist), cy + (r * dy / dist)};
    }
    [[nodiscard]] Vec tangent(const Pt& p) const
    {
        const double dx = p[0] - cx, dy = p[1] - cy;
        const double rn = std::sqrt((dx * dx) + (dy * dy));
        double nx, ny;
        if (rn < 1e-15) {
            nx = 1.0;
            ny = 0.0;  // eye[:, 0]
        } else {
            nx = dx / rn;
            ny = dy / rn;
        }
        return Vec {-ny, nx};
    }
    [[nodiscard]] std::array<Vec, 1> tangent_basis(const Pt& p) const { return {tangent(p)}; }
    [[nodiscard]] Frame<2, 1> project_frame(const Pt& p) const
    { return {.pos = project(p), .basis = {tangent(p)}, .eff_tdim = 1}; }
};
/// @brief An axis-aligned ellipse centred at @f$ (c_x, c_y) @f$ with radii @f$ (r_x, r_y) @f$.
struct Ellipse {
    using Pt = PtN<2>;
    using Vec = VecN<2>;
    static constexpr int tdim = 1;
    double cx, cy, rx, ry;
    [[nodiscard]] Pt project(const Pt& p) const
    {
        const double dx = p[0] - cx, dy = p[1] - cy;
        const double sx = dx / rx, sy = dy / ry;
        const double dist = std::sqrt((sx * sx) + (sy * sy));
        double ux, uy;
        if (dist < 1e-15) {
            ux = uy = 1.0 / std::numbers::sqrt2;  // ones(d)/sqrt(d), d = 2
        } else {
            ux = sx / dist;
            uy = sy / dist;
        }
        return Pt {cx + (ux * rx), cy + (uy * ry)};
    }
    [[nodiscard]] Vec tangent(const Pt& p) const
    {
        const double dx = p[0] - cx, dy = p[1] - cy;
        // Parametric angle from the radial-scaled coordinates (matches Ellipse).
        const double angle = std::atan2(dy / ry, dx / rx);
        double tx = -rx * std::sin(angle);
        double ty = ry * std::cos(angle);
        const double norm = std::sqrt((tx * tx) + (ty * ty));
        if (norm < 1e-15) { return Vec {1.0, 0.0}; }
        return Vec {tx / norm, ty / norm};
    }
    [[nodiscard]] std::array<Vec, 1> tangent_basis(const Pt& p) const { return {tangent(p)}; }
    [[nodiscard]] Frame<2, 1> project_frame(const Pt& p) const
    { return {.pos = project(p), .basis = {tangent(p)}, .eff_tdim = 1}; }
};

/// @brief Representative closed-form curve @ref Parametrization that validates the generic stack.
///
/// `LineParam` drives the generic @ref TrimmedEntity pipeline; its `invert`/`eval`
/// reproduce @ref LineSeg exactly (unclamped @f$ t @f$ from `invert`, @f$ [0,1] @f$
/// clamp from `Trim<1>`). Each additional curve type is just another @ref Parametrization
/// struct (three maps) plus a variant arm; the trimming, orthonormalization, and
/// `project_frame` plumbing is shared and written once here.
struct LineParam {
    static constexpr int edim = 2, idim = 1;
    PtN<2> p0, p1;  ///< Endpoints @f$ P_0 @f$ and @f$ P_1 @f$.
    /// @brief Foot-of-projection parameter (unclamped; `Trim<1>` does the clamp).
    /// @param q Query point.
    /// @return @f$ t = (q - P_0)\cdot(P_1 - P_0) / \lVert P_1 - P_0 \rVert^2 @f$.
    [[nodiscard]] Param<1> invert(const PtN<2>& q) const
    {
        const VecN<2> ab = p1 - p0;
        const double ab_sq = dot(ab, ab);
        return {dot(q - p0, ab) / std::fmax(ab_sq, 1e-30)};
    }
    /// @brief Evaluate the curve point @f$ P_0 + t(P_1 - P_0) @f$.
    /// @param t Curve parameter.
    /// @return The point on the line.
    [[nodiscard]] PtN<2> eval(const Param<1>& t) const { return p0 + t[0] * (p1 - p0); }
    /// @brief The (constant) raw tangent column @f$ P_1 - P_0 @f$.
    /// @return The single tangent column.
    [[nodiscard]] std::array<VecN<2>, 1> frame(const Param<1>&) const { return {p1 - p0}; }
};

static_assert(Parametrization<LineParam>);

// ---------------------------------------------------------------------------
// Curved parametrizations. A point has no closed-form nearest-foot on a general curve, so
// these find the parameter by a seeded Newton iteration on the stationarity
// condition (C(t) - q) · C'(t) = 0 (the foot of the perpendicular). Each curve
// supplies eval / deriv (C') / deriv2 (C''); the shared `project_param` below
// does the coarse seed + Newton, and `frame` returns the single C' tangent.
// ---------------------------------------------------------------------------

/// @brief Seeded-Newton nearest-foot parameter for a curve.
///
/// Coarse-samples the curve over @f$ [t_{lo}, t_{hi}] @f$ for a robust seed, then
/// runs Newton on @f$ f(t) = (C(t) - q)\cdot C'(t) @f$, with
/// @f$ f'(t) = \lVert C'(t)\rVert^2 + (C(t) - q)\cdot C''(t) @f$. The unclamped
/// parameter is returned; the entity's `Trim<1>` clamps it onto the live range.
/// @tparam C Curve type exposing `eval`, `deriv`, and `deriv2` on a `Param<1>`.
/// @param c The curve.
/// @param q Query point.
/// @param t_lo,t_hi Natural parameter domain to seed over.
/// @param n_seed Number of coarse samples for the seed.
/// @param iters Newton iterations.
/// @return The nearest-foot parameter @f$ t @f$.
template <class C>
inline double project_param(
  const C& c, const PtN<2>& q, double t_lo, double t_hi, int n_seed = 16, int iters = 8)
{
    double best_t = t_lo;
    double best_d = std::numeric_limits<double>::infinity();
    for (int i = 0; i <= n_seed; ++i) {
        const double t = t_lo + (((t_hi - t_lo) * i) / n_seed);
        const VecN<2> d = c.eval({t}) - q;
        const double dd = dot(d, d);
        if (dd < best_d) {
            best_d = dd;
            best_t = t;
        }
    }
    double t = best_t;
    for (int it = 0; it < iters; ++it) {
        const VecN<2> d = c.eval({t}) - q;
        const VecN<2> d1 = c.deriv({t});
        const double f = dot(d, d1);
        const double fp = dot(d1, d1) + dot(d, c.deriv2({t}));
        if (std::fabs(fp) < 1e-30) { break; }
        t -= f / fp;
    }
    return t;
}

/// @brief A circular arc of radius @p r centred at @f$ (c_x, c_y) @f$, parametrized
///        by angle; closed-form inverse.
struct CircleArcParam {
    static constexpr int edim = 2, idim = 1;
    PtN<2> c;  ///< Centre.
    double r;  ///< Radius.
    /// @brief Inverse: the polar angle of @p q about the centre.
    [[nodiscard]] Param<1> invert(const PtN<2>& q) const
    { return {std::atan2(q[1] - c[1], q[0] - c[0])}; }
    /// @brief Evaluate @f$ C + r(\cos t, \sin t) @f$.
    [[nodiscard]] PtN<2> eval(const Param<1>& t) const
    { return {c[0] + (r * std::cos(t[0])), c[1] + (r * std::sin(t[0]))}; }
    /// @brief The raw tangent column @f$ C'(t) = r(-\sin t, \cos t) @f$.
    [[nodiscard]] std::array<VecN<2>, 1> frame(const Param<1>& t) const
    { return {VecN<2> {-r * std::sin(t[0]), r * std::cos(t[0])}}; }
};

/// @brief A rotated, axis-scaled elliptical arc; nearest foot via Newton.
struct EllipseArcParam {
    static constexpr int edim = 2, idim = 1;
    PtN<2> c;     ///< Centre.
    double a, b;  ///< Semi-axis lengths along the rotated x/y axes.
    double phi;   ///< Rotation of the major axis from +x.
    /// @brief Evaluate @f$ C + R_\phi (a\cos t, b\sin t) @f$.
    [[nodiscard]] PtN<2> eval(const Param<1>& t) const
    {
        const double cp = std::cos(phi), sp = std::sin(phi);
        const double x = a * std::cos(t[0]), y = b * std::sin(t[0]);
        return {c[0] + (cp * x) - (sp * y), c[1] + (sp * x) + (cp * y)};
    }
    /// @brief First derivative @f$ C'(t) = R_\phi (-a\sin t, b\cos t) @f$.
    [[nodiscard]] VecN<2> deriv(const Param<1>& t) const
    {
        const double cp = std::cos(phi), sp = std::sin(phi);
        const double x = -a * std::sin(t[0]), y = b * std::cos(t[0]);
        return {(cp * x) - (sp * y), (sp * x) + (cp * y)};
    }
    /// @brief Second derivative @f$ C''(t) = R_\phi (-a\cos t, -b\sin t) @f$.
    [[nodiscard]] VecN<2> deriv2(const Param<1>& t) const
    {
        const double cp = std::cos(phi), sp = std::sin(phi);
        const double x = -a * std::cos(t[0]), y = -b * std::sin(t[0]);
        return {(cp * x) - (sp * y), (sp * x) + (cp * y)};
    }
    /// @brief Inverse: seeded-Newton nearest foot over the full angular range.
    [[nodiscard]] Param<1> invert(const PtN<2>& q) const
    { return {project_param(*this, q, 0.0, 2.0 * std::numbers::pi)}; }
    /// @brief The raw tangent column @f$ C'(t) @f$.
    [[nodiscard]] std::array<VecN<2>, 1> frame(const Param<1>& t) const { return {deriv(t)}; }
};

/// @brief A quadratic Bézier curve over three control points; nearest foot via Newton.
struct QuadBezierParam {
    static constexpr int edim = 2, idim = 1;
    std::array<PtN<2>, 3> p;  ///< Control points @f$ P_0, P_1, P_2 @f$.
    /// @brief Evaluate @f$ (1-t)^2 P_0 + 2(1-t)t P_1 + t^2 P_2 @f$.
    [[nodiscard]] PtN<2> eval(const Param<1>& t) const
    {
        const double u = 1.0 - t[0];
        return (u * u) * p[0] + (2.0 * u * t[0]) * p[1] + (t[0] * t[0]) * p[2];
    }
    /// @brief First derivative @f$ 2(1-t)(P_1 - P_0) + 2t(P_2 - P_1) @f$.
    [[nodiscard]] VecN<2> deriv(const Param<1>& t) const
    { return (2.0 * (1.0 - t[0])) * (p[1] - p[0]) + (2.0 * t[0]) * (p[2] - p[1]); }
    /// @brief Second derivative @f$ 2(P_2 - 2P_1 + P_0) @f$ (constant).
    [[nodiscard]] VecN<2> deriv2(const Param<1>&) const
    { return 2.0 * (p[2] - (2.0 * p[1]) + p[0]); }
    /// @brief Inverse: seeded-Newton nearest foot over @f$ [0,1] @f$.
    [[nodiscard]] Param<1> invert(const PtN<2>& q) const
    { return {project_param(*this, q, 0.0, 1.0)}; }
    /// @brief The raw tangent column @f$ C'(t) @f$.
    [[nodiscard]] std::array<VecN<2>, 1> frame(const Param<1>& t) const { return {deriv(t)}; }
};

/// @brief A cubic Bézier curve over four control points; nearest foot via Newton.
struct CubicBezierParam {
    static constexpr int edim = 2, idim = 1;
    std::array<PtN<2>, 4> p;  ///< Control points @f$ P_0 \dots P_3 @f$.
    /// @brief Evaluate the Bernstein form @f$ \sum_i B_i^3(t) P_i @f$.
    [[nodiscard]] PtN<2> eval(const Param<1>& t) const
    {
        const double u = 1.0 - t[0], tt = t[0];
        return (u * u * u) * p[0] + (3.0 * u * u * tt) * p[1] + (3.0 * u * tt * tt) * p[2] +
               (tt * tt * tt) * p[3];
    }
    /// @brief First derivative @f$ 3\sum_i B_i^2(t)(P_{i+1} - P_i) @f$.
    [[nodiscard]] VecN<2> deriv(const Param<1>& t) const
    {
        const double u = 1.0 - t[0], tt = t[0];
        return (3.0 * u * u) * (p[1] - p[0]) + (6.0 * u * tt) * (p[2] - p[1]) +
               (3.0 * tt * tt) * (p[3] - p[2]);
    }
    /// @brief Second derivative @f$ 6\big((1-t)(P_2 - 2P_1 + P_0) + t(P_3 - 2P_2 + P_1)\big) @f$.
    [[nodiscard]] VecN<2> deriv2(const Param<1>& t) const
    {
        const double u = 1.0 - t[0], tt = t[0];
        return (6.0 * u) * (p[2] - (2.0 * p[1]) + p[0]) + (6.0 * tt) * (p[3] - (2.0 * p[2]) + p[1]);
    }
    /// @brief Inverse: seeded-Newton nearest foot over @f$ [0,1] @f$.
    [[nodiscard]] Param<1> invert(const PtN<2>& q) const
    { return {project_param(*this, q, 0.0, 1.0)}; }
    /// @brief The raw tangent column @f$ C'(t) @f$.
    [[nodiscard]] std::array<VecN<2>, 1> frame(const Param<1>& t) const { return {deriv(t)}; }
};

/// @brief Largest B-spline degree the fixed-size de Boor work arrays support.
inline constexpr int kMaxBSplineDegree = 7;
/// @brief Leading dimension of the de Boor basis/derivative work arrays.
inline constexpr int kBSplineCap = kMaxBSplineDegree + 1;

/// @brief Index of the knot span containing @p u (clamped to the live domain).
/// @param degree Basis degree @f$ p @f$.
/// @param n_ctrl Number of control points along this direction.
/// @param knots Knot vector, length @c n_ctrl+degree+1, non-decreasing.
/// @param u Parameter.
/// @return The span index @f$ i @f$ with @f$ knots[i] \le u < knots[i+1] @f$.
inline int bspline_find_span(int degree, int n_ctrl, std::span<const double> knots, double u)
{
    const int p = degree;
    const int n = n_ctrl - 1;
    if (u >= knots[n + 1]) { return n; }
    if (u <= knots[p]) { return p; }
    int lo = p, hi = n + 1, mid = (lo + hi) / 2;
    while (u < knots[mid] || u >= knots[mid + 1]) {
        if (u < knots[mid]) {
            hi = mid;
        } else {
            lo = mid;
        }
        mid = (lo + hi) / 2;
    }
    return mid;
}

/// @brief Nonzero basis functions and their derivatives at @p u (NURBS book A2.3).
///
/// Written once at "one parametric direction" arity: the curve uses it directly
/// and the tensor-product surface calls it per direction.
/// @param degree Basis degree @f$ p @f$.
/// @param knots Knot vector.
/// @param span Knot span containing @p u (from @ref bspline_find_span).
/// @param u Parameter.
/// @param nd Highest derivative order to compute.
/// @param ders Output: `ders[k][j]` is the k-th derivative of the j-th nonzero
///             basis function, @f$ k = 0..nd @f$, @f$ j = 0..degree @f$.
inline void bspline_basis_ders(
  int degree, std::span<const double> knots, int span, double u, int nd, double ders[][kBSplineCap])
{
    const int p = degree;
    double ndu[kBSplineCap][kBSplineCap];
    double a[2][kBSplineCap];
    double left[kBSplineCap], right[kBSplineCap];
    ndu[0][0] = 1.0;
    for (int j = 1; j <= p; ++j) {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            ndu[j][r] = right[r + 1] + left[j - r];
            const double temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + (right[r + 1] * temp);
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    for (int j = 0; j <= p; ++j) { ders[0][j] = ndu[j][p]; }
    for (int r = 0; r <= p; ++r) {
        int s1 = 0, s2 = 1;
        a[0][0] = 1.0;
        for (int k = 1; k <= nd; ++k) {
            double d = 0.0;
            const int rk = r - k, pk = p - k;
            if (r >= k) {
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                d = a[s2][0] * ndu[rk][pk];
            }
            const int j1 = (rk >= -1) ? 1 : -rk;
            const int j2 = (r - 1 <= pk) ? k - 1 : p - r;
            for (int j = j1; j <= j2; ++j) {
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
                d += a[s2][j] * ndu[rk + j][pk];
            }
            if (r <= pk) {
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                d += a[s2][k] * ndu[r][pk];
            }
            ders[k][r] = d;
            const int tmp = s1;
            s1 = s2;
            s2 = tmp;
        }
    }
    int fac = p;
    for (int k = 1; k <= nd; ++k) {
        for (int j = 0; j <= p; ++j) { ders[k][j] *= fac; }
        fac *= (p - k);
    }
}

/// @brief A B-spline / NURBS curve over a knot vector and a flat control net.
///
/// The control points, knots, and (optional) weights live in spans over a
/// device arena (never owned), so the type stays trivially copyable. Evaluation
/// and derivatives use the de Boor basis-function recurrence. An empty
/// @c weights span selects the polynomial path; a non-empty one (length
/// @c n_ctrl) evaluates the rational form via homogeneous sums
/// @f$ A(u) = \sum N_i w_i P_i @f$, @f$ w(u) = \sum N_i w_i @f$ and the
/// quotient rule for @f$ C = A/w @f$ and its first two derivatives.
struct BSplineCurveParam {
    static constexpr int edim = 2, idim = 1;
    int degree;                       ///< Basis degree @f$ p @f$.
    int n_ctrl;                       ///< Number of control points @f$ n+1 @f$.
    std::span<const double> knots;    ///< Knot vector, length @c n_ctrl+degree+1.
    std::span<const double> ctrl;     ///< Control points, length @c 2*n_ctrl (x,y interleaved).
    std::span<const double> weights;  ///< NURBS weights, length @c n_ctrl, or empty (polynomial).

    /// @brief Control point @p i as a 2D point.
    [[nodiscard]] PtN<2> cp(int i) const { return {ctrl[2 * i], ctrl[(2 * i) + 1]}; }

    /// @brief The @p order-th derivative point at @p u (order 0 = the curve point).
    [[nodiscard]] PtN<2> point_at(double u, int order) const
    {
        const int span = bspline_find_span(degree, n_ctrl, knots, u);
        double ders[3][kBSplineCap];
        bspline_basis_ders(degree, knots, span, u, order, ders);
        if (weights.empty()) {
            PtN<2> acc {0.0, 0.0};
            for (int j = 0; j <= degree; ++j) {
                acc = acc + (ders[order][j] * cp(span - degree + j));
            }
            return acc;
        }
        // Rational: homogeneous derivatives A^(k) = Σ N^(k) w P, w^(k) = Σ N^(k) w,
        // then the quotient rule (orders 0..2):
        //   C = A/w; C' = (A' − w'C)/w; C'' = (A'' − 2w'C' − w''C)/w.
        PtN<2> A[3] {};
        double w[3] {};
        for (int k = 0; k <= order; ++k) {
            for (int j = 0; j <= degree; ++j) {
                const int i = span - degree + j;
                const double nw = ders[k][j] * weights[i];
                A[k] = A[k] + (nw * cp(i));
                w[k] += nw;
            }
        }
        PtN<2> C[3];
        C[0] = (1.0 / w[0]) * A[0];
        if (order >= 1) { C[1] = (1.0 / w[0]) * (A[1] - w[1] * C[0]); }
        if (order >= 2) { C[2] = (1.0 / w[0]) * (A[2] - 2.0 * w[1] * C[1] - w[2] * C[0]); }
        return C[order];
    }

    /// @brief Evaluate the curve point @f$ C(u) @f$.
    [[nodiscard]] PtN<2> eval(const Param<1>& u) const { return point_at(u[0], 0); }
    /// @brief First derivative @f$ C'(u) @f$.
    [[nodiscard]] VecN<2> deriv(const Param<1>& u) const { return point_at(u[0], 1); }
    /// @brief Second derivative @f$ C''(u) @f$.
    [[nodiscard]] VecN<2> deriv2(const Param<1>& u) const { return point_at(u[0], 2); }
    /// @brief Inverse: seeded-Newton nearest foot over the live domain
    ///        @f$ [knots[degree], knots[n\_ctrl]] @f$.
    [[nodiscard]] Param<1> invert(const PtN<2>& q) const
    { return {project_param(*this, q, knots[degree], knots[n_ctrl])}; }
    /// @brief The raw tangent column @f$ C'(u) @f$.
    [[nodiscard]] std::array<VecN<2>, 1> frame(const Param<1>& u) const { return {deriv(u)}; }
};

/// @brief A composite path: an ordered sequence of curve segments joined
///        end-to-end (the 2D analogue of an OCCT wire).
///
/// The segments live as fixed-size records in the per-group arena
/// (@ref kCompositeRecSize doubles each: `[tag, params...]`), so the type stays
/// trivially copyable. Projection projects onto every segment and keeps the
/// nearest; the tangent is the matched segment's. The reported effective tdim
/// is always 1: a node clamped onto a segment endpoint sits on an interior
/// joint of the (continuous) path and must stay free to slide across onto the
/// neighbouring segment on the next projection. Nested composites are not
/// supported (such records are skipped).
struct CompositePath {
    using Pt = PtN<2>;              ///< Point type.
    using Vec = VecN<2>;            ///< Vector type.
    static constexpr int tdim = 1;  ///< A path is a curve.
    int n_segs;                     ///< Number of segment records.
    std::span<const double> recs;   ///< Segment records, @c n_segs*kCompositeRecSize doubles.
    const double* arena;            ///< Arena base for span-backed segments (B-splines).

    /// @brief Fused frame of the nearest segment to @p p.
    /// @param p Query point.
    /// @return The nearest segment's projected point and unit tangent; @c eff_tdim is 1.
    [[nodiscard]] Frame<2, 1> project_frame(const Pt& p) const;  // defined after make_entity
    /// @brief Project @p p onto the nearest segment.
    [[nodiscard]] Pt project(const Pt& p) const { return project_frame(p).pos; }
    /// @brief The matched segment's unit tangent column.
    [[nodiscard]] std::array<Vec, 1> tangent_basis(const Pt& p) const
    { return project_frame(p).basis; }
};

// ---------------------------------------------------------------------------
// 3D surface / edge-curve parametrizations. Each is a Parametrization at
// edim=3: surfaces have idim=2 (tdim==2 in TrimmedEntity, exercising the K=2
// Gram–Schmidt branch of orthonormalize and the 2×2 reduced newton_delta);
// edge curves have idim=1. Axis frames (ax, ay[, az]) are orthonormal; az is
// derived as ax × ay at the make_entity boundary so the upload blob stays
// within kParamPad.
// ---------------------------------------------------------------------------

/// @brief Cross product of two 3-vectors.
inline VecN<3> cross3(const VecN<3>& a, const VecN<3>& b)
{
    return {(a[1] * b[2]) - (a[2] * b[1]),
            (a[2] * b[0]) - (a[0] * b[2]),
            (a[0] * b[1]) - (a[1] * b[0])};
}

/// @brief A plane through @c o spanned by the orthonormal axes @c ax, @c ay.
struct PlaneParam {
    static constexpr int edim = 3, idim = 2;
    PtN<3> o;        ///< Origin.
    VecN<3> ax, ay;  ///< Orthonormal in-plane axes.
    /// @brief Inverse: the in-plane coordinates of @p p.
    [[nodiscard]] Param<2> invert(const PtN<3>& p) const
    {
        const VecN<3> d = p - o;
        return {dot(d, ax), dot(d, ay)};
    }
    /// @brief Evaluate @f$ O + u\,a_x + v\,a_y @f$.
    [[nodiscard]] PtN<3> eval(const Param<2>& q) const { return o + q[0] * ax + q[1] * ay; }
    /// @brief The constant tangent columns @f$ \{a_x, a_y\} @f$.
    [[nodiscard]] std::array<VecN<3>, 2> frame(const Param<2>&) const { return {ax, ay}; }
};

/// @brief A sphere of radius @c r centred at @c c with orthonormal frame
///        @c (ax, ay, az); chart @f$ (u, v) @f$ = (azimuth, latitude).
struct SphereParam {
    static constexpr int edim = 3, idim = 2;
    PtN<3> c;            ///< Centre.
    VecN<3> ax, ay, az;  ///< Orthonormal frame.
    double r;            ///< Radius.
    /// @brief Inverse: azimuth/latitude of the radial direction of @p p.
    [[nodiscard]] Param<2> invert(const PtN<3>& p) const
    {
        const VecN<3> m = normalize(p - c);
        return {std::atan2(dot(m, ay), dot(m, ax)), std::asin(std::clamp(dot(m, az), -1.0, 1.0))};
    }
    /// @brief Evaluate @f$ C + r(\cos v\cos u\,a_x + \cos v\sin u\,a_y + \sin v\,a_z) @f$.
    [[nodiscard]] PtN<3> eval(const Param<2>& q) const
    {
        const double cu = std::cos(q[0]), su = std::sin(q[0]);
        const double cv = std::cos(q[1]), sv = std::sin(q[1]);
        return c + r * (cv * cu * ax + cv * su * ay + sv * az);
    }
    /// @brief The raw tangent columns @f$ \partial S/\partial u, \partial S/\partial v @f$.
    [[nodiscard]] std::array<VecN<3>, 2> frame(const Param<2>& q) const
    {
        const double cu = std::cos(q[0]), su = std::sin(q[0]);
        const double cv = std::cos(q[1]), sv = std::sin(q[1]);
        return {r * (cv * -su * ax + cv * cu * ay), r * (-sv * cu * ax - sv * su * ay + cv * az)};
    }
};

/// @brief A right circular cylinder: axis @c az through @c o, radius @c r;
///        chart @f$ (u, v) @f$ = (angle about the axis, height along it).
struct CylinderParam {
    static constexpr int edim = 3, idim = 2;
    PtN<3> o;            ///< A point on the axis.
    VecN<3> ax, ay, az;  ///< Orthonormal frame; @c az is the axis.
    double r;            ///< Radius.
    /// @brief Inverse: angle about the axis and height along it.
    [[nodiscard]] Param<2> invert(const PtN<3>& p) const
    {
        const VecN<3> d = p - o;
        return {std::atan2(dot(d, ay), dot(d, ax)), dot(d, az)};
    }
    /// @brief Evaluate @f$ O + r(\cos u\,a_x + \sin u\,a_y) + v\,a_z @f$.
    [[nodiscard]] PtN<3> eval(const Param<2>& q) const
    { return o + r * (std::cos(q[0]) * ax + std::sin(q[0]) * ay) + q[1] * az; }
    /// @brief The raw tangent columns.
    [[nodiscard]] std::array<VecN<3>, 2> frame(const Param<2>& q) const
    { return {r * (-std::sin(q[0]) * ax + std::cos(q[0]) * ay), az}; }
};

/// @brief A 3D line through @c p0 and @c p1 (an edge curve, tdim==1).
struct Line3Param {
    static constexpr int edim = 3, idim = 1;
    PtN<3> p0, p1;  ///< Endpoints.
    /// @brief Foot-of-projection parameter (unclamped; `Trim<1>` clamps).
    [[nodiscard]] Param<1> invert(const PtN<3>& q) const
    {
        const VecN<3> ab = p1 - p0;
        return {dot(q - p0, ab) / std::fmax(dot(ab, ab), 1e-30)};
    }
    /// @brief Evaluate @f$ P_0 + t(P_1 - P_0) @f$.
    [[nodiscard]] PtN<3> eval(const Param<1>& t) const { return p0 + t[0] * (p1 - p0); }
    /// @brief The (constant) raw tangent column.
    [[nodiscard]] std::array<VecN<3>, 1> frame(const Param<1>&) const { return {p1 - p0}; }
};

/// @brief A tensor-product B-spline / NURBS surface @f$ S(u,v) @f$ embedded in 3D.
///
/// Knots, the control net, and (optional) weights live in spans over a device
/// arena (never owned). The control net is row-major over @f$ (i_u, i_v) @f$
/// with xyz interleaved. An empty @c weights span selects the polynomial path;
/// a non-empty one (length @c nu*nv) evaluates the rational form via the
/// homogeneous sums @f$ A(u,v) = \sum N_i N_j w_{ij} P_{ij} @f$,
/// @f$ w(u,v) = \sum N_i N_j w_{ij} @f$ and the bivariate quotient rule.
/// The inverse is a coarse-grid-seeded Newton iteration on the nearest-foot
/// stationarity @f$ (S-p)\cdot S_u = (S-p)\cdot S_v = 0 @f$ (the surface-arity
/// instantiation of the curve projector).
struct BSplineSurfaceParam {
    static constexpr int edim = 3, idim = 2;
    int pu, pv;                       ///< Basis degrees in u and v.
    int nu, nv;                       ///< Control-net extents in u and v.
    std::span<const double> knots_u;  ///< Knot vector in u, length @c nu+pu+1.
    std::span<const double> knots_v;  ///< Knot vector in v, length @c nv+pv+1.
    std::span<const double> ctrl;     ///< Control net, length @c 3*nu*nv (xyz interleaved).
    std::span<const double> weights;  ///< NURBS weights, length @c nu*nv, or empty.

    /// @brief Control point @f$ P_{i_u, i_v} @f$.
    [[nodiscard]] PtN<3> cp(int iu, int iv) const
    {
        const int b = 3 * ((iu * nv) + iv);
        return {ctrl[b], ctrl[b + 1], ctrl[b + 2]};
    }

    /// @brief All partial derivatives @f$ S^{(a,b)} = \partial^{a+b}S/\partial u^a \partial v^b @f$
    ///        for @f$ a, b \le nd @f$ (the fused evaluation all callers share).
    /// @param q Surface parameters @f$ (u, v) @f$.
    /// @param nd Highest per-direction derivative order (0, 1, or 2).
    /// @param S Output: `S[a][b]` is @f$ S^{(a,b)} @f$; entries with
    ///          @f$ a + b > nd \cdot 2 @f$ beyond what the quotient rule below
    ///          fills are untouched.
    void ders(const Param<2>& q, int nd, PtN<3> S[3][3]) const
    {
        const int su = bspline_find_span(pu, nu, knots_u, q[0]);
        const int sv = bspline_find_span(pv, nv, knots_v, q[1]);
        double du[3][kBSplineCap], dv[3][kBSplineCap];
        bspline_basis_ders(pu, knots_u, su, q[0], nd, du);
        bspline_basis_ders(pv, knots_v, sv, q[1], nd, dv);

        // Homogeneous tensor-product sums A^(a,b) (and w^(a,b) when rational).
        PtN<3> A[3][3] {};
        double w[3][3] {};
        for (int a = 0; a <= nd; ++a) {
            for (int b = 0; b <= nd; ++b) {
                PtN<3> acc {};
                double wacc = 0.0;
                for (int i = 0; i <= pu; ++i) {
                    const int iu = su - pu + i;
                    for (int j = 0; j <= pv; ++j) {
                        const int iv = sv - pv + j;
                        const double nij = du[a][i] * dv[b][j];
                        if (weights.empty()) {
                            acc = acc + (nij * cp(iu, iv));
                        } else {
                            const double nw = nij * weights[(iu * nv) + iv];
                            acc = acc + (nw * cp(iu, iv));
                            wacc += nw;
                        }
                    }
                }
                A[a][b] = acc;
                w[a][b] = wacc;
            }
        }
        if (weights.empty()) {
            for (int a = 0; a <= nd; ++a) {
                for (int b = 0; b <= nd; ++b) { S[a][b] = A[a][b]; }
            }
            return;
        }
        // Bivariate quotient rule for S = A/w, up to second order per direction.
        const double iw = 1.0 / w[0][0];
        S[0][0] = iw * A[0][0];
        if (nd >= 1) {
            S[1][0] = iw * (A[1][0] - w[1][0] * S[0][0]);
            S[0][1] = iw * (A[0][1] - w[0][1] * S[0][0]);
            S[1][1] = iw * (A[1][1] - w[1][0] * S[0][1] - w[0][1] * S[1][0] - w[1][1] * S[0][0]);
        }
        if (nd >= 2) {
            S[2][0] = iw * (A[2][0] - 2.0 * w[1][0] * S[1][0] - w[2][0] * S[0][0]);
            S[0][2] = iw * (A[0][2] - 2.0 * w[0][1] * S[0][1] - w[0][2] * S[0][0]);
        }
    }

    /// @brief Evaluate the surface point @f$ S(u, v) @f$.
    [[nodiscard]] PtN<3> eval(const Param<2>& q) const
    {
        PtN<3> S[3][3];
        ders(q, 0, S);
        return S[0][0];
    }

    /// @brief The raw tangent columns @f$ \{S_u, S_v\} @f$.
    [[nodiscard]] std::array<VecN<3>, 2> frame(const Param<2>& q) const
    {
        PtN<3> S[3][3];
        ders(q, 1, S);
        return {S[1][0], S[0][1]};
    }

    /// @brief Inverse: coarse-grid-seeded Newton on the nearest-foot stationarity.
    ///
    /// Seeds from an 8×8 sample of the live domain, then iterates Newton on
    /// @f$ F = ((S-p)\cdot S_u, (S-p)\cdot S_v) @f$ with the exact Jacobian
    /// (needs the second partials), clamping each iterate to the domain. A
    /// near-singular Jacobian (e.g. at a degenerate corner) stops early.
    [[nodiscard]] Param<2> invert(const PtN<3>& p) const
    {
        const double u0 = knots_u[pu], u1 = knots_u[nu];
        const double v0 = knots_v[pv], v1 = knots_v[nv];
        constexpr int kSeed = 8;
        Param<2> q {u0, v0};
        double best = std::numeric_limits<double>::infinity();
        for (int i = 0; i <= kSeed; ++i) {
            for (int j = 0; j <= kSeed; ++j) {
                const Param<2> t {u0 + (((u1 - u0) * i) / kSeed), v0 + (((v1 - v0) * j) / kSeed)};
                const VecN<3> d = eval(t) - p;
                const double dd = dot(d, d);
                if (dd < best) {
                    best = dd;
                    q = t;
                }
            }
        }
        for (int it = 0; it < 12; ++it) {
            PtN<3> S[3][3];
            ders(q, 2, S);
            const VecN<3> d = S[0][0] - p;
            const double f1 = dot(d, S[1][0]), f2 = dot(d, S[0][1]);
            const double j11 = dot(S[1][0], S[1][0]) + dot(d, S[2][0]);
            const double j12 = dot(S[1][0], S[0][1]) + dot(d, S[1][1]);
            const double j22 = dot(S[0][1], S[0][1]) + dot(d, S[0][2]);
            const double det = (j11 * j22) - (j12 * j12);
            if (std::fabs(det) < 1e-30) { break; }
            q[0] = std::clamp(q[0] - (((j22 * f1) - (j12 * f2)) / det), u0, u1);
            q[1] = std::clamp(q[1] - (((-j12 * f1) + (j11 * f2)) / det), v0, v1);
        }
        return q;
    }
};

static_assert(Parametrization<PlaneParam> && Parametrization<SphereParam> &&
              Parametrization<CylinderParam> && Parametrization<Line3Param> &&
              Parametrization<BSplineSurfaceParam>);
static_assert(GeometryEntity<Free<3>> && GeometryEntity<TrimmedEntity<PlaneParam>> &&
              GeometryEntity<TrimmedEntity<SphereParam>> &&
              GeometryEntity<TrimmedEntity<CylinderParam>> &&
              GeometryEntity<TrimmedEntity<Line3Param>> &&
              GeometryEntity<TrimmedEntity<BSplineSurfaceParam>>);

static_assert(Parametrization<LineParam> && Parametrization<CircleArcParam> &&
              Parametrization<EllipseArcParam> && Parametrization<QuadBezierParam> &&
              Parametrization<CubicBezierParam> && Parametrization<BSplineCurveParam>);
static_assert(GeometryEntity<Free<2>> && GeometryEntity<LineSeg> && GeometryEntity<Circle> &&
              GeometryEntity<Ellipse> && GeometryEntity<TrimmedEntity<LineParam>> &&
              GeometryEntity<TrimmedEntity<CircleArcParam>> &&
              GeometryEntity<TrimmedEntity<EllipseArcParam>> &&
              GeometryEntity<TrimmedEntity<QuadBezierParam>> &&
              GeometryEntity<TrimmedEntity<CubicBezierParam>> && GeometryEntity<CompositePath>);

// --- The closed 2D entity set --------------------------------------------------
//
// The per-DOF entity variant (`EntityKind<D>` / `std::variant<…>`) was retired in
// Phase 4: no value-level variant is materialized anywhere. The closed set now
// lives as (a) the `static_assert(GeometryEntity<…>)` block above (every type
// models the entity interface), (b) the `dispatch_entity_type<2>` switch arms,
// and (c) the `decode_entity_fn<E>` / `EntitySoA<E>` specializations — all three
// must list the same types, which a missing specialization fails to compile.

// ---------------------------------------------------------------------------
// decode_entity<E>: the per-type positional blob decoder (single source).
//
// Each make_entity arm is split into a decode_entity_fn<E> specialization that
// builds a concrete typed E from the positional (params, arena) blob. This is
// the ONE place the untyped `params` pointer is read per type; make_entity
// delegates to it (single source), and the Phase 1a monomorphic sweep kernel
// calls decode_entity<E> directly — no variant, no make_entity on the device
// hot path. Variable-length entities (B-spline net/knots, composite records)
// keep their data in `arena` and store only offsets/counts in `params`;
// fixed-size entities ignore `arena`.
//
// Selected via dispatch_entity_type<D>(tag, [&]<class E>{ decode_entity<E>(...); }),
// so even the cold oracle/test paths never materialize a variant (Phase 4).
// ---------------------------------------------------------------------------

/// @brief Per-type positional blob decoder trait. Specialize per entity type @p E.
/// @tparam E The entity type to decode into.
template <class E> struct decode_entity_fn;  // primary undefined

template <> struct decode_entity_fn<Free<2>> {
    [[nodiscard]] static Free<2> apply(const double*, const double*) { return Free<2> {}; }
};
template <> struct decode_entity_fn<LineSeg> {
    [[nodiscard]] static LineSeg apply(const double* p, const double*)
    { return LineSeg {.sx = p[0], .sy = p[1], .ex = p[2], .ey = p[3]}; }
};
template <> struct decode_entity_fn<Circle> {
    [[nodiscard]] static Circle apply(const double* p, const double*)
    { return Circle {.cx = p[0], .cy = p[1], .r = p[2]}; }
};
template <> struct decode_entity_fn<Ellipse> {
    [[nodiscard]] static Ellipse apply(const double* p, const double*)
    { return Ellipse {.cx = p[0], .cy = p[1], .rx = p[2], .ry = p[3]}; }
};
template <> struct decode_entity_fn<TrimmedEntity<CircleArcParam>> {
    [[nodiscard]] static TrimmedEntity<CircleArcParam> apply(const double* p, const double*)
    {
        return TrimmedEntity<CircleArcParam> {
          .param = {.c = {p[0], p[1]}, .r = p[2]},
          .trim = {.t0 = p[3], .t1 = p[4], .closed = p[5] != 0.0}};
    }
};
template <> struct decode_entity_fn<TrimmedEntity<EllipseArcParam>> {
    [[nodiscard]] static TrimmedEntity<EllipseArcParam> apply(const double* p, const double*)
    {
        return TrimmedEntity<EllipseArcParam> {
          .param = {.c = {p[0], p[1]}, .a = p[2], .b = p[3], .phi = p[4]},
          .trim = {.t0 = p[5], .t1 = p[6], .closed = p[7] != 0.0}};
    }
};
template <> struct decode_entity_fn<TrimmedEntity<QuadBezierParam>> {
    [[nodiscard]] static TrimmedEntity<QuadBezierParam> apply(const double* p, const double*)
    {
        return TrimmedEntity<QuadBezierParam> {
          .param = {.p = {{{p[0], p[1]}, {p[2], p[3]}, {p[4], p[5]}}}},
          .trim = {.t0 = p[6], .t1 = p[7], .closed = false}};
    }
};
template <> struct decode_entity_fn<TrimmedEntity<CubicBezierParam>> {
    [[nodiscard]] static TrimmedEntity<CubicBezierParam> apply(const double* p, const double*)
    {
        return TrimmedEntity<CubicBezierParam> {
          .param = {.p = {{{p[0], p[1]}, {p[2], p[3]}, {p[4], p[5]}, {p[6], p[7]}}}},
          .trim = {.t0 = p[8], .t1 = p[9], .closed = false}};
    }
};
template <> struct decode_entity_fn<TrimmedEntity<BSplineCurveParam>> {
    [[nodiscard]] static TrimmedEntity<BSplineCurveParam> apply(const double* p, const double* arena)
    {
        // Blob: [degree, n_ctrl, knot_off, ctrl_off, t0, t1, w_off, has_w];
        // knots/control points/weights live in the arena. Counts derive from
        // degree and n_ctrl; has_w == 0 selects the polynomial path.
        const int degree = static_cast<int>(p[0]);
        const int n_ctrl = static_cast<int>(p[1]);
        const auto knot_off = static_cast<std::size_t>(p[2]);
        const auto ctrl_off = static_cast<std::size_t>(p[3]);
        const auto n_knots =
          static_cast<std::size_t>(n_ctrl) + static_cast<std::size_t>(degree) + 1;
        const auto n_ctrl_d = 2 * static_cast<std::size_t>(n_ctrl);
        const bool has_w = p[7] != 0.0;
        const auto w_off = static_cast<std::size_t>(p[6]);
        return TrimmedEntity<BSplineCurveParam> {
          .param = {.degree = degree,
                    .n_ctrl = n_ctrl,
                    .knots = {arena + knot_off, n_knots},
                    .ctrl = {arena + ctrl_off, n_ctrl_d},
                    .weights = has_w ? std::span<const double> {arena + w_off,
                                                                static_cast<std::size_t>(n_ctrl)}
                                     : std::span<const double> {}},
          .trim = {.t0 = p[4], .t1 = p[5], .closed = false}};
    }
};
template <> struct decode_entity_fn<CompositePath> {
    [[nodiscard]] static CompositePath apply(const double* p, const double* arena)
    {
        // Blob: [n_segs, rec_off]; the segment records live in the arena.
        const int n_segs = static_cast<int>(p[0]);
        const auto rec_off = static_cast<std::size_t>(p[1]);
        return CompositePath {
          .n_segs = n_segs,
          .recs = {arena + rec_off, static_cast<std::size_t>(n_segs) * kCompositeRecSize},
          .arena = arena};
    }
};

// 3D entity decoders — the per-type builders for the D==3 entity set. Surface
// blobs carry the axis frame as (o/c, ax, ay[, r]); az is derived as ax × ay
// here. Surfaces are untrimmed (empty `Trim<2>`) until the CAD importer bakes
// UV trim polygons into the arena. The blob layouts match the Python
// `entity_encoding` encoder and the `EntitySoA<E>` specializations below.
template <> struct decode_entity_fn<Free<3>> {
    [[nodiscard]] static Free<3> apply(const double*, const double*) { return Free<3> {}; }
};
template <> struct decode_entity_fn<TrimmedEntity<PlaneParam>> {
    [[nodiscard]] static TrimmedEntity<PlaneParam> apply(const double* p, const double*)
    {
        // Blob: [o(3), ax(3), ay(3)].
        const auto pt = [&](int i) { return PtN<3> {p[i], p[i + 1], p[i + 2]}; };
        return TrimmedEntity<PlaneParam> {.param = {.o = pt(0), .ax = pt(3), .ay = pt(6)},
                                          .trim = {}};
    }
};
template <> struct decode_entity_fn<TrimmedEntity<SphereParam>> {
    [[nodiscard]] static TrimmedEntity<SphereParam> apply(const double* p, const double*)
    {
        // Blob: [c(3), r, ax(3), ay(3)].
        const auto pt = [&](int i) { return PtN<3> {p[i], p[i + 1], p[i + 2]}; };
        const VecN<3> ax = pt(4), ay = pt(7);
        return TrimmedEntity<SphereParam> {
          .param = {.c = pt(0), .ax = ax, .ay = ay, .az = cross3(ax, ay), .r = p[3]},
          .trim = {}};
    }
};
template <> struct decode_entity_fn<TrimmedEntity<CylinderParam>> {
    [[nodiscard]] static TrimmedEntity<CylinderParam> apply(const double* p, const double*)
    {
        // Blob: [o(3), ax(3), ay(3), r].
        const auto pt = [&](int i) { return PtN<3> {p[i], p[i + 1], p[i + 2]}; };
        const VecN<3> ax = pt(3), ay = pt(6);
        return TrimmedEntity<CylinderParam> {
          .param = {.o = pt(0), .ax = ax, .ay = ay, .az = cross3(ax, ay), .r = p[9]},
          .trim = {}};
    }
};
template <> struct decode_entity_fn<TrimmedEntity<Line3Param>> {
    [[nodiscard]] static TrimmedEntity<Line3Param> apply(const double* p, const double*)
    {
        // Blob: [p0(3), p1(3), t0, t1].
        const auto pt = [&](int i) { return PtN<3> {p[i], p[i + 1], p[i + 2]}; };
        return TrimmedEntity<Line3Param> {
          .param = {.p0 = pt(0), .p1 = pt(3)},
          .trim = {.t0 = p[6], .t1 = p[7], .closed = false}};
    }
};
template <> struct decode_entity_fn<TrimmedEntity<BSplineSurfaceParam>> {
    [[nodiscard]] static TrimmedEntity<BSplineSurfaceParam> apply(const double* p, const double* arena)
    {
        // Blob: [pu, pv, nu, nv, ku_off, kv_off, ctrl_off, w_off, has_w];
        // knots/control net/weights live in the arena.
        const int pu = static_cast<int>(p[0]);
        const int pv = static_cast<int>(p[1]);
        const int nu = static_cast<int>(p[2]);
        const int nv = static_cast<int>(p[3]);
        const auto ku_off = static_cast<std::size_t>(p[4]);
        const auto kv_off = static_cast<std::size_t>(p[5]);
        const auto ctrl_off = static_cast<std::size_t>(p[6]);
        const auto w_off = static_cast<std::size_t>(p[7]);
        const bool has_w = p[8] != 0.0;
        const auto n_net = static_cast<std::size_t>(nu) * static_cast<std::size_t>(nv);
        return TrimmedEntity<BSplineSurfaceParam> {
          .param = {.pu = pu,
                    .pv = pv,
                    .nu = nu,
                    .nv = nv,
                    .knots_u = {arena + ku_off, static_cast<std::size_t>(nu + pu) + 1},
                    .knots_v = {arena + kv_off, static_cast<std::size_t>(nv + pv) + 1},
                    .ctrl = {arena + ctrl_off, 3 * n_net},
                    .weights = has_w ? std::span<const double> {arena + w_off, n_net}
                                     : std::span<const double> {}},
          .trim = {}};
    }
};

/// @brief Build a typed entity @p E from its positional blob (single source).
///
/// This is the per-type builder that replaces the make_entity arms: it reads the
/// untyped `params` pointer ONCE per type and returns a concrete `E`. Variable-
/// length entities (B-spline, composite) read their payload from @p arena via
/// the offsets stored in @p params; fixed-size entities ignore @p arena.
/// @tparam E The entity type to decode into.
/// @param params Flat parameter blob (`kParamPad` doubles per DOF).
/// @param arena Base of the per-group double arena for span-backed entities, or
///              nullptr when no variable-length entity is in play.
/// @return The typed entity.
template <class E>
[[nodiscard]] inline E decode_entity(const double* params, const double* arena = nullptr)
{
    return decode_entity_fn<E>::apply(params, arena);
}

// The per-DOF entity variant (`make_entity` returning `EntityKind<D>`) was
// retired in Phase 4. Every call site now selects the concrete entity type via
// `dispatch_entity_type<D>(tag, f)` and builds it with `decode_entity<E>` (cold
// blob/oracle paths) or `EntitySoA<E>::load` (the device sweep) — no variant is
// ever materialized. CompositePath::project_frame is defined below
// dispatch_entity_type (it builds each segment the same way).

// ===========================================================================
// Data-oriented entity registry.
//
// The strong `EntityTag` and the per-entity `EntitySoA<E>` trait live in
// entity_soa.hpp; here they are tied to the concrete 2D entity set. `EntityTag`
// is locked to the legacy `TAG_*` integer values (the frozen wire contract), and
// `dispatch_entity_type` is the host-side, run-once tag -> entity-type dispatch
// that selects a monomorphic kernel per launch (the launch-granularity
// counterpart to the per-element `std::visit`). The device sweep, the composite
// inner loop, and the oracle bindings all dispatch through it — there is no
// value-level entity variant left to visit.
// ===========================================================================

// The strong tag must agree, value for value, with the legacy `TAG_*` blob
// constants used by make_entity, the Python encoder, and the golden tables.
static_assert(to_int(EntityTag::Free) == TAG_FREE);
static_assert(to_int(EntityTag::LineSeg) == TAG_LINESEG);
static_assert(to_int(EntityTag::Circle) == TAG_CIRCLE);
static_assert(to_int(EntityTag::Ellipse) == TAG_ELLIPSE);
static_assert(to_int(EntityTag::Sphere) == TAG_SPHERE);
static_assert(to_int(EntityTag::Plane) == TAG_PLANE);
static_assert(to_int(EntityTag::CircleArc) == TAG_CIRCLEARC);
static_assert(to_int(EntityTag::EllipseArc) == TAG_ELLIPSEARC);
static_assert(to_int(EntityTag::QuadBezier) == TAG_QUADBEZIER);
static_assert(to_int(EntityTag::CubicBezier) == TAG_CUBICBEZIER);
static_assert(to_int(EntityTag::BSpline) == TAG_BSPLINE);
static_assert(to_int(EntityTag::Composite) == TAG_COMPOSITE);
static_assert(to_int(EntityTag::Cylinder) == TAG_CYLINDER);
static_assert(to_int(EntityTag::Line3) == TAG_LINE3);
static_assert(to_int(EntityTag::BSplineSurface) == TAG_BSPLINESURF);

/// @brief SoA schema for the interior (free) DOF: no per-entity fields.
///
/// A free node carries no geometry, so its storage is a bare count and its
/// device builder reconstructs a default `Free<D>`. The `View` is an empty
/// `SoAView<const double>` (0×0 extents) so every `EntitySoA<E>` specialization
/// shares the one typed View type — `PartitionView` then holds a
/// `SoAView<const double>` directly, no `const void*`, no type erasure. `load`
/// ignores the view and returns a default-constructed `Free<D>`.
template <int D> struct EntitySoA<Free<D>> {
    static constexpr EntityTag tag = EntityTag::Free;
    static constexpr int kFields = 0;  ///< No fields; `records` is empty.
    static constexpr int kSeg = 0;     ///< No segmented fields.
    struct Host {
        std::vector<double> records;  ///< Empty (kFields == 0).
        std::size_t count = 0;        ///< Number of free DOFs in the partition.
        std::vector<SegmentedHost<double>> seg;  ///< Empty (kSeg == 0).
    };
    struct View {
        SoAView<const double> records{nullptr, 0, 0};  ///< Empty (no fields).
        SegmentedView<double> seg[kMaxSoASeg]{};       ///< Null (no segmented fields).
    };
    /// @brief Reconstruct the (field-less) free entity.
    [[nodiscard]] static Free<D> load(const View&, std::size_t) { return Free<D> {}; }
    /// @brief Scatter a free entity into the host (no-op: no fields).
    static void load_into(Host&, std::size_t, const Free<D>&) {}
    /// @brief Construct the typed View from the generic partition slots.
    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>*)
    { return View{.records = soa}; }
};

static_assert(HasEntitySoA<Free<2>>);

// ---------------------------------------------------------------------------
// Fixed-size 2D entity SoA specializations (Phase 1b-A, uniformized Phase 2-A).
//
// Each is a packed contiguous record store: one flat `double[count*kFields]`
// per partition, stride `kFields` per entity — the layout that matches the
// sweep's per-entity-load access pattern (one coalesced read of entity i's
// kFields doubles per work item, no per-field array indirection). The View is
// a `SoAView<const double>` mdspan with extents `(count, kFields)`; `load`
// reads `view.records(i, FIELD)` via named compile-time offsets and returns via
// designated initializers (matching make_entity's style). `bool` trim fields
// are stored as `0.0`/`1.0` doubles on the wire and reconstituted via `!= 0.0`.
//
// Phase 2-A uniformized all specializations to the same Host/View shape with
// segmented slots (kSeg == 0 for fixed-size — `seg` vectors/views are empty/
// null), `load_into(Host&, i, const E&)`, and `tie_view`. This makes the ctor
// and kernel code fully generic across fixed-size and segmented types.
// ---------------------------------------------------------------------------

/// @brief SoA schema for @ref LineSeg: packed `(sx, sy, ex, ey)` records.
template <> struct EntitySoA<LineSeg> {
    static constexpr EntityTag tag = EntityTag::LineSeg;
    static constexpr int kFields = 4;
    static constexpr int kSeg = 0;
    static constexpr int SX = 0, SY = 1, EX = 2, EY = 3;
    struct Host {
        std::vector<double> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<double>> seg;
    };
    struct View {
        SoAView<const double> records{nullptr, 0, kFields};
        SegmentedView<double> seg[kMaxSoASeg]{};
    };
    [[nodiscard]] static LineSeg load(const View& v, std::size_t i)
    {
        return LineSeg {
          .sx = v.records[i, SX], .sy = v.records[i, SY],
          .ex = v.records[i, EX], .ey = v.records[i, EY]};
    }
    static void load_into(Host& h, std::size_t i, const LineSeg& e)
    {
        double* r = h.records.data() + i * kFields;
        r[SX] = e.sx; r[SY] = e.sy; r[EX] = e.ex; r[EY] = e.ey;
    }
    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>*)
    { return View{.records = soa}; }
};

/// @brief SoA schema for @ref Circle: packed `(cx, cy, r)` records.
template <> struct EntitySoA<Circle> {
    static constexpr EntityTag tag = EntityTag::Circle;
    static constexpr int kFields = 3;
    static constexpr int kSeg = 0;
    static constexpr int CX = 0, CY = 1, R = 2;
    struct Host {
        std::vector<double> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<double>> seg;
    };
    struct View {
        SoAView<const double> records{nullptr, 0, kFields};
        SegmentedView<double> seg[kMaxSoASeg]{};
    };
    [[nodiscard]] static Circle load(const View& v, std::size_t i)
    {
        return Circle {.cx = v.records[i, CX], .cy = v.records[i, CY], .r = v.records[i, R]};
    }
    static void load_into(Host& h, std::size_t i, const Circle& e)
    {
        double* r = h.records.data() + i * kFields;
        r[CX] = e.cx; r[CY] = e.cy; r[R] = e.r;
    }
    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>*)
    { return View{.records = soa}; }
};

/// @brief SoA schema for @ref Ellipse: packed `(cx, cy, rx, ry)` records.
template <> struct EntitySoA<Ellipse> {
    static constexpr EntityTag tag = EntityTag::Ellipse;
    static constexpr int kFields = 4;
    static constexpr int kSeg = 0;
    static constexpr int CX = 0, CY = 1, RX = 2, RY = 3;
    struct Host {
        std::vector<double> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<double>> seg;
    };
    struct View {
        SoAView<const double> records{nullptr, 0, kFields};
        SegmentedView<double> seg[kMaxSoASeg]{};
    };
    [[nodiscard]] static Ellipse load(const View& v, std::size_t i)
    {
        return Ellipse {
          .cx = v.records[i, CX], .cy = v.records[i, CY],
          .rx = v.records[i, RX], .ry = v.records[i, RY]};
    }
    static void load_into(Host& h, std::size_t i, const Ellipse& e)
    {
        double* r = h.records.data() + i * kFields;
        r[CX] = e.cx; r[CY] = e.cy; r[RX] = e.rx; r[RY] = e.ry;
    }
    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>*)
    { return View{.records = soa}; }
};

/// @brief SoA schema for `TrimmedEntity<CircleArcParam>`:
///        packed `(cx, cy, r, t0, t1, closed)` records.
template <> struct EntitySoA<TrimmedEntity<CircleArcParam>> {
    static constexpr EntityTag tag = EntityTag::CircleArc;
    static constexpr int kFields = 6;
    static constexpr int kSeg = 0;
    static constexpr int CX = 0, CY = 1, R = 2, T0 = 3, T1 = 4, CLOSED = 5;
    struct Host {
        std::vector<double> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<double>> seg;
    };
    struct View {
        SoAView<const double> records{nullptr, 0, kFields};
        SegmentedView<double> seg[kMaxSoASeg]{};
    };
    [[nodiscard]] static TrimmedEntity<CircleArcParam> load(const View& v, std::size_t i)
    {
        return TrimmedEntity<CircleArcParam> {
          .param = {.c = {v.records[i, CX], v.records[i, CY]}, .r = v.records[i, R]},
          .trim = {.t0 = v.records[i, T0], .t1 = v.records[i, T1],
                   .closed = v.records[i, CLOSED] != 0.0}};
    }
    static void load_into(Host& h, std::size_t i, const TrimmedEntity<CircleArcParam>& e)
    {
        double* r = h.records.data() + i * kFields;
        r[CX] = e.param.c[0]; r[CY] = e.param.c[1]; r[R] = e.param.r;
        r[T0] = e.trim.t0; r[T1] = e.trim.t1; r[CLOSED] = e.trim.closed ? 1.0 : 0.0;
    }
    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>*)
    { return View{.records = soa}; }
};

/// @brief SoA schema for `TrimmedEntity<EllipseArcParam>`:
///        packed `(cx, cy, a, b, phi, t0, t1, closed)` records.
template <> struct EntitySoA<TrimmedEntity<EllipseArcParam>> {
    static constexpr EntityTag tag = EntityTag::EllipseArc;
    static constexpr int kFields = 8;
    static constexpr int kSeg = 0;
    static constexpr int CX = 0, CY = 1, A = 2, B = 3, PHI = 4, T0 = 5, T1 = 6, CLOSED = 7;
    struct Host {
        std::vector<double> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<double>> seg;
    };
    struct View {
        SoAView<const double> records{nullptr, 0, kFields};
        SegmentedView<double> seg[kMaxSoASeg]{};
    };
    [[nodiscard]] static TrimmedEntity<EllipseArcParam> load(const View& v, std::size_t i)
    {
        return TrimmedEntity<EllipseArcParam> {
          .param = {.c = {v.records[i, CX], v.records[i, CY]},
                    .a = v.records[i, A], .b = v.records[i, B], .phi = v.records[i, PHI]},
          .trim = {.t0 = v.records[i, T0], .t1 = v.records[i, T1],
                   .closed = v.records[i, CLOSED] != 0.0}};
    }
    static void load_into(Host& h, std::size_t i, const TrimmedEntity<EllipseArcParam>& e)
    {
        double* r = h.records.data() + i * kFields;
        r[CX] = e.param.c[0]; r[CY] = e.param.c[1];
        r[A] = e.param.a; r[B] = e.param.b; r[PHI] = e.param.phi;
        r[T0] = e.trim.t0; r[T1] = e.trim.t1; r[CLOSED] = e.trim.closed ? 1.0 : 0.0;
    }
    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>*)
    { return View{.records = soa}; }
};

/// @brief SoA schema for `TrimmedEntity<QuadBezierParam>`:
///        packed `(P0x, P0y, P1x, P1y, P2x, P2y, t0, t1)` records.
template <> struct EntitySoA<TrimmedEntity<QuadBezierParam>> {
    static constexpr EntityTag tag = EntityTag::QuadBezier;
    static constexpr int kFields = 8;
    static constexpr int kSeg = 0;
    static constexpr int P0X = 0, P0Y = 1, P1X = 2, P1Y = 3, P2X = 4, P2Y = 5, T0 = 6, T1 = 7;
    struct Host {
        std::vector<double> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<double>> seg;
    };
    struct View {
        SoAView<const double> records{nullptr, 0, kFields};
        SegmentedView<double> seg[kMaxSoASeg]{};
    };
    [[nodiscard]] static TrimmedEntity<QuadBezierParam> load(const View& v, std::size_t i)
    {
        return TrimmedEntity<QuadBezierParam> {
          .param = {.p = {{{v.records[i, P0X], v.records[i, P0Y]},
                           {v.records[i, P1X], v.records[i, P1Y]},
                           {v.records[i, P2X], v.records[i, P2Y]}}}},
          .trim = {.t0 = v.records[i, T0], .t1 = v.records[i, T1], .closed = false}};
    }
    static void load_into(Host& h, std::size_t i, const TrimmedEntity<QuadBezierParam>& e)
    {
        double* r = h.records.data() + i * kFields;
        r[P0X] = e.param.p[0][0]; r[P0Y] = e.param.p[0][1];
        r[P1X] = e.param.p[1][0]; r[P1Y] = e.param.p[1][1];
        r[P2X] = e.param.p[2][0]; r[P2Y] = e.param.p[2][1];
        r[T0] = e.trim.t0; r[T1] = e.trim.t1;
    }
    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>*)
    { return View{.records = soa}; }
};

/// @brief SoA schema for `TrimmedEntity<CubicBezierParam>`:
///        packed `(P0x, P0y, P1x, P1y, P2x, P2y, P3x, P3y, t0, t1)` records.
template <> struct EntitySoA<TrimmedEntity<CubicBezierParam>> {
    static constexpr EntityTag tag = EntityTag::CubicBezier;
    static constexpr int kFields = 10;
    static constexpr int kSeg = 0;
    static constexpr int P0X = 0, P0Y = 1, P1X = 2, P1Y = 3, P2X = 4, P2Y = 5, P3X = 6, P3Y = 7,
                          T0 = 8, T1 = 9;
    struct Host {
        std::vector<double> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<double>> seg;
    };
    struct View {
        SoAView<const double> records{nullptr, 0, kFields};
        SegmentedView<double> seg[kMaxSoASeg]{};
    };
    [[nodiscard]] static TrimmedEntity<CubicBezierParam> load(const View& v, std::size_t i)
    {
        return TrimmedEntity<CubicBezierParam> {
          .param = {.p = {{{v.records[i, P0X], v.records[i, P0Y]},
                           {v.records[i, P1X], v.records[i, P1Y]},
                           {v.records[i, P2X], v.records[i, P2Y]},
                           {v.records[i, P3X], v.records[i, P3Y]}}}},
          .trim = {.t0 = v.records[i, T0], .t1 = v.records[i, T1], .closed = false}};
    }
    static void load_into(Host& h, std::size_t i, const TrimmedEntity<CubicBezierParam>& e)
    {
        double* r = h.records.data() + i * kFields;
        r[P0X] = e.param.p[0][0]; r[P0Y] = e.param.p[0][1];
        r[P1X] = e.param.p[1][0]; r[P1Y] = e.param.p[1][1];
        r[P2X] = e.param.p[2][0]; r[P2Y] = e.param.p[2][1];
        r[P3X] = e.param.p[3][0]; r[P3Y] = e.param.p[3][1];
        r[T0] = e.trim.t0; r[T1] = e.trim.t1;
    }
    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>*)
    { return View{.records = soa}; }
};

static_assert(HasEntitySoA<LineSeg>);
static_assert(HasEntitySoA<Circle>);
static_assert(HasEntitySoA<Ellipse>);
static_assert(HasEntitySoA<TrimmedEntity<CircleArcParam>>);
static_assert(HasEntitySoA<TrimmedEntity<EllipseArcParam>>);
static_assert(HasEntitySoA<TrimmedEntity<QuadBezierParam>>);
static_assert(HasEntitySoA<TrimmedEntity<CubicBezierParam>>);

// ---------------------------------------------------------------------------
// Segmented 2D entity SoA specialization (Phase 2-A).
//
// The B-spline curve carries variable-length data (knot vector + control net)
// that the packed-record layout cannot hold. The scalar fields (degree,
// n_ctrl, t0, t1, closed) go in the packed records; the variable-length knots
// and ctrl go in two SegmentedHost/SegmentedView CSR slots (kSeg == 2).
//
// `load` constructs `std::span<const double>` from the SegmentedView's data/off
// arrays — the same idiom the existing `decode_entity` uses from the blob arena,
// and unavoidable because `BSplineCurveParam` itself stores `std::span`. The
// toolchain supports `std::span` in device code (GCC 16.1.1 + AdaptiveCpp); the
// golden tests gate correctness.
// ---------------------------------------------------------------------------

/// @brief SoA schema for `TrimmedEntity<BSplineCurveParam>`:
///        packed `(degree, n_ctrl, t0, t1, closed, has_w)` records + up to 3
///        segmented CSR fields (knots, ctrl, weights). The `has_w` flag selects
///        the rational form; `weights` is present only when `has_w != 0`.
template <> struct EntitySoA<TrimmedEntity<BSplineCurveParam>> {
    static constexpr EntityTag tag = EntityTag::BSpline;
    static constexpr int kFields = 6;
    static constexpr int kSeg = 3;  ///< knots (0), ctrl (1), weights (2, optional).
    static constexpr int DEGREE = 0, N_CTRL = 1, T0 = 2, T1 = 3, CLOSED = 4, HAS_W = 5;
    static constexpr int KNOTS = 0, CTRL = 1, WEIGHTS = 2;  ///< Segmented slot indices.

    struct Host {
        std::vector<double> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<double>> seg;
    };
    struct View {
        SoAView<const double> records{nullptr, 0, kFields};
        SegmentedView<double> seg[kMaxSoASeg]{};
    };

    [[nodiscard]] static TrimmedEntity<BSplineCurveParam> load(const View& v, std::size_t i)
    {
        const bool has_w = v.records[i, HAS_W] != 0.0;
        return TrimmedEntity<BSplineCurveParam> {
          .param = {.degree = static_cast<int>(v.records[i, DEGREE]),
                    .n_ctrl = static_cast<int>(v.records[i, N_CTRL]),
                    .knots = v.seg[KNOTS][i],
                    .ctrl = v.seg[CTRL][i],
                    .weights = has_w ? std::span<const double> {v.seg[WEIGHTS][i]}
                                     : std::span<const double> {}},
          .trim = {.t0 = v.records[i, T0], .t1 = v.records[i, T1],
                   .closed = v.records[i, CLOSED] != 0.0}};
    }

    static void load_into(Host& h, std::size_t i, const TrimmedEntity<BSplineCurveParam>& e)
    {
        double* r = h.records.data() + i * kFields;
        r[DEGREE] = static_cast<double>(e.param.degree);
        r[N_CTRL] = static_cast<double>(e.param.n_ctrl);
        r[T0] = e.trim.t0;
        r[T1] = e.trim.t1;
        r[CLOSED] = e.trim.closed ? 1.0 : 0.0;
        const bool has_w = !e.param.weights.empty();
        r[HAS_W] = has_w ? 1.0 : 0.0;
        h.seg[KNOTS].push_back(e.param.knots);
        h.seg[CTRL].push_back(e.param.ctrl);
        h.seg[WEIGHTS].push_back(has_w ? e.param.weights : std::span<const double> {});
    }

    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>* seg)
    {
        View v{.records = soa};
        if (seg != nullptr) {
            v.seg[KNOTS] = seg[KNOTS];
            v.seg[CTRL] = seg[CTRL];
            v.seg[WEIGHTS] = seg[WEIGHTS];
        }
        return v;
    }
};

static_assert(HasEntitySoA<TrimmedEntity<BSplineCurveParam>>);

// ---------------------------------------------------------------------------
// Composite-path SoA specialization (Phase 3; nested arena added later).
//
// A composite owns a single self-contained arena slice — the same positional
// layout the global blob/oracle path uses, but per-composite: any
// variable-length sub-segment data (B-spline knots/ctrl) is laid down first,
// then the `n_segs` fixed-stride segment records `[seg_tag, params(kParamPad)]`
// (kCompositeRecSize doubles each) at offset `rec_off`. The whole slice goes in
// one SegmentedHost/SegmentedView CSR slot (kSeg == 1); the packed fields are
// `n_segs` and `rec_off` (kFields == 2). On load the reconstructed
// `CompositePath` points `recs` at `slice[rec_off..]` and `arena` at the slice
// base, so each segment is decoded by the one `decode_entity<E>` source in
// `project_frame` — including B-spline sub-segments, whose knot_off/ctrl_off in
// their record are offsets into this same self-contained slice.
//
// This makes a Polyline/Spline of *any* curve type (lines, arcs, Béziers,
// B-splines — the NURBS-relevant case) self-describing on the device, with no
// global arena and no blob fallback. Nested composites remain unsupported (a
// recursive device projection is illegal; CompositePath rejects them at
// construction).
// ---------------------------------------------------------------------------

/// @brief SoA schema for `CompositePath`: packed `(n_segs, rec_off)` + one
///        segmented CSR slot holding the per-composite self-contained arena
///        slice (`[sub-segment data | segment records]`).
template <> struct EntitySoA<CompositePath> {
    static constexpr EntityTag tag = EntityTag::Composite;
    static constexpr int kFields = 2;
    static constexpr int kSeg = 1;  ///< self-contained arena slice (slot 0).
    static constexpr int N_SEGS = 0, REC_OFF = 1;
    static constexpr int ARENA = 0;  ///< Segmented slot index.

    struct Host {
        std::vector<double> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<double>> seg;
    };
    struct View {
        SoAView<const double> records{nullptr, 0, kFields};
        SegmentedView<double> seg[kMaxSoASeg]{};
    };

    [[nodiscard]] static CompositePath load(const View& v, std::size_t i)
    {
        const std::span<const double> slice = v.seg[ARENA][i];
        const auto n_segs = static_cast<int>(v.records[i, N_SEGS]);
        const auto rec_off = static_cast<std::size_t>(v.records[i, REC_OFF]);
        return CompositePath {
          .n_segs = n_segs,
          .recs = slice.subspan(rec_off,
                                static_cast<std::size_t>(n_segs) * kCompositeRecSize),
          .arena = slice.data()};
    }

    static void load_into(Host& h, std::size_t i, const CompositePath& e)
    {
        double* r = h.records.data() + (i * kFields);
        r[N_SEGS] = static_cast<double>(e.n_segs);
        // Blob→SoA host path (golden test/bench): the decoded entity's records
        // are self-contained for fixed-size segments, so the slice is exactly
        // the record block at rec_off == 0. (A composite carrying B-spline
        // sub-segment data is built via the Python wire, not this path.)
        r[REC_OFF] = 0.0;
        h.seg[ARENA].push_back(e.recs);
    }

    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>* seg)
    {
        View v{.records = soa};
        if (seg != nullptr) { v.seg[ARENA] = seg[ARENA]; }
        return v;
    }
};

static_assert(HasEntitySoA<CompositePath>);

// ---------------------------------------------------------------------------
// 3D entity SoA specializations.
//
// These mirror the 2D specializations in shape: fixed-size surfaces/edges
// (Plane, Sphere, Cylinder, Line3) pack their blob into `records` (kSeg == 0),
// matching the `decode_entity_fn<E>` blob layout exactly; the B-spline surface
// stores scalars in `records` and knots/ctrl/weights in segmented CSR slots.
// `Free<3>` is already covered by the `EntitySoA<Free<D>>` template above.
//
// The sweep's `sweep_partition_kernel` calls `EntitySoA<E>::load` /
// `tie_view` unconditionally for every dispatched `E`, so every 3D entity type
// the D==3 dispatch arm can select must have a specialization here.
// ---------------------------------------------------------------------------

/// @brief SoA schema for `TrimmedEntity<PlaneParam>`: packed
///        `(o(3), ax(3), ay(3))` records (9 doubles). kSeg == 0.
template <> struct EntitySoA<TrimmedEntity<PlaneParam>> {
    static constexpr EntityTag tag = EntityTag::Plane;
    static constexpr int kFields = 9;
    static constexpr int kSeg = 0;
    struct Host {
        std::vector<double> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<double>> seg;
    };
    struct View {
        SoAView<const double> records{nullptr, 0, kFields};
        SegmentedView<double> seg[kMaxSoASeg]{};
    };
    [[nodiscard]] static TrimmedEntity<PlaneParam> load(const View& v, std::size_t i)
    {
        const auto pt = [&](int j) {
            return PtN<3> {v.records[i, j], v.records[i, j + 1], v.records[i, j + 2]};
        };
        return TrimmedEntity<PlaneParam> {.param = {.o = pt(0), .ax = pt(3), .ay = pt(6)},
                                          .trim = {}};
    }
    static void load_into(Host& h, std::size_t i, const TrimmedEntity<PlaneParam>& e)
    {
        double* r = h.records.data() + i * kFields;
        for (int k = 0; k < 3; ++k) { r[k] = e.param.o[k]; }
        for (int k = 0; k < 3; ++k) { r[3 + k] = e.param.ax[k]; }
        for (int k = 0; k < 3; ++k) { r[6 + k] = e.param.ay[k]; }
    }
    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>*)
    { return View{.records = soa}; }
};

/// @brief SoA schema for `TrimmedEntity<SphereParam>`: packed
///        `(c(3), r, ax(3), ay(3))` records (10 doubles); `az` is derived at
///        load as `ax × ay`. kSeg == 0.
template <> struct EntitySoA<TrimmedEntity<SphereParam>> {
    static constexpr EntityTag tag = EntityTag::Sphere;
    static constexpr int kFields = 10;
    static constexpr int kSeg = 0;
    struct Host {
        std::vector<double> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<double>> seg;
    };
    struct View {
        SoAView<const double> records{nullptr, 0, kFields};
        SegmentedView<double> seg[kMaxSoASeg]{};
    };
    [[nodiscard]] static TrimmedEntity<SphereParam> load(const View& v, std::size_t i)
    {
        const auto pt = [&](int j) {
            return PtN<3> {v.records[i, j], v.records[i, j + 1], v.records[i, j + 2]};
        };
        const VecN<3> ax = pt(4), ay = pt(7);
        return TrimmedEntity<SphereParam> {
          .param = {.c = pt(0), .ax = ax, .ay = ay, .az = cross3(ax, ay), .r = v.records[i, 3]},
          .trim = {}};
    }
    static void load_into(Host& h, std::size_t i, const TrimmedEntity<SphereParam>& e)
    {
        double* r = h.records.data() + i * kFields;
        for (int k = 0; k < 3; ++k) { r[k] = e.param.c[k]; }
        r[3] = e.param.r;
        for (int k = 0; k < 3; ++k) { r[4 + k] = e.param.ax[k]; }
        for (int k = 0; k < 3; ++k) { r[7 + k] = e.param.ay[k]; }
    }
    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>*)
    { return View{.records = soa}; }
};

/// @brief SoA schema for `TrimmedEntity<CylinderParam>`: packed
///        `(o(3), ax(3), ay(3), r)` records (10 doubles); `az` is derived at
///        load as `ax × ay`. kSeg == 0.
template <> struct EntitySoA<TrimmedEntity<CylinderParam>> {
    static constexpr EntityTag tag = EntityTag::Cylinder;
    static constexpr int kFields = 10;
    static constexpr int kSeg = 0;
    struct Host {
        std::vector<double> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<double>> seg;
    };
    struct View {
        SoAView<const double> records{nullptr, 0, kFields};
        SegmentedView<double> seg[kMaxSoASeg]{};
    };
    [[nodiscard]] static TrimmedEntity<CylinderParam> load(const View& v, std::size_t i)
    {
        const auto pt = [&](int j) {
            return PtN<3> {v.records[i, j], v.records[i, j + 1], v.records[i, j + 2]};
        };
        const VecN<3> ax = pt(3), ay = pt(6);
        return TrimmedEntity<CylinderParam> {
          .param = {.o = pt(0), .ax = ax, .ay = ay, .az = cross3(ax, ay), .r = v.records[i, 9]},
          .trim = {}};
    }
    static void load_into(Host& h, std::size_t i, const TrimmedEntity<CylinderParam>& e)
    {
        double* r = h.records.data() + i * kFields;
        for (int k = 0; k < 3; ++k) { r[k] = e.param.o[k]; }
        for (int k = 0; k < 3; ++k) { r[3 + k] = e.param.ax[k]; }
        for (int k = 0; k < 3; ++k) { r[6 + k] = e.param.ay[k]; }
        r[9] = e.param.r;
    }
    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>*)
    { return View{.records = soa}; }
};

/// @brief SoA schema for `TrimmedEntity<Line3Param>`: packed
///        `(p0(3), p1(3), t0, t1)` records (8 doubles). kSeg == 0.
template <> struct EntitySoA<TrimmedEntity<Line3Param>> {
    static constexpr EntityTag tag = EntityTag::Line3;
    static constexpr int kFields = 8;
    static constexpr int kSeg = 0;
    struct Host {
        std::vector<double> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<double>> seg;
    };
    struct View {
        SoAView<const double> records{nullptr, 0, kFields};
        SegmentedView<double> seg[kMaxSoASeg]{};
    };
    [[nodiscard]] static TrimmedEntity<Line3Param> load(const View& v, std::size_t i)
    {
        const auto pt = [&](int j) {
            return PtN<3> {v.records[i, j], v.records[i, j + 1], v.records[i, j + 2]};
        };
        return TrimmedEntity<Line3Param> {
          .param = {.p0 = pt(0), .p1 = pt(3)},
          .trim = {.t0 = v.records[i, 6], .t1 = v.records[i, 7], .closed = false}};
    }
    static void load_into(Host& h, std::size_t i, const TrimmedEntity<Line3Param>& e)
    {
        double* r = h.records.data() + i * kFields;
        for (int k = 0; k < 3; ++k) { r[k] = e.param.p0[k]; }
        for (int k = 0; k < 3; ++k) { r[3 + k] = e.param.p1[k]; }
        r[6] = e.trim.t0;
        r[7] = e.trim.t1;
    }
    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>*)
    { return View{.records = soa}; }
};

/// @brief SoA schema for `TrimmedEntity<BSplineSurfaceParam>`: packed
///        `(pu, pv, nu, nv, ku_off, kv_off, ctrl_off, w_off, has_w)` records +
///        up to 4 segmented CSR fields (knots_u, knots_v, ctrl, weights). The
///        `has_w` flag selects the rational form; `weights` is present only
///        when `has_w != 0`. Offsets in the record are relative to the CSR slot
///        bases (not a global arena), so `load` slices each span directly.
template <> struct EntitySoA<TrimmedEntity<BSplineSurfaceParam>> {
    static constexpr EntityTag tag = EntityTag::BSplineSurface;
    static constexpr int kFields = 9;
    static constexpr int kSeg = 4;  ///< knots_u (0), knots_v (1), ctrl (2), weights (3, optional).
    static constexpr int PU = 0, PV = 1, NU = 2, NV = 3,
                          KU_OFF = 4, KV_OFF = 5, CTRL_OFF = 6, W_OFF = 7, HAS_W = 8;
    static constexpr int KNOTS_U = 0, KNOTS_V = 1, CTRL = 2, WEIGHTS = 3;

    struct Host {
        std::vector<double> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<double>> seg;
    };
    struct View {
        SoAView<const double> records{nullptr, 0, kFields};
        SegmentedView<double> seg[kMaxSoASeg]{};
    };

    [[nodiscard]] static TrimmedEntity<BSplineSurfaceParam> load(const View& v, std::size_t i)
    {
        const int pu = static_cast<int>(v.records[i, PU]);
        const int pv = static_cast<int>(v.records[i, PV]);
        const int nu = static_cast<int>(v.records[i, NU]);
        const int nv = static_cast<int>(v.records[i, NV]);
        const auto n_net = static_cast<std::size_t>(nu) * static_cast<std::size_t>(nv);
        const bool has_w = v.records[i, HAS_W] != 0.0;
        return TrimmedEntity<BSplineSurfaceParam> {
          .param = {.pu = pu,
                    .pv = pv,
                    .nu = nu,
                    .nv = nv,
                    .knots_u = v.seg[KNOTS_U][i],
                    .knots_v = v.seg[KNOTS_V][i],
                    .ctrl = v.seg[CTRL][i],
                    .weights = has_w ? v.seg[WEIGHTS][i] : std::span<const double> {}},
          .trim = {}};
    }

    static void load_into(Host& h, std::size_t i, const TrimmedEntity<BSplineSurfaceParam>& e)
    {
        double* r = h.records.data() + i * kFields;
        r[PU] = static_cast<double>(e.param.pu);
        r[PV] = static_cast<double>(e.param.pv);
        r[NU] = static_cast<double>(e.param.nu);
        r[NV] = static_cast<double>(e.param.nv);
        // Per-entity offsets are implicit in the CSR layout (slot i owns
        // data[off[i]..off[i+1])), so the record offsets are unused on load;
        // they're stored as 0 for layout symmetry with the blob decoder.
        r[KU_OFF] = 0.0;
        r[KV_OFF] = 0.0;
        r[CTRL_OFF] = 0.0;
        r[W_OFF] = 0.0;
        const bool has_w = !e.param.weights.empty();
        r[HAS_W] = has_w ? 1.0 : 0.0;
        h.seg[KNOTS_U].push_back(e.param.knots_u);
        h.seg[KNOTS_V].push_back(e.param.knots_v);
        h.seg[CTRL].push_back(e.param.ctrl);
        h.seg[WEIGHTS].push_back(has_w ? e.param.weights : std::span<const double> {});
    }

    [[nodiscard]] static View tie_view(SoAView<const double> soa, const SegmentedView<double>* seg)
    {
        View v{.records = soa};
        if (seg != nullptr) {
            v.seg[KNOTS_U] = seg[KNOTS_U];
            v.seg[KNOTS_V] = seg[KNOTS_V];
            v.seg[CTRL] = seg[CTRL];
            v.seg[WEIGHTS] = seg[WEIGHTS];
        }
        return v;
    }
};

static_assert(HasEntitySoA<TrimmedEntity<PlaneParam>>);
static_assert(HasEntitySoA<TrimmedEntity<SphereParam>>);
static_assert(HasEntitySoA<TrimmedEntity<CylinderParam>>);
static_assert(HasEntitySoA<TrimmedEntity<Line3Param>>);
static_assert(HasEntitySoA<TrimmedEntity<BSplineSurfaceParam>>);

/// @brief Host-side tag -> concrete entity TYPE dispatch for the 2D entity set.
///
/// Invokes `f.template operator()<E>()` with the entity type `E` that
/// @ref make_entity produces for @p tag. This is the launch-granularity
/// counterpart to the per-element `std::visit`: it lets the sweep instantiate a
/// kernel for ONE entity type per launch (no all-alternatives inlining, no
/// per-lane dispatch). The type list mirrors the per-dimension entity set
/// exactly; in 2D `Sphere`/`Plane` have no entity and map to `Free`, matching
/// the fall-through. `F` is constrained by @ref EntityDispatchFn per dispatched
/// type, so a malformed callable fails at the concept rather than deep in
/// instantiation.
/// @tparam D Embedding dimension (2 or 3).
/// @tparam F Callable with a templated `operator()<E>()` for each entity type.
/// @param tag Entity kind tag.
/// @param f The type-consuming callable.
template <int D = kDefaultDim, class F>
    requires ((D == 2) &&
              EntityDispatchFn<F, Free<2>> && EntityDispatchFn<F, LineSeg> &&
              EntityDispatchFn<F, Circle> && EntityDispatchFn<F, Ellipse> &&
              EntityDispatchFn<F, TrimmedEntity<CircleArcParam>> &&
              EntityDispatchFn<F, TrimmedEntity<EllipseArcParam>> &&
              EntityDispatchFn<F, TrimmedEntity<QuadBezierParam>> &&
              EntityDispatchFn<F, TrimmedEntity<CubicBezierParam>> &&
              EntityDispatchFn<F, TrimmedEntity<BSplineCurveParam>> &&
              EntityDispatchFn<F, CompositePath>) ||
             ((D == 3) &&
              EntityDispatchFn<F, Free<3>> &&
              EntityDispatchFn<F, TrimmedEntity<PlaneParam>> &&
              EntityDispatchFn<F, TrimmedEntity<SphereParam>> &&
              EntityDispatchFn<F, TrimmedEntity<CylinderParam>> &&
              EntityDispatchFn<F, TrimmedEntity<Line3Param>> &&
              EntityDispatchFn<F, TrimmedEntity<BSplineSurfaceParam>>)
inline void dispatch_entity_type(EntityTag tag, F f)
{
    if constexpr (D == 2) {
        switch (tag) {
        case EntityTag::LineSeg: f.template operator()<LineSeg>(); break;
        case EntityTag::Circle: f.template operator()<Circle>(); break;
        case EntityTag::Ellipse: f.template operator()<Ellipse>(); break;
        case EntityTag::CircleArc: f.template operator()<TrimmedEntity<CircleArcParam>>(); break;
        case EntityTag::EllipseArc: f.template operator()<TrimmedEntity<EllipseArcParam>>(); break;
        case EntityTag::QuadBezier: f.template operator()<TrimmedEntity<QuadBezierParam>>(); break;
        case EntityTag::CubicBezier: f.template operator()<TrimmedEntity<CubicBezierParam>>(); break;
        case EntityTag::BSpline: f.template operator()<TrimmedEntity<BSplineCurveParam>>(); break;
        case EntityTag::Composite: f.template operator()<CompositePath>(); break;
        case EntityTag::Free:
        case EntityTag::Sphere:  // 3D surfaces: no 2D entity, fall through to Free.
        case EntityTag::Plane:
        case EntityTag::Cylinder:
        case EntityTag::Line3:
        case EntityTag::BSplineSurface:
        default: f.template operator()<Free<2>>(); break;
        }
    } else if constexpr (D == 3) {
        switch (tag) {
        case EntityTag::Plane:
            f.template operator()<TrimmedEntity<PlaneParam>>(); break;
        case EntityTag::Sphere:
            f.template operator()<TrimmedEntity<SphereParam>>(); break;
        case EntityTag::Cylinder:
            f.template operator()<TrimmedEntity<CylinderParam>>(); break;
        case EntityTag::Line3:
            f.template operator()<TrimmedEntity<Line3Param>>(); break;
        case EntityTag::BSplineSurface:
            f.template operator()<TrimmedEntity<BSplineSurfaceParam>>(); break;
        case EntityTag::Free:
        default: f.template operator()<Free<3>>(); break;
        }
    } else {
        static_assert(D == 2 || D == 3, "dispatch_entity_type: unsupported dimension");
    }
}

/// @brief Fused frame of the nearest segment to @p p (no entity variant).
///
/// Projects onto every non-nested curve segment and keeps the nearest. Each
/// segment is built monomorphically via @ref dispatch_entity_type +
/// @ref decode_entity<E> — the per-segment positional record is decoded into a
/// concrete `E`, with NO `std::visit` and NO `make_entity` variant (the last
/// device-hot-path variant retired in Phase 3). Nested composites and free
/// records are filtered by tag before dispatch (a recursive device call is
/// illegal); the `if constexpr` guard additionally excludes them at compile
/// time so the `CompositePath`/`Free` dispatch arms instantiate to a no-op.
inline Frame<2, 1> CompositePath::project_frame(const Pt& p) const
{
    Frame<2, 1> best {.pos = p, .basis = {Vec {1.0, 0.0}}, .eff_tdim = 1};
    double best_d = std::numeric_limits<double>::infinity();
    for (int s = 0; s < n_segs; ++s) {
        const double* rec = recs.data() + (static_cast<std::size_t>(s) * kCompositeRecSize);
        const auto seg_tag = static_cast<EntityTag>(static_cast<int>(rec[0]));
        if (seg_tag == EntityTag::Composite || seg_tag == EntityTag::Free) {
            continue;  // no nesting; filtered before dispatch (recursion illegal)
        }
        dispatch_entity_type<2>(seg_tag, [&]<class E>() {
            // Only non-composite curve segments (tdim==1) participate; excluding
            // CompositePath is a compile-time necessity (a recursive call is
            // illegal in device code), and the tag skip above keeps it off the
            // runtime path. The Free arm is likewise a compile-time no-op.
            if constexpr (E::tdim == 1 && !std::is_same_v<E, CompositePath>) {
                const E e = decode_entity<E>(rec + 1, arena);
                Frame<2, 1> f = e.project_frame(p);
                const Vec d = f.pos - p;
                const double dd = dot(d, d);
                if (dd < best_d) {
                    best_d = dd;
                    // A clamped segment endpoint is an interior joint of the
                    // continuous path, not a pinned vertex.
                    f.eff_tdim = 1;
                    best = f;
                }
            }
        });
    }
    return best;
}

/// @brief Project @p p onto the entity @p (tag, params).
///
/// Builds the concrete entity monomorphically via @ref dispatch_entity_type +
/// @ref decode_entity<E> (no `std::visit`, no entity variant). Cold oracle path
/// (the `geometry_project` binding); fixed-size entities only (no arena).
/// @param p Query point.
/// @param tag Entity type tag.
/// @param params Flat parameter blob.
/// @return The projected point.
inline Pt project(const Pt& p, Tag tag, const double* params)
{
    Pt out {};
    dispatch_entity_type<2>(static_cast<EntityTag>(tag),
                            [&]<class E>() { out = decode_entity<E>(params).project(p); });
    return out;
}

/// @brief The @f$ (d, 1) @f$ tangent column at @p p on the entity @p (tag, params).
///
/// The first column of the entity tangent basis (the single tangent for a curve;
/// @f$ e_0 @f$ for Free). Built monomorphically via @ref dispatch_entity_type +
/// @ref decode_entity<E> (no `std::visit`, no entity variant).
/// @param p Query point.
/// @param tag Entity type tag.
/// @param params Flat parameter blob.
/// @return The tangent column.
inline Pt tangent_space(const Pt& p, Tag tag, const double* params)
{
    Pt out {};
    dispatch_entity_type<2>(static_cast<EntityTag>(tag),
                            [&]<class E>() { out = decode_entity<E>(params).tangent_basis(p)[0]; });
    return out;
}

}  // namespace egg
