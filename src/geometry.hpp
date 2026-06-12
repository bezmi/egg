#pragma once

#include "core.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <numbers>
#include <span>
#include <type_traits>
#include <variant>

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
inline constexpr Tag TAG_CYLINDER = 12;  // 3D surface
inline constexpr Tag TAG_LINE3 = 13;     // 3D edge curve

inline constexpr int kParamPad = 12;
/// @brief Arena record size of one composite-path segment: `[tag, params(kParamPad)]`.
inline constexpr int kCompositeRecSize = 1 + kParamPad;

// Pt is the D=2 coordinate type (PtN<2>). The 2D entity set is curves only
// (Free, line, circle, ellipse, arcs, Béziers); Sphere/Plane are genuine
// surfaces and live in the 3D entity set, not here.

// ===========================================================================
// Concept-modelled geometry entities (the single source of truth used by the
// kernels). Each entity is a trivially-copyable typed value type: the flat
// upload blob (`const double* params`) is parsed ONCE by make_entity into typed
// fields, and the per-shape math is inlined into each entity (bit-identical to the
// old raw-pointer free functions). The closed set lives in an EntityKind variant;
// the in-kernel dispatch is a per-DOF std::visit over make_entity (geometry is
// per-DOF, so — unlike the run-once Objective visit — the variant is rebuilt and
// visited inside the loop, but the projection/tangent bodies are monomorphic).
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

/// @brief A non-rational B-spline curve over a knot vector and a flat control net.
///
/// The control points and knots live in spans over a device arena (never owned),
/// so the type stays trivially copyable. Evaluation and derivatives use the
/// de Boor basis-function recurrence (the basis derivatives are also reused by the
/// tensor-product surface evaluation). Weights/NURBS are a future extension — the
/// control span is interleaved @f$ (x, y) @f$ pairs, with no weight column yet.
struct BSplineCurveParam {
    static constexpr int edim = 2, idim = 1;
    int degree;                     ///< Polynomial degree @f$ p @f$.
    int n_ctrl;                     ///< Number of control points @f$ n+1 @f$.
    std::span<const double> knots;  ///< Knot vector, length @c n_ctrl+degree+1, non-decreasing.
    std::span<const double> ctrl;   ///< Control points, length @c 2*n_ctrl (x,y interleaved).

    /// @brief Control point @p i as a 2D point.
    [[nodiscard]] PtN<2> cp(int i) const { return {ctrl[2 * i], ctrl[(2 * i) + 1]}; }

    /// @brief Index of the knot span containing @p u (clamped to the live domain).
    /// @param u Curve parameter.
    /// @return The span index @f$ i @f$ with @f$ knots[i] \le u < knots[i+1] @f$.
    [[nodiscard]] int find_span(double u) const
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
    /// @param span Knot span containing @p u (from @ref find_span).
    /// @param u Curve parameter.
    /// @param nd Highest derivative order to compute.
    /// @param ders Output: `ders[k][j]` is the k-th derivative of the j-th nonzero
    ///             basis function, @f$ k = 0..nd @f$, @f$ j = 0..degree @f$.
    void basis_ders(int span, double u, int nd, double ders[][kBSplineCap]) const
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

    /// @brief The @p order-th derivative point at @p u (order 0 = the curve point).
    [[nodiscard]] PtN<2> point_at(double u, int order) const
    {
        const int span = find_span(u);
        double ders[3][kBSplineCap];
        basis_ders(span, u, order, ders);
        PtN<2> acc {0.0, 0.0};
        for (int j = 0; j <= degree; ++j) { acc = acc + (ders[order][j] * cp(span - degree + j)); }
        return acc;
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

static_assert(Parametrization<PlaneParam> && Parametrization<SphereParam> &&
              Parametrization<CylinderParam> && Parametrization<Line3Param>);
static_assert(GeometryEntity<Free<3>> && GeometryEntity<TrimmedEntity<PlaneParam>> &&
              GeometryEntity<TrimmedEntity<SphereParam>> &&
              GeometryEntity<TrimmedEntity<CylinderParam>> &&
              GeometryEntity<TrimmedEntity<Line3Param>>);

static_assert(Parametrization<LineParam> && Parametrization<CircleArcParam> &&
              Parametrization<EllipseArcParam> && Parametrization<QuadBezierParam> &&
              Parametrization<CubicBezierParam> && Parametrization<BSplineCurveParam>);
static_assert(GeometryEntity<Free<2>> && GeometryEntity<LineSeg> && GeometryEntity<Circle> &&
              GeometryEntity<Ellipse> && GeometryEntity<TrimmedEntity<LineParam>> &&
              GeometryEntity<TrimmedEntity<CircleArcParam>> &&
              GeometryEntity<TrimmedEntity<EllipseArcParam>> &&
              GeometryEntity<TrimmedEntity<QuadBezierParam>> &&
              GeometryEntity<TrimmedEntity<CubicBezierParam>> && GeometryEntity<CompositePath>);

// --- Per-dimension closed entity set + the host-side factory ------------------

/// @brief The closed entity set for embedding dimension @p D.
/// @tparam D Embedding dimension; the primary template is left undefined and each
///           supported dimension provides a specialization (only `EntityKind<2>`
///           so far; the 3D set arrives with the surface parametrizations).
template <int D> struct EntityKind;
template <> struct EntityKind<2> {
    using type = std::variant<Free<2>,
                              LineSeg,
                              Circle,
                              Ellipse,
                              TrimmedEntity<CircleArcParam>,
                              TrimmedEntity<EllipseArcParam>,
                              TrimmedEntity<QuadBezierParam>,
                              TrimmedEntity<CubicBezierParam>,
                              TrimmedEntity<BSplineCurveParam>,
                              CompositePath>;
};
template <> struct EntityKind<3> {
    using type = std::variant<Free<3>,
                              TrimmedEntity<PlaneParam>,
                              TrimmedEntity<SphereParam>,
                              TrimmedEntity<CylinderParam>,
                              TrimmedEntity<Line3Param>>;
};
/// @brief Convenience alias for the entity variant at embedding dimension @p D.
template <int D> using EntityKindT = typename EntityKind<D>::type;

/// @brief The single dispatch point: parse the flat upload blob into a typed entity.
///
/// This is the ONE place the untyped `params` pointer is read. Variable-length
/// entities (the B-spline net/knots) keep their data in @p arena and store only
/// offsets/counts in @p params; fixed-size entities ignore @p arena.
/// @param tag Entity type tag (`TAG_*`).
/// @param params Flat parameter blob (`kParamPad` doubles per DOF).
/// @param arena Base of the per-group double arena for span-backed entities, or
///              nullptr when no variable-length entity is in play.
/// @return The typed entity variant.
inline EntityKindT<2> make_entity(Tag tag, const double* params, const double* arena = nullptr)
{
    switch (tag) {
    case TAG_LINESEG:
        return LineSeg {.sx = params[0], .sy = params[1], .ex = params[2], .ey = params[3]};
    case TAG_CIRCLE: return Circle {.cx = params[0], .cy = params[1], .r = params[2]};
    case TAG_ELLIPSE:
        return Ellipse {.cx = params[0], .cy = params[1], .rx = params[2], .ry = params[3]};
    case TAG_CIRCLEARC:
        return TrimmedEntity<CircleArcParam> {
          .param = {.c = {params[0], params[1]}, .r = params[2]},
          .trim = {.t0 = params[3], .t1 = params[4], .closed = params[5] != 0.0}};
    case TAG_ELLIPSEARC:
        return TrimmedEntity<EllipseArcParam> {
          .param = {.c = {params[0], params[1]}, .a = params[2], .b = params[3], .phi = params[4]},
          .trim = {.t0 = params[5], .t1 = params[6], .closed = params[7] != 0.0}};
    case TAG_QUADBEZIER:
        return TrimmedEntity<QuadBezierParam> {
          .param =
            {.p = {{{params[0], params[1]}, {params[2], params[3]}, {params[4], params[5]}}}},
          .trim = {.t0 = params[6], .t1 = params[7], .closed = false}};
    case TAG_CUBICBEZIER:
        return TrimmedEntity<CubicBezierParam> {
          .param = {.p = {{{params[0], params[1]},
                           {params[2], params[3]},
                           {params[4], params[5]},
                           {params[6], params[7]}}}},
          .trim = {.t0 = params[8], .t1 = params[9], .closed = false}};
    case TAG_BSPLINE: {
        // Blob: [degree, n_ctrl, knot_off, ctrl_off, t0, t1]; knots/control points
        // live in the arena. Counts derive from degree and n_ctrl.
        const int degree = static_cast<int>(params[0]);
        const int n_ctrl = static_cast<int>(params[1]);
        const auto knot_off = static_cast<std::size_t>(params[2]);
        const auto ctrl_off = static_cast<std::size_t>(params[3]);
        const auto n_knots =
          static_cast<std::size_t>(n_ctrl) + static_cast<std::size_t>(degree) + 1;
        const auto n_ctrl_d = 2 * static_cast<std::size_t>(n_ctrl);
        return TrimmedEntity<BSplineCurveParam> {
          .param = {.degree = degree,
                    .n_ctrl = n_ctrl,
                    .knots = {arena + knot_off, n_knots},
                    .ctrl = {arena + ctrl_off, n_ctrl_d}},
          .trim = {.t0 = params[4], .t1 = params[5], .closed = false}};
    }
    case TAG_COMPOSITE: {
        // Blob: [n_segs, rec_off]; the segment records live in the arena.
        const int n_segs = static_cast<int>(params[0]);
        const auto rec_off = static_cast<std::size_t>(params[1]);
        return CompositePath {
          .n_segs = n_segs,
          .recs = {arena + rec_off, static_cast<std::size_t>(n_segs) * kCompositeRecSize},
          .arena = arena};
    }
    // TAG_SPHERE / TAG_PLANE are 3D surfaces; they have no 2D entity, so they
    // fall through to Free here.
    case TAG_FREE:
    default: return Free<2> {};
    }
}

inline Frame<2, 1> CompositePath::project_frame(const Pt& p) const
{
    Frame<2, 1> best {.pos = p, .basis = {Vec {1.0, 0.0}}, .eff_tdim = 1};
    double best_d = std::numeric_limits<double>::infinity();
    for (int s = 0; s < n_segs; ++s) {
        const double* rec = recs.data() + (static_cast<std::size_t>(s) * kCompositeRecSize);
        const Tag tag = static_cast<Tag>(rec[0]);
        if (tag == TAG_COMPOSITE || tag == TAG_FREE) { continue; }  // no nesting
        std::visit(
          [&](const auto& e) {
              // Only non-composite curve segments (tdim==1) participate;
              // Free/composite records are filtered by tag above. Excluding
              // CompositePath here is a compile-time necessity, not just an
              // optimisation: a recursive call is illegal in device code.
              using Seg = std::decay_t<decltype(e)>;
              if constexpr (Seg::tdim == 1 && !std::is_same_v<Seg, CompositePath>) {
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
          },
          make_entity(tag, rec + 1, arena));
    }
    return best;
}

/// @brief Parse the flat upload blob into a typed 3D entity.
///
/// Surface blobs carry the axis frame as (o/c, ax, ay[, r]); az is derived as
/// ax × ay here. Surfaces are untrimmed (empty `Trim<2>`) until the CAD
/// importer bakes UV trim polygons into the arena.
/// @param tag Entity type tag (`TAG_*`).
/// @param params Flat parameter blob (`kParamPad` doubles per DOF).
/// @return The typed entity variant.
inline EntityKindT<3> make_entity3(Tag tag, const double* params)
{
    const auto pt = [&](int i) { return PtN<3> {params[i], params[i + 1], params[i + 2]}; };
    switch (tag) {
    case TAG_PLANE:
        // Blob: [o(3), ax(3), ay(3)].
        return TrimmedEntity<PlaneParam> {.param = {.o = pt(0), .ax = pt(3), .ay = pt(6)},
                                          .trim = {}};
    case TAG_SPHERE: {
        // Blob: [c(3), r, ax(3), ay(3)].
        const VecN<3> ax = pt(4), ay = pt(7);
        return TrimmedEntity<SphereParam> {
          .param = {.c = pt(0), .ax = ax, .ay = ay, .az = cross3(ax, ay), .r = params[3]},
          .trim = {}};
    }
    case TAG_CYLINDER: {
        // Blob: [o(3), ax(3), ay(3), r].
        const VecN<3> ax = pt(3), ay = pt(6);
        return TrimmedEntity<CylinderParam> {
          .param = {.o = pt(0), .ax = ax, .ay = ay, .az = cross3(ax, ay), .r = params[9]},
          .trim = {}};
    }
    case TAG_LINE3:
        // Blob: [p0(3), p1(3), t0, t1].
        return TrimmedEntity<Line3Param> {
          .param = {.p0 = pt(0), .p1 = pt(3)},
          .trim = {.t0 = params[6], .t1 = params[7], .closed = false}};
    case TAG_FREE:
    default: return Free<3> {};
    }
}

/// @brief Dimension-templated dispatch over the per-dimension entity factories.
/// @tparam D Embedding dimension (2 or 3).
/// @param tag Entity type tag (`TAG_*`).
/// @param params Flat parameter blob.
/// @param arena Base of the per-group double arena for span-backed entities.
/// @return The typed entity variant.
template <int D = kDefaultDim>
inline EntityKindT<D> make_entity(Tag tag, const double* params, const double* arena = nullptr)
{
    static_assert(D == 2 || D == 3, "make_entity: unsupported dimension");
    if constexpr (D == 2) {
        return make_entity(tag, params, arena);
    } else {
        return make_entity3(tag, params);
    }
}

/// @brief Project @p p onto the entity @p (tag, params).
///
/// Dispatched through the concept entities via `std::visit` — monomorphic per
/// entity type, no virtuals.
/// @param p Query point.
/// @param tag Entity type tag.
/// @param params Flat parameter blob.
/// @return The projected point.
inline Pt project(const Pt& p, Tag tag, const double* params)
{
    return std::visit([&](const auto& e) { return e.project(p); }, make_entity(tag, params));
}

/// @brief The @f$ (d, 1) @f$ tangent column at @p p on the entity @p (tag, params).
///
/// The first column of the entity tangent basis (the single tangent for a curve;
/// @f$ e_0 @f$ for Free).
/// @param p Query point.
/// @param tag Entity type tag.
/// @param params Flat parameter blob.
/// @return The tangent column.
inline Pt tangent_space(const Pt& p, Tag tag, const double* params)
{
    return std::visit([&](const auto& e) { return e.tangent_basis(p)[0]; },
                      make_entity(tag, params));
}

}  // namespace egg
