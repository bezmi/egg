// Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
//
// PolyForm Noncommercial License 2.0.0-pre.2
// https://github.com/bezmi/egg/blob/main/LICENSE.md
// Free to use and redistribute for personal and noncommercial purposes.
// See the license for details.
// For commercial licensing, contact s.imran@tuta.io

#pragma once

#include "core.hpp"
#include "entity_soa.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
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
inline constexpr Tag TAG_LINERAIL = 15;     // 2D open-ended segment (releases beyond the ends)
inline constexpr Tag TAG_LINERAIL3 = 16;    // 3D open-ended segment (releases beyond the ends)

inline constexpr int kParamPad = 12;
/// Arena record size of one composite-path segment: [tag, params(kParamPad)].
inline constexpr int kCompositeRecSize = 1 + kParamPad;

// Pt is the D=2 coordinate type (PtN<2>). The 2D entity set is curves only
// (Free, line, circle, ellipse, arcs, Béziers); Sphere/Plane are genuine
// surfaces and live in the 3D entity set, not here.

// Concept-modelled geometry entities — the single source of truth for the
// kernels. Each entity is a trivially-copyable typed value with its per-shape
// math inlined (bit-identical to the old raw-pointer free functions). The
// concrete type is selected per launch by dispatch_entity_type<D>(tag, f) and
// built with EntitySoA<E>::load (device sweep) or decode_entity<E> (cold
// blob/oracle paths); kernels are fully monomorphic in E — no per-DOF
// std::visit, no value-level entity variant.

/// Parameter-space coordinate: (t) for a curve, (u,v) for a surface.
template <int K> using Param = std::array<real, K>;

/// The D identity columns e_0..e_{D-1} (the free-DOF tangent basis).
template <int D> inline std::array<VecN<D>, D> identity_columns()
{
    std::array<VecN<D>, D> b {};
    for (int i = 0; i < D; ++i) { b[i][i] = 1.0_r; }
    return b;
}

/// Gram–Schmidt orthonormalization of K columns embedded in D (K==1 reduces
/// to a single normalize).
template <int D, int K> inline std::array<VecN<D>, K> orthonormalize(std::array<VecN<D>, K> b)
{
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < i; ++j) { b[i] = b[i] - dot(b[i], b[j]) * b[j]; }
        b[i] = normalize(b[i]);
    }
    return b;
}

/// Fused projection result returned by `project_frame`.
template <int D, int K> struct Frame {
    PtN<D> pos;                    ///< The projected point.
    std::array<VecN<D>, K> basis;  ///< The orthonormal tangent columns.
    int eff_tdim;                  ///< Effective tdim, dropped by one when the
                                   ///< node lands on a trim boundary.
};

/// A parametrization: an intrinsic-idim manifold embedded in edim, given by
/// the three maps invert (point → params), eval (params → point), and frame
/// (params → the idim raw tangent columns).
template <class P>
concept Parametrization = requires(const P param, PtN<P::edim> p, Param<P::idim> q) {
    { P::edim } -> std::convertible_to<int>;  // embedding dimension D
    { P::idim } -> std::convertible_to<int>;  // intrinsic dimension k (== tdim)
    { param.invert(p) } -> std::same_as<Param<P::idim>>;
    { param.eval(q) } -> std::same_as<PtN<P::edim>>;
    { param.frame(q) } -> std::same_as<std::array<VecN<P::edim>, P::idim>>;
};

/// A boundary entity: what the constrained sweep calls. `project_frame` is
/// the fused (pos + orthonormal basis + effective tdim) form; `project` and
/// `tangent_basis` are projections of it.
template <class E>
concept GeometryEntity = requires(const E& e, const typename E::Pt& p) {
    { E::tdim } -> std::convertible_to<int>;
    { e.project(p) } -> std::same_as<typename E::Pt>;
    { e.tangent_basis(p) } -> std::same_as<std::array<typename E::Vec, E::tdim>>;
    { e.project_frame(p) };
};

// --- Trim: a k-box / k-region, one specialization per intrinsic dim -----------

/// Wrap @p x into [a, b) (closed curves); returned unchanged if b <= a.
inline real wrap(real x, real a, real b)
{
    const real L = b - a;
    if (L <= 0.0_r) { return x; }
    real t = sycl::fmod(x - a, L);
    if (t < 0.0_r) { t += L; }
    return a + t;
}

/// k-box / k-region trim in parameter space: Trim<1> a curve interval,
/// Trim<2> a UV region.
template <int K> struct Trim;

/// Curve trim: the parameter interval [t0, t1] (or a closed loop).
template <> struct Trim<1> {
    real t0, t1;          ///< Interval bounds.
    bool closed = false;  ///< If set, the curve is periodic and @c contains is always true.
    /// Inside test (always true when closed).
    [[nodiscard]] bool contains(Param<1> q) const { return closed || (q[0] >= t0 && q[0] <= t1); }
    /// Clamp onto the interval (wrap when closed).
    [[nodiscard]] Param<1> clamp(Param<1> q) const
    { return {closed ? wrap(q[0], t0, t1) : std::clamp(q[0], t0, t1)}; }
};

/// Surface trim: a UV polygon (outer loop minus holes), arena-backed. Empty
/// spans mean untrimmed (the surface's full natural range).
template <> struct Trim<2> {
    std::span<const real> verts;  ///< Loop vertices flattened [u0,v0,u1,v1,...].
    std::span<const real> loops;  ///< Vertex-count offsets [0, n0, n0+n1, ..., total].
    /// Loop vertex i as a UV point.
    [[nodiscard]] PtN<2> vert(int i) const { return {verts[2 * i], verts[(2 * i) + 1]}; }
    /// Even–odd inside test over all loops; untrimmed always contains.
    [[nodiscard]] bool contains(Param<2> uv) const
    {
        if (loops.size() < 2) { return true; }
        bool inside = false;
        for (std::size_t l = 0; l + 1 < loops.size(); ++l) {
            const int lo = static_cast<int>(loops[l]), hi = static_cast<int>(loops[l + 1]);
            for (int i = lo, j = hi - 1; i < hi; j = i++) {
                const PtN<2> a = vert(i), b = vert(j);
                // Even–odd ray cast in +u from uv.
                if ((a[1] > uv[1]) != (b[1] > uv[1]) &&
                    uv[0] < ((b[0] - a[0]) * (uv[1] - a[1]) / (b[1] - a[1])) + a[0]) {
                    inside = !inside;
                }
            }
        }
        return inside;
    }
    /// Nearest point on the loop polylines (closed loops).
    [[nodiscard]] Param<2> clamp(Param<2> uv) const
    {
        Param<2> best = uv;
        real best_d = std::numeric_limits<real>::infinity();
        for (std::size_t l = 0; l + 1 < loops.size(); ++l) {
            const int lo = static_cast<int>(loops[l]), hi = static_cast<int>(loops[l + 1]);
            for (int i = lo, j = hi - 1; i < hi; j = i++) {
                const PtN<2> a = vert(j), b = vert(i);
                const VecN<2> ab = b - a;
                const real ab_sq = dot(ab, ab);
                real t = ab_sq > tol::tiny ? dot(PtN<2> {uv[0], uv[1]} - a, ab) / ab_sq : 0.0_r;
                t = std::clamp(t, 0.0_r, 1.0_r);
                const PtN<2> q = a + t * ab;
                const VecN<2> dq = q - PtN<2> {uv[0], uv[1]};
                const real dd = dot(dq, dq);
                if (dd < best_d) {
                    best_d = dd;
                    best = {q[0], q[1]};
                }
            }
        }
        return best;
    }
};

/// The generic trimmed entity: any @ref Parametrization restricted to its
/// k-region trim.
template <Parametrization P> struct TrimmedEntity {
    using Pt = PtN<P::edim>;              ///< Point type in the embedding space.
    using Vec = VecN<P::edim>;            ///< Vector type in the embedding space.
    static constexpr int tdim = P::idim;  ///< Tangent dimension (== intrinsic dim).
    P param;                              ///< The underlying parametrization.
    Trim<P::idim> trim;                   ///< The parameter-space trim region.

    /// Fused frame at the projection of @p p; eff_tdim drops by one when the
    /// foot lands on the trim boundary.
    [[nodiscard]] Frame<P::edim, tdim> project_frame(const Pt& p) const
    {
        auto q = param.invert(p);
        // Periodic parametrizations return one branch representative; pick
        // the representative nearest the trim interval, so a foot at the
        // period seam clamps to the near interval end, not across the whole
        // range (e.g. an arc over [3/2 pi, 2 pi] queried at angle ~0).
        if constexpr (requires { P::period; }) {
            if (!trim.closed) {
                const real mid = 0.5_r * (trim.t0 + trim.t1);
                q[0] = mid + sycl::remainder(q[0] - mid, P::period);
            }
        }
        const bool inside = trim.contains(q);
        if (!inside) { q = trim.clamp(q); }
        return {.pos = param.eval(q),
                .basis = orthonormalize<P::edim, tdim>(param.frame(q)),
                .eff_tdim = inside ? tdim : tdim - 1};
    }
    /// Projected point.
    [[nodiscard]] Pt project(const Pt& p) const { return project_frame(p).pos; }

    /// Warm-started projection: seed the Newton inverse from @p seed_io (the
    /// previous foot) and write the converged parameter back. Present only
    /// when the parametrization provides invert_seeded (iterative B-splines);
    /// closed-form params fall through to the cold @ref project.
    template <bool Warm = false>
    [[nodiscard]] Pt project_seeded(const Pt& p, Param<tdim>& seed_io, bool has_seed) const
        requires requires(const P& pp, const Pt& pt, Param<tdim>& s, bool b) {
            pp.invert_seeded(pt, s, b);
        }
    {
        Param<tdim> q = param.template invert_seeded<Warm>(p, seed_io, has_seed);
        seed_io = q;  // store the unclamped Newton foot for the next warm start
        const bool inside = trim.contains(q);
        if (!inside) { q = trim.clamp(q); }
        return param.eval(q);
    }
    /// Orthonormal tangent basis at the projection of @p p.
    [[nodiscard]] std::array<Vec, tdim> tangent_basis(const Pt& p) const
    { return project_frame(p).basis; }

    /// Warm-started tangent basis (cf. @ref project_seeded). Present only for
    /// iterative parametrizations; closed-form params fall back to the cold
    /// @ref tangent_basis.
    template <bool Warm = false>
    [[nodiscard]] std::array<Vec, tdim>
      tangent_basis_seeded(const Pt& p, Param<tdim>& seed_io, bool has_seed) const
        requires requires(const P& pp, const Pt& pt, Param<tdim>& s, bool b) {
            pp.invert_seeded(pt, s, b);
        }
    {
        Param<tdim> q = param.template invert_seeded<Warm>(p, seed_io, has_seed);
        seed_io = q;  // store the unclamped Newton foot for the next warm start
        if (!trim.contains(q)) { q = trim.clamp(q); }
        return orthonormalize<P::edim, tdim>(param.frame(q));
    }
};

/// Interior (free) DOF: k==D, identity projection.
template <int D> struct Free {
    using Pt = PtN<D>;              ///< Point type.
    using Vec = VecN<D>;            ///< Vector type.
    static constexpr int tdim = D;  ///< Tangent dimension equals the embedding dimension.
    /// Identity projection (a free node moves anywhere).
    [[nodiscard]] Pt project(const Pt& p) const { return p; }
    /// Full identity tangent basis.
    [[nodiscard]] std::array<Vec, D> tangent_basis(const Pt&) const
    { return identity_columns<D>(); }
    /// Fused frame: the point itself, identity basis, full tdim.
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

/// Line segment (sx, sy) -> (ex, ey), clamped to [0, 1].
struct LineSeg {
    using Pt = PtN<2>;
    using Vec = VecN<2>;
    static constexpr int tdim = 1;
    real sx, sy, ex, ey;
    [[nodiscard]] Pt project(const Pt& p) const
    {
        const real abx = ex - sx, aby = ey - sy;
        const real ab_sq = (abx * abx) + (aby * aby);
        real t = (((p[0] - sx) * abx) + ((p[1] - sy) * aby)) / std::fmax(ab_sq, tol::tiny);
        t = t < 0.0_r ? 0.0_r : t;
        t = t > 1.0_r ? 1.0_r : t;  // clip to [0, 1]
        return Pt {sx + (t * abx), sy + (t * aby)};
    }
    [[nodiscard]] Vec tangent(const Pt&) const
    {
        const real abx = ex - sx, aby = ey - sy;
        const real norm = sycl::sqrt((abx * abx) + (aby * aby));
        if (norm < tol::znorm) {
            return Vec {1.0_r, 0.0_r};  // eye[:, 0]
        }
        return Vec {abx / norm, aby / norm};
    }
    [[nodiscard]] std::array<Vec, 1> tangent_basis(const Pt& p) const { return {tangent(p)}; }
    [[nodiscard]] Frame<2, 1> project_frame(const Pt& p) const
    { return {.pos = project(p), .basis = {tangent(p)}, .eff_tdim = 1}; }
};

/// Open-ended line segment (a "rail"): the projection foot is clamped nowhere —
/// while the foot parameter lies in [0, 1] the node slides on the segment, and
/// beyond either end `project` is the identity, so the node RELEASES and moves
/// freely (newton_delta's on_rail hook switches it to the full-space step).
/// A released node whose foot re-enters the range is recaptured. Declared via
/// the Python entity's `.open_ended()`.
struct LineRail {
    using Pt = PtN<2>;
    using Vec = VecN<2>;
    static constexpr int tdim = 1;
    real sx, sy, ex, ey;
    /// Unclamped foot parameter of @p p on the segment's infinite line.
    [[nodiscard]] real foot_t(const Pt& p) const
    {
        const real abx = ex - sx, aby = ey - sy;
        const real ab_sq = (abx * abx) + (aby * aby);
        return (((p[0] - sx) * abx) + ((p[1] - sy) * aby)) / std::fmax(ab_sq, tol::tiny);
    }
    /// Is the projection foot within the segment (the constraint is active)?
    [[nodiscard]] bool on_rail(const Pt& p) const
    {
        const real t = foot_t(p);
        return t >= 0.0_r && t <= 1.0_r;
    }
    [[nodiscard]] Pt project(const Pt& p) const
    {
        const real t = foot_t(p);
        if (t < 0.0_r || t > 1.0_r) { return p; }  // beyond an end: released
        return Pt {sx + (t * (ex - sx)), sy + (t * (ey - sy))};
    }
    [[nodiscard]] Vec tangent(const Pt&) const
    {
        const real abx = ex - sx, aby = ey - sy;
        const real norm = sycl::sqrt((abx * abx) + (aby * aby));
        if (norm < tol::znorm) {
            return Vec {1.0_r, 0.0_r};  // eye[:, 0]
        }
        return Vec {abx / norm, aby / norm};
    }
    [[nodiscard]] std::array<Vec, 1> tangent_basis(const Pt& p) const { return {tangent(p)}; }
    [[nodiscard]] Frame<2, 1> project_frame(const Pt& p) const
    { return {.pos = project(p), .basis = {tangent(p)}, .eff_tdim = 1}; }
};

/// 3D open-ended line segment — @ref LineRail lifted to D=3 (bespoke
/// closed-form; deliberately NOT the TrimmedEntity pipeline, whose trim
/// clamps rather than releases).
struct LineRail3 {
    using Pt = PtN<3>;
    using Vec = VecN<3>;
    static constexpr int tdim = 1;
    real sx, sy, sz, ex, ey, ez;
    [[nodiscard]] real foot_t(const Pt& p) const
    {
        const real ax = ex - sx, ay = ey - sy, az = ez - sz;
        const real ab_sq = (ax * ax) + (ay * ay) + (az * az);
        return (((p[0] - sx) * ax) + ((p[1] - sy) * ay) + ((p[2] - sz) * az)) /
               std::fmax(ab_sq, tol::tiny);
    }
    [[nodiscard]] bool on_rail(const Pt& p) const
    {
        const real t = foot_t(p);
        return t >= 0.0_r && t <= 1.0_r;
    }
    [[nodiscard]] Pt project(const Pt& p) const
    {
        const real t = foot_t(p);
        if (t < 0.0_r || t > 1.0_r) { return p; }
        return Pt {sx + (t * (ex - sx)), sy + (t * (ey - sy)), sz + (t * (ez - sz))};
    }
    [[nodiscard]] Vec tangent(const Pt&) const
    {
        const real ax = ex - sx, ay = ey - sy, az = ez - sz;
        const real norm = sycl::sqrt((ax * ax) + (ay * ay) + (az * az));
        if (norm < tol::znorm) { return Vec {1.0_r, 0.0_r, 0.0_r}; }
        return Vec {ax / norm, ay / norm, az / norm};
    }
    [[nodiscard]] std::array<Vec, 1> tangent_basis(const Pt& p) const { return {tangent(p)}; }
    [[nodiscard]] Frame<3, 1> project_frame(const Pt& p) const
    { return {.pos = project(p), .basis = {tangent(p)}, .eff_tdim = 1}; }
};
/// Circle of radius r centred at (cx, cy); radial projection.
struct Circle {
    using Pt = PtN<2>;
    using Vec = VecN<2>;
    static constexpr int tdim = 1;
    real cx, cy, r;
    [[nodiscard]] Pt project(const Pt& p) const
    {
        const real dx = p[0] - cx, dy = p[1] - cy;
        const real dist = sycl::sqrt((dx * dx) + (dy * dy));
        if (dist < tol::znorm) {
            return Pt {cx + r, cy};  // arbitrary on-circle point
        }
        return Pt {cx + (r * dx / dist), cy + (r * dy / dist)};
    }
    [[nodiscard]] Vec tangent(const Pt& p) const
    {
        const real dx = p[0] - cx, dy = p[1] - cy;
        const real rn = sycl::sqrt((dx * dx) + (dy * dy));
        real nx, ny;
        if (rn < tol::znorm) {
            nx = 1.0_r;
            ny = 0.0_r;  // eye[:, 0]
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
/// Axis-aligned ellipse centred at (cx, cy) with radii (rx, ry);
/// radial-scaling projection (exact for circles, approximate otherwise).
struct Ellipse {
    using Pt = PtN<2>;
    using Vec = VecN<2>;
    static constexpr int tdim = 1;
    real cx, cy, rx, ry;
    [[nodiscard]] Pt project(const Pt& p) const
    {
        const real dx = p[0] - cx, dy = p[1] - cy;
        const real sx = dx / rx, sy = dy / ry;
        const real dist = sycl::sqrt((sx * sx) + (sy * sy));
        real ux, uy;
        if (dist < tol::znorm) {
            ux = uy = 1.0_r / std::numbers::sqrt2;  // ones(d)/sqrt(d), d = 2
        } else {
            ux = sx / dist;
            uy = sy / dist;
        }
        return Pt {cx + (ux * rx), cy + (uy * ry)};
    }
    [[nodiscard]] Vec tangent(const Pt& p) const
    {
        const real dx = p[0] - cx, dy = p[1] - cy;
        // Parametric angle from the radial-scaled coordinates (matches Ellipse).
        const real angle = sycl::atan2(dy / ry, dx / rx);
        real tx = -rx * sycl::sin(angle);
        real ty = ry * sycl::cos(angle);
        const real norm = sycl::sqrt((tx * tx) + (ty * ty));
        if (norm < tol::znorm) { return Vec {1.0_r, 0.0_r}; }
        return Vec {tx / norm, ty / norm};
    }
    [[nodiscard]] std::array<Vec, 1> tangent_basis(const Pt& p) const { return {tangent(p)}; }
    [[nodiscard]] Frame<2, 1> project_frame(const Pt& p) const
    { return {.pos = project(p), .basis = {tangent(p)}, .eff_tdim = 1}; }
};

/// Representative closed-form curve @ref Parametrization validating the
/// generic stack.
///
/// `LineParam` drives the generic @ref TrimmedEntity pipeline; its `invert`/`eval`
/// reproduce @ref LineSeg exactly (unclamped @f$ t @f$ from `invert`, @f$ [0,1] @f$
/// clamp from `Trim<1>`). Each additional curve type is just another @ref Parametrization
/// struct (three maps) plus a variant arm; the trimming, orthonormalization, and
/// `project_frame` plumbing is shared and written once here.
struct LineParam {
    static constexpr int edim = 2, idim = 1;
    PtN<2> p0, p1;  ///< Endpoints @f$ P_0 @f$ and @f$ P_1 @f$.
    /// Foot-of-projection parameter (unclamped; `Trim<1>` does the clamp).
    /// @return @f$ t = (q - P_0)\cdot(P_1 - P_0) / \lVert P_1 - P_0 \rVert^2 @f$.
    [[nodiscard]] Param<1> invert(const PtN<2>& q) const
    {
        const VecN<2> ab = p1 - p0;
        const real ab_sq = dot(ab, ab);
        return {dot(q - p0, ab) / std::fmax(ab_sq, tol::tiny)};
    }
    /// Evaluate the curve point @f$ P_0 + t(P_1 - P_0) @f$.
    [[nodiscard]] PtN<2> eval(const Param<1>& t) const { return p0 + t[0] * (p1 - p0); }
    /// The (constant) raw tangent column @f$ P_1 - P_0 @f$.
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

/// Seeded-Newton nearest-foot parameter for a curve.
///
/// Coarse-samples the curve over @f$ [t_{lo}, t_{hi}] @f$ for a robust seed, then
/// runs Newton on @f$ f(t) = (C(t) - q)\cdot C'(t) @f$, with
/// @f$ f'(t) = \lVert C'(t)\rVert^2 + (C(t) - q)\cdot C''(t) @f$. The unclamped
/// parameter is returned; the entity's `Trim<1>` clamps it onto the live range.
/// @tparam C Curve type exposing `eval`, `deriv`, and `deriv2` on a `Param<1>`.
/// @param c Curve.
/// @param q Query point.
/// @param t_lo,t_hi Natural parameter domain to seed over.
/// @param seed Warm-start parameter (e.g. the previous sweep's foot).
/// @param has_seed Skip the coarse-sample search and start Newton from @p seed.
/// @param n_seed Number of coarse samples for the seed.
/// @param iters Newton iterations.
/// @return The nearest-foot parameter @f$ t @f$.
template <class C>
inline real project_param_seeded(const C& c,
                                 const PtN<2>& q,
                                 real t_lo,
                                 real t_hi,
                                 real seed,
                                 bool has_seed,
                                 int n_seed = 16,
                                 int iters = 8)
{
    real t;
    if (has_seed) {
        t = std::clamp(seed, t_lo, t_hi);
    } else {
        real best_t = t_lo;
        real best_d = std::numeric_limits<real>::infinity();
        for (int i = 0; i <= n_seed; ++i) {
            const real tc = t_lo + (((t_hi - t_lo) * i) / n_seed);
            const VecN<2> d = c.eval({tc}) - q;
            const real dd = dot(d, d);
            if (dd < best_d) {
                best_d = dd;
                best_t = tc;
            }
        }
        t = best_t;
    }
    for (int it = 0; it < iters; ++it) {
        const VecN<2> d = c.eval({t}) - q;
        const VecN<2> d1 = c.deriv({t});
        const real f = dot(d, d1);
        const real fp = dot(d1, d1) + dot(d, c.deriv2({t}));
        if (sycl::fabs(fp) < tol::tiny) { break; }
        t -= f / fp;
    }
    return t;
}

template <class C>
inline real
  project_param(const C& c, const PtN<2>& q, real t_lo, real t_hi, int n_seed = 16, int iters = 8)
{ return project_param_seeded(c, q, t_lo, t_hi, 0.0_r, false, n_seed, iters); }

/// A circular arc of radius @p r centred at @f$ (c_x, c_y) @f$, parametrized
/// by angle; closed-form inverse.
struct CircleArcParam {
    static constexpr int edim = 2, idim = 1;
    /// Angular period: the inverse returns one branch representative; the
    /// trimmed entity re-represents it nearest the trim interval before
    /// clamping (see TrimmedEntity::project_frame).
    static constexpr real period = static_cast<real>(2.0 * std::numbers::pi);
    PtN<2> c;  ///< Centre.
    real r;    ///< Radius.
    /// Inverse: the polar angle of @p q about the centre.
    [[nodiscard]] Param<1> invert(const PtN<2>& q) const
    { return {sycl::atan2(q[1] - c[1], q[0] - c[0])}; }
    /// Evaluate @f$ C + r(\cos t, \sin t) @f$.
    [[nodiscard]] PtN<2> eval(const Param<1>& t) const
    { return {c[0] + (r * sycl::cos(t[0])), c[1] + (r * sycl::sin(t[0]))}; }
    /// The raw tangent column @f$ C'(t) = r(-\sin t, \cos t) @f$.
    [[nodiscard]] std::array<VecN<2>, 1> frame(const Param<1>& t) const
    { return {VecN<2> {-r * sycl::sin(t[0]), r * sycl::cos(t[0])}}; }
};

/// A rotated, axis-scaled elliptical arc; nearest foot via Newton.
struct EllipseArcParam {
    static constexpr int edim = 2, idim = 1;
    /// Angular period (cf. CircleArcParam::period).
    static constexpr real period = static_cast<real>(2.0 * std::numbers::pi);
    PtN<2> c;   ///< Centre.
    real a, b;  ///< Semi-axis lengths along the rotated x/y axes.
    real phi;   ///< Rotation of the major axis from +x.
    /// Evaluate @f$ C + R_\phi (a\cos t, b\sin t) @f$.
    [[nodiscard]] PtN<2> eval(const Param<1>& t) const
    {
        const real cp = sycl::cos(phi), sp = sycl::sin(phi);
        const real x = a * sycl::cos(t[0]), y = b * sycl::sin(t[0]);
        return {c[0] + (cp * x) - (sp * y), c[1] + (sp * x) + (cp * y)};
    }
    /// First derivative @f$ C'(t) = R_\phi (-a\sin t, b\cos t) @f$.
    [[nodiscard]] VecN<2> deriv(const Param<1>& t) const
    {
        const real cp = sycl::cos(phi), sp = sycl::sin(phi);
        const real x = -a * sycl::sin(t[0]), y = b * sycl::cos(t[0]);
        return {(cp * x) - (sp * y), (sp * x) + (cp * y)};
    }
    /// Second derivative @f$ C''(t) = R_\phi (-a\cos t, -b\sin t) @f$.
    [[nodiscard]] VecN<2> deriv2(const Param<1>& t) const
    {
        const real cp = sycl::cos(phi), sp = sycl::sin(phi);
        const real x = -a * sycl::cos(t[0]), y = -b * sycl::sin(t[0]);
        return {(cp * x) - (sp * y), (sp * x) + (cp * y)};
    }
    /// Inverse: seeded-Newton nearest foot over the full angular range.
    [[nodiscard]] Param<1> invert(const PtN<2>& q) const
    { return {project_param(*this, q, 0.0_r, 2.0_r * std::numbers::pi)}; }
    /// The raw tangent column @f$ C'(t) @f$.
    [[nodiscard]] std::array<VecN<2>, 1> frame(const Param<1>& t) const { return {deriv(t)}; }
};

/// A quadratic Bézier curve over three control points; nearest foot via Newton.
struct QuadBezierParam {
    static constexpr int edim = 2, idim = 1;
    std::array<PtN<2>, 3> p;  ///< Control points @f$ P_0, P_1, P_2 @f$.
    /// Evaluate @f$ (1-t)^2 P_0 + 2(1-t)t P_1 + t^2 P_2 @f$.
    [[nodiscard]] PtN<2> eval(const Param<1>& t) const
    {
        const real u = 1.0_r - t[0];
        return (u * u) * p[0] + (2.0_r * u * t[0]) * p[1] + (t[0] * t[0]) * p[2];
    }
    /// First derivative @f$ 2(1-t)(P_1 - P_0) + 2t(P_2 - P_1) @f$.
    [[nodiscard]] VecN<2> deriv(const Param<1>& t) const
    { return (2.0_r * (1.0_r - t[0])) * (p[1] - p[0]) + (2.0_r * t[0]) * (p[2] - p[1]); }
    /// Second derivative @f$ 2(P_2 - 2P_1 + P_0) @f$ (constant).
    [[nodiscard]] VecN<2> deriv2(const Param<1>&) const
    { return 2.0_r * (p[2] - (2.0_r * p[1]) + p[0]); }
    /// Inverse: seeded-Newton nearest foot over @f$ [0,1] @f$.
    [[nodiscard]] Param<1> invert(const PtN<2>& q) const
    { return {project_param(*this, q, 0.0_r, 1.0_r)}; }
    /// The raw tangent column @f$ C'(t) @f$.
    [[nodiscard]] std::array<VecN<2>, 1> frame(const Param<1>& t) const { return {deriv(t)}; }
};

/// A cubic Bézier curve over four control points; nearest foot via Newton.
struct CubicBezierParam {
    static constexpr int edim = 2, idim = 1;
    std::array<PtN<2>, 4> p;  ///< Control points @f$ P_0 \dots P_3 @f$.
    /// Evaluate the Bernstein form @f$ \sum_i B_i^3(t) P_i @f$.
    [[nodiscard]] PtN<2> eval(const Param<1>& t) const
    {
        const real u = 1.0_r - t[0], tt = t[0];
        return (u * u * u) * p[0] + (3.0_r * u * u * tt) * p[1] + (3.0_r * u * tt * tt) * p[2] +
               (tt * tt * tt) * p[3];
    }
    /// First derivative @f$ 3\sum_i B_i^2(t)(P_{i+1} - P_i) @f$.
    [[nodiscard]] VecN<2> deriv(const Param<1>& t) const
    {
        const real u = 1.0_r - t[0], tt = t[0];
        return (3.0_r * u * u) * (p[1] - p[0]) + (6.0_r * u * tt) * (p[2] - p[1]) +
               (3.0_r * tt * tt) * (p[3] - p[2]);
    }
    /// Second derivative @f$ 6\big((1-t)(P_2 - 2P_1 + P_0) + t(P_3 - 2P_2 + P_1)\big) @f$.
    [[nodiscard]] VecN<2> deriv2(const Param<1>& t) const
    {
        const real u = 1.0_r - t[0], tt = t[0];
        return (6.0_r * u) * (p[2] - (2.0_r * p[1]) + p[0]) +
               (6.0_r * tt) * (p[3] - (2.0_r * p[2]) + p[1]);
    }
    /// Inverse: seeded-Newton nearest foot over @f$ [0,1] @f$.
    [[nodiscard]] Param<1> invert(const PtN<2>& q) const
    { return {project_param(*this, q, 0.0_r, 1.0_r)}; }
    /// The raw tangent column @f$ C'(t) @f$.
    [[nodiscard]] std::array<VecN<2>, 1> frame(const Param<1>& t) const { return {deriv(t)}; }
};

#ifndef EGG_MAX_BSPLINE_DEGREE
    /// Compile-time override for @ref egg::kMaxBSplineDegree.
    #define EGG_MAX_BSPLINE_DEGREE 3
#endif
/// Largest B-spline *surface* degree the fixed-size de Boor work arrays support.
///
/// Sizes the surface de Boor work arrays (`ndu`, `du`, `dv`, the `A*`/`w*` sums).
/// Defaulted to 3 (bicubic — the de-facto CAD NURBS surface standard, covering
/// virtually all 3D STEP) to keep the boundary projection's scratch footprint
/// small: at degree 7 the boundary sweep kernel spilled ~3.3 KB; degree 3 cuts
/// that ~3×. This cap drives the 3D boundary kernel's register/VGPR budget, so it
/// is kept tight on purpose. Override with `-DEGG_MAX_BSPLINE_DEGREE=N` for
/// higher-degree surface models. A surface whose degree exceeds it is rejected at
/// context build (see `bspline_degree_guard` in the `load_into` encoders).
inline constexpr int kMaxBSplineDegree = EGG_MAX_BSPLINE_DEGREE;
/// Leading dimension of the *surface* de Boor work arrays.
inline constexpr int kBSplineCap = kMaxBSplineDegree + 1;

#ifndef EGG_MAX_BSPLINE_CURVE_DEGREE
    /// Compile-time override for @ref egg::kMaxBSplineCurveDegree.
    #define EGG_MAX_BSPLINE_CURVE_DEGREE 7
#endif
/// Largest B-spline *curve* degree the de Boor work arrays support.
///
/// Decoupled from @ref kMaxBSplineDegree — 2D profile/trim curves (and curve
/// sub-segments nested in composites) routinely exceed degree 3 — e.g. a degree-4
/// Bézier arch — whereas 3D surfaces are bicubic in practice. The curve de Boor
/// is not in the VGPR-pinned 3D boundary surface kernel, so its work arrays can
/// carry generous headroom (default 7) without touching the surface scratch the
/// boundary kernel's register budget depends on. Override with
/// `-DEGG_MAX_BSPLINE_CURVE_DEGREE=N`.
inline constexpr int kMaxBSplineCurveDegree = EGG_MAX_BSPLINE_CURVE_DEGREE;
/// Leading dimension of the *curve* de Boor work arrays.
inline constexpr int kBSplineCurveCap = kMaxBSplineCurveDegree + 1;

/// Host-side guard: reject a B-spline whose degree exceeds the compiled
/// @p cap before it can index the fixed de Boor work arrays out of bounds
/// on the device. Throws with an actionable message. Host-only — called
/// from the `load_into` encoders (and the composite encoder), never from
/// device code. Pass @ref kMaxBSplineDegree for surfaces and
/// @ref kMaxBSplineCurveDegree for curves.
inline void bspline_degree_guard(int degree, const char* what, int cap)
{
    if (degree > cap) {
        throw std::runtime_error(
          std::string("B-spline ") + what + " degree " + std::to_string(degree) +
          " exceeds the compiled max degree " + std::to_string(cap) +
          "; rebuild with a larger -DEGG_MAX_BSPLINE_DEGREE / -DEGG_MAX_BSPLINE_CURVE_DEGREE.");
    }
}

/// Index of the knot span containing @p u (clamped to the live domain).
/// @param degree Basis degree @f$ p @f$.
/// @param n_ctrl Number of control points along this direction.
/// @param knots Knot vector, length @c n_ctrl+degree+1, non-decreasing.
/// @param u Parameter.
/// @return The span index @f$ i @f$ with @f$ knots[i] \le u < knots[i+1] @f$.
inline int bspline_find_span(int degree, int n_ctrl, std::span<const real> knots, real u)
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

/// Nonzero basis functions and their derivatives at @p u (NURBS book A2.3).
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
template <int CAP>
inline void bspline_basis_ders(
  int degree, std::span<const real> knots, int span, real u, int nd, real ders[][CAP])
{
    const int p = degree;
    real ndu[CAP][CAP];
    real a[2][CAP];
    real left[CAP], right[CAP];
    ndu[0][0] = 1.0_r;
    for (int j = 1; j <= p; ++j) {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        real saved = 0.0_r;
        for (int r = 0; r < j; ++r) {
            ndu[j][r] = right[r + 1] + left[j - r];
            const real temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + (right[r + 1] * temp);
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    for (int j = 0; j <= p; ++j) { ders[0][j] = ndu[j][p]; }
    for (int r = 0; r <= p; ++r) {
        int s1 = 0, s2 = 1;
        a[0][0] = 1.0_r;
        for (int k = 1; k <= nd; ++k) {
            real d = 0.0_r;
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

/// Compile-time-`nd` de Boor basis-and-derivatives (F2 Lever A).
///
/// Identical to the runtime @ref bspline_basis_ders for `nd == ND`, but the
/// derivative-order bound `ND` is a template parameter so `if constexpr (ND>=1)`
/// statically elides the entire derivative recurrence (and its `a[2][cap]`
/// scratch and the factorial scaling) when `ND==0`. On the SSCP/SYCL path —
/// where `noinline` and unroll pragmas are ignored, so a runtime `nd` keeps the
/// nd≥1 rows live across the whole inlined call — making `ND` static is the only
/// way to keep those temporaries (and registers) out of the nd=0/nd=1 paths.
/// `ders` need only have `ND+1` rows.
template <int ND, int CAP>
inline void
  bspline_basis_ders(int degree, std::span<const real> knots, int span, real u, real ders[][CAP])
{
    const int p = degree;
    real ndu[CAP][CAP];
    real left[CAP], right[CAP];
    ndu[0][0] = 1.0_r;
    for (int j = 1; j <= p; ++j) {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        real saved = 0.0_r;
        for (int r = 0; r < j; ++r) {
            ndu[j][r] = right[r + 1] + left[j - r];
            const real temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + (right[r + 1] * temp);
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    for (int j = 0; j <= p; ++j) { ders[0][j] = ndu[j][p]; }
    if constexpr (ND >= 1) {
        real a[2][CAP];
        for (int r = 0; r <= p; ++r) {
            int s1 = 0, s2 = 1;
            a[0][0] = 1.0_r;
            for (int k = 1; k <= ND; ++k) {
                real d = 0.0_r;
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
        for (int k = 1; k <= ND; ++k) {
            for (int j = 0; j <= p; ++j) { ders[k][j] *= fac; }
            fac *= (p - k);
        }
    }
}

/// A B-spline / NURBS curve over a knot vector and a flat control net.
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
    int degree;                     ///< Basis degree @f$ p @f$.
    int n_ctrl;                     ///< Number of control points @f$ n+1 @f$.
    std::span<const real> knots;    ///< Knot vector, length @c n_ctrl+degree+1.
    std::span<const real> ctrl;     ///< Control points, length @c 2*n_ctrl (x,y interleaved).
    std::span<const real> weights;  ///< NURBS weights, length @c n_ctrl, or empty (polynomial).

    /// Control point @p i as a 2D point.
    [[nodiscard]] PtN<2> cp(int i) const { return {ctrl[2 * i], ctrl[(2 * i) + 1]}; }

    /// The @p order-th derivative point at @p u (order 0 = the curve point).
    [[nodiscard]] PtN<2> point_at(real u, int order) const
    {
        const int span = bspline_find_span(degree, n_ctrl, knots, u);
        real ders[3][kBSplineCurveCap];
        bspline_basis_ders<kBSplineCurveCap>(degree, knots, span, u, order, ders);
        if (weights.empty()) {
            PtN<2> acc {0.0_r, 0.0_r};
            for (int j = 0; j <= degree; ++j) {
                acc = acc + (ders[order][j] * cp(span - degree + j));
            }
            return acc;
        }
        // Rational: homogeneous derivatives A^(k) = Σ N^(k) w P, w^(k) = Σ N^(k) w,
        // then the quotient rule (orders 0..2):
        //   C = A/w; C' = (A' − w'C)/w; C'' = (A'' − 2w'C' − w''C)/w.
        PtN<2> A[3] {};
        real w[3] {};
        for (int k = 0; k <= order; ++k) {
            for (int j = 0; j <= degree; ++j) {
                const int i = span - degree + j;
                const real nw = ders[k][j] * weights[i];
                A[k] = A[k] + (nw * cp(i));
                w[k] += nw;
            }
        }
        PtN<2> C[3];
        C[0] = (1.0_r / w[0]) * A[0];
        if (order >= 1) { C[1] = (1.0_r / w[0]) * (A[1] - w[1] * C[0]); }
        if (order >= 2) { C[2] = (1.0_r / w[0]) * (A[2] - 2.0_r * w[1] * C[1] - w[2] * C[0]); }
        return C[order];
    }

    /// Evaluate the curve point @f$ C(u) @f$.
    [[nodiscard]] PtN<2> eval(const Param<1>& u) const { return point_at(u[0], 0); }
    /// First derivative @f$ C'(u) @f$.
    [[nodiscard]] VecN<2> deriv(const Param<1>& u) const { return point_at(u[0], 1); }
    /// Second derivative @f$ C''(u) @f$.
    [[nodiscard]] VecN<2> deriv2(const Param<1>& u) const { return point_at(u[0], 2); }
    /// Inverse: seeded-Newton nearest foot over the live domain
    /// @f$ [knots[degree], knots[n\_ctrl]] @f$.
    [[nodiscard]] Param<1> invert(const PtN<2>& q) const
    { return {project_param(*this, q, knots[degree], knots[n_ctrl])}; }
    /// Warm-started inverse: skip the coarse seed when @p has_seed. The
    /// @p Warm flag (the 3D surface cold/warm two-kernel split) is accepted
    /// for a uniform `project_seeded`/`tangent_basis_seeded` interface; the
    /// 2D curve solve is already light, so it takes the same path either way.
    template <bool Warm = false>
    [[nodiscard]] Param<1> invert_seeded(const PtN<2>& q, Param<1> seed, bool has_seed) const
    { return {project_param_seeded(*this, q, knots[degree], knots[n_ctrl], seed[0], has_seed)}; }
    /// The raw tangent column @f$ C'(u) @f$.
    [[nodiscard]] std::array<VecN<2>, 1> frame(const Param<1>& u) const { return {deriv(u)}; }
};

/// A composite path: an ordered sequence of curve segments joined
/// end-to-end (the 2D analogue of an OCCT wire).
///
/// The segments live as fixed-size records in the per-group arena
/// (@ref kCompositeRecSize reals each: `[tag, params...]`), so the type stays
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
    std::span<const real> recs;     ///< Segment records, @c n_segs*kCompositeRecSize reals.
    const real* arena;              ///< Arena base for span-backed segments (B-splines).

    /// Fused frame of the nearest segment to @p p.
    /// @return The nearest segment's projected point and unit tangent; @c eff_tdim is 1.
    [[nodiscard]] Frame<2, 1> project_frame(const Pt& p) const;  // defined after make_entity
    /// Project @p p onto the nearest segment.
    [[nodiscard]] Pt project(const Pt& p) const { return project_frame(p).pos; }
    /// The matched segment's unit tangent column.
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

/// Cross product of two 3-vectors.
inline VecN<3> cross3(const VecN<3>& a, const VecN<3>& b)
{
    return {(a[1] * b[2]) - (a[2] * b[1]),
            (a[2] * b[0]) - (a[0] * b[2]),
            (a[0] * b[1]) - (a[1] * b[0])};
}

/// A plane through @c o spanned by the orthonormal axes @c ax, @c ay.
struct PlaneParam {
    static constexpr int edim = 3, idim = 2;
    PtN<3> o;        ///< Origin.
    VecN<3> ax, ay;  ///< Orthonormal in-plane axes.
    /// Inverse: the in-plane coordinates of @p p.
    [[nodiscard]] Param<2> invert(const PtN<3>& p) const
    {
        const VecN<3> d = p - o;
        return {dot(d, ax), dot(d, ay)};
    }
    /// Evaluate @f$ O + u\,a_x + v\,a_y @f$.
    [[nodiscard]] PtN<3> eval(const Param<2>& q) const { return o + q[0] * ax + q[1] * ay; }
    /// The constant tangent columns @f$ \{a_x, a_y\} @f$.
    [[nodiscard]] std::array<VecN<3>, 2> frame(const Param<2>&) const { return {ax, ay}; }
};

/// A sphere of radius @c r centred at @c c with orthonormal frame
/// @c (ax, ay, az); chart @f$ (u, v) @f$ = (azimuth, latitude).
struct SphereParam {
    static constexpr int edim = 3, idim = 2;
    PtN<3> c;            ///< Centre.
    VecN<3> ax, ay, az;  ///< Orthonormal frame.
    real r;              ///< Radius.
    /// Inverse: azimuth/latitude of the radial direction of @p p.
    [[nodiscard]] Param<2> invert(const PtN<3>& p) const
    {
        const VecN<3> m = normalize(p - c);
        return {sycl::atan2(dot(m, ay), dot(m, ax)),
                sycl::asin(std::clamp(dot(m, az), -1.0_r, 1.0_r))};
    }
    /// Evaluate @f$ C + r(\cos v\cos u\,a_x + \cos v\sin u\,a_y + \sin v\,a_z) @f$.
    [[nodiscard]] PtN<3> eval(const Param<2>& q) const
    {
        const real cu = sycl::cos(q[0]), su = sycl::sin(q[0]);
        const real cv = sycl::cos(q[1]), sv = sycl::sin(q[1]);
        return c + r * (cv * cu * ax + cv * su * ay + sv * az);
    }
    /// The raw tangent columns @f$ \partial S/\partial u, \partial S/\partial v @f$.
    [[nodiscard]] std::array<VecN<3>, 2> frame(const Param<2>& q) const
    {
        const real cu = sycl::cos(q[0]), su = sycl::sin(q[0]);
        const real cv = sycl::cos(q[1]), sv = sycl::sin(q[1]);
        return {r * (cv * -su * ax + cv * cu * ay), r * (-sv * cu * ax - sv * su * ay + cv * az)};
    }
};

/// A right circular cylinder: axis @c az through @c o, radius @c r;
/// chart @f$ (u, v) @f$ = (angle about the axis, height along it).
struct CylinderParam {
    static constexpr int edim = 3, idim = 2;
    PtN<3> o;            ///< A point on the axis.
    VecN<3> ax, ay, az;  ///< Orthonormal frame; @c az is the axis.
    real r;              ///< Radius.
    /// Inverse: angle about the axis and height along it.
    [[nodiscard]] Param<2> invert(const PtN<3>& p) const
    {
        const VecN<3> d = p - o;
        return {sycl::atan2(dot(d, ay), dot(d, ax)), dot(d, az)};
    }
    /// Evaluate @f$ O + r(\cos u\,a_x + \sin u\,a_y) + v\,a_z @f$.
    [[nodiscard]] PtN<3> eval(const Param<2>& q) const
    { return o + r * (sycl::cos(q[0]) * ax + sycl::sin(q[0]) * ay) + q[1] * az; }
    /// The raw tangent columns.
    [[nodiscard]] std::array<VecN<3>, 2> frame(const Param<2>& q) const
    { return {r * (-sycl::sin(q[0]) * ax + sycl::cos(q[0]) * ay), az}; }
};

/// A 3D line through @c p0 and @c p1 (an edge curve, tdim==1).
struct Line3Param {
    static constexpr int edim = 3, idim = 1;
    PtN<3> p0, p1;  ///< Endpoints.
    /// Foot-of-projection parameter (unclamped; `Trim<1>` clamps).
    [[nodiscard]] Param<1> invert(const PtN<3>& q) const
    {
        const VecN<3> ab = p1 - p0;
        return {dot(q - p0, ab) / std::fmax(dot(ab, ab), tol::tiny)};
    }
    /// Evaluate @f$ P_0 + t(P_1 - P_0) @f$.
    [[nodiscard]] PtN<3> eval(const Param<1>& t) const { return p0 + t[0] * (p1 - p0); }
    /// The (constant) raw tangent column.
    [[nodiscard]] std::array<VecN<3>, 1> frame(const Param<1>&) const { return {p1 - p0}; }
};

/// A tensor-product B-spline / NURBS surface @f$ S(u,v) @f$ embedded in 3D.
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
    int pu, pv;                     ///< Basis degrees in u and v.
    int nu, nv;                     ///< Control-net extents in u and v.
    std::span<const real> knots_u;  ///< Knot vector in u, length @c nu+pu+1.
    std::span<const real> knots_v;  ///< Knot vector in v, length @c nv+pv+1.
    std::span<const real> ctrl;     ///< Control net, length @c 3*nu*nv (xyz interleaved).
    std::span<const real> weights;  ///< NURBS weights, length @c nu*nv, or empty.

    /// Control point @f$ P_{i_u, i_v} @f$.
    [[nodiscard]] PtN<3> cp(int iu, int iv) const
    {
        const int b = 3 * ((iu * nv) + iv);
        return {ctrl[b], ctrl[b + 1], ctrl[b + 2]};
    }

    /// The (at most) six surface partials the callers actually consume:
    /// @f$ S, S_u, S_v, S_{uu}, S_{vv}, S_{uv} @f$. `nd` selects how many
    /// are valid — `nd=0`→`S00`; `nd=1`→ adds `S10,S01`; `nd=2`→ adds
    /// `S11,S20,S02`. `nd=1` (no second-order trio) is the **warm**
    /// projection's Gauss-Newton path (@ref newton_foot, the steady-state
    /// hot path) and `frame`; `nd=2` is the **cold** exact-Newton path
    /// (first sweep / far queries, not register-critical).
    struct SurfDers {
        PtN<3> S00 {}, S10 {}, S01 {}, S20 {}, S02 {}, S11 {};
    };

    /// Fused compile-time-`nd` evaluation of just the @ref SurfDers partials.
    ///
    /// Flat accumulation: only the homogeneous sums for the six consumed partials
    /// (no full `A[3][3]`/`w[3][3]`/`S[3][3]` grid — that 9-entry nd=2 machinery
    /// was the boundary kernel's 256-VGPR pin). The bivariate quotient rule
    /// (rational case) and the unweighted pass-through are applied inline.
    ///
    /// A static `ND` with `if constexpr` elides the nd≥1 / nd≥2 homogeneous-sum
    /// rows and their `A*`/`w*` accumulators (and, via the templated de Boor, the
    /// derivative recurrence and the extra `du`/`dv` rows). A runtime `nd` would
    /// keep the nd=2 trio (`A11`/`A20`/`A02`, the 9-entry grid's residue) live
    /// across the whole inlined kernel — the boundary kernel's 256-VGPR pin.
    /// Routing the warm GN path through `ders_nd<1>` keeps those out of the warm
    /// kernel; the cold path keeps `ders_nd<2>`.
    template <int ND> [[nodiscard]] SurfDers ders_nd(const Param<2>& q) const
    {
        const int su = bspline_find_span(pu, nu, knots_u, q[0]);
        const int sv = bspline_find_span(pv, nv, knots_v, q[1]);
        real du[ND + 1][kBSplineCap], dv[ND + 1][kBSplineCap];
        bspline_basis_ders<ND, kBSplineCap>(pu, knots_u, su, q[0], du);
        bspline_basis_ders<ND, kBSplineCap>(pv, knots_v, sv, q[1], dv);

        // Homogeneous tensor-product sums A^(a,b) and weight sums w^(a,b) for ONLY
        // (a,b) in {00,10,01,11,20,02}. wt = weight (rational) or 1 (polynomial),
        // so A^(a,b) matches the original both ways and w^(a,b) is the partition
        // of unity in the polynomial case (used only on the rational branch).
        const bool rat = !weights.empty();
        PtN<3> A00 {}, A10 {}, A01 {}, A11 {}, A20 {}, A02 {};
        real w00 = 0.0_r, w10 = 0.0_r, w01 = 0.0_r, w11 = 0.0_r, w20 = 0.0_r, w02 = 0.0_r;
        for (int i = 0; i <= pu; ++i) {
            const int iu = su - pu + i;
            for (int j = 0; j <= pv; ++j) {
                const int iv = sv - pv + j;
                const real wt = rat ? weights[(iu * nv) + iv] : 1.0_r;
                const PtN<3> Pw = wt * cp(iu, iv);
                const real b0u = du[0][i], b0v = dv[0][j];
                const real n00 = b0u * b0v;
                A00 = A00 + (n00 * Pw);
                w00 += n00 * wt;
                if constexpr (ND >= 1) {
                    const real b1u = du[1][i], b1v = dv[1][j];
                    const real n10 = b1u * b0v, n01 = b0u * b1v;
                    A10 = A10 + (n10 * Pw);
                    A01 = A01 + (n01 * Pw);
                    w10 += n10 * wt;
                    w01 += n01 * wt;
                    if constexpr (ND >= 2) {
                        const real b2u = du[2][i], b2v = dv[2][j];
                        const real n11 = b1u * b1v, n20 = b2u * b0v, n02 = b0u * b2v;
                        A11 = A11 + (n11 * Pw);
                        A20 = A20 + (n20 * Pw);
                        A02 = A02 + (n02 * Pw);
                        w11 += n11 * wt;
                        w20 += n20 * wt;
                        w02 += n02 * wt;
                    }
                }
            }
        }
        SurfDers S;
        if (!rat) {
            S.S00 = A00;
            if constexpr (ND >= 1) { S.S10 = A10, S.S01 = A01; }
            if constexpr (ND >= 2) { S.S11 = A11, S.S20 = A20, S.S02 = A02; }
            return S;
        }
        // Bivariate quotient rule for S = A/w, up to second order per direction.
        const real iw = 1.0_r / w00;
        S.S00 = iw * A00;
        if constexpr (ND >= 1) {
            S.S10 = iw * (A10 - (w10 * S.S00));
            S.S01 = iw * (A01 - (w01 * S.S00));
        }
        if constexpr (ND >= 2) {
            S.S11 = iw * (A11 - (w10 * S.S01) - (w01 * S.S10) - (w11 * S.S00));
            S.S20 = iw * (A20 - (2.0_r * w10 * S.S10) - (w20 * S.S00));
            S.S02 = iw * (A02 - (2.0_r * w01 * S.S01) - (w02 * S.S00));
        }
        return S;
    }

    /// Evaluate the surface point @f$ S(u, v) @f$.
    [[nodiscard]] PtN<3> eval(const Param<2>& q) const { return ders_nd<0>(q).S00; }

    /// The raw tangent columns @f$ \{S_u, S_v\} @f$.
    [[nodiscard]] std::array<VecN<3>, 2> frame(const Param<2>& q) const
    {
        const SurfDers S = ders_nd<1>(q);
        return {S.S10, S.S01};
    }

    /// Inverse: coarse-grid-seeded Newton on the nearest-foot stationarity.
    ///
    /// Seeds from an 8×8 sample of the live domain, then iterates Newton on
    /// @f$ F = ((S-p)\cdot S_u, (S-p)\cdot S_v) @f$ with the exact Jacobian
    /// (needs the second partials), clamping each iterate to the domain. A
    /// near-singular Jacobian (e.g. at a degenerate corner) stops early.
    [[nodiscard]] Param<2> invert(const PtN<3>& p) const { return invert_seeded(p, {}, false); }

    /// Newton iteration on the nearest-foot stationarity
    /// @f$ F = ((S-p)\cdot S_u, (S-p)\cdot S_v) = 0 @f$ from start @p q,
    /// clamping each iterate to the domain and stopping once the (clamped)
    /// iterate stops moving (`|Δq| < 1e-9`; quadratic convergence ⇒ ~1e-18
    /// foot error). A near-singular Jacobian (a degenerate corner / cusp)
    /// stops early.
    ///
    /// @tparam Exact selects the Jacobian (W6):
    ///   - `true`  → the **exact** Newton Jacobian, including the curvature terms
    ///     `dot(d, S_uu)`. Robust for large foot residuals (a query far off the
    ///     surface) — used on the **cold** path (first sweep / no warm seed),
    ///     which is not register-critical (runs once).
    ///   - `false` → **Gauss-Newton** (`J = Jbᵀ Jb`, the curvature terms dropped).
    ///     The dropped terms are residual-weighted, so they vanish as `d → 0`;
    ///     on the **warm** path the per-DOF seed sits at the foot every sweep, so
    ///     GN is quadratically Newton-equivalent there while needing only `ders`
    ///     at nd=1 (no `S11/S20/S02`) — a cheaper iter (the boundary kernel's
    ///     steady-state hot path). GN is *not* robust for large residuals (the
    ///     curvature is then first-order), hence the cold path keeps `Exact`.
    ///
    /// @param p Query point.
    /// @param q Start parameter (warm seed or coarse-grid winner).
    /// @param u0,u1 Live u-domain (iterates are clamped to it).
    /// @param v0,v1 Live v-domain.
    /// @param frame_out If non-null, receives the raw tangent frame
    ///   `{S_u, S_v}` from the **last** Newton iterate — at convergence this is
    ///   `< tol::newton` in parameter from the returned foot, so it reuses the
    ///   `ders(nd=1)` already computed here instead of a redundant `frame(q*)`
    ///   call by the caller (F2 Lever C). Parity-gated (not bit-identical: the
    ///   frame is at the pre-final-step iterate).
    template <bool Exact>
    [[nodiscard]] Param<2> newton_foot(const PtN<3>& p,
                                       Param<2> q,
                                       real u0,
                                       real u1,
                                       real v0,
                                       real v1,
                                       std::array<VecN<3>, 2>* frame_out = nullptr) const
    {
        for (int it = 0; it < 12; ++it) {
            const SurfDers S = ders_nd<Exact ? 2 : 1>(q);
            if (frame_out != nullptr) { *frame_out = {S.S10, S.S01}; }
            const VecN<3> d = S.S00 - p;
            const real f1 = dot(d, S.S10), f2 = dot(d, S.S01);
            real j11 = dot(S.S10, S.S10);
            real j12 = dot(S.S10, S.S01);
            real j22 = dot(S.S01, S.S01);
            if constexpr (Exact) {
                j11 += dot(d, S.S20);
                j12 += dot(d, S.S11);
                j22 += dot(d, S.S02);
            }
            const real det = (j11 * j22) - (j12 * j12);
            if (sycl::fabs(det) < tol::tiny) { break; }
            const real qu = std::clamp(q[0] - (((j22 * f1) - (j12 * f2)) / det), u0, u1);
            const real qv = std::clamp(q[1] - (((-j12 * f1) + (j11 * f2)) / det), v0, v1);
            const bool converged = (sycl::fabs(qu - q[0]) + sycl::fabs(qv - q[1])) < tol::newton;
            q[0] = qu;
            q[1] = qv;
            if (converged) { break; }
        }
        return q;
    }

    /// Damped (Levenberg-Marquardt) nearest-foot Newton from one seed.
    ///
    /// Converges to the true foot even where the Jacobian is rank-deficient (a
    /// surface-of-revolution pole, `S_u -> 0`): the damping `lambda` bridges
    /// Gauss-Newton (away from the degeneracy) and gradient descent (at it),
    /// with residual backtracking so every accepted step reduces `|S-p|^2`.
    /// Cold path only (the multi-start placement); the warm kernel stays on the
    /// lean `newton_foot<false>`. Mirrors surfaces3d.py `_lm_foot`.
    [[nodiscard]] Param<2> newton_foot_lm(const PtN<3>& p,
                                          Param<2> q,
                                          real u0,
                                          real u1,
                                          real v0,
                                          real v1,
                                          std::array<VecN<3>, 2>* frame_out = nullptr) const
    {
        q[0] = std::clamp(q[0], u0, u1);
        q[1] = std::clamp(q[1], v0, v1);
        SurfDers S = ders_nd<2>(q);
        VecN<3> d = S.S00 - p;
        real r = dot(d, d);
        real lambda = 1e-3_r * (dot(S.S10, S.S10) + dot(S.S01, S.S01) + tol::tiny);
        for (int it = 0; it < 16; ++it) {
            const real f1 = dot(d, S.S10), f2 = dot(d, S.S01);
            const real j11 = dot(S.S10, S.S10) + dot(d, S.S20);
            const real j12 = dot(S.S10, S.S01) + dot(d, S.S11);
            const real j22 = dot(S.S01, S.S01) + dot(d, S.S02);
            real du = 0, dv = 0;
            bool accepted = false;
            for (int bt = 0; bt < 6; ++bt) {  // backtrack damping until descent
                const real a11 = j11 + lambda, a22 = j22 + lambda;
                const real det = (a11 * a22) - (j12 * j12);
                if (sycl::fabs(det) < tol::tiny) {
                    lambda *= 4.0_r;
                    continue;
                }
                du = -(((a22 * f1) - (j12 * f2)) / det);
                dv = -((((-j12) * f1) + (a11 * f2)) / det);
                const Param<2> qn {std::clamp(q[0] + du, u0, u1), std::clamp(q[1] + dv, v0, v1)};
                const VecN<3> dn = eval(qn) - p;
                const real rn = dot(dn, dn);
                if (rn < r) {
                    q = qn;
                    r = rn;
                    S = ders_nd<2>(q);
                    d = S.S00 - p;
                    lambda = sycl::fmax(lambda * 0.3_r, tol::tiny);
                    accepted = true;
                    break;
                }
                lambda *= 4.0_r;
            }
            if (!accepted || (sycl::fabs(du) + sycl::fabs(dv)) < tol::newton) { break; }
        }
        if (frame_out != nullptr) { *frame_out = {S.S10, S.S01}; }
        return q;
    }

    /// Warm-started inverse.
    ///
    /// @tparam Warm selects the cold/warm two-kernel split (the boundary sweep
    ///   runs a *cold* projection pass once at startup, then a lean *warm* kernel
    ///   for every subsequent sweep):
    ///   - `Warm=false` (cold pass): when @p has_seed start GN from the seed,
    ///     else run the 8×8 coarse-grid search + **exact Newton (nd=2)** — robust
    ///     for the topologically-placed nodes' first projection onto the geometry.
    ///   - `Warm=true` (steady-state hot path): the seed is always live (the cold
    ///     pass populated it and every node already sits on its surface), so this
    ///     **statically elides the coarse grid and the nd=2 path** and runs only
    ///     `newton_foot<false>` (GN nd=1). That keeps the heavy de Boor nd=2 rows
    ///     and the grid out of the warm kernel entirely — the scratch lever.
    template <bool Warm = false>
    [[nodiscard]] Param<2> invert_seeded(const PtN<3>& p,
                                         Param<2> seed,
                                         bool has_seed,
                                         std::array<VecN<3>, 2>* frame_out = nullptr) const
    {
        const real u0 = knots_u[pu], u1 = knots_u[nu];
        const real v0 = knots_v[pv], v1 = knots_v[nv];
        if constexpr (Warm) {
            (void)has_seed;  // the cold pass guarantees a live seed before any warm sweep
            const Param<2> q {std::clamp(seed[0], u0, u1), std::clamp(seed[1], v0, v1)};
            return newton_foot<false>(p, q, u0, u1, v0, v1, frame_out);
        } else {
            if (has_seed) {
                const Param<2> q {std::clamp(seed[0], u0, u1), std::clamp(seed[1], v0, v1)};
                return newton_foot<false>(p, q, u0, u1, v0, v1, frame_out);
            }
            // Multi-start: Newton-polish the best kStarts coarse seeds and keep
            // the globally nearest foot, so a closed surface's seam (the two
            // u-boundary seeds evaluate to the same point) cannot trap Newton on
            // the wrong side of the wrap. Cold path only — runs once per node, so
            // the restarts stay off the warm kernel. Mirrors surfaces3d.py.
            constexpr int kSeed = 8;
            constexpr int kStarts = 4;
            real bestd[kStarts];
            Param<2> bestq[kStarts];
            for (int s = 0; s < kStarts; ++s) {
                bestd[s] = std::numeric_limits<real>::infinity();
                bestq[s] = Param<2> {u0, v0};
            }
            for (int i = 0; i <= kSeed; ++i) {
                for (int j = 0; j <= kSeed; ++j) {
                    const Param<2> t {u0 + (((u1 - u0) * i) / kSeed),
                                      v0 + (((v1 - v0) * j) / kSeed)};
                    const VecN<3> d = eval(t) - p;
                    const real dd = dot(d, d);
                    if (dd < bestd[kStarts - 1]) {
                        int s = kStarts - 1;
                        for (; s > 0 && dd < bestd[s - 1]; --s) {
                            bestd[s] = bestd[s - 1];
                            bestq[s] = bestq[s - 1];
                        }
                        bestd[s] = dd;
                        bestq[s] = t;
                    }
                }
            }
            Param<2> best_foot {u0, v0};
            real best_final = std::numeric_limits<real>::infinity();
            for (int s = 0; s < kStarts; ++s) {
                if (bestd[s] == std::numeric_limits<real>::infinity()) { continue; }
                std::array<VecN<3>, 2> fbuf;
                const Param<2> foot = newton_foot_lm(p,
                                                     bestq[s],
                                                     u0,
                                                     u1,
                                                     v0,
                                                     v1,
                                                     frame_out != nullptr ? &fbuf : nullptr);
                const VecN<3> d = eval(foot) - p;
                const real dd = dot(d, d);
                if (dd < best_final) {
                    best_final = dd;
                    best_foot = foot;
                    if (frame_out != nullptr) { *frame_out = fbuf; }
                }
            }
            return best_foot;
        }
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
static_assert(GeometryEntity<LineRail> && GeometryEntity<LineRail3>);
static_assert(GeometryEntity<Free<2>> && GeometryEntity<LineSeg> && GeometryEntity<Circle> &&
              GeometryEntity<Ellipse> && GeometryEntity<TrimmedEntity<LineParam>> &&
              GeometryEntity<TrimmedEntity<CircleArcParam>> &&
              GeometryEntity<TrimmedEntity<EllipseArcParam>> &&
              GeometryEntity<TrimmedEntity<QuadBezierParam>> &&
              GeometryEntity<TrimmedEntity<CubicBezierParam>> && GeometryEntity<CompositePath>);

// --- The closed 2D entity set --------------------------------------------------
//
// No value-level entity variant exists. The closed set lives as (a) the
// static_assert(GeometryEntity<...>) block above, (b) the
// dispatch_entity_type<2> switch arms, and (c) the decode_entity_fn<E> /
// EntitySoA<E> specializations — all three must list the same types; a
// missing specialization fails to compile.

// ---------------------------------------------------------------------------
// decode_entity<E>: the per-type positional blob decoder — the ONE place the
// untyped `params` pointer is read per type. Variable-length entities
// (B-spline net/knots, composite records) keep their data in `arena` and
// store only offsets/counts in `params`; fixed-size entities ignore `arena`.
// Selected via dispatch_entity_type<D>(tag, [&]<class E>{ ... }), so even the
// cold oracle/test paths never materialize a variant.
// ---------------------------------------------------------------------------

/// Per-type positional blob decoder trait. Specialize per entity type @p E.
/// @tparam E The entity type to decode into.
template <class E> struct decode_entity_fn;  // primary undefined

template <> struct decode_entity_fn<Free<2>> {
    [[nodiscard]] static Free<2> apply(const real*, const real*) { return Free<2> {}; }
};
template <> struct decode_entity_fn<LineSeg> {
    [[nodiscard]] static LineSeg apply(const real* p, const real*)
    { return LineSeg {.sx = p[0], .sy = p[1], .ex = p[2], .ey = p[3]}; }
};
template <> struct decode_entity_fn<LineRail> {
    [[nodiscard]] static LineRail apply(const real* p, const real*)
    { return LineRail {.sx = p[0], .sy = p[1], .ex = p[2], .ey = p[3]}; }
};
template <> struct decode_entity_fn<LineRail3> {
    [[nodiscard]] static LineRail3 apply(const real* p, const real*)
    { return LineRail3 {.sx = p[0], .sy = p[1], .sz = p[2], .ex = p[3], .ey = p[4], .ez = p[5]}; }
};
template <> struct decode_entity_fn<Circle> {
    [[nodiscard]] static Circle apply(const real* p, const real*)
    { return Circle {.cx = p[0], .cy = p[1], .r = p[2]}; }
};
template <> struct decode_entity_fn<Ellipse> {
    [[nodiscard]] static Ellipse apply(const real* p, const real*)
    { return Ellipse {.cx = p[0], .cy = p[1], .rx = p[2], .ry = p[3]}; }
};
template <> struct decode_entity_fn<TrimmedEntity<CircleArcParam>> {
    [[nodiscard]] static TrimmedEntity<CircleArcParam> apply(const real* p, const real*)
    {
        return TrimmedEntity<CircleArcParam> {
          .param = {.c = {p[0], p[1]}, .r = p[2]},
          .trim = {.t0 = p[3], .t1 = p[4], .closed = p[5] != 0.0_r}};
    }
};
template <> struct decode_entity_fn<TrimmedEntity<EllipseArcParam>> {
    [[nodiscard]] static TrimmedEntity<EllipseArcParam> apply(const real* p, const real*)
    {
        return TrimmedEntity<EllipseArcParam> {
          .param = {.c = {p[0], p[1]}, .a = p[2], .b = p[3], .phi = p[4]},
          .trim = {.t0 = p[5], .t1 = p[6], .closed = p[7] != 0.0_r}};
    }
};
template <> struct decode_entity_fn<TrimmedEntity<QuadBezierParam>> {
    [[nodiscard]] static TrimmedEntity<QuadBezierParam> apply(const real* p, const real*)
    {
        return TrimmedEntity<QuadBezierParam> {
          .param = {.p = {{{p[0], p[1]}, {p[2], p[3]}, {p[4], p[5]}}}},
          .trim = {.t0 = p[6], .t1 = p[7], .closed = false}};
    }
};
template <> struct decode_entity_fn<TrimmedEntity<CubicBezierParam>> {
    [[nodiscard]] static TrimmedEntity<CubicBezierParam> apply(const real* p, const real*)
    {
        return TrimmedEntity<CubicBezierParam> {
          .param = {.p = {{{p[0], p[1]}, {p[2], p[3]}, {p[4], p[5]}, {p[6], p[7]}}}},
          .trim = {.t0 = p[8], .t1 = p[9], .closed = false}};
    }
};
template <> struct decode_entity_fn<TrimmedEntity<BSplineCurveParam>> {
    [[nodiscard]] static TrimmedEntity<BSplineCurveParam> apply(const real* p, const real* arena)
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
        const bool has_w = p[7] != 0.0_r;
        const auto w_off = static_cast<std::size_t>(p[6]);
        // Contract: variable-length tags are always dispatched with a live
        // arena (see the decode_entity section note); nullptr never reaches here.
        return TrimmedEntity<BSplineCurveParam> {
          .param = {.degree = degree,
                    .n_ctrl = n_ctrl,
                    // NOLINTNEXTLINE(clang-analyzer-core.NullPointerArithm)
                    .knots = {arena + knot_off, n_knots},
                    .ctrl = {arena + ctrl_off, n_ctrl_d},
                    .weights = has_w ? std::span<const real> {arena + w_off,
                                                              static_cast<std::size_t>(n_ctrl)}
                                     : std::span<const real> {}},
          .trim = {.t0 = p[4], .t1 = p[5], .closed = false}};
    }
};
template <> struct decode_entity_fn<CompositePath> {
    [[nodiscard]] static CompositePath apply(const real* p, const real* arena)
    {
        // Blob: [n_segs, rec_off]; the segment records live in the arena.
        const int n_segs = static_cast<int>(p[0]);
        const auto rec_off = static_cast<std::size_t>(p[1]);
        // Contract: composites always carry a live arena (segment records).
        return CompositePath {
          .n_segs = n_segs,
          // NOLINTNEXTLINE(clang-analyzer-core.NullPointerArithm)
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
    [[nodiscard]] static Free<3> apply(const real*, const real*) { return Free<3> {}; }
};
template <> struct decode_entity_fn<TrimmedEntity<PlaneParam>> {
    [[nodiscard]] static TrimmedEntity<PlaneParam> apply(const real* p, const real*)
    {
        // Blob: [o(3), ax(3), ay(3)].
        const auto pt = [&](int i) { return PtN<3> {p[i], p[i + 1], p[i + 2]}; };
        return TrimmedEntity<PlaneParam> {.param = {.o = pt(0), .ax = pt(3), .ay = pt(6)},
                                          .trim = {}};
    }
};
template <> struct decode_entity_fn<TrimmedEntity<SphereParam>> {
    [[nodiscard]] static TrimmedEntity<SphereParam> apply(const real* p, const real*)
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
    [[nodiscard]] static TrimmedEntity<CylinderParam> apply(const real* p, const real*)
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
    [[nodiscard]] static TrimmedEntity<Line3Param> apply(const real* p, const real*)
    {
        // Blob: [p0(3), p1(3), t0, t1].
        const auto pt = [&](int i) { return PtN<3> {p[i], p[i + 1], p[i + 2]}; };
        return TrimmedEntity<Line3Param> {.param = {.p0 = pt(0), .p1 = pt(3)},
                                          .trim = {.t0 = p[6], .t1 = p[7], .closed = false}};
    }
};
template <> struct decode_entity_fn<TrimmedEntity<BSplineSurfaceParam>> {
    [[nodiscard]] static TrimmedEntity<BSplineSurfaceParam> apply(const real* p, const real* arena)
    {
        // Blob: [pu, pv, nu, nv, ku_off, kv_off, ctrl_off, w_off, has_w,
        // trim_v_off, trim_l_off, n_trim_loop_entries]; knots/control
        // net/weights (and the UV trim polygon, when present) live in the
        // arena. Zero trim fields (older/zero-padded blobs) mean untrimmed.
        const int pu = static_cast<int>(p[0]);
        const int pv = static_cast<int>(p[1]);
        const int nu = static_cast<int>(p[2]);
        const int nv = static_cast<int>(p[3]);
        const auto ku_off = static_cast<std::size_t>(p[4]);
        const auto kv_off = static_cast<std::size_t>(p[5]);
        const auto ctrl_off = static_cast<std::size_t>(p[6]);
        const auto w_off = static_cast<std::size_t>(p[7]);
        const bool has_w = p[8] != 0.0_r;
        const auto n_net = static_cast<std::size_t>(nu) * static_cast<std::size_t>(nv);
        const auto n_loop_entries = static_cast<std::size_t>(p[11]);
        Trim<2> trim {};
        if (n_loop_entries >= 2) {
            const auto v_off = static_cast<std::size_t>(p[9]);
            const auto l_off = static_cast<std::size_t>(p[10]);
            const std::span<const real> loops {arena + l_off, n_loop_entries};
            const auto n_verts = static_cast<std::size_t>(loops[n_loop_entries - 1]);
            trim = {.verts = {arena + v_off, 2 * n_verts}, .loops = loops};
        }
        return TrimmedEntity<BSplineSurfaceParam> {
          .param = {.pu = pu,
                    .pv = pv,
                    .nu = nu,
                    .nv = nv,
                    .knots_u = {arena + ku_off, static_cast<std::size_t>(nu + pu) + 1},
                    .knots_v = {arena + kv_off, static_cast<std::size_t>(nv + pv) + 1},
                    .ctrl = {arena + ctrl_off, 3 * n_net},
                    .weights = has_w ? std::span<const real> {arena + w_off, n_net}
                                     : std::span<const real> {}},
          .trim = trim};
    }
};

/// Build a typed entity @p E from its positional blob (single source).
///
/// This is the per-type builder that replaces the make_entity arms: it reads the
/// untyped `params` pointer ONCE per type and returns a concrete `E`. Variable-
/// length entities (B-spline, composite) read their payload from @p arena via
/// the offsets stored in @p params; fixed-size entities ignore @p arena.
/// @tparam E The entity type to decode into.
/// @param params Flat parameter blob (`kParamPad` reals per DOF).
/// @param arena Base of the per-group real arena for span-backed entities, or
///              nullptr when no variable-length entity is in play.
template <class E>
[[nodiscard]] inline E decode_entity(const real* params, const real* arena = nullptr)
{ return decode_entity_fn<E>::apply(params, arena); }

// There is no per-DOF entity variant: every call site selects the concrete
// entity type via dispatch_entity_type<D>(tag, f) and builds it with
// decode_entity<E> (cold blob/oracle paths) or EntitySoA<E>::load (the device
// sweep). CompositePath::project_frame is defined below dispatch_entity_type
// (it builds each segment the same way).

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
static_assert(to_int(EntityTag::LineRail) == TAG_LINERAIL);
static_assert(to_int(EntityTag::LineRail3) == TAG_LINERAIL3);
static_assert(to_int(EntityTag::Line3) == TAG_LINE3);
static_assert(to_int(EntityTag::BSplineSurface) == TAG_BSPLINESURF);

/// SoA schema for the interior (free) DOF: no per-entity fields.
///
/// A free node carries no geometry, so its storage is a bare count and its
/// device builder reconstructs a default `Free<D>`. The `View` is an empty
/// `SoAView<const real>` (0×0 extents) so every `EntitySoA<E>` specialization
/// shares the one typed View type — `PartitionView` then holds a
/// `SoAView<const real>` directly, no `const void*`, no type erasure. `load`
/// ignores the view and returns a default-constructed `Free<D>`.
template <int D> struct EntitySoA<Free<D>> {
    static constexpr EntityTag tag = EntityTag::Free;
    static constexpr int kFields = 0;  ///< No fields; `records` is empty.
    static constexpr int kSeg = 0;     ///< No segmented fields.
    struct Host {
        std::vector<real> records;             ///< Empty (kFields == 0).
        std::size_t count = 0;                 ///< Number of free DOFs in the partition.
        std::vector<SegmentedHost<real>> seg;  ///< Empty (kSeg == 0).
    };
    struct View {
        SoAView<const real> records {nullptr, 0, 0};  ///< Empty (no fields).
        SegmentedView<real> seg[kMaxSoASeg] {};       ///< Null (no segmented fields).
    };
    /// Reconstruct the (field-less) free entity.
    [[nodiscard]] static Free<D> load(const View&, std::size_t) { return Free<D> {}; }
    /// Scatter a free entity into the host (no-op: no fields).
    static void load_into(Host&, std::size_t, const Free<D>&) {}
    /// Construct the typed View from the generic partition slots.
    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>*)
    { return View {.records = soa}; }
};

static_assert(HasEntitySoA<Free<2>>);

// ---------------------------------------------------------------------------
// Fixed-size 2D entity SoA specializations.
//
// Each is a packed contiguous record store: one flat `real[count*kFields]`
// per partition, stride `kFields` per entity — the layout that matches the
// sweep's per-entity-load access pattern (one coalesced read of entity i's
// kFields reals per work item, no per-field array indirection). The View is
// a `SoAView<const real>` mdspan with extents `(count, kFields)`; `load`
// reads `view.records(i, FIELD)` via named compile-time offsets and returns via
// designated initializers (matching make_entity's style). `bool` trim fields
// are stored as `0.0`/`1.0` reals on the wire and reconstituted via `!= 0.0`.
//
// All specializations share the same Host/View shape with segmented slots
// (kSeg == 0 for fixed-size — seg vectors/views empty/null), load_into, and
// tie_view, keeping the ctor and kernel code generic across fixed-size and
// segmented types.
// ---------------------------------------------------------------------------

/// SoA schema for @ref LineSeg — packed `(sx, sy, ex, ey)` records.
template <> struct EntitySoA<LineSeg> {
    static constexpr EntityTag tag = EntityTag::LineSeg;
    static constexpr int kFields = 4;
    static constexpr int kSeg = 0;
    static constexpr int SX = 0, SY = 1, EX = 2, EY = 3;
    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
    };
    [[nodiscard]] static LineSeg load(const View& v, std::size_t i)
    {
        return LineSeg {.sx = v.records[i, SX],
                        .sy = v.records[i, SY],
                        .ex = v.records[i, EX],
                        .ey = v.records[i, EY]};
    }
    static void load_into(Host& h, std::size_t i, const LineSeg& e)
    {
        real* r = h.records.data() + i * kFields;
        r[SX] = e.sx;
        r[SY] = e.sy;
        r[EX] = e.ex;
        r[EY] = e.ey;
    }
    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>*)
    { return View {.records = soa}; }
};

/// SoA schema for @ref LineRail — packed `(sx, sy, ex, ey)` records.
template <> struct EntitySoA<LineRail> {
    static constexpr EntityTag tag = EntityTag::LineRail;
    static constexpr int kFields = 4;
    static constexpr int kSeg = 0;
    static constexpr int SX = 0, SY = 1, EX = 2, EY = 3;
    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
    };
    [[nodiscard]] static LineRail load(const View& v, std::size_t i)
    {
        return LineRail {.sx = v.records[i, SX],
                         .sy = v.records[i, SY],
                         .ex = v.records[i, EX],
                         .ey = v.records[i, EY]};
    }
    static void load_into(Host& h, std::size_t i, const LineRail& e)
    {
        real* r = h.records.data() + i * kFields;
        r[SX] = e.sx;
        r[SY] = e.sy;
        r[EX] = e.ex;
        r[EY] = e.ey;
    }
    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>*)
    { return View {.records = soa}; }
};

/// SoA schema for @ref LineRail3 — packed `(sx, sy, sz, ex, ey, ez)` records.
template <> struct EntitySoA<LineRail3> {
    static constexpr EntityTag tag = EntityTag::LineRail3;
    static constexpr int kFields = 6;
    static constexpr int kSeg = 0;
    static constexpr int SX = 0, SY = 1, SZ = 2, EX = 3, EY = 4, EZ = 5;
    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
    };
    [[nodiscard]] static LineRail3 load(const View& v, std::size_t i)
    {
        return LineRail3 {.sx = v.records[i, SX],
                          .sy = v.records[i, SY],
                          .sz = v.records[i, SZ],
                          .ex = v.records[i, EX],
                          .ey = v.records[i, EY],
                          .ez = v.records[i, EZ]};
    }
    static void load_into(Host& h, std::size_t i, const LineRail3& e)
    {
        real* r = h.records.data() + i * kFields;
        r[SX] = e.sx;
        r[SY] = e.sy;
        r[SZ] = e.sz;
        r[EX] = e.ex;
        r[EY] = e.ey;
        r[EZ] = e.ez;
    }
    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>*)
    { return View {.records = soa}; }
};

/// SoA schema for @ref Circle — packed `(cx, cy, r)` records.
template <> struct EntitySoA<Circle> {
    static constexpr EntityTag tag = EntityTag::Circle;
    static constexpr int kFields = 3;
    static constexpr int kSeg = 0;
    static constexpr int CX = 0, CY = 1, R = 2;
    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
    };
    [[nodiscard]] static Circle load(const View& v, std::size_t i)
    { return Circle {.cx = v.records[i, CX], .cy = v.records[i, CY], .r = v.records[i, R]}; }
    static void load_into(Host& h, std::size_t i, const Circle& e)
    {
        real* r = h.records.data() + i * kFields;
        r[CX] = e.cx;
        r[CY] = e.cy;
        r[R] = e.r;
    }
    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>*)
    { return View {.records = soa}; }
};

/// SoA schema for @ref Ellipse — packed `(cx, cy, rx, ry)` records.
template <> struct EntitySoA<Ellipse> {
    static constexpr EntityTag tag = EntityTag::Ellipse;
    static constexpr int kFields = 4;
    static constexpr int kSeg = 0;
    static constexpr int CX = 0, CY = 1, RX = 2, RY = 3;
    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
    };
    [[nodiscard]] static Ellipse load(const View& v, std::size_t i)
    {
        return Ellipse {.cx = v.records[i, CX],
                        .cy = v.records[i, CY],
                        .rx = v.records[i, RX],
                        .ry = v.records[i, RY]};
    }
    static void load_into(Host& h, std::size_t i, const Ellipse& e)
    {
        real* r = h.records.data() + i * kFields;
        r[CX] = e.cx;
        r[CY] = e.cy;
        r[RX] = e.rx;
        r[RY] = e.ry;
    }
    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>*)
    { return View {.records = soa}; }
};

/// SoA schema for `TrimmedEntity<CircleArcParam>`:
/// packed `(cx, cy, r, t0, t1, closed)` records.
template <> struct EntitySoA<TrimmedEntity<CircleArcParam>> {
    static constexpr EntityTag tag = EntityTag::CircleArc;
    static constexpr int kFields = 6;
    static constexpr int kSeg = 0;
    static constexpr int CX = 0, CY = 1, R = 2, T0 = 3, T1 = 4, CLOSED = 5;
    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
    };
    [[nodiscard]] static TrimmedEntity<CircleArcParam> load(const View& v, std::size_t i)
    {
        return TrimmedEntity<CircleArcParam> {
          .param = {.c = {v.records[i, CX], v.records[i, CY]}, .r = v.records[i, R]},
          .trim = {.t0 = v.records[i, T0],
                   .t1 = v.records[i, T1],
                   .closed = v.records[i, CLOSED] != 0.0_r}};
    }
    static void load_into(Host& h, std::size_t i, const TrimmedEntity<CircleArcParam>& e)
    {
        real* r = h.records.data() + i * kFields;
        r[CX] = e.param.c[0];
        r[CY] = e.param.c[1];
        r[R] = e.param.r;
        r[T0] = e.trim.t0;
        r[T1] = e.trim.t1;
        r[CLOSED] = e.trim.closed ? 1.0_r : 0.0_r;
    }
    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>*)
    { return View {.records = soa}; }
};

/// SoA schema for `TrimmedEntity<EllipseArcParam>`:
/// packed `(cx, cy, a, b, phi, t0, t1, closed)` records.
template <> struct EntitySoA<TrimmedEntity<EllipseArcParam>> {
    static constexpr EntityTag tag = EntityTag::EllipseArc;
    static constexpr int kFields = 8;
    static constexpr int kSeg = 0;
    static constexpr int CX = 0, CY = 1, A = 2, B = 3, PHI = 4, T0 = 5, T1 = 6, CLOSED = 7;
    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
    };
    [[nodiscard]] static TrimmedEntity<EllipseArcParam> load(const View& v, std::size_t i)
    {
        return TrimmedEntity<EllipseArcParam> {.param = {.c = {v.records[i, CX], v.records[i, CY]},
                                                         .a = v.records[i, A],
                                                         .b = v.records[i, B],
                                                         .phi = v.records[i, PHI]},
                                               .trim = {.t0 = v.records[i, T0],
                                                        .t1 = v.records[i, T1],
                                                        .closed = v.records[i, CLOSED] != 0.0_r}};
    }
    static void load_into(Host& h, std::size_t i, const TrimmedEntity<EllipseArcParam>& e)
    {
        real* r = h.records.data() + i * kFields;
        r[CX] = e.param.c[0];
        r[CY] = e.param.c[1];
        r[A] = e.param.a;
        r[B] = e.param.b;
        r[PHI] = e.param.phi;
        r[T0] = e.trim.t0;
        r[T1] = e.trim.t1;
        r[CLOSED] = e.trim.closed ? 1.0_r : 0.0_r;
    }
    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>*)
    { return View {.records = soa}; }
};

/// SoA schema for `TrimmedEntity<QuadBezierParam>`:
/// packed `(P0x, P0y, P1x, P1y, P2x, P2y, t0, t1)` records.
template <> struct EntitySoA<TrimmedEntity<QuadBezierParam>> {
    static constexpr EntityTag tag = EntityTag::QuadBezier;
    static constexpr int kFields = 8;
    static constexpr int kSeg = 0;
    static constexpr int P0X = 0, P0Y = 1, P1X = 2, P1Y = 3, P2X = 4, P2Y = 5, T0 = 6, T1 = 7;
    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
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
        real* r = h.records.data() + i * kFields;
        r[P0X] = e.param.p[0][0];
        r[P0Y] = e.param.p[0][1];
        r[P1X] = e.param.p[1][0];
        r[P1Y] = e.param.p[1][1];
        r[P2X] = e.param.p[2][0];
        r[P2Y] = e.param.p[2][1];
        r[T0] = e.trim.t0;
        r[T1] = e.trim.t1;
    }
    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>*)
    { return View {.records = soa}; }
};

/// SoA schema for `TrimmedEntity<CubicBezierParam>`:
/// packed `(P0x, P0y, P1x, P1y, P2x, P2y, P3x, P3y, t0, t1)` records.
template <> struct EntitySoA<TrimmedEntity<CubicBezierParam>> {
    static constexpr EntityTag tag = EntityTag::CubicBezier;
    static constexpr int kFields = 10;
    static constexpr int kSeg = 0;
    static constexpr int P0X = 0, P0Y = 1, P1X = 2, P1Y = 3, P2X = 4, P2Y = 5, P3X = 6, P3Y = 7,
                         T0 = 8, T1 = 9;
    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
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
        real* r = h.records.data() + i * kFields;
        r[P0X] = e.param.p[0][0];
        r[P0Y] = e.param.p[0][1];
        r[P1X] = e.param.p[1][0];
        r[P1Y] = e.param.p[1][1];
        r[P2X] = e.param.p[2][0];
        r[P2Y] = e.param.p[2][1];
        r[P3X] = e.param.p[3][0];
        r[P3Y] = e.param.p[3][1];
        r[T0] = e.trim.t0;
        r[T1] = e.trim.t1;
    }
    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>*)
    { return View {.records = soa}; }
};

static_assert(HasEntitySoA<LineSeg>);
static_assert(HasEntitySoA<Circle>);
static_assert(HasEntitySoA<Ellipse>);
static_assert(HasEntitySoA<TrimmedEntity<CircleArcParam>>);
static_assert(HasEntitySoA<TrimmedEntity<EllipseArcParam>>);
static_assert(HasEntitySoA<TrimmedEntity<QuadBezierParam>>);
static_assert(HasEntitySoA<TrimmedEntity<CubicBezierParam>>);

// ---------------------------------------------------------------------------
// Segmented 2D entity SoA specialization.
//
// The B-spline curve carries variable-length data (knot vector + control net)
// that the packed-record layout cannot hold. The scalar fields (degree,
// n_ctrl, t0, t1, closed) go in the packed records; the variable-length knots
// and ctrl go in two SegmentedHost/SegmentedView CSR slots (kSeg == 2).
//
// `load` constructs `std::span<const real>` from the SegmentedView's data/off
// arrays — the same idiom the existing `decode_entity` uses from the blob arena,
// and unavoidable because `BSplineCurveParam` itself stores `std::span`. The
// toolchain supports `std::span` in device code (GCC 16.1.1 + AdaptiveCpp); the
// golden tests gate correctness.
// ---------------------------------------------------------------------------

/// SoA schema for `TrimmedEntity<BSplineCurveParam>`:
/// packed `(degree, n_ctrl, t0, t1, closed, has_w)` records + up to 3
/// segmented CSR fields (knots, ctrl, weights). The `has_w` flag selects
/// the rational form; `weights` is present only when `has_w != 0`.
template <> struct EntitySoA<TrimmedEntity<BSplineCurveParam>> {
    static constexpr EntityTag tag = EntityTag::BSpline;
    static constexpr int kFields = 6;
    static constexpr int kSeg = 3;  ///< knots (0), ctrl (1), weights (2, optional).
    static constexpr int DEGREE = 0, N_CTRL = 1, T0 = 2, T1 = 3, CLOSED = 4, HAS_W = 5;
    static constexpr int KNOTS = 0, CTRL = 1, WEIGHTS = 2;  ///< Segmented slot indices.

    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
    };

    [[nodiscard]] static TrimmedEntity<BSplineCurveParam> load(const View& v, std::size_t i)
    {
        const bool has_w = v.records[i, HAS_W] != 0.0_r;
        return TrimmedEntity<BSplineCurveParam> {
          .param = {.degree = static_cast<int>(v.records[i, DEGREE]),
                    .n_ctrl = static_cast<int>(v.records[i, N_CTRL]),
                    .knots = v.seg[KNOTS][i],
                    .ctrl = v.seg[CTRL][i],
                    .weights =
                      has_w ? std::span<const real> {v.seg[WEIGHTS][i]} : std::span<const real> {}},
          .trim = {.t0 = v.records[i, T0],
                   .t1 = v.records[i, T1],
                   .closed = v.records[i, CLOSED] != 0.0_r}};
    }

    static void load_into(Host& h, std::size_t i, const TrimmedEntity<BSplineCurveParam>& e)
    {
        bspline_degree_guard(e.param.degree, "curve", kMaxBSplineCurveDegree);
        real* r = h.records.data() + i * kFields;
        r[DEGREE] = static_cast<real>(e.param.degree);
        r[N_CTRL] = static_cast<real>(e.param.n_ctrl);
        r[T0] = e.trim.t0;
        r[T1] = e.trim.t1;
        r[CLOSED] = e.trim.closed ? 1.0_r : 0.0_r;
        const bool has_w = !e.param.weights.empty();
        r[HAS_W] = has_w ? 1.0_r : 0.0_r;
        h.seg[KNOTS].push_back(e.param.knots);
        h.seg[CTRL].push_back(e.param.ctrl);
        h.seg[WEIGHTS].push_back(has_w ? e.param.weights : std::span<const real> {});
    }

    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>* seg)
    {
        View v {.records = soa};
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
// Composite-path SoA specialization.
//
// A composite owns a single self-contained arena slice — the same positional
// layout the global blob/oracle path uses, but per-composite: any
// variable-length sub-segment data (B-spline knots/ctrl) is laid down first,
// then the `n_segs` fixed-stride segment records `[seg_tag, params(kParamPad)]`
// (kCompositeRecSize reals each) at offset `rec_off`. The whole slice goes in
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

/// SoA schema for `CompositePath`: packed `(n_segs, rec_off)` + one
/// segmented CSR slot holding the per-composite self-contained arena
/// slice (`[sub-segment data | segment records]`).
template <> struct EntitySoA<CompositePath> {
    static constexpr EntityTag tag = EntityTag::Composite;
    static constexpr int kFields = 2;
    static constexpr int kSeg = 1;  ///< self-contained arena slice (slot 0).
    static constexpr int N_SEGS = 0, REC_OFF = 1;
    static constexpr int ARENA = 0;  ///< Segmented slot index.

    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
    };

    [[nodiscard]] static CompositePath load(const View& v, std::size_t i)
    {
        const std::span<const real> slice = v.seg[ARENA][i];
        const auto n_segs = static_cast<int>(v.records[i, N_SEGS]);
        const auto rec_off = static_cast<std::size_t>(v.records[i, REC_OFF]);
        return CompositePath {
          .n_segs = n_segs,
          .recs = slice.subspan(rec_off, static_cast<std::size_t>(n_segs) * kCompositeRecSize),
          .arena = slice.data()};
    }

    static void load_into(Host& h, std::size_t i, const CompositePath& e)
    {
        real* r = h.records.data() + (i * kFields);
        r[N_SEGS] = static_cast<real>(e.n_segs);
        // Blob→SoA host path (golden test/bench): the decoded entity's records
        // are self-contained for fixed-size segments, so the slice is exactly
        // the record block at rec_off == 0. (A composite carrying B-spline
        // sub-segment data is built via the Python wire, not this path.)
        r[REC_OFF] = 0.0_r;
        h.seg[ARENA].push_back(e.recs);
    }

    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>* seg)
    {
        View v {.records = soa};
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

/// SoA schema for `TrimmedEntity<PlaneParam>`: packed
/// `(o(3), ax(3), ay(3))` records (9 reals). kSeg == 0.
template <> struct EntitySoA<TrimmedEntity<PlaneParam>> {
    static constexpr EntityTag tag = EntityTag::Plane;
    static constexpr int kFields = 9;
    static constexpr int kSeg = 0;
    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
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
        real* r = h.records.data() + i * kFields;
        for (int k = 0; k < 3; ++k) { r[k] = e.param.o[k]; }
        for (int k = 0; k < 3; ++k) { r[3 + k] = e.param.ax[k]; }
        for (int k = 0; k < 3; ++k) { r[6 + k] = e.param.ay[k]; }
    }
    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>*)
    { return View {.records = soa}; }
};

/// SoA schema for `TrimmedEntity<SphereParam>`: packed
/// `(c(3), r, ax(3), ay(3))` records (10 reals); `az` is derived at
/// load as `ax × ay`. kSeg == 0.
template <> struct EntitySoA<TrimmedEntity<SphereParam>> {
    static constexpr EntityTag tag = EntityTag::Sphere;
    static constexpr int kFields = 10;
    static constexpr int kSeg = 0;
    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
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
        real* r = h.records.data() + i * kFields;
        for (int k = 0; k < 3; ++k) { r[k] = e.param.c[k]; }
        r[3] = e.param.r;
        for (int k = 0; k < 3; ++k) { r[4 + k] = e.param.ax[k]; }
        for (int k = 0; k < 3; ++k) { r[7 + k] = e.param.ay[k]; }
    }
    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>*)
    { return View {.records = soa}; }
};

/// SoA schema for `TrimmedEntity<CylinderParam>`: packed
/// `(o(3), ax(3), ay(3), r)` records (10 reals); `az` is derived at
/// load as `ax × ay`. kSeg == 0.
template <> struct EntitySoA<TrimmedEntity<CylinderParam>> {
    static constexpr EntityTag tag = EntityTag::Cylinder;
    static constexpr int kFields = 10;
    static constexpr int kSeg = 0;
    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
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
        real* r = h.records.data() + i * kFields;
        for (int k = 0; k < 3; ++k) { r[k] = e.param.o[k]; }
        for (int k = 0; k < 3; ++k) { r[3 + k] = e.param.ax[k]; }
        for (int k = 0; k < 3; ++k) { r[6 + k] = e.param.ay[k]; }
        r[9] = e.param.r;
    }
    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>*)
    { return View {.records = soa}; }
};

/// SoA schema for `TrimmedEntity<Line3Param>`: packed
/// `(p0(3), p1(3), t0, t1)` records (8 reals). kSeg == 0.
template <> struct EntitySoA<TrimmedEntity<Line3Param>> {
    static constexpr EntityTag tag = EntityTag::Line3;
    static constexpr int kFields = 8;
    static constexpr int kSeg = 0;
    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
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
        real* r = h.records.data() + i * kFields;
        for (int k = 0; k < 3; ++k) { r[k] = e.param.p0[k]; }
        for (int k = 0; k < 3; ++k) { r[3 + k] = e.param.p1[k]; }
        r[6] = e.trim.t0;
        r[7] = e.trim.t1;
    }
    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>*)
    { return View {.records = soa}; }
};

/// SoA schema for `TrimmedEntity<BSplineSurfaceParam>`: packed
/// `(pu, pv, nu, nv, ku_off, kv_off, ctrl_off, w_off, has_w)` records +
/// up to 4 segmented CSR fields (knots_u, knots_v, ctrl, weights). The
/// `has_w` flag selects the rational form; `weights` is present only
/// when `has_w != 0`. Offsets in the record are relative to the CSR slot
/// bases (not a global arena), so `load` slices each span directly.
template <> struct EntitySoA<TrimmedEntity<BSplineSurfaceParam>> {
    static constexpr EntityTag tag = EntityTag::BSplineSurface;
    static constexpr int kFields = 9;
    static constexpr int kSeg = 6;  ///< knots_u, knots_v, ctrl, weights, trim_verts, trim_loops.
    static constexpr int PU = 0, PV = 1, NU = 2, NV = 3, KU_OFF = 4, KV_OFF = 5, CTRL_OFF = 6,
                         W_OFF = 7, HAS_W = 8;
    static constexpr int KNOTS_U = 0, KNOTS_V = 1, CTRL = 2, WEIGHTS = 3, TRIM_VERTS = 4,
                         TRIM_LOOPS = 5;

    struct Host {
        std::vector<real> records;
        std::size_t count = 0;
        std::vector<SegmentedHost<real>> seg;
    };
    struct View {
        SoAView<const real> records {nullptr, 0, kFields};
        SegmentedView<real> seg[kMaxSoASeg] {};
    };

    [[nodiscard]] static TrimmedEntity<BSplineSurfaceParam> load(const View& v, std::size_t i)
    {
        const int pu = static_cast<int>(v.records[i, PU]);
        const int pv = static_cast<int>(v.records[i, PV]);
        const int nu = static_cast<int>(v.records[i, NU]);
        const int nv = static_cast<int>(v.records[i, NV]);
        const auto n_net = static_cast<std::size_t>(nu) * static_cast<std::size_t>(nv);
        const bool has_w = v.records[i, HAS_W] != 0.0_r;
        return TrimmedEntity<BSplineSurfaceParam> {
          .param = {.pu = pu,
                    .pv = pv,
                    .nu = nu,
                    .nv = nv,
                    .knots_u = v.seg[KNOTS_U][i],
                    .knots_v = v.seg[KNOTS_V][i],
                    .ctrl = v.seg[CTRL][i],
                    .weights = has_w ? v.seg[WEIGHTS][i] : std::span<const real> {}},
          .trim = {.verts = v.seg[TRIM_VERTS][i], .loops = v.seg[TRIM_LOOPS][i]}};
    }

    static void load_into(Host& h, std::size_t i, const TrimmedEntity<BSplineSurfaceParam>& e)
    {
        bspline_degree_guard(e.param.pu, "surface (u)", kMaxBSplineDegree);
        bspline_degree_guard(e.param.pv, "surface (v)", kMaxBSplineDegree);
        real* r = h.records.data() + i * kFields;
        r[PU] = static_cast<real>(e.param.pu);
        r[PV] = static_cast<real>(e.param.pv);
        r[NU] = static_cast<real>(e.param.nu);
        r[NV] = static_cast<real>(e.param.nv);
        // Per-entity offsets are implicit in the CSR layout (slot i owns
        // data[off[i]..off[i+1])), so the record offsets are unused on load;
        // they're stored as 0 for layout symmetry with the blob decoder.
        r[KU_OFF] = 0.0_r;
        r[KV_OFF] = 0.0_r;
        r[CTRL_OFF] = 0.0_r;
        r[W_OFF] = 0.0_r;
        const bool has_w = !e.param.weights.empty();
        r[HAS_W] = has_w ? 1.0_r : 0.0_r;
        h.seg[KNOTS_U].push_back(e.param.knots_u);
        h.seg[KNOTS_V].push_back(e.param.knots_v);
        h.seg[CTRL].push_back(e.param.ctrl);
        h.seg[WEIGHTS].push_back(has_w ? e.param.weights : std::span<const real> {});
        h.seg[TRIM_VERTS].push_back(e.trim.verts);
        h.seg[TRIM_LOOPS].push_back(e.trim.loops);
    }

    [[nodiscard]] static View tie_view(SoAView<const real> soa, const SegmentedView<real>* seg)
    {
        View v {.records = soa};
        if (seg != nullptr) {
            v.seg[KNOTS_U] = seg[KNOTS_U];
            v.seg[KNOTS_V] = seg[KNOTS_V];
            v.seg[CTRL] = seg[CTRL];
            v.seg[WEIGHTS] = seg[WEIGHTS];
            v.seg[TRIM_VERTS] = seg[TRIM_VERTS];
            v.seg[TRIM_LOOPS] = seg[TRIM_LOOPS];
        }
        return v;
    }
};

static_assert(HasEntitySoA<TrimmedEntity<PlaneParam>>);
static_assert(HasEntitySoA<TrimmedEntity<SphereParam>>);
static_assert(HasEntitySoA<TrimmedEntity<CylinderParam>>);
static_assert(HasEntitySoA<TrimmedEntity<Line3Param>>);
static_assert(HasEntitySoA<TrimmedEntity<BSplineSurfaceParam>>);
static_assert(HasEntitySoA<LineRail> && HasEntitySoA<LineRail3>);

/// Host-side tag -> concrete entity TYPE dispatch for the 2D entity set.
///
/// Invokes `f.template operator()<E>()` with the entity type `E` that
/// @ref decode_entity decodes for @p tag. This is the launch-granularity
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
    requires((D == 2) && EntityDispatchFn<F, Free<2>> && EntityDispatchFn<F, LineSeg> &&
             EntityDispatchFn<F, LineRail> && EntityDispatchFn<F, Circle> &&
             EntityDispatchFn<F, Ellipse> && EntityDispatchFn<F, TrimmedEntity<CircleArcParam>> &&
             EntityDispatchFn<F, TrimmedEntity<EllipseArcParam>> &&
             EntityDispatchFn<F, TrimmedEntity<QuadBezierParam>> &&
             EntityDispatchFn<F, TrimmedEntity<CubicBezierParam>> &&
             EntityDispatchFn<F, TrimmedEntity<BSplineCurveParam>> &&
             EntityDispatchFn<F, CompositePath>) ||
            ((D == 3) && EntityDispatchFn<F, Free<3>> && EntityDispatchFn<F, LineRail3> &&
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
        case EntityTag::LineRail: f.template operator()<LineRail>(); break;
        case EntityTag::Circle: f.template operator()<Circle>(); break;
        case EntityTag::Ellipse: f.template operator()<Ellipse>(); break;
        case EntityTag::CircleArc: f.template operator()<TrimmedEntity<CircleArcParam>>(); break;
        case EntityTag::EllipseArc: f.template operator()<TrimmedEntity<EllipseArcParam>>(); break;
        case EntityTag::QuadBezier: f.template operator()<TrimmedEntity<QuadBezierParam>>(); break;
        case EntityTag::CubicBezier:
            f.template operator()<TrimmedEntity<CubicBezierParam>>();
            break;
        case EntityTag::BSpline: f.template operator()<TrimmedEntity<BSplineCurveParam>>(); break;
        case EntityTag::Composite: f.template operator()<CompositePath>(); break;
        case EntityTag::Free:
        case EntityTag::Sphere:  // 3D surfaces: no 2D entity, fall through to Free.
        case EntityTag::Plane:
        case EntityTag::Cylinder:
        case EntityTag::Line3:
        case EntityTag::LineRail3:
        case EntityTag::BSplineSurface:
        default: f.template operator()<Free<2>>(); break;
        }
    } else if constexpr (D == 3) {
        switch (tag) {
        case EntityTag::Plane: f.template operator()<TrimmedEntity<PlaneParam>>(); break;
        case EntityTag::Sphere: f.template operator()<TrimmedEntity<SphereParam>>(); break;
        case EntityTag::Cylinder: f.template operator()<TrimmedEntity<CylinderParam>>(); break;
        case EntityTag::Line3: f.template operator()<TrimmedEntity<Line3Param>>(); break;
        case EntityTag::LineRail3: f.template operator()<LineRail3>(); break;
        case EntityTag::BSplineSurface:
            f.template operator()<TrimmedEntity<BSplineSurfaceParam>>();
            break;
        case EntityTag::Free:
        default: f.template operator()<Free<3>>(); break;
        }
    } else {
        static_assert(D == 2 || D == 3, "dispatch_entity_type: unsupported dimension");
    }
}

/// Fused frame of the nearest segment to @p p. Projects onto every non-nested
/// curve segment and keeps the nearest; each segment is built monomorphically
/// via @ref dispatch_entity_type + @ref decode_entity<E> (no std::visit, no
/// variant). Nested composites and free records are filtered by tag before
/// dispatch (a recursive device call is illegal); the `if constexpr` guard
/// also excludes them at compile time, so those arms instantiate to a no-op.
inline Frame<2, 1> CompositePath::project_frame(const Pt& p) const
{
    Frame<2, 1> best {.pos = p, .basis = {Vec {1.0_r, 0.0_r}}, .eff_tdim = 1};
    real best_d = std::numeric_limits<real>::infinity();
    for (int s = 0; s < n_segs; ++s) {
        const real* rec = recs.data() + (static_cast<std::size_t>(s) * kCompositeRecSize);
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
                const real dd = dot(d, d);
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

/// Project @p p onto the entity @p (tag, params).
///
/// Builds the concrete entity monomorphically via @ref dispatch_entity_type +
/// @ref decode_entity<E> (no `std::visit`, no entity variant). Cold oracle path
/// (the `geometry_project` binding); fixed-size entities only (no arena).
/// @param p Query point.
/// @param tag Entity type tag.
/// @param params Flat parameter blob.
inline Pt project(const Pt& p, Tag tag, const real* params)
{
    Pt out {};
    dispatch_entity_type<2>(static_cast<EntityTag>(tag),
                            [&]<class E>() { out = decode_entity<E>(params).project(p); });
    return out;
}

/// The @f$ (d, 1) @f$ tangent column at @p p on the entity @p (tag, params).
///
/// The first column of the entity tangent basis (the single tangent for a curve;
/// @f$ e_0 @f$ for Free). Built monomorphically via @ref dispatch_entity_type +
/// @ref decode_entity<E> (no `std::visit`, no entity variant).
/// @param p Query point.
/// @param tag Entity type tag.
/// @param params Flat parameter blob.
/// @return The tangent column.
inline Pt tangent_space(const Pt& p, Tag tag, const real* params)
{
    Pt out {};
    dispatch_entity_type<2>(static_cast<EntityTag>(tag),
                            [&]<class E>() { out = decode_entity<E>(params).tangent_basis(p)[0]; });
    return out;
}

/// Load DOF @p i's concrete entity from its tag's SoA slot and invoke
/// @p f(ent). The sweep's single device-hot-path tag→type
/// dispatch.
///
/// Callers pass a *small* callable (tangent-reduced Newton or projection), so
/// the heavy patch + backtracking body that surrounds the call stays
/// entity-agnostic and is compiled exactly once. That sidesteps both measured
/// failure modes of the sweep kernel:
///   - wrapping the whole body in `std::visit` inlines every variant
///     alternative into one kernel → register pressure → GPU occupancy
///     collapse (≈4× slower on the bench);
///   - launching one monomorphic kernel per entity type → launch-count
///     explosion on the in-order queue (the per-partition regression).
/// Here the dispatch is one cheap `switch` over @p tag around a small
/// `EntitySoA<E>::load`.
///
/// @tparam D Embedding dimension.
/// @tparam F Callable invoked as `f(const E&)` with the loaded entity.
/// @param tag     Entity tag of the DOF (selects the concrete type).
/// @param records Packed SoA records view of the DOF's tag partition.
/// @param seg     Segmented (CSR) field views of that partition (null for fixed-size).
/// @param i       Row of the DOF within its tag partition.
/// @param f       The small entity-specific callable.
template <int D, class F>
inline void with_entity(
  EntityTag tag, SoAView<const real> records, const SegmentedView<real>* seg, std::size_t i, F&& f)
{
    dispatch_entity_type<D>(tag, [&]<class E>() {
        if constexpr (HasEntitySoA<E>) {
            f(EntitySoA<E>::load(EntitySoA<E>::tie_view(records, seg), i));
        }
    });
}

}  // namespace egg
