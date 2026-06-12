#pragma once

#include "core.hpp"

#include <cmath>
#include <concepts>
#include <variant>

namespace egg
{

using Tag = int;
inline constexpr Tag TAG_FREE = 0;
inline constexpr Tag TAG_LINESEG = 1;
inline constexpr Tag TAG_CIRCLE = 2;
inline constexpr Tag TAG_ELLIPSE = 3;
inline constexpr Tag TAG_SPHERE = 4;
inline constexpr Tag TAG_PLANE = 5;

inline constexpr int kParamPad = 12;

// Pt is the D=2 coordinate type (PtN<2>). The projection / tangent bodies below
// read p[0], p[1] and the 2D analytic shapes, so the entity set here is the 2D
// one (Sphere/Plane are 2D placeholders); a 3D core supplies make_entity<3> with
// the genuine surface projections and its own (Sphere/Plane) closed set.

inline Pt project_free(const Pt& p, const double*) { return p; }

inline Pt project_lineseg(const Pt& p, const double* params)
{
    const double sx = params[0], sy = params[1];
    const double ex = params[2], ey = params[3];
    const double abx = ex - sx, aby = ey - sy;
    const double ab_sq = (abx * abx) + (aby * aby);
    double t = (((p[0] - sx) * abx) + ((p[1] - sy) * aby)) / std::fmax(ab_sq, 1e-30);
    t = t < 0.0 ? 0.0 : t;
    t = t > 1.0 ? 1.0 : t;  // clip to [0, 1]
    return Pt {sx + (t * abx), sy + (t * aby)};
}

inline Pt project_circle(const Pt& p, const double* params)
{
    const double cx = params[0], cy = params[1];
    const double r = params[2];  // params[d] = params[2]
    const double dx = p[0] - cx, dy = p[1] - cy;
    const double dist = std::sqrt((dx * dx) + (dy * dy));
    if (dist < 1e-15) {
        return Pt {cx + r, cy};  // arbitrary on-circle point
    }
    return Pt {cx + (r * dx / dist), cy + (r * dy / dist)};
}

inline Pt project_ellipse(const Pt& p, const double* params)
{
    const double cx = params[0], cy = params[1];
    const double rx = params[2], ry = params[3];
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

inline Pt project_plane(const Pt& p, const double* params)
{
    const double qx = params[0], qy = params[1];
    double nx = params[2], ny = params[3];
    const double nn = std::sqrt((nx * nx) + (ny * ny));
    nx /= nn;
    ny /= nn;
    const double dotp = ((p[0] - qx) * nx) + ((p[1] - qy) * ny);
    return Pt {p[0] - (dotp * nx), p[1] - (dotp * ny)};
}

inline Pt tangent_free(const Pt&, const double*) { return Pt {1.0, 0.0}; }

inline Pt tangent_lineseg(const Pt&, const double* params)
{
    const double abx = params[2] - params[0];
    const double aby = params[3] - params[1];
    const double norm = std::sqrt((abx * abx) + (aby * aby));
    if (norm < 1e-15) {
        return Pt {1.0, 0.0};  // eye[:, 0]
    }
    return Pt {abx / norm, aby / norm};
}

inline Pt tangent_circle(const Pt& p, const double* params)
{
    const double cx = params[0], cy = params[1];
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
    return Pt {-ny, nx};
}

inline Pt tangent_ellipse(const Pt& p, const double* params)
{
    const double cx = params[0], cy = params[1];
    const double rx = params[2], ry = params[3];
    const double dx = p[0] - cx, dy = p[1] - cy;
    // Parametric angle from the radial-scaled coordinates (matches Ellipse).
    const double angle = std::atan2(dy / ry, dx / rx);
    double tx = -rx * std::sin(angle);
    double ty = ry * std::cos(angle);
    const double norm = std::sqrt((tx * tx) + (ty * ty));
    if (norm < 1e-15) return Pt {1.0, 0.0};
    return Pt {tx / norm, ty / norm};
}

// ---------------------------------------------------------------------------
// Concept-modelled geometry entities (the single source of truth used by the
// kernels). The per-shape free functions above hold the math once; each entity
// struct is a thin, trivially-copyable view over its `params` that satisfies the
// GeometryEntity concept. The closed set lives in an EntityKind variant; the
// in-kernel dispatch is a per-DOF std::visit over make_entity (geometry is
// per-DOF, so — unlike the run-once Objective visit — the variant is rebuilt and
// visited inside the loop, but the projection/tangent bodies are monomorphic).
// ---------------------------------------------------------------------------
template <class E>
concept GeometryEntity = requires(const E& e, const Pt& p) {
    { e.project(p) } -> std::convertible_to<Pt>;
    { e.tangent(p) } -> std::convertible_to<Pt>;
};

struct Free {
    const double* params;
    [[nodiscard]] Pt project(const Pt& p) const { return project_free(p, params); }
    [[nodiscard]] Pt tangent(const Pt& p) const { return tangent_free(p, params); }
};
struct LineSeg {
    const double* params;
    [[nodiscard]] Pt project(const Pt& p) const { return project_lineseg(p, params); }
    [[nodiscard]] Pt tangent(const Pt& p) const { return tangent_lineseg(p, params); }
};
struct Circle {
    const double* params;
    [[nodiscard]] Pt project(const Pt& p) const { return project_circle(p, params); }
    [[nodiscard]] Pt tangent(const Pt& p) const { return tangent_circle(p, params); }
};
struct Ellipse {
    const double* params;
    [[nodiscard]] Pt project(const Pt& p) const { return project_ellipse(p, params); }
    [[nodiscard]] Pt tangent(const Pt& p) const { return tangent_ellipse(p, params); }
};

// Sphere reuses the circle projection (radial); its tangent basis is a free
// surface direction. Plane projects onto the plane through (q, n).
struct Sphere {
    const double* params;
    [[nodiscard]] Pt project(const Pt& p) const { return project_circle(p, params); }
    [[nodiscard]] Pt tangent(const Pt& p) const { return tangent_free(p, params); }
};
struct Plane {
    const double* params;
    [[nodiscard]] Pt project(const Pt& p) const { return project_plane(p, params); }
    [[nodiscard]] Pt tangent(const Pt& p) const { return tangent_free(p, params); }
};

static_assert(GeometryEntity<Free> && GeometryEntity<LineSeg> && GeometryEntity<Circle> &&
              GeometryEntity<Ellipse> && GeometryEntity<Sphere> && GeometryEntity<Plane>);

using EntityKind = std::variant<Free, LineSeg, Circle, Ellipse, Sphere, Plane>;

// EntityKindT<D> is the per-dimension closed set; the entity *set differs by D*
// (Circle/LineSeg/Ellipse at 2; Sphere/Plane at 3). Phase 1 supplies only the
// D=2 set, so make_entity<D> is gated until Phase 2 adds make_entity<3>.
template <int D> using EntityKindT = EntityKind;

// The single dispatch point: build the concept-modelled entity for (tag, params).
inline EntityKind make_entity(Tag tag, const double* params)
{
    switch (tag) {
    case TAG_LINESEG: return LineSeg {params};
    case TAG_CIRCLE: return Circle {params};
    case TAG_ELLIPSE: return Ellipse {params};
    case TAG_SPHERE: return Sphere {params};
    case TAG_PLANE: return Plane {params};
    case TAG_FREE:
    default: return Free {params};
    }
}

// Dimension-templated dispatch. make_entity<2> returns exactly today's variant;
// other dimensions are gated until their entity set + projections exist.
template <int D = kDefaultDim> inline EntityKindT<D> make_entity(Tag tag, const double* params)
{
    static_assert(D == 2, "make_entity: only the 2D entity set is implemented");
    return make_entity(tag, params);
}

// Project p onto the entity (type_tag, params), dispatched through the concept
// entities via std::visit — monomorphic per entity type, no virtuals.
inline Pt project(const Pt& p, Tag tag, const double* params)
{
    return std::visit([&](const auto& e) { return e.project(p); }, make_entity(tag, params));
}

// (d, 1) tangent column at p on the entity (type_tag, params).
inline Pt tangent_space(const Pt& p, Tag tag, const double* params)
{
    return std::visit([&](const auto& e) { return e.tangent(p); }, make_entity(tag, params));
}

}  // namespace egg
