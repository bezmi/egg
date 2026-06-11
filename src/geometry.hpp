#pragma once

#include <array>
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

using Pt = std::array<double, 2>;

inline Pt project_free(const Pt& p, const double*) { return p; }

inline Pt project_lineseg(const Pt& p, const double* params)
{
    const double sx = params[0], sy = params[1];
    const double ex = params[2], ey = params[3];
    const double abx = ex - sx, aby = ey - sy;
    const double ab_sq = abx * abx + aby * aby;
    double t = ((p[0] - sx) * abx + (p[1] - sy) * aby) / std::fmax(ab_sq, 1e-30);
    t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);  // clip to [0, 1]
    return Pt {sx + t * abx, sy + t * aby};
}

inline Pt project_circle(const Pt& p, const double* params)
{
    const double cx = params[0], cy = params[1];
    const double r = params[2];  // params[d] = params[2]
    const double dx = p[0] - cx, dy = p[1] - cy;
    const double dist = std::sqrt(dx * dx + dy * dy);
    if (dist < 1e-15) return Pt {cx + r, cy};  // arbitrary on-circle point
    return Pt {cx + r * dx / dist, cy + r * dy / dist};
}

inline Pt project_ellipse(const Pt& p, const double* params)
{
    const double cx = params[0], cy = params[1];
    const double rx = params[2], ry = params[3];
    const double dx = p[0] - cx, dy = p[1] - cy;
    const double sx = dx / rx, sy = dy / ry;
    const double dist = std::sqrt(sx * sx + sy * sy);
    double ux, uy;
    if (dist < 1e-15) {
        ux = uy = 1.0 / std::sqrt(2.0);  // ones(d)/sqrt(d), d = 2
    } else {
        ux = sx / dist;
        uy = sy / dist;
    }
    return Pt {cx + ux * rx, cy + uy * ry};
}

inline Pt project_plane(const Pt& p, const double* params)
{
    const double qx = params[0], qy = params[1];
    double nx = params[2], ny = params[3];
    const double nn = std::sqrt(nx * nx + ny * ny);
    nx /= nn;
    ny /= nn;
    const double dotp = (p[0] - qx) * nx + (p[1] - qy) * ny;
    return Pt {p[0] - dotp * nx, p[1] - dotp * ny};
}

// Project p onto the entity (type_tag, params). In-kernel switch dispatch.
inline Pt project(const Pt& p, Tag tag, const double* params)
{
    switch (tag) {
    case TAG_LINESEG: return project_lineseg(p, params);
    case TAG_CIRCLE: return project_circle(p, params);
    case TAG_ELLIPSE: return project_ellipse(p, params);
    case TAG_SPHERE: return project_circle(p, params);  // identical to circle
    case TAG_PLANE: return project_plane(p, params);
    case TAG_FREE:
    default: return p;
    }
}

inline Pt tangent_free(const Pt&, const double*) { return Pt {1.0, 0.0}; }

inline Pt tangent_lineseg(const Pt&, const double* params)
{
    const double abx = params[2] - params[0];
    const double aby = params[3] - params[1];
    const double norm = std::sqrt(abx * abx + aby * aby);
    if (norm < 1e-15) return Pt {1.0, 0.0};  // eye[:, 0]
    return Pt {abx / norm, aby / norm};
}

inline Pt tangent_circle(const Pt& p, const double* params)
{
    const double cx = params[0], cy = params[1];
    const double dx = p[0] - cx, dy = p[1] - cy;
    const double rn = std::sqrt(dx * dx + dy * dy);
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
    const double norm = std::sqrt(tx * tx + ty * ty);
    if (norm < 1e-15) return Pt {1.0, 0.0};
    return Pt {tx / norm, ty / norm};
}

// (d, 1) tangent column at p on the entity (type_tag, params).
inline Pt tangent_space(const Pt& p, Tag tag, const double* params)
{
    switch (tag) {
    case TAG_LINESEG: return tangent_lineseg(p, params);
    case TAG_CIRCLE: return tangent_circle(p, params);
    case TAG_ELLIPSE: return tangent_ellipse(p, params);
    case TAG_FREE:
    case TAG_SPHERE:
    case TAG_PLANE:
    default: return tangent_free(p, params);
    }
}

template <class E>
concept GeometryEntity = requires(const E& e, const Pt& p) {
    { e.project(p) } -> std::convertible_to<Pt>;
    { e.tangent(p) } -> std::convertible_to<Pt>;
};

struct Free {
    const double* params;
    Pt project(const Pt& p) const { return project_free(p, params); }
    Pt tangent(const Pt& p) const { return tangent_free(p, params); }
};
struct LineSeg {
    const double* params;
    Pt project(const Pt& p) const { return project_lineseg(p, params); }
    Pt tangent(const Pt& p) const { return tangent_lineseg(p, params); }
};
struct Circle {
    const double* params;
    Pt project(const Pt& p) const { return project_circle(p, params); }
    Pt tangent(const Pt& p) const { return tangent_circle(p, params); }
};
struct Ellipse {
    const double* params;
    Pt project(const Pt& p) const { return project_ellipse(p, params); }
    Pt tangent(const Pt& p) const { return tangent_ellipse(p, params); }
};

// placeholders
struct Sphere {
    const double* params;
    Pt project(const Pt& p) const { return project_circle(p, params); }
    Pt tangent(const Pt& p) const { return tangent_free(p, params); }
};
struct Plane {
    const double* params;
    Pt project(const Pt& p) const { return project_plane(p, params); }
    Pt tangent(const Pt& p) const { return tangent_free(p, params); }
};

static_assert(GeometryEntity<Free> && GeometryEntity<LineSeg> && GeometryEntity<Circle> &&
              GeometryEntity<Ellipse> && GeometryEntity<Sphere> && GeometryEntity<Plane>);

using EntityKind = std::variant<Free, LineSeg, Circle, Ellipse, Sphere, Plane>;

// Build the entity kind for (tag, params). Mirrors the in-kernel switch; for
// host-side std::visit over the closed set.
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

}  // namespace egg
