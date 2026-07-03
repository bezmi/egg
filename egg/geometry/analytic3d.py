"""3D analytic geometry primitives: Plane, Sphere, Cylinder, Line3.

These mirror the C++ parametrizations in ``src/geometry.hpp``
(``PlaneParam``, ``SphereParam``, ``CylinderParam``, ``Line3Param``): each is
a parametrization restricted to an (untrimmed for surfaces) trim region.
Projection inverts the parametrization, clamps to the trim, and evaluates;
the tangent space is the orthonormalized frame columns (Gram–Schmidt, as in
the C++ ``orthonormalize``); the normal is the codimension-1 outward vector.
"""

import numpy as np

from .base import GeometryEntity

__all__ = ["Plane", "Sphere", "Cylinder", "Line3"]


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def _orthonormalize2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gram–Schmidt on two 3-vectors → orthonormal 3×2 columns (mirrors C++ orthonormalize<D,2>)."""
    a = _normalize(a)
    b = b - np.dot(b, a) * a
    b = _normalize(b)
    return np.column_stack([a, b])


class Plane(GeometryEntity):
    """A plane through origin ``o`` with orthonormal in-plane axes ``ax``, ``ay``.

    A 2D surface (dim == 2). Projection drops ``p`` onto the plane;
    the tangent space is ``{ax, ay}``; the normal is ``ax × ay``.
    """

    def __init__(self, origin, ax, ay):
        self.o = np.asarray(origin, dtype=float)
        self.ax = _normalize(np.asarray(ax, dtype=float))
        self.ay = np.asarray(ay, dtype=float)
        # Re-orthogonalize ay against ax (mirrors the C++ orthonormalize at load).
        self.ay = _normalize(self.ay - np.dot(self.ay, self.ax) * self.ax)
        self.az = np.cross(self.ax, self.ay)

    @property
    def dim(self) -> int:
        return 2

    def project(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        d = p - self.o
        return self.o + np.dot(d, self.ax) * self.ax + np.dot(d, self.ay) * self.ay

    def tangent_space(self, q: np.ndarray) -> np.ndarray:
        return np.column_stack([self.ax, self.ay])

    def normal(self, q: np.ndarray) -> np.ndarray:
        return self.az


class Sphere(GeometryEntity):
    """A sphere of radius ``r`` centred at ``c`` with orthonormal frame ``(ax, ay, az)``.

    A 2D surface (dim == 2); chart ``(u, v)`` = (azimuth about ``ax`` toward
    ``ay``, latitude from the ``ax``-``ay`` plane). Projection is radial;
    the tangent space is the orthonormalized ``{S_u, S_v}`` columns; the
    normal is the outward radial.
    """

    def __init__(self, center, radius: float, ax, ay):
        self.c = np.asarray(center, dtype=float)
        self.r = float(radius)
        self.ax = _normalize(np.asarray(ax, dtype=float))
        self.ay = np.asarray(ay, dtype=float)
        self.ay = _normalize(self.ay - np.dot(self.ay, self.ax) * self.ax)
        self.az = np.cross(self.ax, self.ay)

    @property
    def dim(self) -> int:
        return 2

    def _invert(self, p: np.ndarray) -> tuple[float, float]:
        m = _normalize(np.asarray(p, dtype=float) - self.c)
        u = float(np.arctan2(np.dot(m, self.ay), np.dot(m, self.ax)))
        v = float(np.arcsin(np.clip(np.dot(m, self.az), -1.0, 1.0)))
        return u, v

    def _eval(self, u: float, v: float) -> np.ndarray:
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        return self.c + self.r * (cv * cu * self.ax + cv * su * self.ay + sv * self.az)

    def _frame(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v), np.sin(v)
        su_col = self.r * (cv * -su * self.ax + cv * cu * self.ay)
        sv_col = self.r * (-sv * cu * self.ax - sv * su * self.ay + cv * self.az)
        return su_col, sv_col

    def project(self, p: np.ndarray) -> np.ndarray:
        u, v = self._invert(p)
        return self._eval(u, v)

    def tangent_space(self, q: np.ndarray) -> np.ndarray:
        u, v = self._invert(q)
        su, sv = self._frame(u, v)
        return _orthonormalize2(su, sv)

    def normal(self, q: np.ndarray) -> np.ndarray:
        return _normalize(np.asarray(q, dtype=float) - self.c)


class Cylinder(GeometryEntity):
    """A right circular cylinder: axis ``az`` through ``o``, radius ``r``.

    A 2D surface (dim == 2); chart ``(u, v)`` = (angle about the axis, height
    along it). Projection is radial onto the cylindrical surface;
    the tangent space is the orthonormalized ``{S_u, S_v}`` columns; the
    normal is the outward radial in the cross-section.
    """

    def __init__(self, origin, ax, ay, radius: float):
        self.o = np.asarray(origin, dtype=float)
        self.r = float(radius)
        self.ax = _normalize(np.asarray(ax, dtype=float))
        self.ay = np.asarray(ay, dtype=float)
        self.ay = _normalize(self.ay - np.dot(self.ay, self.ax) * self.ax)
        self.az = np.cross(self.ax, self.ay)

    @property
    def dim(self) -> int:
        return 2

    def _invert(self, p: np.ndarray) -> tuple[float, float]:
        d = np.asarray(p, dtype=float) - self.o
        u = float(np.arctan2(np.dot(d, self.ay), np.dot(d, self.ax)))
        v = float(np.dot(d, self.az))
        return u, v

    def _eval(self, u: float, v: float) -> np.ndarray:
        return (
            self.o + self.r * (np.cos(u) * self.ax + np.sin(u) * self.ay) + v * self.az
        )

    def _frame(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        su = self.r * (-np.sin(u) * self.ax + np.cos(u) * self.ay)
        sv = self.az
        return su, sv

    def project(self, p: np.ndarray) -> np.ndarray:
        u, v = self._invert(p)
        return self._eval(u, v)

    def tangent_space(self, q: np.ndarray) -> np.ndarray:
        u, v = self._invert(q)
        su, sv = self._frame(u, v)
        return _orthonormalize2(su, sv)

    def normal(self, q: np.ndarray) -> np.ndarray:
        d = np.asarray(q, dtype=float) - self.o
        radial = d - np.dot(d, self.az) * self.az
        return _normalize(radial)


class Line3(GeometryEntity):
    """A 3D line segment from ``p0`` to ``p1`` (an edge curve, dim == 1).

    Projection is foot-of-perpendicular, clamped to ``[t0, t1]``; the tangent
    space is the unit ``(p1 - p0)`` direction.
    """

    def __init__(self, p0, p1, t0: float = 0.0, t1: float = 1.0):
        self.p0 = np.asarray(p0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)
        self.t0 = float(t0)
        self.t1 = float(t1)

    @property
    def dim(self) -> int:
        return 1

    def _invert(self, q: np.ndarray) -> float:
        ab = self.p1 - self.p0
        ab_sq = float(np.dot(ab, ab))
        if ab_sq < 1e-30:
            return 0.0
        return float(np.dot(np.asarray(q, dtype=float) - self.p0, ab) / ab_sq)

    def _clamp(self, t: float) -> float:
        return float(np.clip(t, self.t0, self.t1))

    def project(self, p: np.ndarray) -> np.ndarray:
        t = self._clamp(self._invert(p))
        return self.p0 + t * (self.p1 - self.p0)

    def tangent_space(self, q: np.ndarray) -> np.ndarray:
        ab = self.p1 - self.p0
        n = np.linalg.norm(ab)
        if n < 1e-15:
            return np.array([[1.0], [0.0], [0.0]])
        return (ab / n).reshape(3, 1)
