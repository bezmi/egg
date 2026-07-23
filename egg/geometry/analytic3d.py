# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""3D analytic geometry primitives: Plane, Sphere, Cylinder, Line3.

These mirror the C++ parametrizations in ``src/geometry.hpp``
(``PlaneParam``, ``SphereParam``, ``CylinderParam``, ``Line3Param``) as thin
constructors; projection and tangent bases come from the C++ core itself via
the :class:`~egg.geometry.base.GeometryEntity` base methods, so the
parametrizations are implemented exactly once.
"""

import numpy as np

from .base import GeometryEntity

__all__ = ["Plane", "Sphere", "Cylinder", "Line3"]


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0])
    return v / n


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
