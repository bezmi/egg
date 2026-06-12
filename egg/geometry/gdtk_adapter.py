"""Adapter from gdtk ``geom.path`` objects to egg geometry entities.

The gdtk package (Gas Dynamics Toolkit) is point-and-path based: build
``Vector3`` points and join them into ``Path`` objects (``Line``, ``Arc``,
``Bezier``, ``Polyline``, ``Spline``). :func:`from_gdtk` converts such a path
(z is ignored; the geometry must lie in the z=0 plane) into the matching egg
entity, which ``encode_entity`` can then upload to the C++ core:

- ``Line`` -> :class:`~egg.geometry.analytic2d.LineSegment`
- ``Arc`` (start a, end b, centre c) -> :class:`~egg.geometry.curves2d.CircleArc`
- ``Bezier`` -> :class:`~egg.geometry.curves2d.QuadBezier` /
  :class:`~egg.geometry.curves2d.CubicBezier` for degree 2/3; any other degree
  becomes a :class:`~egg.geometry.curves2d.BSplineCurve` (a degree-n Bézier is
  a B-spline on the clamped knot vector ``{0 x (n+1), 1 x (n+1)}``)
- ``Polyline`` / ``Spline`` -> :class:`~egg.geometry.curves2d.CompositePath`
  over the converted segments

Limitation: an ``Arc`` is converted to absolute angles in ``(-pi, pi]``; arcs
whose angular interval cannot be represented without wrapping past +/-pi (after
a possible direction flip) are rejected, since the C++ inverse returns angles
in that branch.
"""

from __future__ import annotations

import numpy as np

from .analytic2d import LineSegment
from .curves2d import BSplineCurve, CircleArc, CompositePath, CubicBezier, QuadBezier

__all__ = ["from_gdtk"]


def _vec2(v) -> np.ndarray:
    """A gdtk Vector3 as a 2D numpy point (z must be ~0)."""
    if abs(float(v.z)) > 1e-12:
        raise ValueError(f"gdtk point {v} does not lie in the z=0 plane")
    return np.array([float(v.x), float(v.y)])


def _arc_to_circle_arc(arc) -> CircleArc:
    c = _vec2(arc.c)
    a = _vec2(arc.a)
    b = _vec2(arc.b)
    ra, rb = np.linalg.norm(a - c), np.linalg.norm(b - c)
    if abs(ra - rb) > 1e-9 * max(ra, 1.0):
        raise ValueError("Arc radii do not match")
    ta = np.arctan2(a[1] - c[1], a[0] - c[0])
    # Signed sweep from a to b (the same local-frame angle gdtk uses).
    da, db = a - c, b - c
    sweep = np.arctan2(da[0] * db[1] - da[1] * db[0], da @ db)
    t0, t1 = (ta, ta + sweep) if sweep >= 0.0 else (ta + sweep, ta)
    if t1 > np.pi or t0 <= -np.pi:
        # Try the antipodal branch of ta.
        ta2 = ta - 2 * np.pi * np.sign(ta)
        t0, t1 = (ta2, ta2 + sweep) if sweep >= 0.0 else (ta2 + sweep, ta2)
        if t1 > np.pi or t0 <= -np.pi:
            raise ValueError(
                "Arc angular range wraps past the +/-pi branch cut; split it")
    return CircleArc(c, ra, t0, t1)


def _bezier_to_entity(bez):
    pts = [_vec2(p) for p in bez.B]
    n = len(pts) - 1  # degree
    if n == 1:
        return LineSegment(pts[0], pts[1])
    if n == 2:
        return QuadBezier(*pts)
    if n == 3:
        return CubicBezier(*pts)
    # Degree-n Bézier == B-spline on the clamped knot vector.
    knots = np.concatenate([np.zeros(n + 1), np.ones(n + 1)])
    return BSplineCurve(n, knots, np.stack(pts))


def from_gdtk(path):
    """Convert a gdtk ``geom.path.Path`` into an egg geometry entity."""
    # Imported lazily and matched by class name chain so that subclasses
    # (e.g. Spline is a Polyline) resolve to the right conversion.
    from gdtk.geom.path import Arc, Bezier, Line, Polyline

    if isinstance(path, Line):
        return LineSegment(_vec2(path.p0), _vec2(path.p1))
    if isinstance(path, Arc):
        return _arc_to_circle_arc(path)
    if isinstance(path, Bezier):
        return _bezier_to_entity(path)
    if isinstance(path, Polyline):  # includes Spline
        return CompositePath([from_gdtk(s) for s in path.segments])
    raise NotImplementedError(f"gdtk path type {type(path).__name__} not supported")
