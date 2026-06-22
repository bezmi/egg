"""2D geometry construction front-end mirroring the gdtk ``geom`` API.

This module provides gdtk-style constructors (:class:`Vector3`, :class:`Line`,
:class:`Arc`, :class:`Bezier`, :class:`Polyline`, :class:`Spline`) that return
:mod:`egg.geometry` entities directly, so geometry can be authored without a
gdtk dependency. The mapping mirrors the former gdtk adapter:

- :class:`Line`           -> :class:`~egg.geometry.analytic2d.LineSegment`
  (subclass retaining the original ``p0`` / ``p1`` as :class:`Vector3`)
- :class:`Arc`            -> :class:`~egg.geometry.curves2d.CircleArc`
- :class:`Bezier`         -> :class:`~egg.geometry.curves2d.QuadBezier` /
  :class:`~egg.geometry.curves2d.CubicBezier` for degree 2/3 (degree 1 is a
  :class:`~egg.geometry.analytic2d.LineSegment`); any other degree becomes a
  :class:`~egg.geometry.curves2d.BSplineCurve` (a degree-n Bézier is a B-spline
  on the clamped knot vector ``{0 x (n+1), 1 x (n+1)}``).
- :class:`Polyline`       -> :class:`~egg.geometry.curves2d.CompositePath`
  (subclass; when ``closed=True`` a closing segment is appended if needed)
- :class:`Spline`         -> :class:`~egg.geometry.curves2d.CompositePath` of
  :class:`~egg.geometry.curves2d.CubicBezier` segments obtained from a natural
  cubic spline fit through the input points. The closed variant appends the
  first point to the end and solves an open natural spline (matching gdtk's
  algorithm), rather than using periodic boundary conditions.

:class:`Vector3` supports the vector arithmetic (``+``, ``-``, scalar ``*``,
negation, ``abs``) needed for gdtk-style point construction.

The constructed entities are the same ones :mod:`egg.geometry` already
understands, so they can be fed directly into the topology builder and the C++
encoding path without any conversion step.

Limitation (inherited from the C++ ``atan2`` inverse): an :class:`Arc` is
converted to absolute angles in ``(-pi, pi]``; arcs whose angular interval
cannot be represented without wrapping past +/-pi (after a possible direction
flip) are rejected.
"""

from __future__ import annotations

import numpy as np

from .analytic2d import LineSegment
from .curves2d import (
    BSplineCurve,
    CircleArc,
    CompositePath,
    CubicBezier,
    QuadBezier,
)

__all__ = [
    "Vector3",
    "Line",
    "Arc",
    "Bezier",
    "Polyline",
    "Spline",
]


class Vector3:
    """A 2D point with ``.x`` / ``.y`` / ``.z`` attributes and vector arithmetic.

    Supports addition, subtraction, scalar multiplication, negation, and
    magnitude (``abs``), matching the gdtk ``Vector3`` interface for 2D point
    construction. The geometry is planar: ``z`` must be zero (defaults to 0.0).
    """

    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        if abs(float(z)) > 1e-12:
            raise ValueError(f"Vector3 z must be 0 for 2D geometry, got {z}")
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self) -> str:
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __add__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x + other.x, self.y + other.y)
        o = _vec2(other)
        return Vector3(self.x + o[0], self.y + o[1])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x - other.x, self.y - other.y)
        o = _vec2(other)
        return Vector3(self.x - o[0], self.y - o[1])

    def __rsub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(other.x - self.x, other.y - self.y)
        o = _vec2(other)
        return Vector3(o[0] - self.x, o[1] - self.y)

    def __mul__(self, scalar):
        return Vector3(self.x * float(scalar), self.y * float(scalar))

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __neg__(self):
        return Vector3(-self.x, -self.y)

    def __abs__(self):
        return float(np.hypot(self.x, self.y))


def _vec2(v) -> np.ndarray:
    """Coerce ``v`` (Vector3 / tuple / list / array) to a 2D numpy point.

    Vector3 instances must lie in the z=0 plane (checked at construction).
    """
    if isinstance(v, Vector3):
        return np.array([v.x, v.y])
    arr = np.asarray(v, dtype=float).ravel()
    if arr.shape[0] < 2:
        raise ValueError(f"point {v} needs at least 2 coordinates")
    if arr.shape[0] >= 3 and abs(arr[2]) > 1e-12:
        raise ValueError(f"point {v} does not lie in the z=0 plane")
    return arr[:2]


class Line(LineSegment):
    """A straight segment from ``p0`` to ``p1``.

    A subclass of :class:`~egg.geometry.analytic2d.LineSegment` that also
    retains the original ``p0`` / ``p1`` constructor arguments as
    :class:`Vector3` objects, so call sites can read ``line.p0.x`` etc. like a
    gdtk ``Line``.  Passes ``isinstance(..., LineSegment)`` checks for
    encoding and projection.
    """

    def __init__(self, p0, p1):
        self.p0 = p0 if isinstance(p0, Vector3) else Vector3(*_vec2(p0))
        self.p1 = p1 if isinstance(p1, Vector3) else Vector3(*_vec2(p1))
        super().__init__(_vec2(p0), _vec2(p1))


class Arc:
    """A circular arc through endpoints ``a``, ``b`` about centre ``c``.

    Returns a :class:`~egg.geometry.curves2d.CircleArc`. The signed sweep from
    ``a`` to ``b`` (about ``c``) picks the arc direction. As with the former
    gdtk adapter, the C++ inverse returns angles in ``(-pi, pi]``, so the
    angular range is rejected if it cannot be represented in that branch (after
    a possible antipodal flip of the start angle).
    """

    def __new__(cls, a, b, c) -> CircleArc:
        center = _vec2(c)
        pa = _vec2(a)
        pb = _vec2(b)
        ra = float(np.linalg.norm(pa - center))
        rb = float(np.linalg.norm(pb - center))
        if abs(ra - rb) > 1e-9 * max(ra, 1.0):
            raise ValueError("Arc radii do not match")
        ta = float(np.arctan2(pa[1] - center[1], pa[0] - center[0]))
        da, db = pa - center, pb - center
        sweep = float(np.arctan2(da[0] * db[1] - da[1] * db[0], da @ db))
        t0, t1 = (ta, ta + sweep) if sweep >= 0.0 else (ta + sweep, ta)
        if t1 > np.pi or t0 <= -np.pi:
            ta2 = float(ta - 2.0 * np.pi * np.sign(ta))
            t0, t1 = (ta2, ta2 + sweep) if sweep >= 0.0 else (ta2 + sweep, ta2)
            if t1 > np.pi or t0 <= -np.pi:
                raise ValueError(
                    "Arc angular range wraps past the +/-pi branch cut; split it")
        return CircleArc(center, ra, float(t0), float(t1))


class Bezier:
    """A Bézier curve over the given control points.

    Returns a :class:`~egg.geometry.analytic2d.LineSegment` (degree 1),
    :class:`~egg.geometry.curves2d.QuadBezier` (degree 2),
    :class:`~egg.geometry.curves2d.CubicBezier` (degree 3), or
    :class:`~egg.geometry.curves2d.BSplineCurve` (degree >= 4 — a degree-n
    Bézier is a B-spline on the clamped knot vector).
    """

    def __new__(cls, points) -> LineSegment | QuadBezier | CubicBezier | BSplineCurve:
        pts = [_vec2(p) for p in points]
        n = len(pts) - 1  # degree
        if n < 1:
            raise ValueError("Bezier needs at least two control points")
        if n == 1:
            return LineSegment(pts[0], pts[1])
        if n == 2:
            return QuadBezier(pts[0], pts[1], pts[2])
        if n == 3:
            return CubicBezier(pts[0], pts[1], pts[2], pts[3])
        knots = np.concatenate([np.zeros(n + 1), np.ones(n + 1)])
        return BSplineCurve(n, knots, np.stack(pts))


def _seg_start(seg) -> np.ndarray:
    """Start point of any egg curve segment as a 2D numpy array."""
    if hasattr(seg, "start"):
        return np.asarray(seg.start, dtype=float)
    return np.asarray(seg.eval(seg.t0), dtype=float)


def _seg_end(seg) -> np.ndarray:
    """End point of any egg curve segment as a 2D numpy array."""
    if hasattr(seg, "end"):
        return np.asarray(seg.end, dtype=float)
    return np.asarray(seg.eval(seg.t1), dtype=float)


class Polyline(CompositePath):
    """An ordered sequence of egg curve segments.

    A subclass of :class:`~egg.geometry.curves2d.CompositePath`. When
    ``closed=True`` and the end of the last segment does not coincide with the
    start of the first segment (within ``tolerance``), a closing
    :class:`LineSegment` is appended — matching gdtk's ``Polyline`` behaviour.
    Nested composites/polylines are rejected by ``CompositePath`` itself.
    """

    def __init__(self, segments, closed=False, tolerance=1.0e-10):
        segments = list(segments)
        if closed:
            start = _seg_start(segments[0])
            end = _seg_end(segments[-1])
            if np.linalg.norm(end - start) > tolerance:
                segments.append(LineSegment(end, start))
        super().__init__(segments)


def _natural_cubic_second_derivatives(points: np.ndarray) -> np.ndarray:
    """Solve for the per-knot second derivatives of a natural cubic spline.

    Uniform parameterisation (knot spacing ``h = 1``). Natural end conditions
    ``M_0 = M_{n-1} = 0`` are imposed.

    Parameters
    ----------
    points : (N, 2) array of through-points.

    Returns
    -------
    (N, 2) array of second derivatives ``M_i``. ``M_0`` and ``M_{N-1}`` are
    zero.
    """
    n = points.shape[0]
    if n < 3:
        return np.zeros((n, 2))
    rhs = 6.0 * (points[2:] - 2.0 * points[1:-1] + points[:-2])
    A = 4.0 * np.eye(n - 2)
    for i in range(n - 3):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    inner = np.linalg.solve(A, rhs)
    M = np.zeros((n, 2))
    M[1:-1] = inner
    return M


def _spline_to_beziers(points: np.ndarray) -> list[CubicBezier]:
    """Convert a natural cubic spline through ``points`` into cubic Béziers.

    Uses the standard second-derivative → Bézier control-point relation
    (uniform knot spacing ``h = 1``):

    - ``B0 = P_i``
    - ``B1 = P_i + (P_{i+1} - P_i)/3 - (2 M_i + M_{i+1}) / 18``
    - ``B2 = P_{i+1} - (P_{i+1} - P_i)/3 - (M_i + 2 M_{i+1}) / 18``
    - ``B3 = P_{i+1}``
    """
    M = _natural_cubic_second_derivatives(points)
    n = points.shape[0]
    segs: list[CubicBezier] = []
    for i in range(n - 1):
        p_i = points[i]
        p_j = points[i + 1]
        b0 = p_i
        b1 = p_i + (p_j - p_i) / 3.0 - (2.0 * M[i] + M[i + 1]) / 18.0
        b2 = p_j - (p_j - p_i) / 3.0 - (M[i] + 2.0 * M[i + 1]) / 18.0
        b3 = p_j
        segs.append(CubicBezier(b0, b1, b2, b3))
    return segs


class Spline:
    """A natural cubic spline through the given points.

    Returns a :class:`~egg.geometry.curves2d.CompositePath` of
    :class:`~egg.geometry.curves2d.CubicBezier` segments (one per interval).
    The spline uses uniform parameterisation with natural end conditions
    (zero second derivative at the ends).

    For ``closed=True``, the first point is appended to the end (if it does not
    already coincide with the last) and an open natural spline is solved through
    the extended point list — matching gdtk's algorithm.  This means the
    wrap-around joint is C0 but not C1 (the second derivative is zero at both
    ends of the open spline, not periodic).
    """

    def __new__(cls, points, closed=False, tolerance=1.0e-10) -> CompositePath:
        pts = np.stack([_vec2(p) for p in points])
        if pts.shape[0] < 2:
            raise ValueError("Spline needs at least two points")
        if closed and np.linalg.norm(pts[0] - pts[-1]) > tolerance:
            pts = np.vstack([pts, pts[0:1]])
        segs = _spline_to_beziers(pts)
        return CompositePath(segs)
