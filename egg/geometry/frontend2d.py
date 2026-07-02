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
    "Edge",
    "Node",
]


class Vector3:
    """A 2D point with ``.x`` / ``.y`` / ``.z`` attributes and vector arithmetic.

    Supports addition, subtraction, scalar multiplication, negation, and
    magnitude (``abs``), matching the gdtk ``Vector3`` interface for 2D point
    construction. The geometry is planar: ``z`` must be zero (defaults to 0.0).

    ``fixed=True`` marks the point as a pinned grid corner when used as a
    topology corner position; results of arithmetic are never fixed.
    """

    __slots__ = ("x", "y", "z", "fixed")

    def __init__(self, x, y, z=0.0, *, fixed=False):
        if abs(float(z)) > 1e-12:
            raise ValueError(f"Vector3 z must be 0 for 2D geometry, got {z}")
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.fixed = bool(fixed)

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
    """A circular arc from ``p0`` to ``p1`` about ``centre``.

    Keyword names match the gdtk/Eilmer ``Arc:new{p0=, p1=, centre=}`` form.
    Returns a :class:`~egg.geometry.curves2d.CircleArc` whose parameter runs
    from the angle of ``p0`` to the angle of ``p1`` (``t1 < t0`` for a
    clockwise sweep), so ``eval(t0) == p0`` and composites traverse the arc
    in the authored direction. As with the former gdtk adapter, the C++
    inverse returns angles in ``(-pi, pi]``, so the angular range is rejected
    if it cannot be represented in that branch (after a possible antipodal
    flip of the start angle).
    """

    def __new__(cls, p0, p1, centre) -> CircleArc:
        center = _vec2(centre)
        pa = _vec2(p0)
        pb = _vec2(p1)
        ra = float(np.linalg.norm(pa - center))
        rb = float(np.linalg.norm(pb - center))
        if abs(ra - rb) > 1e-9 * max(ra, 1.0):
            raise ValueError("Arc radii do not match")
        ta = float(np.arctan2(pa[1] - center[1], pa[0] - center[0]))
        da, db = pa - center, pb - center
        sweep = float(np.arctan2(da[0] * db[1] - da[1] * db[0], da @ db))
        t0, t1 = ta, ta + sweep
        if max(t0, t1) > np.pi or min(t0, t1) <= -np.pi:
            ta2 = float(ta - 2.0 * np.pi * np.sign(ta))
            t0, t1 = ta2, ta2 + sweep
            if max(t0, t1) > np.pi or min(t0, t1) <= -np.pi:
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


class Edge:
    """A grid edge: a 1D entity re-parameterized over normalized t in [0, 1].

    Wraps any egg 2D curve entity that exposes the standalone-Python
    parametric interface (``eval``/``t0``/``t1``; every entity constructed by
    this module, plus :class:`~egg.geometry.analytic2d.Circle` /
    :class:`~egg.geometry.analytic2d.Ellipse`, does). Nodes can then be placed
    along the edge in parametric space::

        bottom = Edge(Line(p0=Vector3(0, 0), p1=Vector3(4, 0)))
        n = bottom.place_node(0.25)  # Node a quarter of the way along
        p = bottom.point_at(0.25)    # plain Vector3, no grid identity

    Both methods take the fractional parameter by default; pass
    ``param="native"`` to use the wrapped entity's own parameter instead
    (e.g. placing a node on an arc at an exact angle). The entities
    themselves expose the same pair as ``eval`` (native) / ``eval_frac``
    (fractional).

    ``arc_length=True`` re-parameterizes by normalized arc length (numerically
    sampled), so equal steps in t give equal steps in distance along curves
    with non-uniform native speed (Béziers, splines, composite wires).

    The wrapper is pure Python and never reaches the C++ core: consumers that
    encode entities (e.g. the topology builder) unwrap :attr:`entity` first.
    Projection-style queries delegate to the wrapped entity.
    """

    def __init__(self, entity, arc_length: bool = False, samples: int = 256):
        for attr in ("eval", "t0", "t1"):
            if not hasattr(entity, attr):
                raise TypeError(
                    f"Edge needs an entity with a parametric interface "
                    f"(missing '{attr}'): {entity!r}"
                )
        self.entity = entity
        self._table = None
        if arc_length:
            ts = np.linspace(entity.t0, entity.t1, samples + 1)
            pts = np.stack([np.asarray(entity.eval(t), dtype=float)
                            for t in ts])
            s = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0),
                                                 axis=1))])
            if s[-1] <= 0.0:
                raise ValueError("Edge has zero arc length")
            self._table = (s / s[-1], ts)

    @property
    def dim(self) -> int:
        return 1

    def _tau(self, t: float) -> float:
        """Native parameter for fractional t in [0, 1]."""
        t = float(np.clip(t, 0.0, 1.0))
        if self._table is not None:
            frac, ts = self._table
            return float(np.interp(t, frac, ts))
        return self.entity.t0 + t * (self.entity.t1 - self.entity.t0)

    def _frac(self, tau: float) -> float:
        """Fractional parameter in [0, 1] for native parameter tau."""
        lo, hi = sorted((self.entity.t0, self.entity.t1))
        tau = float(np.clip(tau, lo, hi))
        if self._table is not None:
            frac, ts = self._table
            if ts[0] > ts[-1]:  # reversed traversal: np.interp needs xp ascending
                return float(np.interp(tau, ts[::-1], frac[::-1]))
            return float(np.interp(tau, ts, frac))
        return (tau - self.entity.t0) / (self.entity.t1 - self.entity.t0)

    @staticmethod
    def _as_frac_param(t: float, param: str, frac_of_native) -> float:
        if param == "frac":
            return float(t)
        if param == "native":
            return frac_of_native(t)
        raise ValueError(f"param must be 'frac' or 'native', got {param!r}")

    def point_at(self, t: float, param: str = "frac") -> Vector3:
        """Physical point at parameter t (a plain Vector3).

        ``param`` selects the parametrisation of ``t``: ``"frac"`` (default)
        is the fraction in [0, 1] along the edge; ``"native"`` is the wrapped
        entity's own parameter (radians for arcs, knot values for B-splines).
        """
        t = self._as_frac_param(t, param, self._frac)
        p = np.asarray(self.entity.eval(self._tau(t)), dtype=float)
        return Vector3(p[0], p[1])

    def place_node(self, t: float, param: str = "frac",
                   *, fixed: bool = False) -> "Node":
        """A grid node placed on this edge at parameter t.

        ``param`` selects the parametrisation of ``t`` as in :meth:`point_at`.
        The node's ``t`` attribute is always stored fractionally.
        ``fixed=True`` pins the node when used as a topology corner.
        """
        return Node(self, self._as_frac_param(t, param, self._frac),
                    fixed=fixed)

    def point_at_native(self, t: float) -> Vector3:
        """Convenience for ``point_at(t, param="native")``."""
        return self.point_at(t, param="native")

    def place_node_native(self, t: float, *, fixed: bool = False) -> "Node":
        """Convenience for ``place_node(t, param="native")``."""
        return self.place_node(t, param="native", fixed=fixed)

    # Entity-protocol queries delegate to the wrapped entity.
    def project(self, p):
        return self.entity.project(p)

    def tangent_space(self, q):
        return self.entity.tangent_space(q)

    def normal(self, q):
        return self.entity.normal(q)


class Node(Vector3):
    """A point placed on an :class:`Edge` at normalized parameter ``t``.

    Behaves as a :class:`Vector3` (so it can be used anywhere a point is
    expected, e.g. as a topology corner position) while remembering its host
    ``edge`` and parameter ``t``.
    """

    __slots__ = ("edge", "t")

    def __init__(self, edge: Edge, t: float, *, fixed: bool = False):
        p = edge.point_at(t)
        super().__init__(p.x, p.y, fixed=fixed)
        self.edge = edge
        self.t = float(t)

    def __repr__(self) -> str:
        return f"Node({self.x}, {self.y}, t={self.t})"
