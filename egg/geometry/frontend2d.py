# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""gdtk-style 2D geometry construction front-end.

Provides the gdtk ``geom`` constructors (:class:`Vector3`, :class:`Line`,
:func:`Arc`, :func:`Bezier`, :class:`Polyline`, :func:`Spline`) as thin
factories over :mod:`egg.geometry` entities, so geometry can be authored
without a gdtk dependency. The results are ordinary entities and feed
directly into the topology builder and the C++ encoding path.

:class:`Edge` wraps any parametric entity for normalized-parameter node
placement; :class:`Node` records its host edge, which drives association
inference in :meth:`egg.topology.builder.TopologyBuilder.build`.
"""

from __future__ import annotations

import numpy as np

from .base import _INHERIT
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
    "split_cells",
    "tfi_point",
]


def split_cells(n: int, k: int) -> list[int]:
    """Split ``n`` cells into ``k`` contiguous per-block counts.

    As even as possible; the counts always sum to ``n``. Used to divide a
    total resolution across a row of blocks (cf.
    :meth:`egg.topology.builder.TopologyBuilder.add_block_array`).
    """
    return [round(n * (t + 1) / k) - round(n * t / k) for t in range(k)]


def tfi_point(
    u: float, v: float, south: "Edge", north: "Edge", west: "Edge", east: "Edge"
) -> "Vector3":
    """Bilinear transfinite interpolation of four bounding edges at (u, v).

    Standard Coons-patch formula; used to place interior block corners of a
    block array parametrically. ``south``/``north`` are parameterised
    west -> east (by ``u``), ``west``/``east`` south -> north (by ``v``);
    the patch corners are the shared edge endpoints.
    """
    s, n = south.point_at(u), north.point_at(u)
    w, e = west.point_at(v), east.point_at(v)
    p00, p10 = south.point_at(0.0), south.point_at(1.0)
    p01, p11 = north.point_at(0.0), north.point_at(1.0)
    x = (
        (1 - v) * s.x
        + v * n.x
        + (1 - u) * w.x
        + u * e.x
        - (
            (1 - u) * (1 - v) * p00.x
            + u * (1 - v) * p10.x
            + (1 - u) * v * p01.x
            + u * v * p11.x
        )
    )
    y = (
        (1 - v) * s.y
        + v * n.y
        + (1 - u) * w.y
        + u * e.y
        - (
            (1 - u) * (1 - v) * p00.y
            + u * (1 - v) * p10.y
            + (1 - u) * v * p01.y
            + u * v * p11.y
        )
    )
    return Vector3(x, y)


class Vector3:
    """2D point with gdtk ``Vector3`` semantics.

    Supports ``+``, ``-``, scalar ``*``, negation, and ``abs`` (magnitude);
    arithmetic results are never ``fixed``. The geometry is planar: ``z``
    must be zero.

    Parameters
    ----------
    x, y, z : float
        Coordinates. ``z`` must be 0 (default 0.0).
    fixed : bool, optional
        Mark the point as a pinned grid corner when used as a topology
        corner position.
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

    Points with a third coordinate must lie in the z=0 plane.
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
    """Straight segment from ``p0`` to ``p1``.

    A :class:`~egg.geometry.analytic2d.LineSegment` that also retains the
    constructor arguments as :class:`Vector3` attributes ``p0`` / ``p1``
    (gdtk-style access, e.g. ``line.p0.x``).
    """

    def __init__(self, p0, p1):
        self.p0 = p0 if isinstance(p0, Vector3) else Vector3(*_vec2(p0))
        self.p1 = p1 if isinstance(p1, Vector3) else Vector3(*_vec2(p1))
        super().__init__(_vec2(p0), _vec2(p1))


def Arc(p0, p1, centre) -> CircleArc:
    """Circular arc from ``p0`` to ``p1`` about ``centre``.

    Keyword names match the gdtk/Eilmer ``Arc:new{p0=, p1=, centre=}`` form.

    Parameters
    ----------
    p0, p1 : Vector3 or array_like
        Endpoints; must be equidistant from ``centre``.
    centre : Vector3 or array_like
        Arc centre.

    Returns
    -------
    CircleArc
        Parameter runs from the angle of ``p0`` to the angle of ``p1``
        (``t1 < t0`` for a clockwise sweep), so ``eval(t0) == p0`` and
        composites traverse the arc in the authored direction.

    Raises
    ------
    ValueError
        If the endpoint radii differ, or the angular interval cannot be
        represented in the C++ ``atan2`` inverse branch ``(-pi, pi]``
        (after a possible antipodal flip of the start angle) — split such
        arcs.
    """
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
                "Arc angular range wraps past the +/-pi branch cut; split it"
            )
    arc = CircleArc(center, ra, float(t0), float(t1))
    # Retain the construction points as Vector3s (gdtk-style, like Line.p0/p1).
    arc.p0 = p0 if isinstance(p0, Vector3) else Vector3(*pa)
    arc.p1 = p1 if isinstance(p1, Vector3) else Vector3(*pb)
    arc.centre = centre if isinstance(centre, Vector3) else Vector3(*center)
    return arc


def Bezier(points) -> LineSegment | QuadBezier | CubicBezier | BSplineCurve:
    """Bézier curve over the given control points.

    Parameters
    ----------
    points : sequence of Vector3 or array_like
        Control points; the degree is ``len(points) - 1 >= 1``.

    Returns
    -------
    LineSegment or QuadBezier or CubicBezier or BSplineCurve
        By degree 1 / 2 / 3 / >= 4 respectively. A degree-n Bézier is
        returned as a B-spline on the clamped knot vector
        ``{0 x (n+1), 1 x (n+1)}``.
    """
    pts = [_vec2(p) for p in points]
    n = len(pts) - 1  # degree
    if n < 1:
        raise ValueError("Bezier needs at least two control points")
    if n == 1:
        curve = LineSegment(pts[0], pts[1])
    elif n == 2:
        curve = QuadBezier(pts[0], pts[1], pts[2])
    elif n == 3:
        curve = CubicBezier(pts[0], pts[1], pts[2], pts[3])
    else:
        knots = np.concatenate([np.zeros(n + 1), np.ones(n + 1)])
        curve = BSplineCurve(n, knots, np.stack(pts))
    # Retain the control points as Vector3s (gdtk-style, like Line.p0/p1).
    curve.points = [
        p if isinstance(p, Vector3) else Vector3(*q) for p, q in zip(points, pts)
    ]
    return curve


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
    """Ordered sequence of curve segments joined into one path.

    Parameters
    ----------
    segments : sequence of curve entities
        Joined end-to-end. Nested composites/polylines are rejected by
        :class:`~egg.geometry.curves2d.CompositePath` itself.
    closed : bool, optional
        Append a closing :class:`~egg.geometry.analytic2d.LineSegment` when
        the last segment's end is farther than ``tolerance`` from the first
        segment's start (gdtk ``Polyline`` behaviour).
    tolerance : float, optional
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
    """Per-knot second derivatives of a natural cubic spline.

    Uniform parameterisation (knot spacing ``h = 1``); natural end
    conditions ``M_0 = M_{n-1} = 0``.

    Parameters
    ----------
    points : (N, 2) array of through-points.

    Returns
    -------
    (N, 2) array of second derivatives ``M_i``.
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

    Standard second-derivative -> Bézier control-point relation (uniform
    knot spacing ``h = 1``):

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


def Spline(points, closed=False, tolerance=1.0e-10) -> CompositePath:
    """Natural cubic spline through the given points.

    Uniform parameterisation with natural end conditions (zero second
    derivative at the ends).

    Parameters
    ----------
    points : sequence of Vector3 or array_like
        At least two through-points.
    closed : bool, optional
        Append the first point to the end (if farther than ``tolerance``)
        and solve an *open* natural spline through the extended list —
        gdtk's algorithm, not periodic end conditions. The wrap-around
        joint is therefore C0 but not C1.
    tolerance : float, optional

    Returns
    -------
    CompositePath
        One :class:`~egg.geometry.curves2d.CubicBezier` per interval.
    """
    pts = np.stack([_vec2(p) for p in points])
    if pts.shape[0] < 2:
        raise ValueError("Spline needs at least two points")
    if closed and np.linalg.norm(pts[0] - pts[-1]) > tolerance:
        pts = np.vstack([pts, pts[0:1]])
    segs = _spline_to_beziers(pts)
    path = CompositePath(segs)
    # Retain the through-points as Vector3s (gdtk-style, like Line.p0/p1),
    # so tooling (e.g. the web UI) can show the construction points.
    path.points = [p if isinstance(p, Vector3) else Vector3(*_vec2(p)) for p in points]
    return path


class Edge:
    """Grid edge: a 1D entity re-parameterized over normalized t in [0, 1].

    Wraps any entity exposing the standalone-Python parametric interface
    (``eval``/``t0``/``t1`` — everything this module constructs, plus
    :class:`~egg.geometry.analytic2d.Circle` /
    :class:`~egg.geometry.analytic2d.Ellipse`). The wrapper is pure Python
    and never reaches the C++ core: consumers that encode entities (e.g.
    the topology builder) unwrap :attr:`entity` first, and projection-style
    queries delegate to it.

    Parameters
    ----------
    entity : geometry entity
        Must expose ``eval`` / ``t0`` / ``t1``.
    arc_length : bool, optional
        Re-parameterize by normalized arc length (numerically sampled with
        ``samples`` subdivisions), so equal steps in t give equal distance
        along curves with non-uniform native speed (Béziers, splines,
        composite wires).
    samples : int, optional

    Examples
    --------
    >>> bottom = Edge(Line(Vector3(0, 0), Vector3(4, 0)))
    >>> n = bottom.place_node(0.25)  # Node a quarter of the way along
    >>> p = bottom.point_at(0.25)    # plain Vector3, no grid identity

    Both methods take the fractional parameter by default; pass
    ``param="native"`` to use the wrapped entity's own parameter instead
    (e.g. placing a node on an arc at an exact angle). The entities
    themselves expose the same pair as ``eval`` (native) / ``eval_frac``
    (fractional).
    """

    def __init__(
        self,
        entity,
        arc_length: bool = False,
        samples: int = 256,
        *,
        name=None,
        tag=_INHERIT,
    ):
        for attr in ("eval", "t0", "t1"):
            if not hasattr(entity, attr):
                raise TypeError(
                    f"Edge needs an entity with a parametric interface "
                    f"(missing '{attr}'): {entity!r}"
                )
        self.entity = entity
        if name is not None:
            entity.name = name
        if tag is not _INHERIT:
            entity.tag = tag
        self._table = None
        if arc_length:
            ts = np.linspace(entity.t0, entity.t1, samples + 1)
            pts = np.stack([np.asarray(entity.eval(t), dtype=float) for t in ts])
            s = np.concatenate(
                [[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
            )
            if s[-1] <= 0.0:
                raise ValueError("Edge has zero arc length")
            self._table = (s / s[-1], ts)

    @property
    def dim(self) -> int:
        """1 (a curve)."""
        return 1

    @property
    def name(self):
        """The wrapped :attr:`entity`'s :attr:`~egg.geometry.base.GeometryEntity.name`."""
        return self.entity.name

    @name.setter
    def name(self, value):
        self.entity.name = value

    @property
    def tag(self):
        """The wrapped :attr:`entity`'s :attr:`~egg.geometry.base.GeometryEntity.tag`."""
        return self.entity.tag

    @tag.setter
    def tag(self, value):
        self.entity.tag = value

    def named(self, name, *, tag=_INHERIT):
        """Name the wrapped :attr:`entity` (and optionally set its
        :attr:`~egg.geometry.base.GeometryEntity.tag`); return this ``Edge``
        (fluent)."""
        self.entity.name = name
        if tag is not _INHERIT:
            self.entity.tag = tag
        return self

    @property
    def boundary_layer(self):
        """The wrapped :attr:`entity`'s
        :attr:`~egg.geometry.base.GeometryEntity.boundary_layer`."""
        return self.entity.boundary_layer

    def clustered(self, **kw):
        """Request clustering on the wrapped :attr:`entity`
        (see :meth:`~egg.geometry.base.GeometryEntity.clustered`); return this
        ``Edge`` (fluent)."""
        self.entity.clustered(**kw)
        return self

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
        """Normalize ``t`` to the fractional parametrisation."""
        if param == "frac":
            return float(t)
        if param == "native":
            return frac_of_native(t)
        raise ValueError(f"param must be 'frac' or 'native', got {param!r}")

    def point_at(self, t: float, param: str = "frac") -> Vector3:
        """Physical point at parameter t (a plain :class:`Vector3`).

        ``param`` selects the parametrisation of ``t``: ``"frac"`` (default)
        is the fraction in [0, 1] along the edge; ``"native"`` is the wrapped
        entity's own parameter (radians for arcs, knot values for B-splines).
        """
        t = self._as_frac_param(t, param, self._frac)
        p = np.asarray(self.entity.eval(self._tau(t)), dtype=float)
        return Vector3(p[0], p[1])

    def place_node(
        self, t: float, param: str = "frac", *, fixed: bool = False
    ) -> "Node":
        """Grid node placed on this edge at parameter t.

        ``param`` as in :meth:`point_at`; the node's ``t`` attribute is
        always stored fractionally. ``fixed=True`` pins the node when used
        as a topology corner.
        """
        return Node(self, self._as_frac_param(t, param, self._frac), fixed=fixed)

    def point_at_native(self, t: float) -> Vector3:
        """Convenience for ``point_at(t, param="native")``."""
        return self.point_at(t, param="native")

    def place_node_native(self, t: float, *, fixed: bool = False) -> "Node":
        """Convenience for ``place_node(t, param="native")``."""
        return self.place_node(t, param="native", fixed=fixed)

    # Entity-protocol queries delegate to the wrapped entity.
    def project(self, p):
        """Closest point on the wrapped entity to p."""
        return self.entity.project(p)

    def tangent_space(self, q):
        """Tangent space of the wrapped entity at q."""
        return self.entity.tangent_space(q)

    def normal(self, q):
        """Normal of the wrapped entity at q."""
        return self.entity.normal(q)


class Node(Vector3):
    """Point placed on an :class:`Edge` at normalized parameter ``t``.

    Behaves as a :class:`Vector3` (usable anywhere a point is expected,
    e.g. as a topology corner position) while remembering its host
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
