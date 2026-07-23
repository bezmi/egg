# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""2D curve primitives: arcs, Béziers, B-splines, and composite paths.

These mirror the C++ parametrizations in ``src/geometry.hpp`` (``CircleArcParam``,
``EllipseArcParam``, ``QuadBezierParam``, ``CubicBezierParam``,
``BSplineCurveParam``, ``CompositePath``): each curve is a parametrization
``C(t)`` restricted to an interval trim ``[t0, t1]``, carried in Python only
for construction and parametric evaluation (node placement, sampling).
Projection and tangents come from the C++ core itself via the
:class:`~egg.geometry.base.GeometryEntity` base methods — the
parametrizations are implemented exactly once, in ``src/geometry.hpp``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import GeometryEntity

if TYPE_CHECKING:
    from egg.geometry.frontend2d import Vector3

__all__ = [
    "CircleArc",
    "EllipseArc",
    "QuadBezier",
    "CubicBezier",
    "BSplineCurve",
    "CompositePath",
]


class _TrimmedCurve(GeometryEntity):
    """Shared parametric-evaluation base for interval-trimmed curves.

    Subclasses supply ``eval``/``deriv``/``deriv2`` plus ``t0``/``t1``/
    ``closed``. Those three are numpy-broadcasting: a scalar ``t`` returns shape
    ``(2,)`` and an array of ``n`` parameters returns ``(n, 2)``. The batched
    ``*_many`` forms below therefore just call them with the whole array.
    """

    t0: float
    t1: float
    closed: bool = False

    @property
    def dim(self) -> int:
        return 1

    @abstractmethod
    def eval(self, t: float | np.ndarray) -> np.ndarray:
        """Point(s) at native parameter(s) ``t``. Shape ``np.shape(t) + (2,)``."""
        ...

    @abstractmethod
    def deriv(self, t: float | np.ndarray) -> np.ndarray:
        """d/dt of :meth:`eval` at ``t``. Shape ``np.shape(t) + (2,)``."""
        ...

    def deriv2(self, t: float | np.ndarray) -> np.ndarray:
        """d2/dt2 of :meth:`eval` at ``t``. Shape ``np.shape(t) + (2,)``.

        Optional second derivative (curvature): the curve kinds that support it
        override this; the others raise."""
        raise NotImplementedError(f"{type(self).__name__} has no second derivative")

    def eval_many(self, ts: np.ndarray) -> np.ndarray:
        """Points at each parameter in ``ts``. Shape (n, 2)."""
        return self.eval(np.asarray(ts, dtype=float))

    def deriv_many(self, ts: np.ndarray) -> np.ndarray:
        """d/dt of :meth:`eval` at each parameter in ``ts``. Shape (n, 2)."""
        return self.deriv(np.asarray(ts, dtype=float))

    def deriv2_many(self, ts: np.ndarray) -> np.ndarray:
        """d2/dt2 of :meth:`eval` at each parameter in ``ts``. Shape (n, 2)."""
        return self.deriv2(np.asarray(ts, dtype=float))


class CircleArc(_TrimmedCurve):
    """A circular arc ``C + r(cos t, sin t)``, t from t0 to t1 (radians).

    ``t1 < t0`` is allowed and traverses the arc clockwise (the parameter
    still varies linearly from ``t0`` to ``t1``); encoders normalise the
    interval before it reaches the C++ projection kernels.
    """

    # gdtk-style construction provenance: set by frontend2d.Arc, read by
    # egg.webui.scene to recover a topology from local construction.
    p0: Vector3 | None = None
    p1: Vector3 | None = None
    centre: Vector3 | None = None

    def __init__(
        self, center, radius: float, t0: float, t1: float, closed: bool = False
    ):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.t0, self.t1, self.closed = float(t0), float(t1), bool(closed)

    def eval(self, t: float | np.ndarray) -> np.ndarray:
        """Point at angle t (radians)."""
        return self.center + self.radius * np.stack([np.cos(t), np.sin(t)], axis=-1)

    def deriv(self, t: float | np.ndarray) -> np.ndarray:
        """d/dt of :meth:`eval`."""
        return self.radius * np.stack([-np.sin(t), np.cos(t)], axis=-1)


class EllipseArc(_TrimmedCurve):
    """A rotated elliptical arc ``C + R_phi (a cos t, b sin t)``."""

    def __init__(
        self,
        center,
        a: float,
        b: float,
        phi: float,
        t0: float,
        t1: float,
        closed: bool = False,
    ):
        self.center = np.asarray(center, dtype=float)
        self.a, self.b, self.phi = float(a), float(b), float(phi)
        self.t0, self.t1, self.closed = float(t0), float(t1), bool(closed)

    def _rot(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Rotate ``(x, y)`` by ``phi``, stacking the coordinate on the last
        axis so scalar and array inputs both broadcast."""
        cp, sp = np.cos(self.phi), np.sin(self.phi)
        return np.stack([cp * x - sp * y, sp * x + cp * y], axis=-1)

    def eval(self, t: float | np.ndarray) -> np.ndarray:
        """Point at parametric angle t (radians)."""
        return self.center + self._rot(self.a * np.cos(t), self.b * np.sin(t))

    def deriv(self, t: float | np.ndarray) -> np.ndarray:
        """d/dt of :meth:`eval`."""
        return self._rot(-self.a * np.sin(t), self.b * np.cos(t))

    def deriv2(self, t: float | np.ndarray) -> np.ndarray:
        """d2/dt2 of :meth:`eval`."""
        return self._rot(-self.a * np.cos(t), -self.b * np.sin(t))


class QuadBezier(_TrimmedCurve):
    """A quadratic Bézier over control points P0, P1, P2."""

    # Construction provenance retained by frontend2d.Bezier (see CircleArc).
    points: list[Vector3] | None = None

    def __init__(self, p0, p1, p2, t0: float = 0.0, t1: float = 1.0):
        self.p = np.stack([np.asarray(v, dtype=float) for v in (p0, p1, p2)])
        self.t0, self.t1, self.closed = float(t0), float(t1), False

    def eval(self, t: float | np.ndarray) -> np.ndarray:
        """Point at parameter t."""
        t = np.asarray(t, dtype=float)[..., None]
        u = 1.0 - t
        return u * u * self.p[0] + 2 * u * t * self.p[1] + t * t * self.p[2]

    def deriv(self, t: float | np.ndarray) -> np.ndarray:
        """d/dt of :meth:`eval`."""
        t = np.asarray(t, dtype=float)[..., None]
        return 2 * (1 - t) * (self.p[1] - self.p[0]) + 2 * t * (self.p[2] - self.p[1])

    def deriv2(self, t: float | np.ndarray) -> np.ndarray:
        """d2/dt2 of :meth:`eval` (constant)."""
        c = 2 * (self.p[2] - 2 * self.p[1] + self.p[0])
        return np.broadcast_to(c, np.shape(np.asarray(t, dtype=float)) + (2,))


class CubicBezier(_TrimmedCurve):
    """A cubic Bézier over control points P0..P3."""

    # Construction provenance retained by frontend2d.Bezier (see CircleArc).
    points: list[Vector3] | None = None

    def __init__(self, p0, p1, p2, p3, t0: float = 0.0, t1: float = 1.0):
        self.p = np.stack([np.asarray(v, dtype=float) for v in (p0, p1, p2, p3)])
        self.t0, self.t1, self.closed = float(t0), float(t1), False

    def eval(self, t: float | np.ndarray) -> np.ndarray:
        """Point at parameter t."""
        t = np.asarray(t, dtype=float)[..., None]
        u = 1.0 - t
        return (
            u**3 * self.p[0]
            + 3 * u * u * t * self.p[1]
            + 3 * u * t * t * self.p[2]
            + t**3 * self.p[3]
        )

    def deriv(self, t: float | np.ndarray) -> np.ndarray:
        """d/dt of :meth:`eval`."""
        t = np.asarray(t, dtype=float)[..., None]
        u = 1.0 - t
        return (
            3 * u * u * (self.p[1] - self.p[0])
            + 6 * u * t * (self.p[2] - self.p[1])
            + 3 * t * t * (self.p[3] - self.p[2])
        )

    def deriv2(self, t: float | np.ndarray) -> np.ndarray:
        """d2/dt2 of :meth:`eval`."""
        t = np.asarray(t, dtype=float)[..., None]
        u = 1.0 - t
        return 6 * u * (self.p[2] - 2 * self.p[1] + self.p[0]) + 6 * t * (
            self.p[3] - 2 * self.p[2] + self.p[1]
        )


class BSplineCurve(_TrimmedCurve):
    """A B-spline / NURBS curve over a knot vector and 2D control points.

    ``weights`` (length ``n_ctrl``) selects the rational form, evaluated via the
    homogeneous splines ``A(t) = sum N_i w_i P_i``, ``w(t) = sum N_i w_i`` and
    the quotient rule (mirrors the C++ ``BSplineCurveParam``); ``None`` is the
    polynomial path. The live domain is ``[knots[degree], knots[n_ctrl]]``; the
    C++ side caps the degree at ``kMaxBSplineDegree = 7``.
    """

    # Construction provenance retained by frontend2d.Bezier (see CircleArc).
    points: list[Vector3] | None = None

    def __init__(self, degree: int, knots, ctrl, weights=None):
        from scipy.interpolate import BSpline as _SciBSpline

        self.degree = int(degree)
        self.knots = np.asarray(knots, dtype=float)
        self.ctrl = np.asarray(ctrl, dtype=float).reshape(-1, 2)
        n_ctrl = self.ctrl.shape[0]
        if self.knots.shape[0] != n_ctrl + self.degree + 1:
            raise ValueError("knot vector length must be n_ctrl + degree + 1")
        if self.degree > 7:
            raise ValueError("degree exceeds the C++ kMaxBSplineDegree = 7")
        self.weights = None
        if weights is not None:
            self.weights = np.asarray(weights, dtype=float)
            if self.weights.shape != (n_ctrl,):
                raise ValueError("weights length must equal n_ctrl")
        coeffs = (
            self.ctrl if self.weights is None else self.weights[:, None] * self.ctrl
        )
        self._spl = _SciBSpline(self.knots, coeffs, self.degree)
        self._d1 = self._spl.derivative(1)
        self._d2 = self._spl.derivative(2)
        if self.weights is not None:
            self._w = _SciBSpline(self.knots, self.weights, self.degree)
            self._w1 = self._w.derivative(1)
            self._w2 = self._w.derivative(2)
        self.t0 = float(self.knots[self.degree])
        self.t1 = float(self.knots[n_ctrl])
        self.closed = False

    # scipy's BSpline evaluates a scalar to (2,) and an array of n params to
    # (n, 2); the rational quotient rule broadcasts the weight splines on the
    # trailing coordinate axis, so scalar and array both work.
    def eval(self, t: float | np.ndarray) -> np.ndarray:
        """Point(s) at knot-space parameter(s) t."""
        A = np.asarray(self._spl(t), dtype=float)
        if self.weights is None:
            return A
        return A / np.asarray(self._w(t), dtype=float)[..., None]

    def deriv(self, t: float | np.ndarray) -> np.ndarray:
        """d/dt of :meth:`eval` (quotient rule for the rational form)."""
        A1 = np.asarray(self._d1(t), dtype=float)
        if self.weights is None:
            return A1
        w = np.asarray(self._w(t), dtype=float)[..., None]
        w1 = np.asarray(self._w1(t), dtype=float)[..., None]
        return (A1 - w1 * self.eval(t)) / w

    def deriv2(self, t: float | np.ndarray) -> np.ndarray:
        """d2/dt2 of :meth:`eval` (quotient rule for the rational form)."""
        A2 = np.asarray(self._d2(t), dtype=float)
        if self.weights is None:
            return A2
        w = np.asarray(self._w(t), dtype=float)[..., None]
        w1 = np.asarray(self._w1(t), dtype=float)[..., None]
        w2 = np.asarray(self._w2(t), dtype=float)[..., None]
        C, C1 = self.eval(t), self.deriv(t)
        return (A2 - 2.0 * w1 * C1 - w2 * C) / w


class CompositePath(GeometryEntity):
    """An ordered sequence of curve segments joined end-to-end (a 2D wire).

    Projection projects onto every segment and keeps the nearest; the tangent
    is the matched segment's. Nested composites are not supported.

    Parametric form: ``eval(t)`` over t in [0, 1], with sub-intervals allotted
    to segments in proportion to their (numerically sampled) arc lengths and
    mapped linearly onto each segment's own parameter — standalone Python, so
    grid edges can place nodes along the wire without touching the C++ core.
    """

    t0: float = 0.0
    t1: float = 1.0
    closed: bool = False
    # Construction provenance retained by frontend2d.Spline/Polyline; empty for
    # a composite built directly from segments (no point-based constructor).
    points: list[Vector3]

    def __init__(self, segments):
        segments = list(segments)
        if not segments:
            raise ValueError("CompositePath needs at least one segment")
        if any(isinstance(s, CompositePath) for s in segments):
            raise ValueError("nested CompositePath segments are not supported")
        self.segments = segments
        self.points = []
        self._breaks = None  # cumulative arc-length fractions, computed lazily

    @property
    def dim(self) -> int:
        return 1

    def _segment_breaks(self, samples_per_segment: int = 64) -> np.ndarray:
        """Cumulative arc-length fractions [0, ..., 1] over the segments."""
        if self._breaks is None:
            lengths = []
            for seg in self.segments:
                ts = seg.t0 + (seg.t1 - seg.t0) * np.linspace(
                    0.0, 1.0, samples_per_segment + 1
                )
                pts = np.stack([seg.eval(t) for t in ts])
                lengths.append(
                    float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
                )
            total = sum(lengths)
            if total <= 0.0:
                raise ValueError("CompositePath has zero total arc length")
            self._breaks = np.concatenate([[0.0], np.cumsum(lengths) / total])
            self._breaks[-1] = 1.0
        return self._breaks

    def _locate(self, t: float) -> tuple[Any, float, float]:
        """Segment, local parameter, and d(local)/d(global) at global t."""
        br = self._segment_breaks()
        t = float(np.clip(t, 0.0, 1.0))
        i = min(int(np.searchsorted(br, t, side="right")) - 1, len(self.segments) - 1)
        i = max(i, 0)
        seg = self.segments[i]
        width = br[i + 1] - br[i]
        u = (t - br[i]) / width
        scale = (seg.t1 - seg.t0) / width
        return seg, seg.t0 + u * (seg.t1 - seg.t0), scale

    def eval(self, t: float) -> np.ndarray:
        """Point at global parameter t in [0, 1]."""
        seg, tl, _ = self._locate(t)
        return seg.eval(tl)

    def deriv(self, t: float) -> np.ndarray:
        """d/dt of :meth:`eval` (segment derivative, chain-rule scaled)."""
        seg, tl, scale = self._locate(t)
        return seg.deriv(tl) * scale

    # Projection, tangents, and normals come from the C++ core through the
    # GeometryEntity base methods (one batched nearest-segment contest per
    # call, regardless of segment count).
