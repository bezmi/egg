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

from typing import Any

import numpy as np

from .base import GeometryEntity

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
    ``closed``; the vector counterparts below default to scalar loops and
    are overridden where the expressions broadcast.
    """

    t0: float
    t1: float
    closed: bool = False

    @property
    def dim(self) -> int:
        return 1

    def eval_many(self, ts: np.ndarray) -> np.ndarray:
        """Points at each parameter in ``ts``. Shape (n, 2)."""
        return np.stack([self.eval(float(t)) for t in ts])

    def deriv_many(self, ts: np.ndarray) -> np.ndarray:
        """d/dt of :meth:`eval` at each parameter in ``ts``. Shape (n, 2)."""
        return np.stack([self.deriv(float(t)) for t in ts])

    def deriv2_many(self, ts: np.ndarray) -> np.ndarray:
        """d2/dt2 of :meth:`eval` at each parameter in ``ts``. Shape (n, 2)."""
        return np.stack([self.deriv2(float(t)) for t in ts])


class CircleArc(_TrimmedCurve):
    """A circular arc ``C + r(cos t, sin t)``, t from t0 to t1 (radians).

    ``t1 < t0`` is allowed and traverses the arc clockwise (the parameter
    still varies linearly from ``t0`` to ``t1``); encoders normalise the
    interval before it reaches the C++ projection kernels.
    """

    def __init__(
        self, center, radius: float, t0: float, t1: float, closed: bool = False
    ):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.t0, self.t1, self.closed = float(t0), float(t1), bool(closed)

    def eval(self, t: float) -> np.ndarray:
        """Point at angle t (radians)."""
        return self.center + self.radius * np.array([np.cos(t), np.sin(t)])

    def deriv(self, t: float) -> np.ndarray:
        """d/dt of :meth:`eval`."""
        return self.radius * np.array([-np.sin(t), np.cos(t)])

    def eval_many(self, ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, dtype=float)
        return self.center + self.radius * np.stack([np.cos(ts), np.sin(ts)], -1)

    def deriv_many(self, ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, dtype=float)
        return self.radius * np.stack([-np.sin(ts), np.cos(ts)], -1)


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

    def _rot(self, x: float, y: float) -> np.ndarray:
        cp, sp = np.cos(self.phi), np.sin(self.phi)
        return np.array([cp * x - sp * y, sp * x + cp * y])

    def eval(self, t: float) -> np.ndarray:
        """Point at parametric angle t (radians)."""
        return self.center + self._rot(self.a * np.cos(t), self.b * np.sin(t))

    def deriv(self, t: float) -> np.ndarray:
        """d/dt of :meth:`eval`."""
        return self._rot(-self.a * np.sin(t), self.b * np.cos(t))

    def deriv2(self, t: float) -> np.ndarray:
        """d2/dt2 of :meth:`eval`."""
        return self._rot(-self.a * np.cos(t), -self.b * np.sin(t))

    def _rot_many(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cp, sp = np.cos(self.phi), np.sin(self.phi)
        return np.stack([cp * x - sp * y, sp * x + cp * y], -1)

    def eval_many(self, ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, dtype=float)
        return self.center + self._rot_many(self.a * np.cos(ts), self.b * np.sin(ts))

    def deriv_many(self, ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, dtype=float)
        return self._rot_many(-self.a * np.sin(ts), self.b * np.cos(ts))

    def deriv2_many(self, ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, dtype=float)
        return self._rot_many(-self.a * np.cos(ts), -self.b * np.sin(ts))


class QuadBezier(_TrimmedCurve):
    """A quadratic Bézier over control points P0, P1, P2."""

    def __init__(self, p0, p1, p2, t0: float = 0.0, t1: float = 1.0):
        self.p = np.stack([np.asarray(v, dtype=float) for v in (p0, p1, p2)])
        self.t0, self.t1, self.closed = float(t0), float(t1), False

    def eval(self, t: float) -> np.ndarray:
        """Point at parameter t."""
        u = 1.0 - t
        return u * u * self.p[0] + 2 * u * t * self.p[1] + t * t * self.p[2]

    def deriv(self, t: float) -> np.ndarray:
        """d/dt of :meth:`eval`."""
        return 2 * (1 - t) * (self.p[1] - self.p[0]) + 2 * t * (self.p[2] - self.p[1])

    def deriv2(self, t: float) -> np.ndarray:
        """d2/dt2 of :meth:`eval` (constant)."""
        return 2 * (self.p[2] - 2 * self.p[1] + self.p[0])

    def eval_many(self, ts: np.ndarray) -> np.ndarray:
        return self.eval(np.asarray(ts, dtype=float)[:, None])

    def deriv_many(self, ts: np.ndarray) -> np.ndarray:
        return self.deriv(np.asarray(ts, dtype=float)[:, None])

    def deriv2_many(self, ts: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.deriv2(0.0), (len(ts), 2))


class CubicBezier(_TrimmedCurve):
    """A cubic Bézier over control points P0..P3."""

    def __init__(self, p0, p1, p2, p3, t0: float = 0.0, t1: float = 1.0):
        self.p = np.stack([np.asarray(v, dtype=float) for v in (p0, p1, p2, p3)])
        self.t0, self.t1, self.closed = float(t0), float(t1), False

    def eval(self, t: float) -> np.ndarray:
        """Point at parameter t."""
        u = 1.0 - t
        return (
            u**3 * self.p[0]
            + 3 * u * u * t * self.p[1]
            + 3 * u * t * t * self.p[2]
            + t**3 * self.p[3]
        )

    def deriv(self, t: float) -> np.ndarray:
        """d/dt of :meth:`eval`."""
        u = 1.0 - t
        return (
            3 * u * u * (self.p[1] - self.p[0])
            + 6 * u * t * (self.p[2] - self.p[1])
            + 3 * t * t * (self.p[3] - self.p[2])
        )

    def deriv2(self, t: float) -> np.ndarray:
        """d2/dt2 of :meth:`eval`."""
        u = 1.0 - t
        return 6 * u * (self.p[2] - 2 * self.p[1] + self.p[0]) + 6 * t * (
            self.p[3] - 2 * self.p[2] + self.p[1]
        )

    # The polynomial expressions above broadcast over a column of
    # parameters: (n, 1) * (2,) -> (n, 2).
    def eval_many(self, ts: np.ndarray) -> np.ndarray:
        return self.eval(np.asarray(ts, dtype=float)[:, None])

    def deriv_many(self, ts: np.ndarray) -> np.ndarray:
        return self.deriv(np.asarray(ts, dtype=float)[:, None])

    def deriv2_many(self, ts: np.ndarray) -> np.ndarray:
        return self.deriv2(np.asarray(ts, dtype=float)[:, None])


class BSplineCurve(_TrimmedCurve):
    """A B-spline / NURBS curve over a knot vector and 2D control points.

    ``weights`` (length ``n_ctrl``) selects the rational form, evaluated via the
    homogeneous splines ``A(t) = sum N_i w_i P_i``, ``w(t) = sum N_i w_i`` and
    the quotient rule (mirrors the C++ ``BSplineCurveParam``); ``None`` is the
    polynomial path. The live domain is ``[knots[degree], knots[n_ctrl]]``; the
    C++ side caps the degree at ``kMaxBSplineDegree = 7``.
    """

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

    def eval(self, t: float) -> np.ndarray:
        """Point at knot-space parameter t."""
        A = np.asarray(self._spl(t), dtype=float)
        if self.weights is None:
            return A
        return A / float(self._w(t))

    def deriv(self, t: float) -> np.ndarray:
        """d/dt of :meth:`eval` (quotient rule for the rational form)."""
        A1 = np.asarray(self._d1(t), dtype=float)
        if self.weights is None:
            return A1
        w = float(self._w(t))
        return (A1 - float(self._w1(t)) * self.eval(t)) / w

    def deriv2(self, t: float) -> np.ndarray:
        """d2/dt2 of :meth:`eval` (quotient rule for the rational form)."""
        A2 = np.asarray(self._d2(t), dtype=float)
        if self.weights is None:
            return A2
        w = float(self._w(t))
        C, C1 = self.eval(t), self.deriv(t)
        return (A2 - 2.0 * float(self._w1(t)) * C1 - float(self._w2(t)) * C) / w

    # scipy's BSpline evaluates arrays natively; the rational form applies
    # the same quotient rule with the weight splines as columns.
    def eval_many(self, ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, dtype=float)
        A = np.asarray(self._spl(ts), dtype=float)
        if self.weights is None:
            return A
        return A / np.asarray(self._w(ts), dtype=float)[:, None]

    def deriv_many(self, ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, dtype=float)
        A1 = np.asarray(self._d1(ts), dtype=float)
        if self.weights is None:
            return A1
        w = np.asarray(self._w(ts), dtype=float)[:, None]
        w1 = np.asarray(self._w1(ts), dtype=float)[:, None]
        return (A1 - w1 * self.eval_many(ts)) / w

    def deriv2_many(self, ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, dtype=float)
        A2 = np.asarray(self._d2(ts), dtype=float)
        if self.weights is None:
            return A2
        w = np.asarray(self._w(ts), dtype=float)[:, None]
        w1 = np.asarray(self._w1(ts), dtype=float)[:, None]
        w2 = np.asarray(self._w2(ts), dtype=float)[:, None]
        return (A2 - 2.0 * w1 * self.deriv_many(ts) - w2 * self.eval_many(ts)) / w


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

    def __init__(self, segments):
        segments = list(segments)
        if not segments:
            raise ValueError("CompositePath needs at least one segment")
        if any(isinstance(s, CompositePath) for s in segments):
            raise ValueError("nested CompositePath segments are not supported")
        self.segments = segments
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
