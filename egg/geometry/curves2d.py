"""2D curve primitives: arcs, Béziers, B-splines, and composite paths.

These mirror the C++ parametrizations in ``src/geometry.hpp`` (``CircleArcParam``,
``EllipseArcParam``, ``QuadBezierParam``, ``CubicBezierParam``,
``BSplineCurveParam``, ``CompositePath``): each curve is a parametrization
``C(t)`` restricted to an interval trim ``[t0, t1]``. Projection inverts the
parametrization (closed form where possible, seeded Newton on the nearest-foot
condition ``(C(t) - q) . C'(t) = 0`` otherwise), clamps the parameter onto the
trim, and evaluates; the tangent space is the normalized ``C'(t)``.
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


def _wrap(x: float, a: float, b: float) -> float:
    """Wrap x into [a, b) (mirrors C++ ``wrap``)."""
    L = b - a
    if L <= 0.0:
        return x
    t = np.fmod(x - a, L)
    if t < 0.0:
        t += L
    return a + t


def _newton_foot(
    curve, q: np.ndarray, t_lo: float, t_hi: float, n_seed: int = 16, iters: int = 8
) -> float:
    """Seeded-Newton nearest-foot parameter (mirrors C++ ``project_param``)."""
    ts = t_lo + (t_hi - t_lo) * np.arange(n_seed + 1) / n_seed
    pts = np.stack([curve.eval(t) for t in ts])
    t = float(ts[np.argmin(((pts - q) ** 2).sum(axis=1))])
    for _ in range(iters):
        d = curve.eval(t) - q
        d1 = curve.deriv(t)
        f = float(d @ d1)
        fp = float(d1 @ d1 + d @ curve.deriv2(t))
        if abs(fp) < 1e-30:
            break
        t -= f / fp
    return t


class _TrimmedCurve(GeometryEntity):
    """Shared invert → clamp → eval pipeline for interval-trimmed curves.

    Subclasses supply ``eval``/``deriv`` (and ``invert``; the default is the
    seeded-Newton foot, requiring ``deriv2``) plus ``t0``/``t1``/``closed``.
    """

    t0: float
    t1: float
    closed: bool = False

    @property
    def dim(self) -> int:
        return 1

    def invert(self, q: np.ndarray) -> float:
        """Unclamped nearest-foot parameter of q (seeded Newton)."""
        lo, hi = sorted((self.t0, self.t1))
        return _newton_foot(self, np.asarray(q, dtype=float), lo, hi)

    def _clamp(self, t: float) -> float:
        # t1 < t0 encodes a direction-reversed traversal (see the frontend
        # Arc); the point set is the same, so clamp to the sorted interval.
        lo, hi = sorted((self.t0, self.t1))
        if self.closed:
            return _wrap(t, lo, hi)
        return float(np.clip(t, lo, hi))

    def project(self, p: np.ndarray) -> np.ndarray:
        """Closest point on the trimmed curve to p (invert -> clamp -> eval)."""
        return self.eval(self._clamp(self.invert(np.asarray(p, dtype=float))))

    def tangent_space(self, q: np.ndarray) -> np.ndarray:
        """Normalized C'(t) at the clamped inverse of q. Shape (2, 1)."""
        t = self._clamp(self.invert(np.asarray(q, dtype=float)))
        d = self.deriv(t)
        norm = np.linalg.norm(d)
        if norm < 1e-15:
            return np.array([[1.0], [0.0]])
        return (d / norm).reshape(2, 1)

    def normal(self, q: np.ndarray) -> np.ndarray:
        """Tangent rotated 90 deg CCW. Shape (2,)."""
        t = self.tangent_space(q)[:, 0]
        return np.array([-t[1], t[0]])


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

    def invert(self, q) -> float:
        """Angle of q about the centre, in (-pi, pi] (closed-form atan2)."""
        d = np.asarray(q, dtype=float) - self.center
        return float(np.arctan2(d[1], d[0]))

    def eval(self, t: float) -> np.ndarray:
        """Point at angle t (radians)."""
        return self.center + self.radius * np.array([np.cos(t), np.sin(t)])

    def deriv(self, t: float) -> np.ndarray:
        """d/dt of :meth:`eval`."""
        return self.radius * np.array([-np.sin(t), np.cos(t)])


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

    def invert(self, q) -> float:
        """Nearest-foot parameter over the full period (seeded Newton)."""
        return _newton_foot(self, np.asarray(q, dtype=float), 0.0, 2.0 * np.pi)


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

    def _nearest(self, p: np.ndarray):
        p = np.asarray(p, dtype=float)
        best, best_d = None, np.inf
        for seg in self.segments:
            pr = seg.project(p)
            dd = float(((pr - p) ** 2).sum())
            if dd < best_d:
                best, best_d = seg, dd
        return best

    def project(self, p: np.ndarray) -> np.ndarray:
        """Closest point over all segments."""
        return self._nearest(p).project(p)

    def tangent_space(self, q: np.ndarray) -> np.ndarray:
        """Tangent space of the segment nearest to q."""
        return self._nearest(q).tangent_space(q)

    def normal(self, q: np.ndarray) -> np.ndarray:
        """Normal of the segment nearest to q."""
        return self._nearest(q).normal(q)
