"""3D parametric surface primitives: B-spline / NURBS surfaces.

These mirror the C++ ``BSplineSurfaceParam`` in ``src/geometry.hpp``: a
tensor-product B-spline (or NURBS, when ``weights`` is set) surface
@f$ S(u,v) @f$ embedded in 3D. Evaluation uses the row-then-column tensor
product (each row is a v-direction ``BSpline``, the row results form a
u-direction ``BSpline``); the rational form evaluates in homogeneous
coordinates and applies the quotient rule, matching the C++ ``ders`` method.
Projection inverts the parametrization via coarse-grid-seeded Newton on the
nearest-foot stationarity (mirrors ``BSplineSurfaceParam::invert``).
"""

from __future__ import annotations

import numpy as np

from .base import GeometryEntity

__all__ = ["BSplineSurface"]


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def _orthonormalize2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _normalize(a)
    b = b - np.dot(b, a) * a
    b = _normalize(b)
    return np.column_stack([a, b])


class BSplineSurface(GeometryEntity):
    """A tensor-product B-spline / NURBS surface in 3D.

    ``weights`` (shape ``(nu, nv)``) selects the rational form (NURBS),
    evaluated via homogeneous tensor-product sums and the bivariate quotient
    rule (mirrors the C++ ``BSplineSurfaceParam``); ``None`` is the polynomial
    path. The live domain is ``[knots_u[pu], knots_u[nu]] ×
    [knots_v[pv], knots_v[nv]]``; the C++ side caps each degree at
    ``kMaxBSplineDegree = 7``.
    """

    def __init__(self, pu: int, pv: int, knots_u, knots_v, ctrl, weights=None):
        from scipy.interpolate import BSpline as _SciBSpline

        self.pu = int(pu)
        self.pv = int(pv)
        self.knots_u = np.asarray(knots_u, dtype=float)
        self.knots_v = np.asarray(knots_v, dtype=float)
        self.ctrl = np.asarray(ctrl, dtype=float)
        if self.ctrl.ndim != 3 or self.ctrl.shape[2] != 3:
            raise ValueError("ctrl must have shape (nu, nv, 3)")
        self.nu = self.ctrl.shape[0]
        self.nv = self.ctrl.shape[1]
        if self.knots_u.shape[0] != self.nu + self.pu + 1:
            raise ValueError("knots_u length must be nu + pu + 1")
        if self.knots_v.shape[0] != self.nv + self.pv + 1:
            raise ValueError("knots_v length must be nv + pv + 1")
        if self.pu > 7 or self.pv > 7:
            raise ValueError("degree exceeds the C++ kMaxBSplineDegree = 7")

        self.weights = None
        if weights is not None:
            self.weights = np.asarray(weights, dtype=float).reshape(self.nu, self.nv)
            if self.weights.shape != (self.nu, self.nv):
                raise ValueError("weights shape must be (nu, nv)")

        # Pre-build the v-direction BSplines for each row (and their derivatives).
        # For the rational form, work in homogeneous coords (x*w, y*w, z*w, w).
        self._u0 = float(self.knots_u[self.pu])
        self._u1 = float(self.knots_u[self.nu])
        self._v0 = float(self.knots_v[self.pv])
        self._v1 = float(self.knots_v[self.nv])

    def _row_splines(self, deriv: int = 0):
        """Build v-direction BSplines for each row (optionally derivative order).

        Derivative order is clamped to the spline degree (higher orders return a
        zero-valued callable, matching C++ bspline_basis_ders).
        """
        from scipy.interpolate import BSpline as _SciBSpline
        if self.weights is None:
            coeffs = self.ctrl  # (nu, nv, 3)
        else:
            coeffs = np.concatenate(
                [self.weights[:, :, None] * self.ctrl,
                 self.weights[:, :, None]],
                axis=2,
            )  # (nu, nv, 4)
        edim = coeffs.shape[-1]
        if deriv > self.pv:
            # Beyond the degree: all derivatives are zero.
            zero = np.zeros(edim)
            return [(lambda v, _z=zero: _z.copy()) for _ in range(self.nu)], edim
        splines = []
        for i in range(self.nu):
            spl = _SciBSpline(self.knots_v, coeffs[i, :, :], self.pv)
            if deriv > 0:
                spl = spl.derivative(deriv)
            splines.append(spl)
        return splines, edim

    def _eval_ders(self, u: float, v: float, nd: int):
        """Evaluate S and partial derivatives up to order nd via row-then-column.

        Returns a 3×3 array S[a][b] = ∂^(a+b)S/∂u^a ∂v^b (a,b ≤ nd),
        matching the C++ BSplineSurfaceParam::ders output layout.
        """
        from scipy.interpolate import BSpline as _SciBSpline

        # Evaluate the v-direction at v for each derivative order 0..nd.
        # row_vals[a] = array (nu, edim) of the a-th v-derivative of each row at v.
        row_vals = []
        _, edim = self._row_splines(0)
        for a in range(nd + 1):
            spls, _ = self._row_splines(a)
            vals = np.array([spl(v) for spl in spls])  # (nu, edim)
            row_vals.append(vals)

        # Now evaluate the u-direction at u for each derivative order 0..nd,
        # using the row results as control points. U-derivative order is clamped
        # to pu (higher orders are zero, matching C++ bspline_basis_ders).
        S = np.zeros((3, 3, max(edim, 3)))
        for b in range(nd + 1):
            u_spl = _SciBSpline(self.knots_u, row_vals[b], self.pu)
            for a in range(nd + 1):
                if a == 0:
                    val = np.asarray(u_spl(u), dtype=float)
                elif a <= self.pu:
                    val = np.asarray(u_spl.derivative(a)(u), dtype=float)
                else:
                    val = np.zeros(edim)
                S[a, b, :edim] = val

        if self.weights is None:
            return S[:, :, :3]

        # Rational: apply the bivariate quotient rule for S = A/w.
        # A[a][b] and w[a][b] are in S[.., :3] and S[.., 3] respectively.
        A = S[:, :, :3].copy()
        w = S[:, :, 3].copy()
        out = np.zeros((3, 3, 3))
        iw = 1.0 / w[0, 0]
        out[0, 0] = iw * A[0, 0]
        if nd >= 1:
            out[1, 0] = iw * (A[1, 0] - w[1, 0] * out[0, 0])
            out[0, 1] = iw * (A[0, 1] - w[0, 1] * out[0, 0])
            out[1, 1] = iw * (A[1, 1] - w[1, 0] * out[0, 1]
                              - w[0, 1] * out[1, 0] - w[1, 1] * out[0, 0])
        if nd >= 2:
            out[2, 0] = iw * (A[2, 0] - 2.0 * w[1, 0] * out[1, 0] - w[2, 0] * out[0, 0])
            out[0, 2] = iw * (A[0, 2] - 2.0 * w[0, 1] * out[0, 1] - w[0, 2] * out[0, 0])
        return out

    def eval(self, u: float, v: float) -> np.ndarray:
        return self._eval_ders(u, v, 0)[0, 0]

    def frame(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        S = self._eval_ders(u, v, 1)
        return S[1, 0], S[0, 1]

    def invert(self, p: np.ndarray) -> tuple[float, float]:
        """Coarse-grid-seeded Newton on (S-p)·S_u = (S-p)·S_v = 0 (mirrors C++)."""
        p = np.asarray(p, dtype=float)
        u0, u1, v0, v1 = self._u0, self._u1, self._v0, self._v1
        k_seed = 8
        best_q = (u0, v0)
        best_d = np.inf
        for i in range(k_seed + 1):
            for j in range(k_seed + 1):
                u = u0 + (u1 - u0) * i / k_seed
                v = v0 + (v1 - v0) * j / k_seed
                d = self.eval(u, v) - p
                dd = float(np.dot(d, d))
                if dd < best_d:
                    best_d = dd
                    best_q = (u, v)
        u, v = best_q
        for _ in range(12):
            S = self._eval_ders(u, v, 2)
            d = S[0, 0] - p
            f1 = float(np.dot(d, S[1, 0]))
            f2 = float(np.dot(d, S[0, 1]))
            j11 = float(np.dot(S[1, 0], S[1, 0]) + np.dot(d, S[2, 0]))
            j12 = float(np.dot(S[1, 0], S[0, 1]) + np.dot(d, S[1, 1]))
            j22 = float(np.dot(S[0, 1], S[0, 1]) + np.dot(d, S[0, 2]))
            det = j11 * j22 - j12 * j12
            if abs(det) < 1e-30:
                break
            u = float(np.clip(u - (j22 * f1 - j12 * f2) / det, u0, u1))
            v = float(np.clip(v - (-j12 * f1 + j11 * f2) / det, v0, v1))
        return u, v

    @property
    def dim(self) -> int:
        return 2

    def project(self, p: np.ndarray) -> np.ndarray:
        u, v = self.invert(p)
        return self.eval(u, v)

    def tangent_space(self, q: np.ndarray) -> np.ndarray:
        u, v = self.invert(q)
        su, sv = self.frame(u, v)
        return _orthonormalize2(su, sv)

    def normal(self, q: np.ndarray) -> np.ndarray:
        u, v = self.invert(q)
        su, sv = self.frame(u, v)
        return _normalize(np.cross(su, sv))
