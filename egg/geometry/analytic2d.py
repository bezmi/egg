"""2D analytic geometry primitives: Circle, LineSegment, Ellipse.

Each entity also carries a standalone-Python parametric form ``eval(t)`` /
``deriv(t)`` over ``[t0, t1]`` (with ``closed`` marking periodic curves) so
grid edges can place nodes along it without touching the C++ core.
"""

import numpy as np

from .base import GeometryEntity

__all__ = ["Circle", "Ellipse", "LineSegment"]


class Circle(GeometryEntity):
    """A circle in 2D — a 1D curve.

    Parametric form: ``C + r (cos t, sin t)``, t in [0, 2*pi), periodic.
    """

    t0: float = 0.0
    t1: float = 2.0 * np.pi
    closed: bool = True

    @property
    def dim(self) -> int:
        return 1

    def __init__(self, center, radius: float):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

    def eval(self, t: float) -> np.ndarray:
        return self.center + self.radius * np.array([np.cos(t), np.sin(t)])

    def deriv(self, t: float) -> np.ndarray:
        return self.radius * np.array([-np.sin(t), np.cos(t)])

    def project(self, p: np.ndarray) -> np.ndarray:
        """Closest point on the circle to p."""
        diff = np.asarray(p, dtype=float) - self.center
        dist = np.linalg.norm(diff)
        if dist < 1e-15:
            return self.center + np.array([self.radius, 0.0])
        return self.center + self.radius * diff / dist

    def tangent_space(self, q: np.ndarray) -> np.ndarray:
        """Unit tangent vector at q (90 deg CCW from outward radial). Shape (2, 1)."""
        radial = np.asarray(q, dtype=float) - self.center
        r_norm = np.linalg.norm(radial)
        if r_norm < 1e-15:
            return np.array([[1.0], [0.0]])
        n = radial / r_norm
        t = np.array([-n[1], n[0]])
        return t.reshape(2, 1)

    def normal(self, q: np.ndarray) -> np.ndarray:
        """Outward radial unit normal. Shape (2,)."""
        radial = np.asarray(q, dtype=float) - self.center
        r_norm = np.linalg.norm(radial)
        if r_norm < 1e-15:
            return np.array([1.0, 0.0])
        return radial / r_norm


class LineSegment(GeometryEntity):
    """A line segment in 2D — a 1D curve.

    Parametric form: ``start + t (end - start)``, t in [0, 1].
    """

    t0: float = 0.0
    t1: float = 1.0
    closed: bool = False

    @property
    def dim(self) -> int:
        return 1

    def __init__(self, start, end):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)

    def eval(self, t: float) -> np.ndarray:
        return self.start + t * (self.end - self.start)

    def deriv(self, t: float) -> np.ndarray:
        return self.end - self.start

    def project(self, p: np.ndarray) -> np.ndarray:
        """Closest point on the segment to p."""
        p = np.asarray(p, dtype=float)
        ab = self.end - self.start
        ab_sq = np.dot(ab, ab)
        if ab_sq < 1e-30:
            return self.start.copy()
        t = np.dot(p - self.start, ab) / ab_sq
        t = np.clip(t, 0.0, 1.0)
        return self.start + t * ab

    def tangent_space(self, q: np.ndarray) -> np.ndarray:
        """Unit direction vector (end - start) / |end - start|. Shape (2, 1)."""
        ab = self.end - self.start
        norm = np.linalg.norm(ab)
        if norm < 1e-15:
            return np.array([[1.0], [0.0]])
        return (ab / norm).reshape(2, 1)

    def normal(self, q: np.ndarray) -> np.ndarray:
        """Unit normal (tangent rotated 90 deg CCW). Shape (2,)."""
        t = self.tangent_space(q)[:, 0]
        return np.array([-t[1], t[0]])


class Ellipse(GeometryEntity):
    """An ellipse in 2D — a 1D curve.

    Uses radial-scaling projection (exact for circles, approximate for ellipses).

    Parametric form: ``C + (rx cos t, ry sin t)``, t in [0, 2*pi), periodic.
    """

    t0: float = 0.0
    t1: float = 2.0 * np.pi
    closed: bool = True

    @property
    def dim(self) -> int:
        return 1

    def __init__(self, center, rx: float, ry: float):
        self.center = np.asarray(center, dtype=float)
        self.rx = float(rx)
        self.ry = float(ry)

    def eval(self, t: float) -> np.ndarray:
        return self.center + np.array([self.rx * np.cos(t), self.ry * np.sin(t)])

    def deriv(self, t: float) -> np.ndarray:
        return np.array([-self.rx * np.sin(t), self.ry * np.cos(t)])

    def project(self, p: np.ndarray) -> np.ndarray:
        """Closest point on the ellipse (radial-scaling approximation)."""
        p = np.asarray(p, dtype=float)
        diff = p - self.center
        scaled = np.array([diff[0] / self.rx, diff[1] / self.ry])
        dist = np.linalg.norm(scaled)
        if dist < 1e-15:
            scaled = np.array([1.0, 0.0])
        else:
            scaled = scaled / dist
        return self.center + scaled * np.array([self.rx, self.ry])

    def tangent_space(self, q: np.ndarray) -> np.ndarray:
        """Unit tangent via parametric angle. Shape (2, 1)."""
        q = np.asarray(q, dtype=float)
        diff = q - self.center
        angle = np.arctan2(diff[1] / self.ry, diff[0] / self.rx)
        t = np.array([-self.rx * np.sin(angle), self.ry * np.cos(angle)])
        norm = np.linalg.norm(t)
        if norm < 1e-15:
            return np.array([[1.0], [0.0]])
        return (t / norm).reshape(2, 1)

    def normal(self, q: np.ndarray) -> np.ndarray:
        """Outward normal (gradient of implicit form). Shape (2,)."""
        q = np.asarray(q, dtype=float)
        diff = q - self.center
        n = np.array([diff[0] / (self.rx * self.rx), diff[1] / (self.ry * self.ry)])
        nrm = np.linalg.norm(n)
        if nrm < 1e-15:
            return np.array([1.0, 0.0])
        return n / nrm
