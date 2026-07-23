# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""2D analytic geometry primitives: Circle, LineSegment, Ellipse.

Each entity carries the parametric form in standalone Python (so grid
edges can place nodes along it without the compiled core); projection and
tangents come from the C++ core via the
:class:`~egg.geometry.base.GeometryEntity` base methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import GeometryEntity

if TYPE_CHECKING:
    from egg.geometry.frontend2d import Vector3

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
        """Point at angle t (radians)."""
        return self.center + self.radius * np.array([np.cos(t), np.sin(t)])

    def deriv(self, t: float) -> np.ndarray:
        """d/dt of :meth:`eval`."""
        return self.radius * np.array([-np.sin(t), np.cos(t)])

    def normal(self, q: np.ndarray) -> np.ndarray:
        """Outward radial unit normal (a convention override: the base
        tangent-CCW rotation points inward on a CCW closed curve). Shape (2,)."""
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
    # Construction provenance retained by frontend2d.Bezier (a 2-point Bézier
    # returns a LineSegment); read by egg.webui.scene.
    points: list[Vector3] | None = None

    @property
    def dim(self) -> int:
        return 1

    def __init__(self, start, end):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)

    def eval(self, t: float) -> np.ndarray:
        """Point at parameter t in [0, 1]."""
        return self.start + t * (self.end - self.start)

    def deriv(self, t: float) -> np.ndarray:
        """d/dt of :meth:`eval` (constant)."""
        return self.end - self.start


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
        """Point at parametric angle t (radians)."""
        return self.center + np.array([self.rx * np.cos(t), self.ry * np.sin(t)])

    def deriv(self, t: float) -> np.ndarray:
        """d/dt of :meth:`eval`."""
        return np.array([-self.rx * np.sin(t), self.ry * np.cos(t)])

    def normal(self, q: np.ndarray) -> np.ndarray:
        """Outward unit normal, the implicit-form gradient (a convention
        override: the base tangent-CCW rotation points inward on a CCW
        closed curve). Shape (2,)."""
        q = np.asarray(q, dtype=float)
        diff = q - self.center
        n = np.array([diff[0] / (self.rx * self.rx), diff[1] / (self.ry * self.ry)])
        nrm = np.linalg.norm(n)
        if nrm < 1e-15:
            return np.array([1.0, 0.0])
        return n / nrm
