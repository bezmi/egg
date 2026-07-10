# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""3D geometry construction front-end: points, curves, grid-node placement.

Mirrors :mod:`egg.geometry.frontend2d` for 3D. :class:`Vector3` is a genuine
3D point (no planar guard). :class:`Edge` wraps a 3D parametric curve for
normalized-parameter node placement; :class:`Node` records its host edge and
parameter, which drives curve-aware edge spacing in
:meth:`egg.topology.block_topology.BlockTopology.initialize_grid`.
"""

from __future__ import annotations

import numpy as np

from .base import _INHERIT

__all__ = ["Vector3", "Edge", "Node", "Bezier3"]


class Vector3:
    """3D point with gdtk ``Vector3`` semantics (add/sub/scalar-mul/abs)."""

    __slots__ = ("x", "y", "z", "fixed")

    def __init__(self, x, y, z=0.0, *, fixed=False):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.fixed = bool(fixed)

    def _arr(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __repr__(self) -> str:
        return f"Vector3({self.x}, {self.y}, {self.z})"

    def __add__(self, other):
        o = other._arr() if isinstance(other, Vector3) else np.asarray(other, float)
        return Vector3(*(self._arr() + o))

    def __sub__(self, other):
        o = other._arr() if isinstance(other, Vector3) else np.asarray(other, float)
        return Vector3(*(self._arr() - o))

    def __mul__(self, s):
        return Vector3(*(self._arr() * float(s)))

    __rmul__ = __mul__

    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)

    def __abs__(self) -> float:
        return float(np.linalg.norm(self._arr()))


class Bezier3:
    """Bézier curve in 3D from control points (de Casteljau), t in [0, 1]."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("Bezier3 needs an (n, 3) control-point array")
        self.t0 = 0.0
        self.t1 = 1.0
        self.name = None
        self.tag = None

    def eval(self, t: float) -> np.ndarray:
        pts = self.points
        while len(pts) > 1:
            pts = (1.0 - t) * pts[:-1] + t * pts[1:]
        return pts[0]

    def eval_frac(self, t: float) -> np.ndarray:
        return self.eval(self.t0 + t * (self.t1 - self.t0))


class Edge:
    """3D grid edge: a parametric curve re-parameterized over t in [0, 1].

    Wraps any entity exposing ``eval`` / ``t0`` / ``t1``. Pure Python; the
    wrapped :attr:`entity` is what topology association and encoding consume.
    """

    def __init__(self, entity, *, name=None, tag=_INHERIT):
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

    @property
    def dim(self) -> int:
        return 1

    @property
    def name(self):
        return self.entity.name

    @name.setter
    def name(self, value):
        self.entity.name = value

    def named(self, name, *, tag=_INHERIT):
        self.entity.name = name
        if tag is not _INHERIT:
            self.entity.tag = tag
        return self

    def _tau(self, t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        return self.entity.t0 + t * (self.entity.t1 - self.entity.t0)

    def point_at(self, t: float) -> Vector3:
        """Physical point at fractional parameter t (a plain :class:`Vector3`)."""
        p = np.asarray(self.entity.eval(self._tau(t)), dtype=float)
        return Vector3(p[0], p[1], p[2])

    def place_node(self, t: float, *, fixed: bool = False) -> "Node":
        """Grid node placed on this edge at fractional parameter t."""
        return Node(self, float(np.clip(t, 0.0, 1.0)), fixed=fixed)

    def project(self, p):
        return self.entity.project(p)

    def tangent_space(self, q):
        return self.entity.tangent_space(q)

    def normal(self, q):
        return self.entity.normal(q)


class Node(Vector3):
    """3D point placed on an :class:`Edge` at normalized parameter ``t``.

    Behaves as a :class:`Vector3` while remembering its host ``edge`` and
    parameter ``t``, which the topology reads to space edges along the curve.
    """

    __slots__ = ("edge", "t")

    def __init__(self, edge: Edge, t: float, *, fixed: bool = False):
        p = edge.point_at(t)
        super().__init__(p.x, p.y, p.z, fixed=fixed)
        self.edge = edge
        self.t = float(t)
