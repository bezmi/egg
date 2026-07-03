"""Abstract GeometryEntity interface shared by all 2D/3D entities."""

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["GeometryEntity"]


class GeometryEntity(ABC):
    """Abstract geometry entity of dimension ``dim`` embedded in R^d.

    Entities implement two protocols:

    **Projection protocol** (all entities; used by boundary snapping and
    the C++ encoding path): :meth:`project`, :meth:`tangent_space`, and —
    for codimension-1 entities — :meth:`normal`.

    **Parametric protocol** (curve entities constructed in Python; used by
    :class:`~egg.geometry.frontend2d.Edge` for node placement): ``eval(t)``
    and ``deriv(t)`` over the native parameter interval ``[t0, t1]``, with
    ``closed = True`` marking periodic curves and :meth:`eval_frac` the
    normalized-parameter counterpart. The native parameter is
    curve-specific: radians for arcs/circles, knot values for B-splines,
    [0, 1] for Béziers, segments and composite paths.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the entity (0=point, 1=curve, 2=surface)."""
        ...

    @abstractmethod
    def project(self, p: np.ndarray) -> np.ndarray:
        """Closest point on the entity to p. Shape (d,)."""
        ...

    @abstractmethod
    def tangent_space(self, q: np.ndarray) -> np.ndarray:
        """Orthonormal basis of the tangent space at q. Shape (d, dim)."""
        ...

    def normal(self, q: np.ndarray) -> np.ndarray:
        """Normal vector at q, shape (d,). Defined for codimension 1."""
        raise NotImplementedError

    def eval_frac(self, t: float) -> np.ndarray:
        """Evaluate at fractional parameter t in [0, 1] mapped onto [t0, t1].

        The fractional counterpart of ``eval`` (which takes the native
        parameter). Available on entities implementing the parametric
        protocol; raises AttributeError otherwise.
        """
        t = min(max(float(t), 0.0), 1.0)
        return self.eval(self.t0 + t * (self.t1 - self.t0))
