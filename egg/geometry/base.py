"""Abstract GeometryEntity interface (project, tangent_space, normal)."""

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["GeometryEntity"]


class GeometryEntity(ABC):
    """Abstract geometry entity with dim (0=point, 1=curve, 2=surface)."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the entity (0=point, 1=curve, 2=surface)."""
        ...

    @abstractmethod
    def project(self, p: np.ndarray) -> np.ndarray:
        """Return the closest point on the entity to p."""
        ...

    @abstractmethod
    def tangent_space(self, q: np.ndarray) -> np.ndarray:
        """Return orthonormal basis of the entity's tangent space at q, shape (d, k)."""
        ...

    def normal(self, q: np.ndarray) -> np.ndarray:
        """Normal vector at q, shape (d,). Defined for entities of codimension 1."""
        raise NotImplementedError

    def eval_frac(self, t: float) -> np.ndarray:
        """Evaluate at fractional parameter t in [0, 1] mapped onto [t0, t1].

        The fractional counterpart of ``eval``, which takes the curve's native
        parameter (radians for arcs, knot values for B-splines, ...). Available
        on entities that expose the standalone parametric interface
        (``eval``/``t0``/``t1``); raises AttributeError otherwise.
        """
        t = min(max(float(t), 0.0), 1.0)
        return self.eval(self.t0 + t * (self.t1 - self.t0))
