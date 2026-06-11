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
