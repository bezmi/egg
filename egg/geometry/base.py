# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Abstract GeometryEntity interface shared by all 2D/3D entities."""

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["GeometryEntity"]

# Sentinel: a tag left to inherit the entity's name (distinct from tag=None,
# which explicitly suppresses the export marker).
_INHERIT = object()


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
    def name(self) -> str | None:
        """Optional label for this entity (``None`` if unnamed).

        Naming an entity is the single source of truth for its identity in a
        topology: the built topology exposes ``{name: entity}`` as
        :attr:`~egg.topology.block_topology.BlockTopology.entities` for
        rendering, and (via :attr:`tag`)
        :meth:`~egg.topology.builder.TopologyBuilder.build` auto-derives a
        boundary marker for every associated face. Set it fluently with
        :meth:`named`.
        """
        return getattr(self, "_egg_name", None)

    @name.setter
    def name(self, value: str | None) -> None:
        self._egg_name = value

    @property
    def tag(self) -> str | None:
        """Export/boundary marker for this entity's associated faces.

        Defaults to :attr:`name`, so naming a curve is enough:
        :meth:`~egg.topology.builder.TopologyBuilder.build` tags every face
        associated with this entity under this marker (an explicit
        :meth:`~egg.topology.builder.TopologyBuilder.tag_boundary` on a
        specific face still overrides it). Set it distinct from the name to
        export several entities under one marker (e.g. two ``wall_*`` curves
        both tagged ``"wall"``), or to ``None`` to associate/draw the curve
        without emitting any marker.
        """
        t = getattr(self, "_egg_tag", _INHERIT)
        return self.name if t is _INHERIT else t

    @tag.setter
    def tag(self, value: str | None) -> None:
        self._egg_tag = value

    def named(self, name: str, *, tag=_INHERIT):
        """Set :attr:`name` (and optionally :attr:`tag`) and return ``self``
        (fluent), e.g. ``Spline(ring, closed=True).named("egg")`` or
        ``Line(...).named("wall_top", tag="wall")``."""
        self._egg_name = name
        if tag is not _INHERIT:
            self._egg_tag = tag
        return self

    @property
    def open_ends(self) -> bool:
        """Whether this segment releases sliding nodes beyond its ends.

        Set with :meth:`open_ended`. A bound node projects onto the segment
        while its foot parameter is within range and moves as a *free* node
        beyond either end (recaptured if its foot re-enters), so a guide can
        cover part of a rail of nodes without clamping the overflow at its
        endpoints. Straight segments only (Line / Line3).
        """
        return bool(getattr(self, "_egg_open_ends", False))

    def open_ended(self):
        """Make this segment an open-ended rail; return ``self`` (fluent).

        Nodes bound to it slide along the segment but drop off its ends and
        become free instead of clamping at the endpoints — see
        :attr:`open_ends`. Supported for straight segments (``Line`` /
        ``Line3``); other entity kinds raise at encode time.
        """
        self._egg_open_ends = True
        return self

    @property
    def boundary_layer(self) -> dict | None:
        """Wall-normal clustering requested on this entity, or ``None``.

        Set with :meth:`clustered`. Collected for every associated face by
        :meth:`~egg.topology.builder.TopologyBuilder.build` — exactly like
        :attr:`tag` — into the built topology's ``boundary_layer_specs``, which
        the pipeline's ``cluster_boundary_layers`` reads to build the clustering
        target. So a drawn / by-name topology clusters with no base builder. An
        explicit :meth:`~egg.topology.builder.TopologyBuilder.set_boundary_layer`
        for the same entity overrides it.
        """
        return getattr(self, "_egg_bl", None)

    def clustered(
        self,
        *,
        first_height: float,
        growth: float,
        n_layers: int = 8,
        max_height: float | None = None,
        tangential_spacing: float | None = None,
        n_fixed: int = 1,
        relax_orthogonality=(),
    ):
        """Request wall-normal boundary-layer clustering on this entity; return
        ``self`` (fluent). Mirrors
        :meth:`~egg.topology.builder.TopologyBuilder.set_boundary_layer`, but
        travels with the geometry, so a by-name / drawn topology needs no base
        builder::

            egg = Spline(ring, closed=True).named("egg").clustered(
                first_height=5e-3, growth=1.5, n_fixed=2
            )

        ``relax_orthogonality`` is deprecated here: declare it on the topology
        (``TopologyBuilder.relax_orthogonality(...)`` or
        ``ExplicitTopology(relax_orthogonality=...)``); entries given here are
        forwarded at build time.
        """
        if relax_orthogonality:
            import warnings

            warnings.warn(
                "clustered(relax_orthogonality=...) is deprecated; use "
                "TopologyBuilder.relax_orthogonality(...) or "
                "ExplicitTopology(relax_orthogonality=...)",
                DeprecationWarning,
                stacklevel=2,
            )
        self._egg_bl = dict(
            first_height=first_height,
            growth=growth,
            n_layers=n_layers,
            max_height=max_height,
            tangential_spacing=tangential_spacing,
            n_fixed=int(n_fixed),
            relax_orthogonality=tuple(relax_orthogonality),
        )
        return self

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the entity (0=point, 1=curve, 2=surface)."""
        ...

    def project(self, p: np.ndarray) -> np.ndarray:
        """Closest point on the entity to p. Shape (d,)."""
        return self.project_many(np.asarray(p, dtype=float)[None])[0]

    def project_many(self, pts: np.ndarray) -> np.ndarray:
        """Closest point on the entity to each row of ``pts``. Shape (n, d).

        Projection is owned by the C++ core (the same SoA entity
        reconstruction the sweep kernels use); Python keeps no per-entity
        mirror. One batched call regardless of entity complexity.
        """
        from .entity_soa import project_frame_batch

        return project_frame_batch(self, np.asarray(pts, dtype=float))[0]

    def tangent_space(self, q: np.ndarray) -> np.ndarray:
        """Orthonormal basis of the tangent space at q. Shape (d, dim)."""
        return self.tangent_space_many(np.asarray(q, dtype=float)[None])[0]

    def tangent_space_many(self, Q: np.ndarray) -> np.ndarray:
        """Tangent basis at the foot of each row of ``Q``. Shape (n, d, dim)."""
        from .entity_soa import project_frame_batch

        return project_frame_batch(self, np.asarray(Q, dtype=float))[1]

    def normal(self, q: np.ndarray) -> np.ndarray:
        """Normal vector at q, shape (d,). Defined for codimension 1.

        Derived from the tangent basis: the 90-deg-CCW rotation for a 2D
        curve, the (unit) cross product of the basis columns for a 3D
        surface.
        """
        B = np.asarray(self.tangent_space(np.asarray(q, dtype=float)), dtype=float)
        d, k = B.shape
        if d == 2 and k == 1:
            t = B[:, 0]
            return np.array([-t[1], t[0]])
        if d == 3 and k == 2:
            n = np.cross(B[:, 0], B[:, 1])
            nrm = np.linalg.norm(n)
            if nrm < 1e-15:
                raise ValueError("degenerate tangent basis: normal undefined here")
            return n / nrm
        raise NotImplementedError("normal requires codimension 1")

    def eval_frac(self, t: float) -> np.ndarray:
        """Evaluate at fractional parameter t in [0, 1] mapped onto [t0, t1].

        The fractional counterpart of ``eval`` (which takes the native
        parameter). Available on entities implementing the parametric
        protocol; raises AttributeError otherwise.
        """
        t = min(max(float(t), 0.0), 1.0)
        return self.eval(self.t0 + t * (self.t1 - self.t0))
