# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Projection + tangential sliding constraints."""

from __future__ import annotations

import numpy as np

__all__ = ["project_nodes", "tangential_slide", "tangent_projector"]


def tangent_projector(entity, q: np.ndarray) -> np.ndarray:
    """Projector ``P = T Tᵀ`` onto the entity's tangent space at ``q``.

    ``T = entity.tangent_space(q)`` is the ``(d, k)`` orthonormal tangent basis
    (``k = 1`` for a curve in 2D). ``P`` removes the normal component of a step.
    """
    T = np.asarray(entity.tangent_space(q), dtype=float)  # (d, k)
    return T @ T.T


def tangential_slide(pos: np.ndarray, delta: np.ndarray, entity) -> np.ndarray:
    """Slide ``pos`` by the tangential part of ``delta``, then re-project.

    The step is restricted to the tangent space (so the motion has no normal
    component to first order), then ``entity.project`` snaps the result back
    exactly onto the entity (orthogonal projection). For a ``LineSegment`` the
    projection clamps to the segment, so a node cannot slide past an endpoint
    (i.e. it cannot traverse a domain corner).
    """
    pos = np.asarray(pos, dtype=float)
    P = tangent_projector(entity, pos)
    moved = pos + P @ np.asarray(delta, dtype=float)
    return np.asarray(entity.project(moved), dtype=float)


def project_nodes(grid, dof_constraints: dict | None = None) -> None:
    """Snap every sliding DOF back onto its entity (orthogonal projection).

    Propagates the corrected positions into the per-block node arrays.
    """
    if dof_constraints is None:
        dof_constraints = getattr(grid, "dof_constraints", {})
    # ``global_nodes`` may be a read-only view (e.g. fresh from ``np.asarray`` of
    # a JAX array after a fused sweep); ensure it is writable before snapping.
    if not grid.global_nodes.flags.writeable:
        grid.global_nodes = np.array(grid.global_nodes)
    for gidx, entity in dof_constraints.items():
        grid.global_nodes[gidx] = entity.project(grid.global_nodes[gidx])
    for bi, block in enumerate(grid.blocks):
        block.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]
