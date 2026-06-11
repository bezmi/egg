"""Transfinite interpolation (Boolean sum, d-general)."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from egg.core.types import Block

__all__ = ["tfi_fill_interior"]


def tfi_fill_interior(block: Block) -> None:
    """Fill all interior nodes of a block using d-dimensional TFI.

    Boundary nodes (corners, edges, faces) must already be set before calling.
    The block.nodes array is modified in-place.

    Uses the Boolean sum formula:
        TFI = corner_interp + Σ_axis Σ_side w*(face_val - corner_interp_at_face)

    Parameters
    ----------
    block : Block
        Block whose interior nodes will be filled.
    """
    nodes = block.nodes
    d = block.d
    shape = block.logical_shape
    if d == 0:
        return

    # Only fill nodes strictly interior (not on any boundary)
    for idx in product(*(range(1, s - 1) for s in shape)):
        nodes[idx] = _tfi_point(nodes, idx, shape, d)


def _tfi_point(
    nodes: np.ndarray, idx: tuple[int, ...], shape: tuple[int, ...], d: int
) -> np.ndarray:
    """Evaluate TFI at a single interior point."""
    xi = np.array([i / (s - 1) for i, s in zip(idx, shape)], dtype=float)

    # Base: multilinear corner interpolation
    result = _corner_interp(nodes, xi, shape, d)

    # Face corrections: for each axis, add weighted difference between
    # the actual face node value and the corner interp evaluated at that face
    for axis in range(d):
        for side in (0, 1):
            fixed = 0 if side == 0 else shape[axis] - 1

            # Face node value (from the already-filled boundary array)
            face_idx = list(idx)
            face_idx[axis] = fixed
            face_val = nodes[tuple(face_idx)]

            # Corner interpolation evaluated with this axis fixed at side
            xi_fixed = xi.copy()
            xi_fixed[axis] = float(side)
            corner_val = _corner_interp(nodes, xi_fixed, shape, d)

            w = (1.0 - xi[axis]) if side == 0 else xi[axis]
            result = result + w * (face_val - corner_val)

    return result


def _corner_interp(
    nodes: np.ndarray, xi: np.ndarray, shape: tuple[int, ...], d: int
) -> np.ndarray:
    """Multilinear interpolation from the 2**d corner nodes.

    Parameters
    ----------
    nodes : ndarray
        Shape (n_0, ..., n_{d-1}, spatial_dim).
    xi : ndarray
        Parametric coordinates in [0, 1] for each axis.
    shape : tuple
        Logical node counts per axis.
    d : int
        Spatial dimension.

    Returns
    -------
    ndarray of shape (spatial_dim,)
    """
    result = np.zeros(nodes.shape[-1])
    for offset in product((0, 1), repeat=d):
        weight = 1.0
        corner_idx = [0] * d
        for axis in range(d):
            if offset[axis] == 0:
                weight *= 1.0 - xi[axis]
                corner_idx[axis] = 0
            else:
                weight *= xi[axis]
                corner_idx[axis] = shape[axis] - 1
        result = result + weight * nodes[tuple(corner_idx)]
    return result
