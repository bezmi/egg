# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""A from node patch; sample points (dimension-agnostic)."""

from __future__ import annotations

from itertools import product

import numpy as np

__all__ = [
    "compute_jacobian",
    "compute_corner_jacobian",
    "corner_sample",
    "compute_all_jacobians",
    "iter_sample_points",
]


def compute_jacobian(nodes: np.ndarray, cell_base: tuple[int, ...]) -> np.ndarray:
    """Forward-difference Jacobian at a cell's lower-left corner.

    A[:, k] = nodes[base + e_k] - nodes[base]   for each axis k.

    Parameters
    ----------
    nodes : ndarray, shape (n_0, ..., n_{d-1}, spatial_dim)
    cell_base : tuple of int
        Lower-corner logical index of the cell.

    Returns
    -------
    A : ndarray, shape (d, d) where d = nodes.ndim - 1
    """
    d = nodes.ndim - 1
    A = np.zeros((d, d))
    base = np.asarray(cell_base)
    for axis in range(d):
        neighbor = base.copy()
        neighbor[axis] += 1
        A[:, axis] = nodes[tuple(neighbor)] - nodes[tuple(base)]
    return A


def corner_sample(
    nodes: np.ndarray,
    cell_base: tuple[int, ...],
    corner_offset: tuple[int, ...],
):
    """One-sided Jacobian at a single cell corner, with its scatter stencil.

    For corner offset ``o`` (each component 0/1), the in-cell neighbour along
    axis ``k`` is one step toward the cell interior; with sign
    ``s_k = +1 if o_k == 0 else -1``::

        A[:, k] = s_k * (nodes[nbr_k] - nodes[corner]).

    For an affine cell this reproduces the affine Jacobian at *every* corner with
    ``det(A) > 0``, so the ``1/det`` barrier and the metric see all corners — not
    just the lower-left one. This is what makes folds at any corner visible.

    Returns
    -------
    A : ndarray, shape (d, d)
    corner_idx : tuple of int
        Logical index of the corner node (``cell_base + corner_offset``).
    nbrs : list of (tuple, int)
        ``(neighbour logical index, sign s_k)`` for each axis ``k``.
    """
    d = nodes.ndim - 1
    base = np.asarray(cell_base, dtype=int)
    o = np.asarray(corner_offset, dtype=int)
    corner_idx = tuple(base + o)
    corner_pos = nodes[corner_idx]
    A = np.zeros((d, d))
    nbrs: list[tuple[tuple, int]] = []
    for k in range(d):
        s = 1 if o[k] == 0 else -1
        nbr = (base + o).copy()
        nbr[k] += s
        nbr_idx = tuple(nbr)
        A[:, k] = s * (nodes[nbr_idx] - corner_pos)
        nbrs.append((nbr_idx, s))
    return A, corner_idx, nbrs


def compute_corner_jacobian(
    nodes: np.ndarray,
    cell_base: tuple[int, ...],
    corner_offset: tuple[int, ...],
) -> np.ndarray:
    """One-sided Jacobian at a single cell corner (see :func:`corner_sample`)."""
    A, _, _ = corner_sample(nodes, cell_base, corner_offset)
    return A


def compute_all_jacobians(nodes: np.ndarray) -> np.ndarray:
    """Compute Jacobian at every cell (lower-left corner).

    Returns
    -------
    jacobians : ndarray, shape (cell_count_0, ..., cell_count_{d-1}, d, d)
    """
    d = nodes.ndim - 1
    shape = nodes.shape[:-1]
    cell_counts = tuple(s - 1 for s in shape)
    jacobians = np.zeros(cell_counts + (d, d))

    for cell_base in product(*[range(c) for c in cell_counts]):
        A = compute_jacobian(nodes, cell_base)
        jacobians[cell_base] = A

    return jacobians


def iter_sample_points(nodes: np.ndarray):
    """Iterate over all sample points: every corner of every cell.

    Each cell contributes ``2**d`` ``(cell_base, corner_offset)`` pairs, one per
    corner. Sampling all corners (rather than only the lower-left) is what lets
    the metric/barrier detect distortion and folds anywhere in a cell.

    Parameters
    ----------
    nodes : ndarray, shape (n_0, ..., n_{d-1}, spatial_dim)

    Yields
    ------
    cell_base : tuple of int
    corner_offset : tuple of int  (each component 0 or 1)
    """
    d = nodes.ndim - 1
    shape = nodes.shape[:-1]
    cell_counts = tuple(s - 1 for s in shape)

    for cell_base in product(*[range(c) for c in cell_counts]):
        for corner_offset in product((0, 1), repeat=d):
            yield cell_base, corner_offset
