"""Assemble F(x), gradient, (optional) Hessian.

Uses the DOF map to scatter contributions from each cell's sample
points into the global energy and gradient arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from .jacobian import compute_corner_jacobian, corner_sample, iter_sample_points
from .metrics import metric_value, metric_value_and_grad

if TYPE_CHECKING:
    from egg.core.types import MultiBlockGrid

__all__ = [
    "assemble_energy",
    "assemble_energy_vec",
    "assemble_gradient",
    "assemble_energy_and_gradient",
    "pack_x",
    "unpack_x",
    "copy_grid_state",
    "restore_grid_state",
]


def assemble_energy_vec(X, gc, gn0, gn1, s0, s1, W_inv) -> float:
    """Vectorized ``shape_2d`` global energy over precomputed stencil arrays.

    One NumPy pass over every (cell, corner) sample of the grid, reading
    positions straight from the flat ``global_nodes`` array ``X`` through
    integer index stencils (the same ``gc/gn0/gn1/s0/s1/W_inv`` layout
    :mod:`egg.smoothing.batch` uses, precomputed once in
    :func:`egg.smoothing.solver.build_sweep_context`).

    Numerically identical to :func:`assemble_energy` with ``metric="shape_2d"``.
    """
    Xc = X[gc]
    A = np.empty((gc.shape[0], 2, 2))
    A[:, :, 0] = s0[:, None] * (X[gn0] - Xc)
    A[:, :, 1] = s1[:, None] * (X[gn1] - Xc)
    T = np.einsum("pij,pjk->pik", A, W_inv)
    a, b, c, d = T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1]
    D = a * d - b * c
    s = a * a + b * b + c * c + d * d
    return float((s / (2.0 * D) - 1.0).sum())


def assemble_energy(
    grid: MultiBlockGrid,
    target_fn: Callable[..., np.ndarray],
    metric: str = "shape_2d",
) -> float:
    """Compute global energy F(x) over all blocks.

    Parameters
    ----------
    grid : MultiBlockGrid
    target_fn : callable
        Function (cell_base, corner_offset) -> W matrix of shape (d, d).
    metric : str
        One of "shape", "shape_2d", "shape_size". Defaults to ``"shape_2d"`` to
        match the solver's fast path (:func:`assemble_energy_vec` and
        :func:`egg.smoothing.solver.local_relaxation_sweep`), so the two energies
        agree by default.

    Returns
    -------
    F : float
    """
    total = 0.0

    for bi, block in enumerate(grid.blocks):
        nodes = block.nodes
        for cell_base, corner_offset in iter_sample_points(nodes):
            A = compute_corner_jacobian(nodes, cell_base, corner_offset)
            W = target_fn(bi, block, cell_base, corner_offset)
            T = A @ np.linalg.inv(W)
            total += metric_value(T, metric)

    return float(total)


def assemble_gradient(
    grid: MultiBlockGrid,
    target_fn: Callable[..., np.ndarray],
    metric: str = "shape",
) -> np.ndarray:
    """Compute ∇F(x) as (M, d) array for all global DOFs.

    Fixed DOF entries are set to zero.

    Parameters
    ----------
    grid : MultiBlockGrid
    target_fn : callable
    metric : str

    Returns
    -------
    grad : ndarray, shape (M, d)
    """
    d = grid.topology.d
    M = grid.global_node_count
    grad = np.zeros((M, d))

    for bi, block in enumerate(grid.blocks):
        dof_map = grid.block_dof_maps[bi]
        nodes = block.nodes

        for cell_base, corner_offset in iter_sample_points(nodes):
            A, corner_idx, nbrs = corner_sample(nodes, cell_base, corner_offset)
            W = target_fn(bi, block, cell_base, corner_offset)
            W_inv = np.linalg.inv(W)
            T = A @ W_inv

            _, dmu_dT = metric_value_and_grad(T, metric)
            dmu_dA = dmu_dT @ W_inv.T  # (d, d)

            # A[:, k] = s_k * (nbr_k - corner) -> nbr_k gets +s_k*col, corner -s_k*col
            gidx_corner = int(dof_map[corner_idx])
            for k, (nbr_idx, s) in enumerate(nbrs):
                col = s * dmu_dA[:, k]
                grad[int(dof_map[nbr_idx])] += col
                grad[gidx_corner] -= col

    # Zero out fixed DOFs
    grad[~grid.free_mask] = 0.0
    return grad


def assemble_energy_and_gradient(
    grid: MultiBlockGrid,
    target_fn: Callable[..., np.ndarray],
    metric: str = "shape",
) -> tuple[float, np.ndarray]:
    """Compute F(x) and ∇F(x) in a single pass."""
    d = grid.topology.d
    M = grid.global_node_count
    total = 0.0
    grad = np.zeros((M, d))

    for bi, block in enumerate(grid.blocks):
        dof_map = grid.block_dof_maps[bi]
        nodes = block.nodes

        for cell_base, corner_offset in iter_sample_points(nodes):
            A, corner_idx, nbrs = corner_sample(nodes, cell_base, corner_offset)
            W = target_fn(bi, block, cell_base, corner_offset)
            W_inv = np.linalg.inv(W)
            T = A @ W_inv

            val, dmu_dT = metric_value_and_grad(T, metric)
            dmu_dA = dmu_dT @ W_inv.T
            total += val

            gidx_corner = int(dof_map[corner_idx])
            for k, (nbr_idx, s) in enumerate(nbrs):
                col = s * dmu_dA[:, k]
                grad[int(dof_map[nbr_idx])] += col
                grad[gidx_corner] -= col

    grad[~grid.free_mask] = 0.0
    return float(total), grad


# ---- Pack / Unpack ----


def pack_x(grid: MultiBlockGrid) -> np.ndarray:
    """Flatten free DOF coordinates into a 1D optimization vector.

    Returns
    -------
    x : ndarray, shape (n_free * d,)
    """
    return grid.global_nodes[grid.free_mask].ravel()


def unpack_x(x: np.ndarray, grid: MultiBlockGrid) -> None:
    """Write x back into global_nodes and propagate to block.nodes.

    Parameters
    ----------
    x : ndarray, shape (n_free * d,)
    grid : MultiBlockGrid
    """
    d = grid.topology.d
    n_free = int(np.sum(grid.free_mask))
    x_reshaped = x.reshape(n_free, d)
    grid.global_nodes[grid.free_mask] = x_reshaped

    # Propagate to block node arrays via DOF maps
    for bi, block in enumerate(grid.blocks):
        dof_map = grid.block_dof_maps[bi]
        block.nodes[...] = grid.global_nodes[dof_map]


# ---- Grid state save/restore for visual snapshots ----


def copy_grid_state(grid: MultiBlockGrid) -> tuple[np.ndarray, list[np.ndarray]]:
    """Deep-copy the grid's node state.

    Returns (global_nodes_copy, block_nodes_copies).
    """
    return (
        grid.global_nodes.copy(),
        [b.nodes.copy() for b in grid.blocks],
    )


def restore_grid_state(
    grid: MultiBlockGrid, state: tuple[np.ndarray, list[np.ndarray]]
) -> None:
    """Restore grid state from a copy returned by copy_grid_state."""
    global_copy, block_copies = state
    grid.global_nodes[:] = global_copy
    for bi, nodes_copy in enumerate(block_copies):
        grid.blocks[bi].nodes[:] = nodes_copy
