"""Modified-determinant continuation (delta schedule).

Recovers a folded start (some ``det A <= 0``) to an all-positive-determinant mesh
by minimising the smooth-determinant *untangling* metric at a decreasing sequence
of continuation parameters ``delta``. Each ``delta`` is large enough that the
surrogate ``Dh(delta) = 0.5*(D + sqrt(D^2 + 4*delta^2))`` is smooth and the
energy finite everywhere; as the mesh unfolds, ``delta`` shrinks geometrically
until ``min det A > margin``.

The inner loop runs on the C++ backend via
:func:`egg.smoothing.cpp_backend.cpp_untangle`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import batch as _batch

__all__ = ["untangle", "UntangleResult"]


@dataclass
class UntangleResult:
    """Outcome of an :func:`untangle` run."""

    converged: bool
    min_det: float          # final raw min det A over the grid
    delta: float            # final continuation parameter
    outer_iters: int        # delta-steps taken
    no_op: bool = False     # input already valid (margin met), nothing done


def grid_min_det(X: np.ndarray, energy_stencil: dict) -> float:
    """Raw ``min det A`` over every (cell, corner) sample (host, NumPy)."""
    es = energy_stencil
    A = _batch.assemble_A(X, es["gc"], es["gn0"], es["gn1"], es["s0"], es["s1"])
    det_A = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    return float(det_A.min())


def untangle(
    grid,
    ctx,
    *,
    shrink: float = 0.8,
    margin: float = 1e-9,
    sweeps_per_delta: int = 20,
    delta0_factor: float = 2.0,
    max_outer: int = 60,
    verbose: bool = False,
) -> UntangleResult:
    """delta-continuation untangler on the C++ backend.

    Parameters
    ----------
    grid : MultiBlockGrid
        Mutated in place; on return its nodes hold the (hopefully) untangled mesh.
    ctx : SweepContext
        Built via :func:`egg.smoothing.solver.build_sweep_context`.
    shrink : float
        Geometric delta decrease per outer step.
    margin : float
        Target validity margin: stop once ``min det A > margin``.
    sweeps_per_delta : int
        Untangle sweeps per fixed delta.
    delta0_factor : float
        delta_0 = ``delta0_factor * max(|min det A|, eps)`` — comfortably above
        the worst inversion so the surrogate is smooth.
    max_outer : int
        Maximum delta-steps before reporting a stall.

    Returns
    -------
    UntangleResult
        ``converged`` is False on a stall (mesh left at its best effort, never
        hung). ``no_op`` is True when the input already met ``margin``.
    """
    from .cpp_backend import cpp_untangle

    es = ctx.energy_stencil

    md = grid_min_det(grid.global_nodes, es)
    if md > margin:
        return UntangleResult(True, md, 0.0, 0, no_op=True)

    X_out, md_final, outer_iters, delta_final = cpp_untangle(
        ctx, grid, grid.global_nodes,
        sweeps_per_delta=sweeps_per_delta,
        delta0_factor=delta0_factor,
        shrink=shrink,
        max_outer=max_outer,
        margin=margin,
    )

    grid.global_nodes = X_out
    for bi, block in enumerate(grid.blocks):
        block.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]

    converged = md_final > margin
    return UntangleResult(converged, md_final, float(delta_final), outer_iters, no_op=False)
