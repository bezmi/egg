"""Local Gauss-Seidel relaxation with damped Newton steps.

Dimension-agnostic: all gradient/Hessian computations use range(d).

A :class:`SweepContext` precomputes the per-DOF incidence maps and the target
``W_inv`` once (the target is a function of logical position only, so it never
changes during a sweep), keeping the hot path free of repeated map rebuilds,
matrix inverses, and O(N) node-propagation scans.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Callable

import numpy as np

from . import batch as _batch

if TYPE_CHECKING:
    from egg.core.types import MultiBlockGrid

__all__ = [
    "local_relaxation",
    "local_relaxation_sweep",
    "SweepContext",
    "build_sweep_context",
]


@dataclass
class SweepContext:
    """Precomputed, position-independent data reused across sweeps.

    Attributes
    ----------
    dof_to_cells : list[list[(block_idx, cell_base)]]
        Cells whose energy depends on each global DOF (all 2**d corners).
    dof_to_locals : list[list[(block_idx, logical_idx)]]
        Block-node slots that alias each global DOF (for O(incidence) moves).
    w_inv : dict[(block_idx, cell_base, corner_offset), ndarray]
        Cached ``inv(target_fn(cell_base, corner_offset))`` per sample.
    dof_patches : list[dict]
        Per-DOF batched stencil arrays for vectorized patch evaluation
        (see :mod:`egg.smoothing.batch`). Each dict contains ``gc``, ``gn0``,
        ``gn1``, ``s0``, ``s1``, ``W_inv``, ``role`` arrays.
    energy_stencil : dict
        Global stencil arrays over *every* (cell, corner) sample of the grid,
        for one-pass vectorized energy assembly
        (:func:`egg.smoothing.objective.assemble_energy_vec`). Contains
        ``gc``, ``gn0``, ``gn1``, ``s0``, ``s1``, ``W_inv``.
    dof_colours : list[int]
        Greedy (Welsh-Powell) colour of each global DOF on the share-a-cell
        coupling graph. The graph is *not* bipartite (K4 per cell + O-grid
        valence-5 singularities), so this yields ~6 colours, not 2. DOFs of the
        same colour share no cell, so their relaxation moves are independent
        (Gauss-Seidel-exact when applied simultaneously).
    dof_patch_sizes : list[int]
        Number of corner samples ``P`` in each DOF's patch (0 if it has no
        incident cells, e.g. a fixed corner). The P values that actually occur
        on the demo grid are ``{8, 16, 20}``.
    colour_P_groups : dict[tuple[int, int], list[int]]
        ``(colour, P) -> sorted free-DOF indices`` for uniform-shape vmap
        batches. Only free DOFs with at least one incident cell are included.
    """

    dof_to_cells: list[list[tuple[int, tuple]]]
    dof_to_locals: list[list[tuple[int, tuple]]]
    w_inv: dict[tuple[int, tuple, tuple], np.ndarray]
    dof_patches: list[dict]
    energy_stencil: dict
    dof_colours: list[int]
    dof_patch_sizes: list[int]
    colour_P_groups: dict[tuple[int, int], list[int]]
    dof_constraint_tags: np.ndarray  # (M,) int32 entity type tag per DOF
    dof_constraint_params: np.ndarray  # (M, PARAM_PAD_SIZE) float64 per DOF
    # Lazily-built, position-independent stacked patch arrays per (colour, P) group.
    # Position-independent (indices/W_inv/J/role), so cached once and reused across
    # all sweeps. Populated on first sweep.
    jax_group_batches: dict | None = None

    @property
    def num_colours(self) -> int:
        """Number of distinct colours used (max colour index + 1)."""
        return (max(self.dof_colours) + 1) if self.dof_colours else 0

    def get_colour_P_groups(self, colour: int) -> dict[int, list[int]]:
        """Return ``{P: [dof_indices]}`` for free DOFs of the given colour."""
        groups: dict[int, list[int]] = {}
        for (c, P), dofs in self.colour_P_groups.items():
            if c == colour:
                groups[P] = dofs
        return groups


def _greedy_colour(adj: list[set[int]]) -> list[int]:
    """Welsh-Powell greedy colouring (highest degree first).

    The share-a-cell graph contains a K4 per quad cell, so this needs >= 4
    colours; on the demo grid it yields ~6. Returns a colour (0..k-1) per node.
    """
    M = len(adj)
    order = sorted(range(M), key=lambda v: -len(adj[v]))
    colours = [-1] * M
    for v in order:
        used = {colours[u] for u in adj[v] if colours[u] != -1}
        c = 0
        while c in used:
            c += 1
        colours[v] = c
    return colours


def build_sweep_context(
    grid: MultiBlockGrid, target_fn: Callable[..., np.ndarray]
) -> SweepContext:
    """Build the per-sweep incidence maps and cached target inverses once.

    With all-corner sampling a cell's energy depends on *all* 2**d of its corner
    nodes, so every corner DOF lists the cell as incident.
    """
    M = grid.global_node_count
    d = grid.topology.d
    dof_to_cells: list[list[tuple[int, tuple]]] = [[] for _ in range(M)]
    dof_to_locals: list[list[tuple[int, tuple]]] = [[] for _ in range(M)]
    w_inv: dict[tuple[int, tuple, tuple], np.ndarray] = {}
    corners = list(product((0, 1), repeat=d))

    # Batch stencil accumulators (one list per global DOF)
    dof_gc: list[list[int]] = [[] for _ in range(M)]
    dof_gn0: list[list[int]] = [[] for _ in range(M)]
    dof_gn1: list[list[int]] = [[] for _ in range(M)]
    dof_s0: list[list[int]] = [[] for _ in range(M)]
    dof_s1: list[list[int]] = [[] for _ in range(M)]
    dof_roles: list[list[int]] = [[] for _ in range(M)]
    dof_w_inv: list[list[np.ndarray]] = [[] for _ in range(M)]

    # Global energy stencil accumulators (one entry per (cell, corner) sample)
    en_gc: list[int] = []
    en_gn0: list[int] = []
    en_gn1: list[int] = []
    en_s0: list[int] = []
    en_s1: list[int] = []
    en_w_inv: list[np.ndarray] = []

    # Share-a-cell adjacency for DOF colouring (Step 2). Built across all blocks
    # from the unified dof maps, so shared-interface DOFs colour consistently.
    adj: list[set[int]] = [set() for _ in range(M)]

    for bi, block in enumerate(grid.blocks):
        dof_map = grid.block_dof_maps[bi]

        for logical_idx in product(*[range(s) for s in block.logical_shape]):
            dof = int(dof_map[logical_idx])
            dof_to_locals[dof].append((bi, logical_idx))

        for cell_base in block.iter_cells():
            corner_dofs = {int(dof_map[ci]) for ci in block.corner_indices(cell_base)}
            for dof in corner_dofs:
                dof_to_cells[dof].append((bi, cell_base))
            # Every pair of this cell's corner DOFs couples (K4 per quad).
            for i in corner_dofs:
                adj[i].update(corner_dofs)
                adj[i].discard(i)

            base_arr = np.asarray(cell_base, dtype=int)

            for co in corners:
                w_inv[(bi, cell_base, co)] = np.linalg.inv(
                    target_fn(bi, block, cell_base, co))

                o_arr = np.asarray(co, dtype=int)
                corner_idx = tuple(base_arr + o_arr)
                gidx_c = int(dof_map[corner_idx])

                s0 = 1 if o_arr[0] == 0 else -1
                nbr0 = (base_arr + o_arr).copy()
                nbr0[0] += s0
                gidx_n0 = int(dof_map[tuple(nbr0)])

                s1 = 1 if o_arr[1] == 0 else -1
                nbr1 = (base_arr + o_arr).copy()
                nbr1[1] += s1
                gidx_n1 = int(dof_map[tuple(nbr1)])

                wi = w_inv[(bi, cell_base, co)]

                # Each (cell, corner) sample contributes once to global energy
                en_gc.append(gidx_c)
                en_gn0.append(gidx_n0)
                en_gn1.append(gidx_n1)
                en_s0.append(s0)
                en_s1.append(s1)
                en_w_inv.append(wi)

                for dof in corner_dofs:
                    dof_gc[dof].append(gidx_c)
                    dof_gn0[dof].append(gidx_n0)
                    dof_gn1[dof].append(gidx_n1)
                    dof_s0[dof].append(s0)
                    dof_s1[dof].append(s1)
                    dof_w_inv[dof].append(wi)
                    if gidx_c == dof:
                        dof_roles[dof].append(0)
                    elif gidx_n0 == dof:
                        dof_roles[dof].append(1)
                    elif gidx_n1 == dof:
                        dof_roles[dof].append(2)
                    else:
                        dof_roles[dof].append(-1)

    dof_patches: list[dict] = []
    for dof_idx in range(M):
        p = dof_w_inv[dof_idx]
        patch = {
            "gc": np.array(dof_gc[dof_idx], dtype=np.intp),
            "gn0": np.array(dof_gn0[dof_idx], dtype=np.intp),
            "gn1": np.array(dof_gn1[dof_idx], dtype=np.intp),
            "s0": np.array(dof_s0[dof_idx], dtype=np.intp),
            "s1": np.array(dof_s1[dof_idx], dtype=np.intp),
            "W_inv": np.stack(p) if p else np.zeros((0, 2, 2)),
            "role": np.array(dof_roles[dof_idx], dtype=np.intp),
        }
        P = patch["gc"].shape[0]
        patch["J"] = _batch.make_chain_J(
            patch["s0"], patch["s1"], patch["W_inv"],
        ) if P > 0 else np.zeros((0, 4, 6))
        dof_patches.append(patch)

    energy_stencil = {
        "gc": np.array(en_gc, dtype=np.intp),
        "gn0": np.array(en_gn0, dtype=np.intp),
        "gn1": np.array(en_gn1, dtype=np.intp),
        "s0": np.array(en_s0, dtype=np.intp),
        "s1": np.array(en_s1, dtype=np.intp),
        "W_inv": np.stack(en_w_inv) if en_w_inv else np.zeros((0, 2, 2)),
    }

    # --- Step 2: greedy colouring + (colour, P) grouping ---
    dof_colours = _greedy_colour(adj)
    dof_patch_sizes = [dof_patches[d]["gc"].shape[0] for d in range(M)]

    colour_P_groups: dict[tuple[int, int], list[int]] = {}
    for dof_idx in range(M):
        if not grid.free_mask[dof_idx]:
            continue
        P = dof_patch_sizes[dof_idx]
        if P == 0:
            continue
        colour_P_groups.setdefault((dof_colours[dof_idx], P), []).append(dof_idx)

    # --- Constraint tags/params per DOF ---
    from egg.geometry.entity_encoding import PARAM_PAD_SIZE, encode_entity

    dof_constraint_tags = np.zeros(M, dtype=np.int32)
    dof_constraint_params = np.zeros((M, PARAM_PAD_SIZE), dtype=np.float64)
    for dof_idx, entity in grid.dof_constraints.items():
        tag, params = encode_entity(entity, d=d)
        dof_constraint_tags[dof_idx] = tag
        dof_constraint_params[dof_idx] = params

    return SweepContext(
        dof_to_cells, dof_to_locals, w_inv, dof_patches, energy_stencil,
        dof_colours, dof_patch_sizes, colour_P_groups,
        dof_constraint_tags, dof_constraint_params,
    )


def _patch_grad_hess(
    grid: MultiBlockGrid,
    dof_idx: int,
    ctx: SweepContext,
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Local gradient (d,) and Hessian (d, d) for a single global DOF.

    Uses precomputed batched stencils to evaluate all corner samples of the DOF's
    incident cells at once.
    """
    patch = ctx.dof_patches[dof_idx]
    return _batch.dof_grad_hess(
        grid.global_nodes,
        patch["gc"], patch["gn0"], patch["gn1"],
        patch["s0"], patch["s1"], patch["W_inv"], patch["role"],
        patch["J"],
    )


def _move_node(grid: MultiBlockGrid, dof_idx: int, new_pos: np.ndarray, ctx: SweepContext) -> None:
    """Move a single global DOF to a new position (O(incidence))."""
    grid.global_nodes[dof_idx] = new_pos
    for bi, logical_idx in ctx.dof_to_locals[dof_idx]:
        grid.blocks[bi].nodes[logical_idx] = new_pos


def _patch_energy_and_mindet(
    grid: MultiBlockGrid,
    dof_idx: int,
    ctx: SweepContext,
    metric: str,
) -> tuple[float, float]:
    """Sum of mu and min det(A) over all corner samples of all incident cells.

    Uses precomputed batched stencils for vectorized evaluation.
    """
    patch = ctx.dof_patches[dof_idx]
    return _batch.energy_and_mindet(
        grid.global_nodes,
        patch["gc"], patch["gn0"], patch["gn1"],
        patch["s0"], patch["s1"], patch["W_inv"],
    )


def _patch_fused(
    grid: MultiBlockGrid,
    dof_idx: int,
    ctx: SweepContext,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Combined (grad, hess, energy, mindet) in a single batch evaluation.

    Avoids recomputing ``A`` and ``T`` that :func:`_patch_grad_hess` and
    :func:`_patch_energy_and_mindet` would duplicate when called back-to-back.
    """
    patch = ctx.dof_patches[dof_idx]
    return _batch.patch_eval(
        grid.global_nodes,
        patch["gc"], patch["gn0"], patch["gn1"],
        patch["s0"], patch["s1"], patch["W_inv"], patch["role"],
        patch["J"],
    )


def local_relaxation_sweep(
    grid: MultiBlockGrid,
    target_fn: Callable[..., np.ndarray],
    metric: str = "shape_2d",
    ctx: SweepContext | None = None,
    quadratic_filter: bool = False,
) -> float:
    """Perform one Gauss-Seidel sweep over all free DOFs.

    Parameters
    ----------
    grid, target_fn, metric
        As elsewhere.
    ctx : SweepContext, optional
        Precomputed incidence/target data; built on demand if not supplied.
        Pass the same context across sweeps to avoid rebuilding it each time.
    quadratic_filter : bool
        If True, use a quadratic model (from the already-computed gradient and
        Hessian) to skip full patch-energy evaluations in the backtracking loop
        when the model predicts an energy increase.  Numerically identical to
        ``quadratic_filter=False`` — the filter only suppresses evaluations that
        would have been rejected anyway.

    Returns
    -------
    max_movement : float
        Maximum node movement during this sweep.
    """
    if ctx is None:
        ctx = build_sweep_context(grid, target_fn)

    max_movement = 0.0

    for dof_idx in range(grid.global_node_count):
        if not grid.free_mask[dof_idx]:
            continue
        if not ctx.dof_to_cells[dof_idx]:
            continue

        old_pos = grid.global_nodes[dof_idx].copy()
        # Sliding geometry constraint (None -> free 2-D node)
        entity = grid.dof_constraints.get(dof_idx)

        # Compute local gradient, Hessian, and initial patch energy in one pass
        g, H, e0, _ = _patch_fused(grid, dof_idx, ctx, metric)

        # For a sliding DOF, reduce the system to the entity's tangent subspace
        if entity is None:
            basis = None
            A_mat, rhs = H, g
        else:
            basis = np.asarray(entity.tangent_space(old_pos), dtype=float)  # (d, k)
            A_mat = basis.T @ H @ basis
            rhs = basis.T @ g

        # Solve for the (possibly reduced) Newton step
        try:
            step = -np.linalg.solve(A_mat, rhs)
        except np.linalg.LinAlgError:
            # Fall back to gradient descent in the same space
            rhs_norm = np.linalg.norm(rhs)
            if rhs_norm < 1e-15:
                continue
            step = -0.1 * rhs / rhs_norm
        delta = step if basis is None else basis @ step

        # Precompute H·delta for the quadratic filter (cost is negligible)
        H_delta = H @ delta

        # Damping: reduce step if energy increases. Sliding DOFs re-project onto
        # their entity each trial (orthogonal projection; LineSegment.project
        # clamps to the segment so a wall node cannot cross a corner).
        alpha = 1.0
        accepted = False
        new_pos = old_pos.copy()

        if quadratic_filter:
            for _ in range(10):
                e_quad = e0 + alpha * g.dot(delta) + 0.5 * alpha * alpha * delta.dot(H_delta)
                if e_quad > e0 + 1e-12:
                    alpha *= 0.5
                    continue
                trial = old_pos + alpha * delta
                new_pos = trial if entity is None else np.asarray(entity.project(trial))
                _move_node(grid, dof_idx, new_pos, ctx)
                e_new, min_det = _patch_energy_and_mindet(grid, dof_idx, ctx, metric)
                if np.isfinite(e_new) and e_new <= e0 + 1e-12 and min_det > 0:
                    accepted = True
                    break
                alpha *= 0.5
        else:
            for _ in range(10):
                trial = old_pos + alpha * delta
                new_pos = trial if entity is None else np.asarray(entity.project(trial))
                _move_node(grid, dof_idx, new_pos, ctx)
                e_new, min_det = _patch_energy_and_mindet(grid, dof_idx, ctx, metric)
                if np.isfinite(e_new) and e_new <= e0 + 1e-12 and min_det > 0:
                    accepted = True
                    break
                alpha *= 0.5

        if accepted:
            movement = np.linalg.norm(new_pos - old_pos)
            max_movement = max(max_movement, movement)
        else:
            # Restore original position
            _move_node(grid, dof_idx, old_pos, ctx)

    return float(max_movement)


def _newton_backtrack_dof(
    grid: MultiBlockGrid,
    dof_idx: int,
    g: np.ndarray,
    H: np.ndarray,
    e0: float,
    ctx: SweepContext,
    metric: str,
    quadratic_filter: bool,
) -> float:
    """Sequential Newton + backtracking for one DOF (NumPy).

    Identical logic to the body of :func:`local_relaxation_sweep`, factored out
    so both the NumPy and JAX sweeps share the exact backtracking path. ``g`` and
    ``H`` are the already-computed local gradient/Hessian. Returns the node
    movement (0.0 if no move was accepted).
    """
    old_pos = grid.global_nodes[dof_idx].copy()
    entity = grid.dof_constraints.get(dof_idx)

    if entity is None:
        basis = None
        A_mat, rhs = H, g
    else:
        basis = np.asarray(entity.tangent_space(old_pos), dtype=float)
        A_mat = basis.T @ H @ basis
        rhs = basis.T @ g

    try:
        step = -np.linalg.solve(A_mat, rhs)
    except np.linalg.LinAlgError:
        rhs_norm = np.linalg.norm(rhs)
        if rhs_norm < 1e-15:
            return 0.0
        step = -0.1 * rhs / rhs_norm
    delta = step if basis is None else basis @ step

    H_delta = H @ delta
    alpha = 1.0
    accepted = False
    new_pos = old_pos.copy()

    for _ in range(10):
        if quadratic_filter:
            e_quad = e0 + alpha * g.dot(delta) + 0.5 * alpha * alpha * delta.dot(H_delta)
            if e_quad > e0 + 1e-12:
                alpha *= 0.5
                continue
        trial = old_pos + alpha * delta
        new_pos = trial if entity is None else np.asarray(entity.project(trial))
        _move_node(grid, dof_idx, new_pos, ctx)
        e_new, min_det = _patch_energy_and_mindet(grid, dof_idx, ctx, metric)
        if np.isfinite(e_new) and e_new <= e0 + 1e-12 and min_det > 0:
            accepted = True
            break
        alpha *= 0.5

    if accepted:
        return float(np.linalg.norm(new_pos - old_pos))
    _move_node(grid, dof_idx, old_pos, ctx)
    return 0.0


def local_relaxation(
    grid: MultiBlockGrid,
    target_fn: Callable[..., np.ndarray],
    metric: str = "shape_2d",
    max_sweeps: int = 100,
    tol: float = 1e-6,
) -> list[float]:
    """Run local Gauss-Seidel relaxation to convergence.

    Parameters
    ----------
    grid : MultiBlockGrid
    target_fn : callable
    metric : str
    max_sweeps : int
    tol : float
        Stop when max node movement < tol.

    Returns
    -------
    energy_history : list[float]
        Global energy after each sweep.
    """
    from .objective import assemble_energy

    ctx = build_sweep_context(grid, target_fn)
    energy_history = []

    for sweep in range(max_sweeps):
        movement = local_relaxation_sweep(grid, target_fn, metric, ctx)
        F = assemble_energy(grid, target_fn, metric)
        energy_history.append(F)

        if movement < tol:
            break

    return energy_history
