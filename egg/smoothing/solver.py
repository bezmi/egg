"""Sweep-context construction for the TMOP block-Jacobi backend.

A :class:`SweepContext` precomputes the per-DOF incidence maps and the target
``W_inv`` once (the target is a function of logical position only, so it never
changes during a sweep). The C++ wire itself is built by the single
dimension-generic builder :func:`egg.smoothing.flat_context.build_flat_context`
and carried on ``SweepContext.wire``; the per-DOF ``dof_patches`` here are the
NumPy parity reference only (see :mod:`egg.smoothing.batch` tests).

Dimension note: this builder is 2-D (it names the two axis neighbours directly).
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
        ``gn1``, ``s0``, ``s1``, ``W_inv``, ``role``, ``J`` arrays.
    energy_stencil : dict
        Global stencil arrays over *every* (cell, corner) sample of the grid,
        for one-pass vectorized energy assembly
        (:func:`egg.smoothing.objective.assemble_energy_vec`).
    dof_patch_sizes : list[int]
        Number of corner samples ``P`` in each DOF's patch (0 if it has no
        incident cells, e.g. a fixed corner).
    free_dofs : np.ndarray
        The moving DOFs that own at least one incident cell (``P > 0``), in
        ascending id — the single relaxation group the block-Jacobi sweep runs.
    """

    dof_to_cells: list[list[tuple[int, tuple]]]
    dof_to_locals: list[list[tuple[int, tuple]]]
    w_inv: dict[tuple[int, tuple, tuple], np.ndarray]
    dof_patches: list[dict]
    energy_stencil: dict
    dof_patch_sizes: list[int]
    free_dofs: np.ndarray  # (F,) int moving DOFs with >= 1 incident cell
    dof_constraint_tags: np.ndarray  # (M,) int32 entity type tag per DOF
    dof_constraint_params: np.ndarray  # (M, PARAM_PAD_SIZE) float64 per DOF
    # Variable-length entity data (B-spline knots/nets, composite-path segment
    # records); offsets in dof_constraint_params index into this. Empty when
    # only fixed-size entities are present.
    entity_arena: np.ndarray = None  # (A,) float64
    # Original entity objects per DOF (for the typed SoA wire boundary).
    # None when no constraints are present.
    dof_entities: dict = None
    # Embedding dimension (2 or 3). Defaults to 2.
    d: int = 2
    # The C++ wire ({"groups", "energy_stencil"}) built once by
    # egg.smoothing.flat_context.build_flat_context — the single wire builder for
    # both this (reference-carrying) path and the direct structured path. The
    # dof_patches above stay as the NumPy parity reference only.
    wire: dict = None


def build_sweep_context(
    grid: MultiBlockGrid, target_fn: Callable[..., np.ndarray]
) -> SweepContext:
    """Build the per-sweep incidence maps and cached target inverses once.

    With all-corner sampling a cell's energy depends on *all* 2**d of its corner
    nodes, so every corner DOF lists the cell as incident.
    """
    from egg.smoothing.flat_context import cell_stencil

    M = grid.global_node_count
    d = grid.topology.d
    corners = list(product((0, 1), repeat=d))

    # Shared vectorized cell stencil + node->sample membership over each block's
    # global-dof id array. cell_stencil enumerates cells in iter_cells (C) order
    # and corners in `corners` (product) order, so the per-sample W_inv computed
    # below aligns index-for-index (sample id = 2**d * cell + corner). gn[k]/s[k]
    # are the k-th axis neighbour's global id and sign.
    st = cell_stencil([np.asarray(dm) for dm in grid.block_dof_maps], d)
    ns = st["ns"]
    en_gc = st["gc"].astype(np.intp)
    en_gn0, en_gn1 = st["gn"][0].astype(np.intp), st["gn"][1].astype(np.intp)
    en_s0, en_s1 = st["s"][0].astype(np.intp), st["s"][1].astype(np.intp)

    # Per-(cell, corner) target inverse in sample order, the raw w_inv dict, and
    # the incidence maps (dof -> cells / locals), built across all blocks.
    w_inv: dict[tuple[int, tuple, tuple], np.ndarray] = {}
    dof_to_cells: list[list[tuple[int, tuple]]] = [[] for _ in range(M)]
    dof_to_locals: list[list[tuple[int, tuple]]] = [[] for _ in range(M)]
    samp_w_inv = np.empty((ns, d, d))

    for bi, block in enumerate(grid.blocks):
        dof_map = grid.block_dof_maps[bi]
        for logical_idx in product(*[range(s) for s in block.logical_shape]):
            dof_to_locals[int(dof_map[logical_idx])].append((bi, logical_idx))

    sid = 0
    for bi, block in enumerate(grid.blocks):
        dof_map = grid.block_dof_maps[bi]
        for cell_base in block.iter_cells():
            corner_dofs = {int(dof_map[ci]) for ci in block.corner_indices(cell_base)}
            for dof in corner_dofs:
                dof_to_cells[dof].append((bi, cell_base))
            for co in corners:
                wi = np.linalg.inv(target_fn(bi, block, cell_base, co))
                w_inv[(bi, cell_base, co)] = wi
                samp_w_inv[sid] = wi
                sid += 1

    # Per-DOF patches: group the node->sample membership by DOF. A stable sort
    # keeps each DOF's samples in cell/corner traversal order (the reference
    # order); patches are built for every node (the C++ wire indexes by id).
    order = np.argsort(st["m_node"], kind="stable")
    m_node = st["m_node"][order]
    m_sid = st["m_sid"][order]
    m_role = st["m_role"][order].astype(np.intp)
    starts = np.searchsorted(m_node, np.arange(M), side="left")
    ends = np.searchsorted(m_node, np.arange(M), side="right")

    dof_patches: list[dict] = []
    for dof_idx in range(M):
        lo, hi = starts[dof_idx], ends[dof_idx]
        sids = m_sid[lo:hi]
        P = int(sids.shape[0])
        patch = {
            "gc": en_gc[sids], "gn0": en_gn0[sids], "gn1": en_gn1[sids],
            "s0": en_s0[sids], "s1": en_s1[sids],
            "W_inv": samp_w_inv[sids] if P else np.zeros((0, 2, 2)),
            "role": m_role[lo:hi],
        }
        patch["J"] = _batch.make_chain_J(
            patch["s0"], patch["s1"], patch["W_inv"],
        ) if P > 0 else np.zeros((0, 4, 6))
        dof_patches.append(patch)

    energy_stencil = {
        "gc": en_gc, "gn0": en_gn0, "gn1": en_gn1,
        "s0": en_s0, "s1": en_s1,
        "W_inv": samp_w_inv if ns else np.zeros((0, 2, 2)),
    }

    dof_patch_sizes = [dof_patches[dof]["gc"].shape[0] for dof in range(M)]
    # The single block-Jacobi group: every moving DOF that owns a cell.
    free_dofs = np.array(
        [dof for dof in range(M)
         if grid.free_mask[dof] and dof_patch_sizes[dof] > 0],
        dtype=np.int64,
    )

    # --- Constraint tags/params per DOF ---
    from egg.geometry.entity_encoding import PARAM_PAD_SIZE, encode_entity

    dof_constraint_tags = np.zeros(M, dtype=np.int32)
    dof_constraint_params = np.zeros((M, PARAM_PAD_SIZE), dtype=np.float64)
    arena: list[float] = []
    for dof_idx, entity in grid.dof_constraints.items():
        tag, params = encode_entity(entity, d=d, arena=arena)
        dof_constraint_tags[dof_idx] = tag
        dof_constraint_params[dof_idx] = params

    # The C++ wire comes from the single dimension-generic builder, fed the same
    # per-sample target inverse (samp_w_inv) this function already computed.
    from egg.smoothing.flat_context import build_flat_context
    wire = build_flat_context(
        [np.asarray(dm) for dm in grid.block_dof_maps],
        grid.free_mask, dict(grid.dof_constraints), d, w_inv=samp_w_inv)

    return SweepContext(
        dof_to_cells, dof_to_locals, w_inv, dof_patches, energy_stencil,
        dof_patch_sizes, free_dofs,
        dof_constraint_tags, dof_constraint_params,
        np.asarray(arena, dtype=np.float64),
        dict(grid.dof_constraints), d=d, wire=wire,
    )


def _patch_grad_hess(
    grid: MultiBlockGrid,
    dof_idx: int,
    ctx: SweepContext,
) -> tuple[np.ndarray, np.ndarray]:
    """Local gradient (d,) and Hessian (d, d) for a single global DOF.

    Uses precomputed batched stencils to evaluate all corner samples of the DOF's
    incident cells at once. A NumPy reference for the objective gradient/Hessian
    the C++ backend computes (see :mod:`tests.smoothing.test_batch`).
    """
    patch = ctx.dof_patches[dof_idx]
    return _batch.dof_grad_hess(
        grid.global_nodes,
        patch["gc"], patch["gn0"], patch["gn1"],
        patch["s0"], patch["s1"], patch["W_inv"], patch["role"],
        patch["J"],
    )


def _patch_energy_and_mindet(
    grid: MultiBlockGrid,
    dof_idx: int,
    ctx: SweepContext,
) -> tuple[float, float]:
    """Sum of mu and min det(A) over all corner samples of all incident cells.

    Uses precomputed batched stencils for vectorized evaluation. A NumPy
    reference for the C++ patch energy / min-det.
    """
    patch = ctx.dof_patches[dof_idx]
    return _batch.energy_and_mindet(
        grid.global_nodes,
        patch["gc"], patch["gn0"], patch["gn1"],
        patch["s0"], patch["s1"], patch["W_inv"],
    )
