# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Sweep-context construction for the TMOP block-Jacobi backend.

A :class:`SweepContext` precomputes the target ``W_inv`` once (the target is a
function of logical position only, so it never changes during a sweep). The
C++ wire itself is built by the single dimension-generic builder
:func:`egg.smoothing.flat_context.build_flat_context` and carried on
``SweepContext.wire``. The per-DOF NumPy parity-reference views
(``dof_patches``, ``dof_to_cells``, ``dof_to_locals``, ``w_inv``) are built
LAZILY on first access — they are consumed only by the parity tests (see
:mod:`egg.smoothing.batch`), and their O(N) pure-Python construction must not
tax the production pipeline path.

Dimension note: this builder is 2-D (it names the two axis neighbours directly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
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

    The eager fields below are what the pipeline consumes (the wire, the
    energy stencil, the constraint encoding). The NumPy parity-reference
    views — :attr:`dof_patches`, :attr:`dof_to_cells`, :attr:`dof_to_locals`,
    :attr:`w_inv`, :attr:`dof_patch_sizes` — are ``cached_property`` accessors
    built on first use from the retained grid/stencil ingredients, so tests
    keep their API while production runs never pay for them.

    Attributes
    ----------
    energy_stencil : dict
        Global stencil arrays over *every* (cell, corner) sample of the grid,
        for one-pass vectorized energy assembly
        (:func:`egg.smoothing.objective.assemble_energy_vec`).
    free_dofs : np.ndarray
        The moving DOFs that own at least one incident cell (``P > 0``), in
        ascending id — the single relaxation group the block-Jacobi sweep runs.
    dof_patches : list[dict]  (lazy)
        Per-DOF batched stencil arrays for vectorized patch evaluation
        (see :mod:`egg.smoothing.batch`). Each dict contains ``gc``, ``gn0``,
        ``gn1``, ``s0``, ``s1``, ``W_inv``, ``role``, ``J`` arrays.
    dof_to_cells : list[list[(block_idx, cell_base)]]  (lazy)
        Cells whose energy depends on each global DOF (all 2**d corners).
    dof_to_locals : list[list[(block_idx, logical_idx)]]  (lazy)
        Block-node slots that alias each global DOF (for O(incidence) moves).
    w_inv : dict[(block_idx, cell_base, corner_offset), ndarray]  (lazy)
        ``inv(target_fn(cell_base, corner_offset))`` per sample.
    dof_patch_sizes : list[int]  (lazy)
        Number of corner samples ``P`` in each DOF's patch (0 if it has no
        incident cells, e.g. a fixed corner).
    """

    energy_stencil: dict
    free_dofs: np.ndarray  # (F,) int moving DOFs with >= 1 incident cell
    dof_constraint_tags: np.ndarray  # (M,) int32 entity type tag per DOF
    dof_constraint_params: np.ndarray  # (M, PARAM_PAD_SIZE) float64 per DOF
    # Variable-length entity data (B-spline knots/nets, composite-path segment
    # records); offsets in dof_constraint_params index into this. Empty when
    # only fixed-size entities are present.
    entity_arena: np.ndarray
    # Original entity objects per DOF (for the typed SoA wire boundary).
    # None when no constraints are present.
    dof_entities: dict
    # Embedding dimension (2 or 3).
    d: int
    # The C++ wire ({"groups", "energy_stencil"}) built once by
    # egg.smoothing.flat_context.build_flat_context — the single wire builder
    # for both this (reference-carrying) path and the direct structured path.
    wire: dict

    # Lazy-build ingredients for the parity-reference views: the grid (cell /
    # logical-index iteration), the cell_stencil result, and the per-sample
    # target inverses. Position-independent, so retaining the grid is safe.
    _grid: MultiBlockGrid = field(repr=False, default=None)
    _stencil: dict = field(repr=False, default=None)
    _samp_w_inv: np.ndarray = field(repr=False, default=None)

    @cached_property
    def dof_patch_sizes(self) -> list[int]:
        """Corner-sample count P per DOF, from the membership table."""
        M = self._grid.global_node_count
        return np.bincount(self._stencil["m_node"], minlength=M)[:M].tolist()

    @cached_property
    def dof_patches(self) -> list[dict]:
        """Per-DOF batched stencil arrays (NumPy parity reference only)."""
        st = self._stencil
        M = self._grid.global_node_count
        en_gc = st["gc"].astype(np.intp)
        en_gn0, en_gn1 = st["gn"][0].astype(np.intp), st["gn"][1].astype(np.intp)
        en_s0, en_s1 = st["s"][0].astype(np.intp), st["s"][1].astype(np.intp)

        # Group the node->sample membership by DOF. A stable sort keeps each
        # DOF's samples in cell/corner traversal order (the reference order);
        # patches are built for every node (indexed by id).
        order = np.argsort(st["m_node"], kind="stable")
        m_node = st["m_node"][order]
        m_sid = st["m_sid"][order]
        m_role = st["m_role"][order].astype(np.intp)
        starts = np.searchsorted(m_node, np.arange(M), side="left")
        ends = np.searchsorted(m_node, np.arange(M), side="right")

        patches: list[dict] = []
        for dof_idx in range(M):
            lo, hi = starts[dof_idx], ends[dof_idx]
            sids = m_sid[lo:hi]
            P = int(sids.shape[0])
            patch = {
                "gc": en_gc[sids],
                "gn0": en_gn0[sids],
                "gn1": en_gn1[sids],
                "s0": en_s0[sids],
                "s1": en_s1[sids],
                "W_inv": self._samp_w_inv[sids] if P else np.zeros((0, 2, 2)),
                "role": m_role[lo:hi],
            }
            patch["J"] = (
                _batch.make_chain_J(
                    patch["s0"],
                    patch["s1"],
                    patch["W_inv"],
                )
                if P > 0
                else np.zeros((0, 4, 6))
            )
            patches.append(patch)
        return patches

    @cached_property
    def dof_to_cells(self) -> list[list[tuple[int, tuple]]]:
        """Cells whose energy depends on each global DOF."""
        grid = self._grid
        out: list[list[tuple[int, tuple]]] = [[] for _ in range(grid.global_node_count)]
        for bi, block in enumerate(grid.blocks):
            dof_map = grid.block_dof_maps[bi]
            for cell_base in block.iter_cells():
                corner_dofs = {
                    int(dof_map[ci]) for ci in block.corner_indices(cell_base)
                }
                for dof in corner_dofs:
                    out[dof].append((bi, cell_base))
        return out

    @cached_property
    def dof_to_locals(self) -> list[list[tuple[int, tuple]]]:
        """Block-node slots aliasing each global DOF."""
        grid = self._grid
        out: list[list[tuple[int, tuple]]] = [[] for _ in range(grid.global_node_count)]
        for bi, block in enumerate(grid.blocks):
            dof_map = grid.block_dof_maps[bi]
            for logical_idx in product(*[range(s) for s in block.logical_shape]):
                out[int(dof_map[logical_idx])].append((bi, logical_idx))
        return out

    @cached_property
    def w_inv(self) -> dict[tuple[int, tuple, tuple], np.ndarray]:
        """Per-(block, cell, corner) target inverse, keyed for the tests.
        Rows of the already-inverted per-sample stack, walked in the same
        sample order the build used."""
        grid = self._grid
        corners = list(product((0, 1), repeat=self.d))
        out: dict[tuple[int, tuple, tuple], np.ndarray] = {}
        sid = 0
        for bi, block in enumerate(grid.blocks):
            for cell_base in block.iter_cells():
                for co in corners:
                    out[(bi, cell_base, co)] = self._samp_w_inv[sid]
                    sid += 1
        return out


def build_sweep_context(
    grid: MultiBlockGrid,
    target_fn: Callable[..., np.ndarray],
    *,
    interface_ortho: dict | None = None,
) -> SweepContext:
    """Build the sweep context: energy stencil, constraint encoding, C++ wire.

    With all-corner sampling a cell's energy depends on *all* 2**d of its corner
    nodes, so every corner DOF lists the cell as incident. The per-DOF NumPy
    parity-reference views are NOT built here — they materialise lazily on
    first attribute access (see :class:`SweepContext`).

    ``interface_ortho`` (optional) enables the composed block-boundary term: a
    dict of keyword args for
    :func:`egg.smoothing.interface_ortho.interface_ortho_samples` (``mode``,
    ``weight``). The frame is frozen from ``grid``'s current node positions.
    """
    from egg.smoothing.flat_context import cell_stencil

    iface = None
    if interface_ortho:
        from egg.smoothing.interface_ortho import interface_ortho_samples

        iface = interface_ortho_samples(grid, **interface_ortho)

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

    # Per-(cell, corner) target in sample order, inverted in ONE stacked
    # np.linalg.inv over the (ns, d, d) batch — per-sample Python-loop
    # inversion dominated the build on large grids. The target_fn call itself
    # stays a per-cell Python call (its signature is per-(block, cell, corner)).
    samp_w = np.empty((ns, d, d))
    sid = 0
    for bi, block in enumerate(grid.blocks):
        for cell_base in block.iter_cells():
            for co in corners:
                samp_w[sid] = target_fn(bi, block, cell_base, co)
                sid += 1
    samp_w_inv = np.linalg.inv(samp_w) if ns else np.zeros((0, d, d))

    energy_stencil = {
        "gc": st["gc"].astype(np.intp),
        "gn0": st["gn"][0].astype(np.intp),
        "gn1": st["gn"][1].astype(np.intp),
        "s0": st["s"][0].astype(np.intp),
        "s1": st["s"][1].astype(np.intp),
        "W_inv": samp_w_inv if ns else np.zeros((0, 2, 2)),
    }

    # The single block-Jacobi group: every moving DOF that owns a cell. Patch
    # size per node comes straight from the membership table (vectorized).
    counts = np.bincount(st["m_node"], minlength=M)[:M]
    free_dofs = np.flatnonzero(np.asarray(grid.free_mask) & (counts > 0)).astype(
        np.int64
    )

    # --- Constraint tags/params per DOF ---
    from egg.geometry.entity_encoding import PARAM_PAD_SIZE, encode_entity

    dof_constraint_tags = np.zeros(M, dtype=np.int32)
    dof_constraint_params = np.zeros((M, PARAM_PAD_SIZE), dtype=np.float64)
    arena: list[float] = []
    for dof_idx, entity in grid.dof_constraints.items():
        try:
            tag, params = encode_entity(entity, d=d, arena=arena)
        except NotImplementedError:
            # dof_constraint_tags is a 2D diagnostic field; the C++ wire encodes
            # every entity (2D and 3D) via build_flat_context below.
            continue
        dof_constraint_tags[dof_idx] = tag
        dof_constraint_params[dof_idx] = params

    # The C++ wire comes from the single dimension-generic builder, fed the same
    # per-sample target inverse (samp_w_inv) this function already computed.
    from egg.smoothing.flat_context import build_flat_context

    wire = build_flat_context(
        [np.asarray(dm) for dm in grid.block_dof_maps],
        grid.free_mask,
        dict(grid.dof_constraints),
        d,
        w_inv=samp_w_inv,
        interface=iface,
    )

    return SweepContext(
        energy_stencil=energy_stencil,
        free_dofs=free_dofs,
        dof_constraint_tags=dof_constraint_tags,
        dof_constraint_params=dof_constraint_params,
        entity_arena=np.asarray(arena, dtype=np.float64),
        dof_entities=dict(grid.dof_constraints),
        d=d,
        wire=wire,
        _grid=grid,
        _stencil=st,
        _samp_w_inv=samp_w_inv,
    )


def _patch_grad_hess(
    grid: MultiBlockGrid,
    dof_idx: int,
    ctx: SweepContext,
    metric: str = "shape_2d",
) -> tuple[np.ndarray, np.ndarray]:
    """Local gradient (d,) and Hessian (d, d) for a single global DOF.

    Uses precomputed batched stencils to evaluate all corner samples of the DOF's
    incident cells at once. A NumPy reference for the objective gradient/Hessian
    the C++ backend computes (see :mod:`tests.smoothing.test_batch`).
    """
    patch = ctx.dof_patches[dof_idx]
    return _batch.dof_grad_hess(
        grid.global_nodes,
        patch["gc"],
        patch["gn0"],
        patch["gn1"],
        patch["s0"],
        patch["s1"],
        patch["W_inv"],
        patch["role"],
        patch["J"],
        metric=metric,
    )


def _patch_energy_and_mindet(
    grid: MultiBlockGrid,
    dof_idx: int,
    ctx: SweepContext,
    metric: str = "shape_2d",
) -> tuple[float, float]:
    """Sum of mu and min det(A) over all corner samples of all incident cells.

    Uses precomputed batched stencils for vectorized evaluation. A NumPy
    reference for the C++ patch energy / min-det.
    """
    patch = ctx.dof_patches[dof_idx]
    return _batch.energy_and_mindet(
        grid.global_nodes,
        patch["gc"],
        patch["gn0"],
        patch["gn1"],
        patch["s0"],
        patch["s1"],
        patch["W_inv"],
        metric=metric,
    )
