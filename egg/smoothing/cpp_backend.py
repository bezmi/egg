"""C++ backend bridge for the colored Gauss-Seidel sweep.

Provides :func:`cpp_sweep` — a drop-in replacement for
``build_fused_multisweep.run`` that calls the device-resident C++ sweep via
``egg._cpp.cpp_core``. Also provides :func:`flatten_context` to pack a
:class:`~egg.smoothing.solver.SweepContext` into the dict format the
binding expects, and :func:`build_block_structured_context` to pack a
:class:`~egg.topology.block_topology.BlockTopology` grid into the structured
halo-padded layout tables the Phase 1 GPU path consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from egg.core.types import MultiBlockGrid
    from egg.smoothing.solver import SweepContext

__all__ = [
    "BlockStructuredContext",
    "CppStructuredSweepSession",
    "CppSweepSession",
    "build_block_structured_context",
    "build_structured_context_from_block_maps",
    "cpp_structured_sweep",
    "cpp_sweep",
    "cpp_untangle",
    "flatten_context",
    "structured_arrays",
]


def _grid_mindet(X: np.ndarray, es: dict) -> float:
    """Raw min det A over the energy stencil (mirrors solver._grid_mindet)."""
    gc, gn0, gn1 = es["gc"], es["gn0"], es["gn1"]
    s0, s1 = es["s0"], es["s1"]
    Xc = X[gc]
    col0 = s0[:, None] * (X[gn0] - Xc)
    col1 = s1[:, None] * (X[gn1] - Xc)
    det = col0[:, 0] * col1[:, 1] - col1[:, 0] * col0[:, 1]
    return float(det.min())


def _ensure_group_batches(ctx: SweepContext) -> None:
    """Build jax_group_batches if not already populated (host-side NumPy path).

    The JAX path does this lazily on first sweep. We replicate it here with
    pure NumPy so the C++ backend does not require JAX.
    """
    if ctx.jax_group_batches is not None:
        return

    from .batch import make_chain_J

    ctx.jax_group_batches = {}
    for (colour, P), dof_indices in ctx.colour_P_groups.items():
        dof_indices = list(dof_indices)
        D = len(dof_indices)
        if D == 0:
            continue

        patches = [ctx.dof_patches[d] for d in dof_indices]

        gc = np.stack([p["gc"] for p in patches])       # (D, P)
        gn0 = np.stack([p["gn0"] for p in patches])
        gn1 = np.stack([p["gn1"] for p in patches])
        s0 = np.stack([p["s0"] for p in patches])
        s1 = np.stack([p["s1"] for p in patches])
        W_inv = np.stack([p["W_inv"] for p in patches])  # (D, P, 2, 2)
        role = np.stack([p["role"] for p in patches])

        J = make_chain_J(s0[0], s1[0], W_inv[0])[None].repeat(D, axis=0)
        # Recompute per-DOF J (signs/W_inv may vary per DOF)
        J = np.stack([make_chain_J(s0[i], s1[i], W_inv[i]) for i in range(D)])

        ctx.jax_group_batches[(colour, P)] = {
            "gc": gc.astype(np.int32),
            "gn0": gn0.astype(np.int32),
            "gn1": gn1.astype(np.int32),
            "s0": s0.astype(np.float64),
            "s1": s1.astype(np.float64),
            "W_inv": W_inv.astype(np.float64),
            "role": role.astype(np.int32),
            "J": J.astype(np.float64),
            "dof_idx": np.asarray(dof_indices, dtype=np.int32),
            "tag": ctx.dof_constraint_tags[dof_indices].astype(np.int32),
            "params": ctx.dof_constraint_params[dof_indices].astype(np.float64),
        }


def flatten_context(ctx: SweepContext) -> dict:
    """Pack a SweepContext into the ctx_arrays dict the C++ binding expects.

    Returns a dict with keys ``"groups"`` (list of group dicts) and
    ``"energy_stencil"`` (dict of stencil arrays). Each group dict contains
    the batched arrays the C++ ``Executor`` consumes.

    Parameters
    ----------
    ctx : SweepContext
        The sweep context from :func:`build_sweep_context`.

    Returns
    -------
    dict
        Flattened context ready for ``cpp_core.cpp_sweep``.
    """
    _ensure_group_batches(ctx)

    from egg.geometry.entity_soa import group_entities_by_type

    # One ragged group per colour: merge the colour's (colour, P) batches into a
    # single launch (#1 — see cpp_backend_plan.md §5). Per-sample arrays are
    # concatenated DOF-major over the colour's DOFs (each DOF contributes its P
    # samples contiguously); the C++ side derives sample_offset from P_of. This
    # cuts the per-sweep launch count from ~(colours × distinct-P) to ~colours.
    groups = []
    for colour in range(ctx.num_colours):
        cg = ctx.get_colour_P_groups(colour)  # {P: [dof_indices]}
        if not cg:
            continue

        gc, gn0, gn1, s0, s1, role = [], [], [], [], [], []
        W_inv, J = [], []
        dof_idx, P_of = [], []

        # Collect the colour group's DOF indices in order (for entity typing).
        group_dof_indices: list[int] = []

        for P, _dofs in cg.items():
            b = ctx.jax_group_batches[(colour, P)]
            D = int(b["dof_idx"].shape[0])
            # (D, P, ...) -> flat (D*P, ...) is DOF-major: DOF d's P samples are
            # contiguous, matching the ragged sample_offset layout.
            gc.append(b["gc"].reshape(D * P))
            gn0.append(b["gn0"].reshape(D * P))
            gn1.append(b["gn1"].reshape(D * P))
            s0.append(b["s0"].reshape(D * P))
            s1.append(b["s1"].reshape(D * P))
            role.append(b["role"].reshape(D * P))
            W_inv.append(b["W_inv"].reshape(D * P, 4))     # (D,P,2,2) -> (D*P,4)
            J.append(b["J"].reshape(D * P, 24))            # (D,P,4,6) -> (D*P,24)
            dof_idx.append(b["dof_idx"])
            P_of.append(np.full(D, P, dtype=np.int32))
            group_dof_indices.extend(int(x) for x in b["dof_idx"])

        group_dict = {
            "D": int(np.concatenate(dof_idx).shape[0]),
            "gc": np.ascontiguousarray(np.concatenate(gc), dtype=np.int32),
            "gn0": np.ascontiguousarray(np.concatenate(gn0), dtype=np.int32),
            "gn1": np.ascontiguousarray(np.concatenate(gn1), dtype=np.int32),
            "s0": np.ascontiguousarray(np.concatenate(s0), dtype=np.float64),
            "s1": np.ascontiguousarray(np.concatenate(s1), dtype=np.float64),
            "W_inv": np.ascontiguousarray(np.concatenate(W_inv), dtype=np.float64),
            "role": np.ascontiguousarray(np.concatenate(role), dtype=np.int32),
            "J": np.ascontiguousarray(np.concatenate(J), dtype=np.float64),
            "dof_idx": np.ascontiguousarray(np.concatenate(dof_idx), dtype=np.int32),
            "P_of": np.ascontiguousarray(np.concatenate(P_of), dtype=np.int32),
        }

        # Typed per-entity-type SoA sub-dict — the sole entity transport (Phase 4
        # retired the positional tag/params/arena blob). Python partitions the
        # group's DOFs by entity type and emits packed-record arrays (+ CSR seg
        # fields) matching the C++ EntitySoA<E>::Host layout, covering every DOF
        # (free + constrained). All 2D types are covered, including B-spline
        # (segmented) and composite (Phase 3).
        entities = ctx.dof_entities
        if entities is None:
            raise ValueError(
                "flatten_context: ctx.dof_entities is required (the entity SoA wire "
                "has no positional-blob fallback since Phase 4); build the context "
                "via build_sweep_context"
            )
        typed = group_entities_by_type(group_dof_indices, entities, d=ctx.d)
        if "__blob__" in typed:
            # Every 2D entity now has an SoA encoder (composites carry any curve
            # type, incl. B-splines, via a self-contained per-composite arena), so
            # this is reachable only by a future/unencodable entity. The positional
            # blob path was retired in Phase 4, so there is no fallback.
            raise NotImplementedError(
                "flatten_context: a DOF has an entity with no SoA encoder; the "
                "positional blob path was retired in Phase 4 — add an EntitySoA "
                "encoder for it (see egg.geometry.entity_soa)"
            )
        group_dict["entities"] = typed

        groups.append(group_dict)

    es = ctx.energy_stencil
    n = int(es["gc"].shape[0])
    energy_stencil = {
        "num_samples": n,
        "gc": np.ascontiguousarray(es["gc"], dtype=np.int32),
        "gn0": np.ascontiguousarray(es["gn0"], dtype=np.int32),
        "gn1": np.ascontiguousarray(es["gn1"], dtype=np.int32),
        "s0": np.ascontiguousarray(es["s0"], dtype=np.float64),
        "s1": np.ascontiguousarray(es["s1"], dtype=np.float64),
        "W_inv": np.ascontiguousarray(es["W_inv"], dtype=np.float64),
    }

    return {
        "groups": groups,
        "energy_stencil": energy_stencil,
    }


@dataclass
class BlockStructuredContext:
    """Host-built tables for the Phase 1 halo-padded structured layout.

    These are the *only* things the C++ structured path uploads; all
    connectivity (shared-DOF identification, interface orientation, singularity
    detection) is resolved here from the already-validated
    :class:`~egg.topology.block_topology.BlockTopology`. The C++ side never
    re-derives a connection — it indexes the layout with ``BlockLayout<D>`` and
    copies the slots these tables name.

    Index conventions (shared with ``src/structured.hpp``):

    * Blocks are in ``block_specs`` insertion order (== ``block_dof_maps`` order).
    * ``interior_shapes[b]`` is the node count per axis (``BlockSpec.logical_shape``).
    * A *padded* index has the one-node ghost shell: each component is in
      ``[0, n_k + 1]``; interior logical index ``l`` maps to padded ``l + 1``.
    * ``block_global_dof[b]`` flattens the block's interior in row-major
      (C-order) logical order — the same order ``BlockLayout`` walks — so entry
      ``ravel_multi_index(logical, shape)`` is the global DOF of that node. It is
      the scatter/gather map between the flat global ``X`` and the block store.

    Attributes
    ----------
    d : int
        Spatial dimension.
    interior_shapes : list[tuple[int, ...]]
        Per-block interior (node-count) shapes, block order.
    block_global_dof : list[np.ndarray]
        Per-block int32 array (length ``prod(shape)``, row-major) of global DOFs.
    block_colour : list[np.ndarray]
        Per-block int32 array (length ``prod(shape)``, row-major) of the structured
        ``2**d`` parity colour ``sum_k (logical[k] & 1) << k`` of each node — the
        mask that replaces the per-DOF Welsh-Powell ``dof_colours``. Same-colour
        nodes share no cell (the GS independence guarantee), so a colour's DOFs
        relax simultaneously without racing.
    num_colours : int
        ``2**d`` — the number of structured parity colours.
    colour_consistent : bool
        True iff every shared interface node has the same parity colour in all
        blocks that own it. When False, exact per-colour GS across blocks would
        miscolour the interface; the per-sweep halo cadence (1.4b, additive
        Schwarz with frozen halos) sidesteps it. Even per-axis resolutions with
        aligned orientation keep this True.
    halo_src_padded, halo_dst_padded : np.ndarray
        ``(E, d)`` int32 padded indices: ghost slot ``dst`` is filled by copying
        the ``d`` coordinates of interior slot ``src``. ``src`` lives in
        ``halo_src_block[e]``, ``dst`` in ``halo_dst_block[e]``.
    halo_src_block, halo_dst_block : np.ndarray
        ``(E,)`` int32 block indices for each halo copy.
    sing_block : np.ndarray
        ``(S,)`` int32 block index of each singular node.
    sing_logical : np.ndarray
        ``(S, d)`` int32 interior logical index of each singular node.
    sing_valence : np.ndarray
        ``(S,)`` int32 neighbour count (≠ 2·d) of each singular node.
    owner_block : np.ndarray
        ``(M,)`` int32 owner block of each global DOF: the lowest-index block
        that contains it. A shared interface DOF is relaxed by exactly this block;
        the other blocks hold a copy kept in sync by the broadcast below. For an
        unshared DOF this is simply its only block.
    share_src_block, share_dst_block : np.ndarray
        ``(K,)`` int32 block indices of the owner→non-owner shared-node broadcast.
    share_src_padded, share_dst_padded : np.ndarray
        ``(K, d)`` int32 *interior* padded indices: each shared interface node's
        owner interior slot (``src``) is copied into the non-owner block's
        interior slot (``dst``). Unlike the ghost halo (``halo_*``, interior→ghost),
        this is interior→interior — it keeps the duplicated copies of a shared face
        node consistent (the frozen-per-sweep additive-Schwarz treatment of the
        shared shell, applied alongside the ghost exchange).
    """

    d: int
    interior_shapes: list[tuple[int, ...]]
    block_global_dof: list[np.ndarray]
    block_colour: list[np.ndarray]
    num_colours: int
    colour_consistent: bool
    halo_src_block: np.ndarray
    halo_src_padded: np.ndarray
    halo_dst_block: np.ndarray
    halo_dst_padded: np.ndarray
    sing_block: np.ndarray
    sing_logical: np.ndarray
    sing_valence: np.ndarray
    owner_block: np.ndarray
    share_src_block: np.ndarray
    share_src_padded: np.ndarray
    share_dst_block: np.ndarray
    share_dst_padded: np.ndarray

    @property
    def num_blocks(self) -> int:
        return len(self.interior_shapes)

    @property
    def num_halo_entries(self) -> int:
        return int(self.halo_dst_block.shape[0])

    @property
    def num_share_entries(self) -> int:
        return int(self.share_dst_block.shape[0])


def _parity_colouring(
    d: int, interior_shapes: list[tuple[int, ...]], dof_maps: list[np.ndarray]
) -> tuple[list[np.ndarray], bool]:
    """The structured ``2**d`` parity colouring + cross-block consistency flag.

    ``colour(node) = sum_k (logical[k] & 1) << k``. Within a block every cell's
    ``2**d`` corners differ by 0/1 per axis, so they get distinct colours -> no
    same-colour pair shares a cell (the GS independence guarantee). Across blocks
    the colour is well-defined only if a shared node has matching parity in every
    owner block; the returned flag records whether that holds.
    """
    block_colour: list[np.ndarray] = []
    global_colour: dict[int, int] = {}
    colour_consistent = True
    for bi, shape in enumerate(interior_shapes):
        dof_map = dof_maps[bi]
        carr = np.zeros(shape, dtype=np.int32)
        for logical in product(*[range(s) for s in shape]):
            colour = 0
            for k in range(d):
                colour |= (logical[k] & 1) << k
            carr[logical] = colour
            gd = int(dof_map[logical])
            if gd in global_colour:
                if global_colour[gd] != colour:
                    colour_consistent = False
            else:
                global_colour[gd] = colour
        block_colour.append(np.ascontiguousarray(carr.reshape(-1), dtype=np.int32))
    return block_colour, colour_consistent


def _owner_and_broadcast(
    d: int,
    interior_shapes: list[tuple[int, ...]],
    dof_maps: list[np.ndarray],
    global_node_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Owner block of each global DOF + the owner->non-owner shared-node broadcast.

    A global DOF that lives in more than one block is a shared interface node: the
    lowest-index block owns (relaxes) it, and each other block holds a copy
    refreshed by an owner-interior -> non-owner-interior broadcast (interior ->
    interior, distinct from the ghost halo). Returns ``(owner_block,
    share_src_block, share_src_padded, share_dst_block, share_dst_padded)``.
    """
    def interior_padded(logical: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(c + 1 for c in logical)

    global_to_locals: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    for bi, shape in enumerate(interior_shapes):
        dof_map = dof_maps[bi]
        for logical in product(*[range(s) for s in shape]):
            global_to_locals.setdefault(int(dof_map[logical]), []).append((bi, logical))

    owner_block = np.full(global_node_count, -1, dtype=np.int32)
    share_src_block: list[int] = []
    share_dst_block: list[int] = []
    share_src_padded: list[tuple[int, ...]] = []
    share_dst_padded: list[tuple[int, ...]] = []
    for gd, locs in global_to_locals.items():
        locs_sorted = sorted(locs)  # by block index, then logical
        owner_bi, owner_logical = locs_sorted[0]
        owner_block[gd] = owner_bi
        for bi, logical in locs_sorted[1:]:
            share_src_block.append(owner_bi)
            share_src_padded.append(interior_padded(owner_logical))
            share_dst_block.append(bi)
            share_dst_padded.append(interior_padded(logical))

    k = len(share_src_block)
    return (
        owner_block,
        np.array(share_src_block, dtype=np.int32),
        np.array(share_src_padded, dtype=np.int32).reshape(k, d),
        np.array(share_dst_block, dtype=np.int32),
        np.array(share_dst_padded, dtype=np.int32).reshape(k, d),
    )


def build_structured_context_from_block_maps(
    d: int,
    block_global_dof: list[np.ndarray],
    global_node_count: int,
) -> BlockStructuredContext:
    """Structured tables from raw per-block global-DOF arrays (no BlockTopology).

    For hand-assembled multiblock grids (e.g. the ``sphere_in_cube`` benchmark)
    that carry only the per-block structured id arrays, not a
    :class:`BlockTopology`. Derives owner + shared-node broadcast + parity colour
    generically; the ghost **halo and singularity tables are left empty**, because
    the C++ structured remap mirrors every cross-block patch neighbour into a spare
    ghost slot (the singular-fan fallback) — so a correct sweep needs no
    precomputed face-ghost connectivity. The canonical, fully-coalesced path with
    face ghosts is :func:`build_block_structured_context`.

    Parameters
    ----------
    d : int
        Spatial dimension.
    block_global_dof : list[np.ndarray]
        Per-block ``d``-D int arrays of global node ids (each block's structured
        id array; ``.shape`` is its interior node-count shape).
    global_node_count : int
        Total number of distinct global nodes.
    """
    interior_shapes = [tuple(int(s) for s in b.shape) for b in block_global_dof]
    dof_maps = [np.ascontiguousarray(b, dtype=np.int64) for b in block_global_dof]
    flat_dof = [np.ascontiguousarray(b.reshape(-1), dtype=np.int32) for b in block_global_dof]

    block_colour, colour_consistent = _parity_colouring(d, interior_shapes, dof_maps)
    owner_block, ssb, ssp, sdb, sdp = _owner_and_broadcast(
        d, interior_shapes, dof_maps, global_node_count
    )

    empty_i = np.zeros((0,), dtype=np.int32)
    empty_rows = np.zeros((0, d), dtype=np.int32)
    return BlockStructuredContext(
        d=d,
        interior_shapes=interior_shapes,
        block_global_dof=flat_dof,
        block_colour=block_colour,
        num_colours=2**d,
        colour_consistent=colour_consistent,
        halo_src_block=empty_i,
        halo_src_padded=empty_rows,
        halo_dst_block=empty_i,
        halo_dst_padded=empty_rows,
        sing_block=empty_i,
        sing_logical=empty_rows,
        sing_valence=empty_i,
        owner_block=owner_block,
        share_src_block=ssb,
        share_src_padded=ssp,
        share_dst_block=sdb,
        share_dst_padded=sdp,
    )


def build_block_structured_context(grid: MultiBlockGrid) -> BlockStructuredContext:
    """Build the halo-padded structured tables from a multiblock grid.

    Derives, from ``grid.topology`` and ``grid.block_dof_maps``:

    * per-block interior shapes and the interior→global-DOF scatter map, and
    * one ghost-fill entry per (interface node, neighbour) pair: the ghost slot
      just outside a shared face and the neighbour block's first interior layer
      that fills it.

    Only *axis-aligned* face ghosts are emitted — the TMOP metric stencil reads
    a node's ±1 neighbour on each axis separately (never a diagonal), so corner
    ghosts are never touched and need no fill. Restricted to conforming
    multiblock interfaces (matching node counts across a face); the neighbour on
    the far side of a face node is found by *shared global DOF*, so no
    orientation logic is re-implemented here.

    Parameters
    ----------
    grid : MultiBlockGrid
        A grid whose ``topology`` is a :class:`BlockTopology` (i.e. built from a
        ``TopologyBuilder``), with ``block_dof_maps`` populated.

    Returns
    -------
    BlockStructuredContext
    """
    topo = grid.topology
    d = topo.d
    block_names = list(topo.block_specs.keys())
    name_to_idx = {n: i for i, n in enumerate(block_names)}

    interior_shapes = [topo.block_specs[n].logical_shape for n in block_names]
    block_global_dof = [
        np.ascontiguousarray(grid.block_dof_maps[bi].reshape(-1), dtype=np.int32)
        for bi in range(len(block_names))
    ]

    # Structured 2**d parity colouring (+ cross-block consistency flag).
    block_colour, colour_consistent = _parity_colouring(
        d, interior_shapes, grid.block_dof_maps
    )

    # Per-block {global_dof: logical_idx} for nodes on a given (axis, side) face,
    # so a face node in block A can locate its twin (same global DOF) in block B.
    def face_dof_to_logical(bi: int, axis: int, side: int) -> dict[int, tuple[int, ...]]:
        spec = topo.block_specs[block_names[bi]]
        shape = spec.logical_shape
        fixed = 0 if side == 0 else shape[axis] - 1
        dof_map = grid.block_dof_maps[bi]
        out: dict[int, tuple[int, ...]] = {}
        free_axes = [a for a in range(d) if a != axis]
        for free_idx in product(*[range(shape[a]) for a in free_axes]):
            logical = [0] * d
            logical[axis] = fixed
            for j, a in enumerate(free_axes):
                logical[a] = free_idx[j]
            out[int(dof_map[tuple(logical)])] = tuple(logical)
        return out

    src_block: list[int] = []
    dst_block: list[int] = []
    src_padded: list[tuple[int, ...]] = []
    dst_padded: list[tuple[int, ...]] = []

    def ghost_padded(logical: tuple[int, ...], axis: int, side: int, shape) -> tuple[int, ...]:
        """Padded index of the ghost one step *outside* `logical`'s shared face."""
        idx = [c + 1 for c in logical]  # interior -> padded
        idx[axis] = 0 if side == 0 else shape[axis] + 1  # step into the ghost shell
        return tuple(idx)

    def interior_padded(logical: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(c + 1 for c in logical)

    def step_inward(logical: tuple[int, ...], axis: int, side: int) -> tuple[int, ...]:
        idx = list(logical)
        idx[axis] += 1 if side == 0 else -1  # one layer inside the shared face
        return tuple(idx)

    for conn in topo.interface_connections:
        fa, fb = conn.face_a, conn.face_b
        ai, bi = name_to_idx[fa.block_name], name_to_idx[fb.block_name]
        shape_a = topo.block_specs[fa.block_name].logical_shape
        shape_b = topo.block_specs[fb.block_name].logical_shape
        dof_a = grid.block_dof_maps[ai]
        b_face = face_dof_to_logical(bi, fb.axis, fb.side)

        fixed_a = 0 if fa.side == 0 else shape_a[fa.axis] - 1
        free_axes_a = [a for a in range(d) if a != fa.axis]
        for free_idx in product(*[range(shape_a[a]) for a in free_axes_a]):
            logical_a = [0] * d
            logical_a[fa.axis] = fixed_a
            for j, a in enumerate(free_axes_a):
                logical_a[a] = free_idx[j]
            logical_a = tuple(logical_a)

            # Twin on B's face = same global DOF (set identification did the work).
            logical_b = b_face.get(int(dof_a[logical_a]))
            if logical_b is None:
                # Non-conforming interface: a face node has no matching twin.
                raise NotImplementedError(
                    "build_block_structured_context: non-conforming interface "
                    f"between '{fa.block_name}' and '{fb.block_name}' (face node "
                    f"{logical_a} has no shared-DOF twin). Phase 1 supports only "
                    "conforming multiblock (see plan §1.7)."
                )

            # A's ghost (outside A's face) <- B's first interior layer inside the face.
            src_block.append(bi)
            src_padded.append(interior_padded(step_inward(logical_b, fb.axis, fb.side)))
            dst_block.append(ai)
            dst_padded.append(ghost_padded(logical_a, fa.axis, fa.side, shape_a))

            # B's ghost <- A's first interior layer (symmetric).
            src_block.append(ai)
            src_padded.append(interior_padded(step_inward(logical_a, fa.axis, fa.side)))
            dst_block.append(bi)
            dst_padded.append(ghost_padded(logical_b, fb.axis, fb.side, shape_b))

    sing_block = np.array(
        [name_to_idx[s.block_name] for s in topo.singularities], dtype=np.int32
    )
    sing_logical = np.array(
        [s.logical_idx for s in topo.singularities], dtype=np.int32
    ).reshape(-1, d)
    sing_valence = np.array([s.valence for s in topo.singularities], dtype=np.int32)

    # Owner block + shared-node broadcast (the lowest-index block owns/relaxes each
    # shared node; the others hold a copy refreshed by an interior->interior
    # broadcast, distinct from the ghost halo above).
    owner_block, ssb, ssp, sdb, sdp = _owner_and_broadcast(
        d, interior_shapes, grid.block_dof_maps, grid.global_node_count
    )

    e = len(src_block)
    return BlockStructuredContext(
        d=d,
        interior_shapes=interior_shapes,
        block_global_dof=block_global_dof,
        block_colour=block_colour,
        num_colours=2**d,
        colour_consistent=colour_consistent,
        halo_src_block=np.array(src_block, dtype=np.int32),
        halo_src_padded=np.array(src_padded, dtype=np.int32).reshape(e, d),
        halo_dst_block=np.array(dst_block, dtype=np.int32),
        halo_dst_padded=np.array(dst_padded, dtype=np.int32).reshape(e, d),
        sing_block=sing_block,
        sing_logical=sing_logical,
        sing_valence=sing_valence,
        owner_block=owner_block,
        share_src_block=ssb,
        share_src_padded=ssp,
        share_dst_block=sdb,
        share_dst_padded=sdp,
    )


class CppSweepSession:
    """Persistent device-resident smoothing session.

    The flattened context is uploaded once at construction and ``X`` is kept
    device-resident across :meth:`run` calls (no per-call upload/download or
    re-JIT), so chunked driving and warm steady-state timing avoid re-staging.

    Parameters
    ----------
    ctx : SweepContext
        The sweep context from :func:`build_sweep_context`.
    X : ndarray, shape (N, 2)
        Initial node positions (uploaded once).
    device : str, optional
        ``"auto"`` (default), ``"cpu"``, or ``"gpu"``.
    """

    def __init__(
        self,
        ctx: SweepContext,
        X: np.ndarray,
        *,
        device: str = "auto",
    ) -> None:
        from egg._cpp import cpp_core

        self._shape = X.shape
        ctx_arrays = flatten_context(ctx)
        X_flat = np.ascontiguousarray(X, dtype=np.float64).ravel()
        self._session = cpp_core.CppSweepSession(ctx_arrays, X_flat, device=device)

    def run(
        self,
        n_sweeps: int,
        *,
        phase: str = "barrier",
        delta: float = 0.0,
        report_every: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run ``n_sweeps`` on the resident X; returns ``(energies, mindets)``.

        Parameters
        ----------
        report_every : int, optional
            0 (default) reports a single ``(energy, min_det)`` pair for the
            final sweep of this call (chunk-end cadence — minimal reduction
            launches, ideal for the small-n launch-overhead-bound regime);
            1 reports every sweep (legacy per-sweep contract); ``k > 1``
            reports every ``k``-th sweep plus the final one. The returned
            arrays carry exactly the reported values, i.e. length
            ``ceil(n_sweeps / k)`` (or ``1`` when ``report_every <= 0``).
        """
        return self._session.run(
            n_sweeps, phase=phase, delta=delta, report_every=report_every
        )

    def get_X(self) -> np.ndarray:
        """Return a host copy of the resident X, reshaped to ``(N, 2)``."""
        return self._session.get_X().reshape(self._shape)

    def set_X(self, X: np.ndarray) -> None:
        """Re-upload X to the device."""
        self._session.set_X(np.ascontiguousarray(X, dtype=np.float64).ravel())


class CppStructuredSweepSession:
    """Persistent device-resident *structured* smoothing session.

    The structured analogue of :class:`CppSweepSession`: the context is re-homed
    onto the halo-padded per-block store and uploaded once, and the packed ``X``
    stays device-resident across :meth:`run` calls, so the coalesced stencil reads
    and the per-sweep halo exchange are measured warm without re-staging.

    Parameters
    ----------
    ctx : SweepContext
        The sweep context from :func:`build_sweep_context` (global indices).
    bsc : BlockStructuredContext
        The halo-padded structured tables (from
        :func:`build_block_structured_context` for a BlockTopology grid, or
        :func:`build_structured_context_from_block_maps` for a hand-built one).
    X : ndarray, shape (N, d)
        Initial node positions (global node order).
    device : str, optional
        ``"auto"`` (default), ``"cpu"``, or ``"gpu"``.
    """

    def __init__(
        self,
        ctx: SweepContext,
        bsc: BlockStructuredContext,
        X: np.ndarray,
        *,
        device: str = "auto",
    ) -> None:
        from egg._cpp import cpp_core

        self._shape = X.shape
        ctx_arrays = flatten_context(ctx)
        structured = structured_arrays(bsc)
        X_flat = np.ascontiguousarray(X, dtype=np.float64).ravel()
        self._session = cpp_core.CppStructuredSweepSession(
            ctx_arrays, structured, X_flat, device=device, dim=ctx.d
        )

    def run(
        self,
        n_sweeps: int,
        *,
        phase: str = "barrier",
        delta: float = 0.0,
        smoother: str = "colored-gs",
        omega: float = 1.0,
        report_every: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run ``n_sweeps`` on the resident packed X; returns ``(energies, mindets)``.

        Parameters
        ----------
        smoother : str, optional
            ``"colored-gs"`` (default, the in-place per-colour chain) or
            ``"block-jacobi"`` (one merged double-buffered launch per sweep).
        omega : float, optional
            SOR/damping weight for block-Jacobi (``1.0`` = undamped). Ignored by
            colored-GS.
        report_every : int, optional
            0 (default) reports a single ``(energy, min_det)`` pair for the
            final sweep of this call (chunk-end cadence — minimal reduction
            launches, ideal for the small-n launch-overhead-bound regime);
            1 reports every sweep (legacy per-sweep contract); ``k > 1``
            reports every ``k``-th sweep plus the final one. The returned
            arrays carry exactly the reported values, i.e. length
            ``ceil(n_sweeps / k)`` (or ``1`` when ``report_every <= 0``).
        """
        return self._session.run(
            n_sweeps, phase=phase, delta=delta,
            smoother=smoother, omega=omega, report_every=report_every,
        )

    def get_X(self) -> np.ndarray:
        """Return X gathered back to global node order, reshaped to the input shape."""
        return self._session.get_X().reshape(self._shape)


def cpp_sweep(
    ctx: SweepContext,
    X: np.ndarray,
    n_sweeps: int,
    *,
    device: str = "auto",
    phase: str = "barrier",
    delta: float = 0.0,
    report_every: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``n_sweeps`` of colored GS barrier smoothing via the C++ backend.

    Drop-in replacement for ``build_fused_multisweep(ctx).run(X, n_sweeps)``.

    Parameters
    ----------
    ctx : SweepContext
        The sweep context from :func:`build_sweep_context`.
    X : ndarray, shape (N, 2)
        Initial node positions.
    n_sweeps : int
        Number of sweeps to run.
    device : str, optional
        ``"auto"`` (default), ``"cpu"``, or ``"gpu"``.
    report_every : int, optional
        0 (default) reports a single ``(energy, min_det)`` pair for the final
        sweep of this call (chunk-end cadence — minimal reduction launches);
        1 reports every sweep (legacy per-sweep contract); ``k > 1`` reports
        every ``k``-th sweep plus the final one. The returned ``energies`` /
        ``mindets`` arrays carry exactly the reported values (length
        ``ceil(n_sweeps / k)`` or ``1`` when ``report_every <= 0``).

    Returns
    -------
    X_out : ndarray, shape (N, 2)
        Final node positions.
    energies, mindets : ndarray
        Per-report total energy and min det A (see ``report_every``).
    """
    from egg._cpp import cpp_core

    ctx_arrays = flatten_context(ctx)
    X_flat = np.ascontiguousarray(X, dtype=np.float64).ravel()

    X_out_flat, energies, mindets = cpp_core.cpp_sweep(
        ctx_arrays, X_flat, n_sweeps,
        device=device, phase=phase, delta=delta, report_every=report_every,
    )

    return X_out_flat.reshape(X.shape), energies, mindets


def structured_arrays(bsc: BlockStructuredContext) -> dict:
    """Pack a BlockStructuredContext into the ``structured`` dict the C++ binding
    expects (the halo-padded tables; ``BlockLayout`` owns the offset math)."""
    return {
        "interior_shapes": [np.asarray(s, dtype=np.int32) for s in bsc.interior_shapes],
        "block_global_dof": bsc.block_global_dof,
        "halo_src_block": bsc.halo_src_block,
        "halo_src_padded": bsc.halo_src_padded,
        "halo_dst_block": bsc.halo_dst_block,
        "halo_dst_padded": bsc.halo_dst_padded,
        "sing_block": bsc.sing_block,
        "sing_logical": bsc.sing_logical,
        # Conforming-multiblock (step 2): owner of each global DOF + the owner ->
        # non-owner shared-node broadcast that keeps the duplicated copies in sync.
        "owner_block": bsc.owner_block,
        "share_src_block": bsc.share_src_block,
        "share_src_padded": bsc.share_src_padded,
        "share_dst_block": bsc.share_dst_block,
        "share_dst_padded": bsc.share_dst_padded,
    }


def cpp_structured_sweep(
    ctx: SweepContext,
    grid: MultiBlockGrid,
    X: np.ndarray,
    n_sweeps: int,
    *,
    device: str = "auto",
    phase: str = "barrier",
    delta: float = 0.0,
    smoother: str = "colored-gs",
    omega: float = 1.0,
    report_every: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``n_sweeps`` of colored GS over the halo-padded structured store.

    Same colored Gauss-Seidel as :func:`cpp_sweep`, but the context is re-homed
    onto per-block halo-padded arrays (coalesced stencil reads) with a per-sweep
    halo exchange and shared-node broadcast. Converges to the same minimiser as
    :func:`cpp_sweep`; for a single block it is bit-for-bit identical (the halo
    exchange is a no-op).

    Conforming multiblock (shared interfaces) is supported: a shared interface
    node is relaxed only by its owner block (lowest index), cross-block patch
    neighbours read the owner's frozen ghost copy, and the non-owner copies are
    refreshed each sweep (frozen-halo additive Schwarz, cadence 1.4b).

    Parameters
    ----------
    ctx : SweepContext
        The sweep context from :func:`build_sweep_context`.
    grid : MultiBlockGrid
        The grid whose ``topology`` is a ``BlockTopology`` (for the structured
        tables via :func:`build_block_structured_context`).
    X : ndarray, shape (N, d)
        Initial node positions (global node order).
    n_sweeps : int
        Number of sweeps to run.
    device : str, optional
        ``"auto"`` (default), ``"cpu"``, or ``"gpu"``.
    smoother : str, optional
        ``"colored-gs"`` (default, the in-place per-colour chain) or
        ``"block-jacobi"`` (one merged double-buffered launch per sweep). Both
        converge to the same minimiser under the frozen-halo cadence.
    omega : float, optional
        SOR/damping weight for block-Jacobi (``1.0`` = undamped). Ignored by
        colored-GS.
    report_every : int, optional
        0 (default) reports a single ``(energy, min_det)`` pair for the final
        sweep of this call (chunk-end cadence — minimal reduction launches);
        1 reports every sweep (legacy per-sweep contract); ``k > 1`` reports
        every ``k``-th sweep plus the final one. The returned ``energies`` /
        ``mindets`` arrays carry exactly the reported values (length
        ``ceil(n_sweeps / k)`` or ``1`` when ``report_every <= 0``).

    Returns
    -------
    X_out : ndarray, shape (N, d)
        Final node positions (global node order).
    energies, mindets : ndarray
        Per-report total energy and min det A (see ``report_every``).
    """
    from egg._cpp import cpp_core

    ctx_arrays = flatten_context(ctx)
    structured = structured_arrays(build_block_structured_context(grid))
    X_flat = np.ascontiguousarray(X, dtype=np.float64).ravel()

    X_out_flat, energies, mindets = cpp_core.cpp_structured_sweep(
        ctx_arrays, structured, X_flat, n_sweeps,
        device=device, phase=phase, delta=delta, dim=ctx.d,
        smoother=smoother, omega=omega, report_every=report_every,
    )

    return X_out_flat.reshape(X.shape), energies, mindets


def cpp_untangle(
    ctx: SweepContext,
    X: np.ndarray,
    *,
    device: str = "auto",
    sweeps_per_delta: int = 20,
    delta0_factor: float = 2.0,
    shrink: float = 0.5,
    max_outer: int = 60,
    margin: float = 1e-9,
) -> tuple[np.ndarray, float, int, float]:
    """δ-continuation untangle via the C++ backend.

    Same δ-continuation as the stepped loop in
    ``egg.pipeline.generate_steps`` (``untangle_direct=False``), but the whole
    schedule runs in one call here: a
    persistent device-resident session runs ``sweeps_per_delta`` untangle sweeps
    per δ on a geometric schedule ``δ_k = δ_0 · shrink^k`` (starting from
    ``delta0_factor · |min det A|``) until ``min det A > margin``.

    Parameters
    ----------
    ctx : SweepContext
        The (isotropic) sweep context from :func:`build_sweep_context`.
    X : ndarray, shape (N, 2)
        Initial (possibly folded) node positions.
    device : str, optional
        ``"auto"`` (default), ``"cpu"``, or ``"gpu"``.
    sweeps_per_delta, delta0_factor, max_outer, margin : optional
        Continuation schedule controls (defaults match the JAX driver).
    shrink : float, optional
        Geometric δ decrease per outer step (``δ ← shrink·δ``); default 0.5.

    Returns
    -------
    X_out : ndarray, shape (N, 2)
        Untangled node positions (or the best found if the schedule stalls).
    mindet : float
        Final raw min det A.
    outer_iters : int
        Number of δ-steps actually taken.
    delta_final : float
        The δ value at the last step taken.
    """
    es = ctx.energy_stencil
    md = _grid_mindet(X, es)
    if md > margin:
        return X, md, 0, 0.0

    session = CppSweepSession(ctx, X, device=device)
    delta = delta0_factor * max(abs(md), 1e-12)
    outer_iters = 0
    for _ in range(max_outer):
        _e, mds = session.run(sweeps_per_delta, phase="untangle", delta=delta)
        outer_iters += 1
        md = float(np.asarray(mds)[-1])
        if md > margin:
            break
        delta *= shrink

    return session.get_X(), md, outer_iters, delta
