# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""C++ backend bridge for the structured block-Jacobi sweep.

Provides :func:`cpp_structured_sweep` / :class:`CppStructuredSweepSession`, which
call the device-resident C++ block-Jacobi sweep via ``egg._cpp.cpp_core``. The
wire dict they pass comes from the single builder
:func:`egg.smoothing.flat_context.build_flat_context` (carried on
``SweepContext.wire``); :func:`build_block_structured_context` packs a
:class:`~egg.topology.block_topology.BlockTopology` grid into the structured
halo-padded layout tables the GPU path consumes.
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
    "build_block_structured_context",
    "build_structured_context_from_block_maps",
    "cpp_structured_sweep",
    "cpp_untangle",
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


@dataclass
class BlockStructuredContext:
    """Host-built tables for the halo-padded structured layout.

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

    The owner block of each global DOF and the owner→non-owner shared-node
    broadcast are derived in the C++ core from ``block_global_dof`` (a single
    interior pass), so they are not carried here.
    """

    d: int
    interior_shapes: list[tuple[int, ...]]
    block_global_dof: list[np.ndarray]
    halo_src_block: np.ndarray
    halo_src_padded: np.ndarray
    halo_dst_block: np.ndarray
    halo_dst_padded: np.ndarray
    sing_block: np.ndarray
    sing_logical: np.ndarray

    @property
    def num_blocks(self) -> int:
        return len(self.interior_shapes)

    @property
    def num_halo_entries(self) -> int:
        return int(self.halo_dst_block.shape[0])


def build_structured_context_from_block_maps(
    d: int,
    block_global_dof: list[np.ndarray],
) -> BlockStructuredContext:
    """Structured tables from raw per-block global-DOF arrays (no BlockTopology).

    For hand-assembled multiblock grids (e.g. the ``sphere_in_cube`` benchmark)
    that carry only the per-block structured id arrays, not a
    :class:`BlockTopology`. Conforming **face-ghost entries are derived here by
    shared-global-DOF twin matching** (no orientation logic): for each face node
    with a copy in another block, the ghost just outside the face is filled by
    the node one step behind the twin — resolved purely through the id arrays,
    including "patchwork" contacts where one face touches several neighbour
    blocks. Anything that does not resolve (block self-contacts, singular-edge
    neighbourhoods) is left to the C++ structured remap, which mirrors every
    remaining cross-block patch neighbour into a spare ghost slot (the
    singular-fan fallback) — so the sweep is correct either way; the derived
    entries buy coalesced ghost reads and give the FAS hierarchy the tables it
    needs to relax coarse interface DOFs. The singularity tables stay
    empty. The canonical path from a :class:`BlockTopology` is
    :func:`build_block_structured_context`.

    Parameters
    ----------
    d : int
        Spatial dimension.
    block_global_dof : list[np.ndarray]
        Per-block ``d``-D int arrays of global node ids (each block's structured
        id array; ``.shape`` is its interior node-count shape).
    """
    interior_shapes = [tuple(int(s) for s in b.shape) for b in block_global_dof]
    flat_dof = [
        np.ascontiguousarray(b.reshape(-1), dtype=np.int32) for b in block_global_dof
    ]

    # gid -> [(block, logical), ...] over the SHARED nodes only (a face node
    # duplicated across blocks), plus per-block gid sets (to reject a "behind"
    # candidate that runs parallel to the contact).
    n_gids = max(int(f.max()) for f in flat_dof if f.size) + 1 if flat_dof else 0
    counts = np.zeros(n_gids, dtype=np.int64)
    for f in flat_dof:
        counts += np.bincount(f, minlength=n_gids)
    shared = counts > 1
    where: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    for bi, dof in enumerate(block_global_dof):
        for logical in zip(*np.nonzero(shared[dof])):
            li = tuple(int(c) for c in logical)
            where.setdefault(int(dof[li]), []).append((bi, li))
    in_block = [set(int(g) for g in f) for f in flat_dof]

    def behind_gid(dst_block: int, gid: int) -> tuple[int, tuple[int, ...]] | None:
        """(block, logical) of the node one step behind the contact face —
        the OWNER (lowest-block) copy, so the halo source slot is never also a
        broadcast destination — or None when no twin resolves it unambiguously."""
        g_behind = None
        loc = None
        for bj, lj in where.get(gid, ()):
            if bj == dst_block:
                continue
            shape_j = block_global_dof[bj].shape
            for axis in range(d):
                if lj[axis] == 0:
                    step = 1
                elif lj[axis] == shape_j[axis] - 1:
                    step = -1
                else:
                    continue
                inner = list(lj)
                inner[axis] += step
                g_in = int(block_global_dof[bj][tuple(inner)])
                # The contact normal is the axis whose inward step leaves the
                # receiving block; a parallel (seam) step stays inside it.
                if g_in in in_block[dst_block]:
                    continue
                if g_behind is not None and g_behind != g_in:
                    return None  # ambiguous (self-contact / singular edge)
                g_behind = g_in
                loc = (bj, tuple(inner))
        if g_behind is None:
            return None
        if shared[g_behind]:
            return min(where[g_behind])  # owner copy
        return loc  # interior to exactly one block — the candidate's own slot

    halo_src_block: list[int] = []
    halo_src_padded: list[tuple[int, ...]] = []
    halo_dst_block: list[int] = []
    halo_dst_padded: list[tuple[int, ...]] = []
    for bi, dof in enumerate(block_global_dof):
        shape = dof.shape
        for axis in range(d):
            for side in (0, 1):
                fixed = 0 if side == 0 else shape[axis] - 1
                face = np.take(dof, fixed, axis=axis)
                for free_idx in np.ndindex(face.shape):
                    gid = int(face[free_idx])
                    if not shared[gid]:
                        continue  # physical-boundary face node — no twin
                    logical = list(free_idx[:axis]) + [fixed] + list(free_idx[axis:])
                    hit = behind_gid(bi, gid)
                    if hit is None:
                        continue
                    bj, inner = hit
                    ghost = [c + 1 for c in logical]
                    ghost[axis] = 0 if side == 0 else shape[axis] + 1
                    halo_src_block.append(bj)
                    halo_src_padded.append(tuple(c + 1 for c in inner))
                    halo_dst_block.append(bi)
                    halo_dst_padded.append(tuple(ghost))

    empty_i = np.zeros((0,), dtype=np.int32)
    empty_rows = np.zeros((0, d), dtype=np.int32)
    return BlockStructuredContext(
        d=d,
        interior_shapes=interior_shapes,
        block_global_dof=flat_dof,
        halo_src_block=np.asarray(halo_src_block, dtype=np.int32),
        halo_src_padded=(
            np.asarray(halo_src_padded, dtype=np.int32)
            if halo_src_padded
            else empty_rows
        ),
        halo_dst_block=np.asarray(halo_dst_block, dtype=np.int32),
        halo_dst_padded=(
            np.asarray(halo_dst_padded, dtype=np.int32)
            if halo_dst_padded
            else empty_rows
        ),
        sing_block=empty_i,
        sing_logical=empty_rows,
    )


def _capacity_aware_sing_hosts(
    topo, name_to_idx, interior_shapes, block_global_dof, dst_block, dst_padded, d
) -> list[tuple[int, tuple[int, ...]]]:
    """Choose, for each singular node, a host block that has spare ghost-ring
    slots for its fan mirror — instead of whatever block the topology's
    singularity detector emitted.

    A singular node is a shared corner of every block in its fan, so it may be
    relaxed in any of them; the C++ core mirrors its cross-interface fan
    neighbours into spare *ring* slots of the host block. A fully-interior block
    has only its 4 corners free (regardless of resolution), so an arbitrary host
    can run out even when a sibling fan block (one with a domain-boundary face)
    has ample room. This picks the highest-capacity fan block per node, which
    makes the structured backend robust to block ordering / drawn topologies.

    Returns ``[(block_idx, logical_idx), ...]`` in ``topo.singularities`` order.
    """
    sings = topo.singularities
    if not sings:
        return []

    n_blk = len(interior_shapes)
    # Free ghost-ring slots per block: ring size minus the distinct face-ghost
    # destinations (mirrors the C++ ``free_ghosts`` pool exactly).
    free = []
    for shp in interior_shapes:
        interior = int(np.prod(shp))
        padded = int(np.prod([s + 2 for s in shp]))
        free.append(padded - interior)  # ring count; face-ghosts subtracted below
    used: list[set] = [set() for _ in range(n_blk)]
    for e in range(len(dst_block)):
        used[int(dst_block[e])].add(tuple(int(x) for x in dst_padded[e]))
    free = [free[b] - len(used[b]) for b in range(n_blk)]

    # Fan membership: global-DOF of every block corner -> [(block, logical), ...].
    # Singular nodes are block corners, so this captures each node's full fan.
    fan: dict[int, list[tuple[int, tuple[int, ...]]]] = {}
    for b in range(n_blk):
        shp = interior_shapes[b]
        dofmap = block_global_dof[b].reshape(shp)
        for corner in product(*[(0, s - 1) for s in shp]):
            fan.setdefault(int(dofmap[corner]), []).append((b, corner))

    result: list = [None] * len(sings)
    # Minimal perturbation: keep each node's natural (lowest-index) host — which is
    # what the C++ core picks by default — unless it lacks the room its fan needs,
    # only then moving to the highest-capacity fan block. Leaving already-fine hosts
    # untouched keeps the common O-grid case bit-identical to before. Hardest
    # (highest-valence) nodes first; reserve only on a block we newly load.
    for i in sorted(range(len(sings)), key=lambda k: -sings[k].valence):
        s = sings[i]
        host0 = name_to_idx[s.block_name]
        logical0 = tuple(int(x) for x in s.logical_idx)
        gid = int(block_global_dof[host0].reshape(interior_shapes[host0])[logical0])
        cands = fan.get(gid, [(host0, logical0)])
        default = min(cands, key=lambda bl: bl[0])  # lowest-index = the C++ default
        if free[default[0]] >= s.valence:  # valence upper-bounds the foreign nodes
            result[i] = default  # ample room — do not perturb
        else:
            best = max(cands, key=lambda bl: free[bl[0]])
            result[i] = best
            free[best[0]] -= s.valence  # reserve on the roomier host we moved onto
    return result


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

    # Per-block {global_dof: logical_idx} for nodes on a given (axis, side) face,
    # so a face node in block A can locate its twin (same global DOF) in block B.
    def face_dof_to_logical(
        bi: int, axis: int, side: int
    ) -> dict[int, tuple[int, ...]]:
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

    def ghost_padded(
        logical: tuple[int, ...], axis: int, side: int, shape
    ) -> tuple[int, ...]:
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
                    f"{logical_a} has no shared-DOF twin). The structured layout supports "
                    "only conforming multiblock interfaces."
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

    hosts = _capacity_aware_sing_hosts(
        topo, name_to_idx, interior_shapes, block_global_dof, dst_block, dst_padded, d
    )
    sing_block = np.array([h[0] for h in hosts], dtype=np.int32)
    sing_logical = np.array([h[1] for h in hosts], dtype=np.int32).reshape(-1, d)

    e = len(src_block)
    return BlockStructuredContext(
        d=d,
        interior_shapes=interior_shapes,
        block_global_dof=block_global_dof,
        halo_src_block=np.array(src_block, dtype=np.int32),
        halo_src_padded=np.array(src_padded, dtype=np.int32).reshape(e, d),
        halo_dst_block=np.array(dst_block, dtype=np.int32),
        halo_dst_padded=np.array(dst_padded, dtype=np.int32).reshape(e, d),
        sing_block=sing_block,
        sing_logical=sing_logical,
    )


class CppStructuredSweepSession:
    """Persistent device-resident *structured* smoothing session.

    The context is re-homed onto the halo-padded per-block store and uploaded
    once, and the packed ``X`` stays device-resident across :meth:`run` calls, so
    the coalesced stencil reads and the per-sweep halo exchange are measured warm
    without re-staging (block-Jacobi, one merged launch per sweep).

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
        ctx_arrays = ctx.wire
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
        omega: float = 0.8,
        report_every: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run ``n_sweeps`` of block-Jacobi on the resident packed X; returns
        ``(energies, mindets)``.

        Parameters
        ----------
        phase : str, optional
            Objective selector: ``"barrier"`` (default, the shape metric),
            ``"shape_size"`` (shape + ``(det T - 1)^2`` — size-aware, the
            context's targets must carry physical scale), or ``"untangle"``
            (the δ-surrogate; give ``delta``).
        omega : float, optional
            Block-Jacobi SOR/damping weight (``1.0`` = undamped).
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
            n_sweeps,
            phase=phase,
            delta=delta,
            omega=omega,
            report_every=report_every,
        )

    def run_fas(
        self,
        n_cycles: int,
        *,
        nu_pre: int = 2,
        nu_post: int = 2,
        nu_coarse: int = 32,
        omega: float = 0.8,
        max_levels: int = 32,
        phase: str = "barrier",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run ``n_cycles`` of FAS (nonlinear geometric multigrid) V-cycles on
        the resident packed X; returns per-cycle ``(energies, mindets)``.

        Each cycle runs ``nu_pre`` block-Jacobi sweeps, a tau-corrected coarse
        solve that recurses down the factor-2 hierarchy (at most
        ``max_levels`` levels, fine included; the hierarchy stops on its own
        when blocks stop coarsening) to a coarsest level solved with Newton
        sweeps scaled to its interior diameter (capped by ``nu_coarse``), a
        safeguarded coarse-grid correction
        (backtracking line search on the global energy, so energy decrease and
        ``det > 0`` are preserved), and ``nu_post`` sweeps. Barrier phases
        only (``"barrier"``/``"shape_size"``): ``phase="untangle"`` raises
        ``ValueError``. Blocks whose node counts
        admit no coarse level (or ``max_levels < 2``) fall back to plain
        block-Jacobi with the equivalent fine-sweep budget.
        """
        return self._session.run_fas(
            n_cycles,
            nu_pre=nu_pre,
            nu_post=nu_post,
            nu_coarse=nu_coarse,
            omega=omega,
            max_levels=max_levels,
            phase=phase,
        )

    def mg_levels(self) -> list:
        """Coarse-hierarchy shape: one list per level of per-block interior
        node-count tuples; empty when no coarse level can be built."""
        return self._session.mg_levels()

    def get_X(self) -> np.ndarray:
        """Return X gathered back to global node order, reshaped to the input shape."""
        return self._session.get_X().reshape(self._shape)


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
    omega: float = 0.8,
    report_every: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``n_sweeps`` of block-Jacobi over the halo-padded structured store.

    The context is re-homed onto per-block halo-padded arrays (coalesced stencil
    reads) with a per-sweep halo exchange and shared-node broadcast; one merged
    double-buffered launch per sweep, SOR weight ``omega``.

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
    phase : str, optional
        Objective selector: ``"barrier"`` (default, shape), ``"shape_size"``
        (size-aware shape+size), or ``"untangle"`` (δ-surrogate, give
        ``delta``).
    omega : float, optional
        Block-Jacobi SOR/damping weight (``1.0`` = undamped).
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

    ctx_arrays = ctx.wire
    structured = structured_arrays(build_block_structured_context(grid))
    X_flat = np.ascontiguousarray(X, dtype=np.float64).ravel()

    X_out_flat, energies, mindets = cpp_core.cpp_structured_sweep(
        ctx_arrays,
        structured,
        X_flat,
        n_sweeps,
        device=device,
        phase=phase,
        delta=delta,
        dim=ctx.d,
        omega=omega,
        report_every=report_every,
    )

    return X_out_flat.reshape(X.shape), energies, mindets


def cpp_untangle(
    ctx: SweepContext,
    grid: MultiBlockGrid,
    X: np.ndarray,
    *,
    device: str = "auto",
    sweeps_per_delta: int = 20,
    delta0_factor: float = 2.0,
    shrink: float = 0.8,
    max_outer: int = 60,
    margin: float = 1e-9,
    omega: float = 0.5,
) -> tuple[np.ndarray, float, int, float]:
    """δ-continuation untangle via the structured block-Jacobi backend.

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
    grid : MultiBlockGrid
        The grid whose ``topology`` is a ``BlockTopology`` (for the halo-padded
        structured store the block-Jacobi sweep runs over).
    X : ndarray, shape (N, d)
        Initial (possibly folded) node positions.
    device : str, optional
        ``"auto"`` (default), ``"cpu"``, or ``"gpu"``.
    sweeps_per_delta, delta0_factor, max_outer, margin : optional
        Continuation schedule controls.
    shrink : float, optional
        Geometric δ decrease per outer step (``δ ← shrink·δ``); default 0.8.
        Block-Jacobi smooths less per sweep than a sequential relaxation, so it
        needs a gentler schedule — a steeper shrink retires the barrier before
        the simultaneous update can act on it.
    omega : float, optional
        Block-Jacobi relaxation weight for the untangle sweeps; default 0.5.
        Must be damped (``< 1``): an undamped simultaneous update overshoots the
        barrier and can deepen a fold instead of clearing it.

    Returns
    -------
    X_out : ndarray, shape (N, d)
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

    bsc = build_block_structured_context(grid)
    session = CppStructuredSweepSession(ctx, bsc, X, device=device)
    delta = delta0_factor * max(abs(md), 1e-12)
    outer_iters = 0
    for _ in range(max_outer):
        _e, mds = session.run(
            sweeps_per_delta, phase="untangle", delta=delta, omega=omega
        )
        outer_iters += 1
        md = float(np.asarray(mds)[-1])
        if md > margin:
            break
        delta *= shrink

    return session.get_X(), md, outer_iters, delta
