# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Directional soft-energy samples: builder + NumPy reference evaluator.

The topology's :class:`~egg.topology.block_topology.ParallelChain` and
:class:`~egg.topology.block_topology.FanFrame` declarations become a flat
SoA table of *directional samples* — small node tuples with a frozen
reference vector and a composed weight — added to the TMOP objective as soft
penalties:

- ``parallel``: per chain segment ``(i, j)``, ``E = w (t̂·n̂)²`` with ``t̂``
  the live unit segment direction and ``n̂`` the frozen boundary normal at
  the segment's corresponding wall nodes; ``w`` carries the frozen segment
  length, so the term is orientation-only (spacing stays free).
- ``line``: per framed fan vertex, nodes ``(a, c, b)`` (first through-rail
  nodes and the corner), ``E = w |q̂1 + q̂2|²`` with ``q̂k`` the live unit
  directions away from ``c`` — the through legs stay opposed.
- ``stem``: per framed fan vertex, nodes ``(n1, c)``,
  ``E = w (q̂3·â)²`` with ``â`` the frozen unit through axis at the vertex —
  the normal leg stays perpendicular to the through rail.

All unit normalisations are ε-regularised (``x/sqrt(|x|²+ε²)``), so the
terms and gradients stay finite through degenerate (zero-length) states.
The table layout is one schema for every kind, including the 3D-only kinds
that land with the 3D work (sheet-normal alignment and the cross-sectional
line/stem terms along singular edges, which additionally use the frozen
projector direction ``axis``): up to four node ids per sample, a d-component
reference, a d-component projector direction (zeros = unused), one weight.

References are FROZEN at build time: rebuild the samples wherever the sweep
context is rebuilt. The reference evaluator here is the NumPy parity target
for the C++ kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "KIND_PARALLEL",
    "KIND_LINE",
    "KIND_STEM",
    "KIND_SHEET_NORMAL",
    "KIND_EDGE_LINE",
    "KIND_EDGE_STEM",
    "MAX_ARITY",
    "DirectionalSamples",
    "build_directional_samples",
    "directional_energy_grad",
    "directional_dof_table",
    "directional_energy_table",
    "kind_arity",
]

KIND_PARALLEL = 0
KIND_LINE = 1
KIND_STEM = 2
# Reserved for the 3D-only kinds: quad-face sheet-normal alignment and the
# cross-sectional line/stem terms along singular edges (projector axis used).
KIND_SHEET_NORMAL = 3
KIND_EDGE_LINE = 4
KIND_EDGE_STEM = 5

MAX_ARITY = 4


@dataclass
class DirectionalSamples:
    """SoA table of directional samples (one layout for every kind)."""

    nodes: np.ndarray  # (n, MAX_ARITY) int32, -1 padded
    kind: np.ndarray  # (n,) int32
    ref: np.ndarray  # (n, d) frozen reference vector (zeros when unused)
    axis: np.ndarray  # (n, d) frozen projector direction (zeros = unused)
    weight: np.ndarray  # (n,) composed weight
    eps: float = 1e-12

    @property
    def n(self) -> int:
        return int(self.kind.shape[0])

    @property
    def d(self) -> int:
        return int(self.ref.shape[1])


def _unit(v: np.ndarray, eps: float) -> np.ndarray:
    """Rows of ``v`` ε-normalised."""
    s = np.sqrt(np.einsum("ij,ij->i", v, v) + eps * eps)
    return v / s[:, None]


def _normals_at(entity, W: np.ndarray) -> np.ndarray:
    """Boundary normals at (the feet of) the rows of ``W``, unit, (n, d).

    Codimension 1 only: the 90°-CCW rotation of the tangent for a 2D curve,
    the unit cross product of the tangent basis for a 3D surface.
    """
    B = np.asarray(entity.tangent_space_many(np.asarray(W, dtype=float)))
    d, k = B.shape[1], B.shape[2]
    if d == 2 and k == 1:
        t = B[:, :, 0]
        return np.stack([-t[:, 1], t[:, 0]], axis=1)
    if d == 3 and k == 2:
        n = np.cross(B[:, :, 0], B[:, :, 1])
        return _unit(n, 0.0)
    raise NotImplementedError("directional samples need a codimension-1 boundary")


def _chain_consistent_normals(
    n_node: np.ndarray, cos_thresh: float = 0.5
) -> np.ndarray:
    """De-outlier the per-node frozen wall normals along a chain.

    A chain node's frozen normal is the wall entity's normal at that node's
    projection. Where the projection lands on a degenerate or extrapolated wall
    endpoint (e.g. a spline run past its data to meet an exit plane), the
    analytic normal can curl ~90° off the chain's trend — and that single bad
    value then corrupts the adjacent segment reference, pulling that segment's
    cells off in the wrong direction. Any node whose normal agrees (up to sign)
    with NEITHER neighbour beyond ``cos_thresh`` is replaced by the nearest
    consistent node's; orientation is then aligned so a sign flip cannot cancel
    a segment average ``n[k] + n[k+1]``.
    """
    n = np.asarray(n_node, dtype=float).copy()
    N = len(n)
    if N < 3:
        return n
    good = np.ones(N, dtype=bool)
    for i in range(N):
        agree = -1.0
        if i > 0:
            agree = max(agree, abs(float(np.dot(n[i], n[i - 1]))))
        if i < N - 1:
            agree = max(agree, abs(float(np.dot(n[i], n[i + 1]))))
        good[i] = agree >= cos_thresh
    gi = np.flatnonzero(good)
    if not good.all() and gi.size:
        for i in np.flatnonzero(~good):
            j = int(gi[np.argmin(np.abs(gi - i))])
            n[i] = n[j]
    for i in range(1, N):
        if float(np.dot(n[i], n[i - 1])) < 0.0:
            n[i] = -n[i]
    return n


def _singular_dofs(topo) -> set[int]:
    """Global DOFs of framed fan corners and detected singular nodes."""
    out = {f.dof for f in topo.fan_frames}
    block_names = list(topo.block_specs.keys())
    for s in topo.singularities:
        dm = topo.grid.block_dof_maps[block_names.index(s.block_name)]
        out.add(int(dm[tuple(s.logical_idx)]))
    return out


def build_directional_samples(
    grid,
    *,
    lambda_parallel: float = 1.0,
    lambda_line: float = 1.0,
    lambda_stem: float = 1.0,
    lambda_fair: float = 0.0,
    energy_scale: float = 1.0,
    eps: float = 1e-12,
) -> DirectionalSamples | None:
    """Build the directional sample table for ``grid``'s topology, freezing
    every reference (wall normals, segment lengths, vertex axes) from the
    current ``grid.global_nodes``.

    Every weight is dimensionless: a parallel segment carries
    ``λ_parallel · energy_scale · (ℓ/ℓ̄) · taper`` with ℓ̄ the mean segment
    length over every declared chain (longer segments count proportionally
    more, but the absolute domain scale cancels), and the vertex line/stem
    samples carry ``λ · energy_scale``. λ values are therefore O(1) and both
    resolution- and domain-scale-independent. Returns ``None`` when the
    topology declares neither parallel chains nor fan frames.

    ``lambda_fair`` > 0 adds an orientation-only fairness term along each
    chain: a line sample ``w·|q̂1+q̂2|²`` at every interior chain node (skipping
    singular fan corners, whose legs meet at topology-set angles), penalising
    kinks and the short-wavelength ripples a sparse parallelism term can
    otherwise leave, without constraining the spacing along the rail. Keep it
    secondary to ``lambda_parallel`` — a very smooth curve can still be an
    S-curve.
    """
    topo = grid.topology
    chains = getattr(topo, "parallel_chains", [])
    frames = getattr(topo, "fan_frames", [])
    if not chains and not frames:
        return None
    X = np.asarray(grid.global_nodes, dtype=float)
    d = X.shape[1]

    nodes: list[tuple[int, ...]] = []
    kinds: list[int] = []
    refs: list[np.ndarray] = []
    weights: list[float] = []

    need_singular = any(c.taper is not None for c in chains) or (
        lambda_fair > 0.0 and bool(chains)
    )
    singular = _singular_dofs(topo) if need_singular else set()

    # Frozen per-chain geometry first: the segment-length weighting must be
    # DIMENSIONLESS (ℓ/ℓ̄ with ℓ̄ the mean segment length over every declared
    # chain), or the term silently vanishes on small physical domains — a
    # raw-metre ℓ on a centimetre-scale grid is ~1e-4 against shape energies
    # of order one. The relative weighting between long and short segments
    # is what matters; the absolute scale belongs to λ and energy_scale.
    chain_geo = []
    for c in chains:
        dofs = np.asarray(c.dofs, dtype=int)
        P = X[dofs]
        # Frozen wall point per chain node: the column-walk correspondence
        # where it resolved, nearest-foot projection where it did not.
        wall = np.asarray(c.wall_dofs, dtype=int)
        W = np.empty_like(P)
        hit = wall >= 0
        W[hit] = X[wall[hit]]
        if np.any(~hit):
            W[~hit] = np.asarray(c.to.project_many(P[~hit]), dtype=float)
        n_node = _chain_consistent_normals(_normals_at(c.to, W))
        seg = P[1:] - P[:-1]
        ell = np.linalg.norm(seg, axis=1)
        n_seg = _unit(n_node[:-1] + n_node[1:], eps)
        chain_geo.append((c, dofs, ell, n_seg))
    ell_all = np.concatenate([g[2] for g in chain_geo]) if chain_geo else np.zeros(0)
    ell_bar = float(np.mean(ell_all)) if ell_all.size else 1.0

    for c, dofs, ell, n_seg in chain_geo:
        w_seg = lambda_parallel * energy_scale * (ell / ell_bar)

        arc = np.concatenate([[0.0], np.cumsum(ell)])
        marks = [arc[k] for k, g in enumerate(c.dofs) if int(g) in singular]
        if c.taper is not None and marks:
            # Gaussian falloff with chain arclength from the nearest
            # singular vertex on the chain (uniform when there is none).
            mid = 0.5 * (arc[:-1] + arc[1:])
            dist = np.min(np.abs(mid[:, None] - np.asarray(marks)[None, :]), axis=1)
            w_seg = w_seg * np.exp(-((dist / c.taper) ** 2))

        for k in range(len(dofs) - 1):
            nodes.append((int(dofs[k]), int(dofs[k + 1])))
            kinds.append(KIND_PARALLEL)
            refs.append(n_seg[k])
            weights.append(float(c.weight) * float(w_seg[k]))

        # Chain fairness: an orientation-only anti-kink line sample at every
        # interior chain node. Singular fan corners are skipped — their legs
        # meet at angles the topology sets, and forcing straightness there
        # would fight the fan's own frame terms.
        if lambda_fair > 0.0 and len(dofs) >= 3:
            w_node = np.full(len(dofs), lambda_fair * energy_scale * float(c.weight))
            if c.taper is not None and marks:
                nd = np.min(np.abs(arc[:, None] - np.asarray(marks)[None, :]), axis=1)
                w_node = w_node * np.exp(-((nd / c.taper) ** 2))
            for k in range(1, len(dofs) - 1):
                if int(dofs[k]) in singular:
                    continue
                nodes.append((int(dofs[k - 1]), int(dofs[k]), int(dofs[k + 1])))
                kinds.append(KIND_LINE)
                refs.append(np.zeros(d))
                weights.append(float(w_node[k]))

    for f in frames:
        a = int(f.through_rails[0][1])
        b = int(f.through_rails[1][1])
        cdof = int(f.dof)
        n1 = int(f.normal_rail[1])
        nodes.append((a, cdof, b))
        kinds.append(KIND_LINE)
        refs.append(np.zeros(d))
        weights.append(lambda_line * energy_scale)
        axis_vec = _unit((X[b] - X[a])[None], eps)[0]
        nodes.append((n1, cdof))
        kinds.append(KIND_STEM)
        refs.append(axis_vec)
        weights.append(lambda_stem * energy_scale)

    n = len(nodes)
    node_arr = np.full((n, MAX_ARITY), -1, dtype=np.int32)
    for i, t in enumerate(nodes):
        node_arr[i, : len(t)] = t
    return DirectionalSamples(
        nodes=node_arr,
        kind=np.asarray(kinds, dtype=np.int32),
        ref=np.asarray(refs, dtype=float),
        axis=np.zeros((n, d)),
        weight=np.asarray(weights, dtype=float),
        eps=float(eps),
    )


def _dot_pen_grad(u: np.ndarray, ref: np.ndarray, w: np.ndarray, eps: float):
    """Energy and ∂/∂u of ``w ((u·ref)/s)²`` with ``s = sqrt(|u|²+ε²)``.

    The shared core of the parallel and stem kinds.
    """
    s2 = np.einsum("ij,ij->i", u, u) + eps * eps
    s = np.sqrt(s2)
    c = np.einsum("ij,ij->i", u, ref) / s
    e = w * c * c
    gu = (2.0 * w * c)[:, None] * (ref / s[:, None] - (c / s2)[:, None] * u)
    return e, gu


def directional_energy_grad(
    X: np.ndarray, samples: DirectionalSamples
) -> tuple[float, np.ndarray]:
    """Total directional energy and its gradient w.r.t. every node, (n, d).

    Pure NumPy reference implementation — the parity target for the C++
    kernels. Raises on the reserved (not yet implemented) kinds.
    """
    X = np.asarray(X, dtype=float)
    grad = np.zeros_like(X)
    if samples is None or samples.n == 0:
        return 0.0, grad
    if np.any(samples.kind > KIND_STEM):
        raise NotImplementedError("reserved directional kind in sample table")
    eps = samples.eps
    energy = 0.0

    m = samples.kind == KIND_PARALLEL
    if np.any(m):
        i, j = samples.nodes[m, 0], samples.nodes[m, 1]
        e, gu = _dot_pen_grad(X[j] - X[i], samples.ref[m], samples.weight[m], eps)
        energy += float(np.sum(e))
        np.add.at(grad, j, gu)
        np.add.at(grad, i, -gu)

    m = samples.kind == KIND_LINE
    if np.any(m):
        a, c, b = samples.nodes[m, 0], samples.nodes[m, 1], samples.nodes[m, 2]
        w = samples.weight[m]
        q1, q2 = X[a] - X[c], X[b] - X[c]
        s1 = np.sqrt(np.einsum("ij,ij->i", q1, q1) + eps * eps)
        s2 = np.sqrt(np.einsum("ij,ij->i", q2, q2) + eps * eps)
        r = q1 / s1[:, None] + q2 / s2[:, None]
        energy += float(np.sum(w * np.einsum("ij,ij->i", r, r)))
        g1 = (2.0 * w / s1)[:, None] * (
            r - q1 * (np.einsum("ij,ij->i", q1, r) / (s1 * s1))[:, None]
        )
        g2 = (2.0 * w / s2)[:, None] * (
            r - q2 * (np.einsum("ij,ij->i", q2, r) / (s2 * s2))[:, None]
        )
        np.add.at(grad, a, g1)
        np.add.at(grad, b, g2)
        np.add.at(grad, c, -(g1 + g2))

    m = samples.kind == KIND_STEM
    if np.any(m):
        n1, c = samples.nodes[m, 0], samples.nodes[m, 1]
        e, gu = _dot_pen_grad(X[n1] - X[c], samples.ref[m], samples.weight[m], eps)
        energy += float(np.sum(e))
        np.add.at(grad, n1, gu)
        np.add.at(grad, c, -gu)

    return energy, grad


def kind_arity(kind: int) -> int:
    """Occupied node slots of a directional kind (0 for reserved kinds)."""
    if kind == KIND_LINE:
        return 3
    if kind in (KIND_PARALLEL, KIND_STEM):
        return 2
    return 0


def directional_dof_table(free_mask, samples: DirectionalSamples | None):
    """Per-free-DOF fixed-K directional sample table (wire layout).

    Aligns with the flat context's group DOF order (``np.flatnonzero(free_mask)``):
    for each free DOF, every sample it participates in is listed with the DOF's
    slot (0..arity−1), padded to a common width ``K`` (slot ``-1`` marks an empty
    pad). Node ids are GLOBAL — the structured remap rewrites them to owner node
    indices, and the packed X (read frozen per sweep) is global-readable.

    Returns a dict of wire arrays: ``dir_k`` (int), ``dir_eps`` (float),
    ``dir_nodes`` ``(ndof, K, 4)`` int32, ``dir_slot``/``dir_kind`` ``(ndof, K)``
    int32, ``dir_ref``/``dir_axis`` ``(ndof, K, d)`` float64, ``dir_weight``
    ``(ndof, K)`` float64. Empty dict when there is nothing to pack.
    """
    free = np.asarray(free_mask)
    dofs = np.flatnonzero(free)
    if samples is None or samples.n == 0 or dofs.size == 0:
        return {}
    pos = np.full(free.shape[0], -1, dtype=np.int64)
    pos[dofs] = np.arange(dofs.size)

    d = samples.d
    lists: list[list[tuple]] = [[] for _ in range(dofs.size)]
    for si in range(samples.n):
        arity = kind_arity(int(samples.kind[si]))
        for slot in range(arity):
            node = int(samples.nodes[si, slot])
            if node >= 0 and free[node]:
                lists[int(pos[node])].append((si, slot))

    K = max((len(entry) for entry in lists), default=0)
    if K == 0:
        return {}
    dir_nodes = np.full((dofs.size, K, MAX_ARITY), -1, dtype=np.int32)
    dir_slot = np.full((dofs.size, K), -1, dtype=np.int32)
    dir_kind = np.zeros((dofs.size, K), dtype=np.int32)
    dir_ref = np.zeros((dofs.size, K, d), dtype=np.float64)
    dir_axis = np.zeros((dofs.size, K, d), dtype=np.float64)
    dir_weight = np.zeros((dofs.size, K), dtype=np.float64)
    for i, entry in enumerate(lists):
        for j, (si, slot) in enumerate(entry):
            dir_nodes[i, j] = samples.nodes[si]
            dir_slot[i, j] = slot
            dir_kind[i, j] = samples.kind[si]
            dir_ref[i, j] = samples.ref[si]
            dir_axis[i, j] = samples.axis[si]
            dir_weight[i, j] = samples.weight[si]
    return {
        "dir_k": int(K),
        "dir_eps": float(samples.eps),
        "dir_nodes": np.ascontiguousarray(dir_nodes),
        "dir_slot": np.ascontiguousarray(dir_slot),
        "dir_kind": np.ascontiguousarray(dir_kind),
        "dir_ref": np.ascontiguousarray(dir_ref),
        "dir_axis": np.ascontiguousarray(dir_axis),
        "dir_weight": np.ascontiguousarray(dir_weight),
    }


def directional_energy_table(samples: DirectionalSamples | None):
    """Whole-sample directional table for the energy stencil (wire layout).

    Every sample rides once so the energy/line-search reductions report the
    composed objective. Node ids are GLOBAL (remapped alongside the stencil).
    Returns ``dir_num`` (int), ``dir_eps`` (float), ``dir_nodes`` ``(n, 4)``
    int32, ``dir_kind`` ``(n,)`` int32, ``dir_ref``/``dir_axis`` ``(n, d)``
    float64, ``dir_weight`` ``(n,)`` float64; empty dict when no samples.
    """
    if samples is None or samples.n == 0:
        return {}
    return {
        "dir_num": int(samples.n),
        "dir_eps": float(samples.eps),
        "dir_nodes": np.ascontiguousarray(samples.nodes.astype(np.int32)),
        "dir_kind": np.ascontiguousarray(samples.kind.astype(np.int32)),
        "dir_ref": np.ascontiguousarray(samples.ref.astype(np.float64)),
        "dir_axis": np.ascontiguousarray(samples.axis.astype(np.float64)),
        "dir_weight": np.ascontiguousarray(samples.weight.astype(np.float64)),
    }
