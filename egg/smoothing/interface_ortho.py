# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Block-interface orthogonality / continuity as composed TMOP samples (2D).

The seam between two blocks is smoothed by the shape metric like any other
region, so grid lines cross it continuously but neither orthogonally nor
kink-free once an anisotropic (wall-clustering) target skews the near-seam
cells. This module adds an *extra* set of weighted shape samples at the
interface-adjacent cells whose oriented target ``W`` rewards the desired seam
behaviour, composed additively with the base energy (E += Σ w·mu):

* ``mode="normal"``     — the cross-seam edge is pulled perpendicular to the
  seam tangent (which also makes the two crossing half-edges collinear, i.e.
  continuous).
* ``mode="continuous"`` — the cross-seam edge is pulled onto the straight line
  through the seam node (kink-free) at its natural angle, without forcing it
  normal.

Both are the same construction: a corner sample at the seam node ``P`` with
columns (cross edge ``P->Q``, tangent edge ``P->R``) and a target
``W = [c_hat*Lc | f]``, where ``f`` is the current tangent edge (so the metric
applies no tangential force and preserves seam spacing), ``Lc`` the current
cross-edge length (preserves cell size), and ``c_hat`` the target cross
direction — the seam normal for ``"normal"``, the straight-crossing direction
for ``"continuous"``. The frame is frozen from the current node positions, so a
converging run re-derives it between passes (like ``BoundaryLayerTarget``).

The samples are returned in the same layout the flat sweep context consumes
(gc/gn/s/W_inv per sample, plus per-sample participants and roles for gradient
scatter), so the same generator feeds both the NumPy reference and the C++
backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["InterfaceSamples", "interface_ortho_samples"]


@dataclass
class InterfaceSamples:
    """Extra weighted corner samples for the interface term (energy-order).

    Arrays are (P,) unless noted; ``W_inv`` is (P, 2, 2), ``weight`` (P,).
    ``part_node``/``part_role`` are (P, 3): the three participating global DOFs
    (corner, axis-0 neighbour, axis-1 neighbour) and their sweep roles
    (0, 1, 2). A participant that is a fixed DOF is still listed; the consumer
    drops it when scattering the gradient.
    """

    gc: np.ndarray
    gn0: np.ndarray
    gn1: np.ndarray
    s0: np.ndarray
    s1: np.ndarray
    W_inv: np.ndarray
    weight: np.ndarray
    part_node: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int64))
    part_role: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int32))

    def __len__(self) -> int:
        return int(self.gc.shape[0])


def _oriented_map(dm: np.ndarray, axis: int, side: int) -> np.ndarray:
    """Block DOF map with ``axis`` first and the face at row 0 (2D)."""
    m = np.moveaxis(dm, axis, 0)
    return m[::-1] if side == 1 else m


def _side_frames(grid, block_names, face) -> dict[int, dict]:
    """Per interior seam node on ``face``: its cross/tangent neighbour DOFs.

    Returns ``{P_id: {"Q", "Rp", "Rm", "cross_axis", "tan_axis"}}`` with global
    DOF ids; ``Rp``/``Rm`` are the +/- tangent neighbours (``-1`` if off-block).
    """
    bi = block_names.index(face.block_name)
    dm = np.asarray(grid.block_dof_maps[bi])
    m = _oriented_map(dm, face.axis, face.side)  # (n_cross, n_tan), row0 = seam
    n_tan = m.shape[1]
    cross_axis, tan_axis = face.axis, 1 - face.axis
    out: dict[int, dict] = {}
    for j in range(1, n_tan - 1):  # skip seam-corner endpoints
        out[int(m[0, j])] = {
            "Q": int(m[1, j]),
            "Rp": int(m[0, j + 1]),
            "Rm": int(m[0, j - 1]),
            "cross_axis": cross_axis,
            "tan_axis": tan_axis,
        }
    return out


def _smoothstep(t: float) -> float:
    """Hermite smoothstep on [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _sample_for(X, corner, Q, Rp, Rm, ca, ta, c_hat, frozen) -> tuple | None:
    """One corner sample (columns cross, tangent) with target W = [c_hat*Lc | f].

    ``corner`` is the sample corner (a seam node at layer 0, else a near-seam
    interior node), ``Q`` its cross neighbour one layer deeper, ``Rp``/``Rm``
    its +/- tangent neighbours (-1 if off-block). ``c_hat`` is the target cross
    direction; picks the tangent neighbour that makes det W > 0. Returns
    (gc, gn0, gn1, s0, s1, W_inv(2,2), participants) or None.

    A participant in ``frozen`` (the block-boundary nodes) is registered with
    role -1, so the term never scatters force onto it — the seam's smoothness
    is the base shape metric's job, and the term orthogonalises by moving only
    the interior nodes. The seam node is still read for the sample geometry.
    """
    p = X[corner]
    e = X[Q] - p  # current cross edge
    Lc = float(np.linalg.norm(e))
    if Lc == 0.0:
        return None
    c = np.asarray(c_hat, dtype=float)
    nc = float(np.linalg.norm(c))
    if nc == 0.0:
        return None
    c = c / nc
    if float(np.dot(c, e)) < 0.0:
        c = -c

    # Try each available tangent neighbour, keep the one giving det W > 0.
    for R in (Rp, Rm):
        if R < 0:
            continue
        f = X[R] - p  # current tangent edge (used verbatim as W's tan column)
        W = np.empty((2, 2))
        W[:, ca] = c * Lc
        W[:, ta] = f
        if np.linalg.det(W) <= 0.0:
            continue
        gn = [0, 0]
        s = [1.0, 1.0]
        gn[ca], gn[ta] = Q, R
        W_inv = np.linalg.inv(W)
        part_node = np.array([corner, Q, R], dtype=np.int64)
        part_role = np.array([0, 1 + ca, 1 + ta], dtype=np.int32)
        for i, node in enumerate(part_node):
            if int(node) in frozen:
                part_role[i] = -1  # block-boundary node: read, never pushed
        return (corner, gn[0], gn[1], s[0], s[1], W_inv, part_node, part_role)
    return None


def interface_ortho_samples(
    grid,
    *,
    mode: str = "normal",
    weight: float = 1.0,
    n_layers: int = 3,
    cluster_relax: float = 0.0,
    topology=None,
) -> InterfaceSamples:
    """Build the interface orthogonality/continuity samples for a 2D grid.

    Parameters
    ----------
    grid : MultiBlockGrid
    mode : {"normal", "continuous"}
    weight : float
        Per-sample weight of the composed term (soft; larger = more dominant).
    n_layers : int
        Depth of the near-seam band the term acts on. The crossing edge is
        oriented not just at the seam (layer 0) but through the first
        ``n_layers`` layers on each side; each layer's target direction is
        blended from the seam constraint toward the edge's *current* direction
        deeper in (smoothstep in ``k/n_layers``), so the grid line curves
        *gradually* to the target seam angle instead of snapping the first edge
        (which relocates the kink one layer in). ``n_layers=1`` acts on the
        first edge only — the concentrating behaviour; kept for parity/tests.
    cluster_relax : float
        In ``[0, 1]``, how strongly to relax the orthogonality target where the
        near-seam cell is anisotropic. A wall-normal-clustered cell is a thin
        sliver whose shape is set by the clustering, not the blocking; forcing
        its crossing edge perpendicular to an oblique seam would tilt it off the
        wall-normal. This fades the target toward the crossing edge's *current*
        direction by ``cluster_relax * aniso`` (``aniso`` = 0 for a square cell,
        →1 for a sliver), so the term backs off in the clustered band and only
        acts where the blocking, not the clustering, owns the cell shape. ``0``
        (default) keeps the plain orthogonality target.
    topology : BlockTopology, optional
        Defaults to ``grid.topology``.
    """
    topo = topology if topology is not None else grid.topology
    if topo.d != 2:
        raise NotImplementedError("interface_ortho_samples is 2D-only for now")
    if mode not in ("normal", "continuous"):
        raise ValueError(f"unknown mode {mode!r}")
    X = np.asarray(grid.global_nodes, dtype=float)
    block_names = list(topo.block_specs.keys())

    # Every block-boundary node: the term reads these but never pushes them, so
    # the seam stays as smooth as the base shape metric makes it.
    frozen: set[int] = set()
    for conn in topo.interface_connections:
        for face in (conn.face_a, conn.face_b):
            bi = block_names.index(face.block_name)
            m = _oriented_map(np.asarray(grid.block_dof_maps[bi]), face.axis, face.side)
            frozen.update(int(x) for x in m[0, :])

    rows: list[tuple] = []
    wts: list[float] = []

    def side(m, j, ca, ta, base_dir, k):
        """Emit the layer-k crossing sample of column j on one oriented map.

        The target cross direction is blended from ``base_dir`` (the seam
        constraint: normal, or the straight-crossing direction) at the seam
        toward the edge's *current* direction deeper in the band, so the grid
        line curves smoothly to meet the seam instead of being straightened to
        a fixed direction (which just relocates the kink to the band edge)."""
        if k + 1 >= m.shape[0]:
            return
        corner, Q = int(m[k, j]), int(m[k + 1, j])
        e = X[Q] - X[corner]
        ne = float(np.linalg.norm(e))
        if ne == 0.0:
            return
        e_hat = e / ne
        base = np.asarray(base_dir, dtype=float)
        nb = float(np.linalg.norm(base))
        if nb == 0.0:
            return
        base = base / nb
        if float(np.dot(base, e_hat)) < 0.0:
            base = -base
        Rp = int(m[k, j + 1]) if j + 1 < m.shape[1] else -1
        Rm = int(m[k, j - 1]) if j - 1 >= 0 else -1
        t = _smoothstep(k / n_layers)  # 0 at seam → 1 at band edge
        # Clustering relax: a thin near-seam cell (its shape set by wall-normal
        # clustering, not the blocking) fades the orthogonality target toward the
        # crossing edge's current direction — which follows the clustering — so
        # the term does not tilt the clustered cells off the wall-normal. aniso
        # is the cross/tangent length imbalance: 0 for a square cell, →1 sliver.
        # Cubed so only genuine slivers (near-wall aspect ~20+, aniso≈0.95) relax
        # and a merely sheared/moderate-aspect cell (aniso≈0.3) is left alone.
        if cluster_relax > 0.0:
            tl = [float(np.linalg.norm(X[R] - X[corner])) for R in (Rp, Rm) if R >= 0]
            if tl:
                lt = float(np.mean(tl))
                hi = max(ne, lt)
                aniso = 0.0 if hi == 0.0 else 1.0 - (min(ne, lt) / hi)
                t = t + (1.0 - t) * cluster_relax * (aniso**3)
        c_hat = (1.0 - t) * base + t * e_hat
        got = _sample_for(X, corner, Q, Rp, Rm, ca, ta, c_hat, frozen)
        if got is not None:
            rows.append(got)
            wts.append(weight)

    for conn in topo.interface_connections:
        biA = block_names.index(conn.face_a.block_name)
        biB = block_names.index(conn.face_b.block_name)
        mA = _oriented_map(
            np.asarray(grid.block_dof_maps[biA]), conn.face_a.axis, conn.face_a.side
        )
        mB = _oriented_map(
            np.asarray(grid.block_dof_maps[biB]), conn.face_b.axis, conn.face_b.side
        )
        caA, taA = conn.face_a.axis, 1 - conn.face_a.axis
        caB, taB = conn.face_b.axis, 1 - conn.face_b.axis
        posB = {int(mB[0, j]): j for j in range(mB.shape[1])}
        for jA in range(1, mA.shape[1] - 1):  # skip seam-corner endpoints
            P = int(mA[0, jA])
            jB = posB.get(P)
            if jB is None or jB in (0, mB.shape[1] - 1):
                continue  # non-conforming or corner seam node
            if mode == "normal":
                # Seam normal (rotate the seam tangent), shared by both sides so
                # orthogonality-both-sides gives a straight crossing (continuity).
                tan = X[int(mA[0, jA + 1])] - X[int(mA[0, jA - 1])]
                cA = cB = np.array([-tan[1], tan[0]])
            else:  # continuous: straight crossing direction through the seam
                cross = X[int(mA[1, jA])] - X[int(mB[1, jB])]
                cA, cB = cross, -cross
            for k in range(n_layers):
                side(mA, jA, caA, taA, cA, k)
                side(mB, jB, caB, taB, cB, k)

    if not rows:
        z = np.zeros(0)
        return InterfaceSamples(z, z, z, z, z, np.zeros((0, 2, 2)), z)

    gc = np.array([r[0] for r in rows], dtype=np.int64)
    gn0 = np.array([r[1] for r in rows], dtype=np.int64)
    gn1 = np.array([r[2] for r in rows], dtype=np.int64)
    s0 = np.array([r[3] for r in rows], dtype=float)
    s1 = np.array([r[4] for r in rows], dtype=float)
    W_inv = np.stack([r[5] for r in rows])
    part_node = np.stack([r[6] for r in rows])
    part_role = np.stack([r[7] for r in rows])
    return InterfaceSamples(
        gc, gn0, gn1, s0, s1, W_inv, np.asarray(wts, dtype=float), part_node, part_role
    )
