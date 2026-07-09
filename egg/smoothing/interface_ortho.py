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


def _sample_for(X, P, fr, c_hat) -> tuple | None:
    """One corner sample (columns cross, tangent) with target W = [c_hat*Lc | f].

    ``c_hat`` is the target cross direction; picks the tangent neighbour that
    makes det W > 0. Returns (gc, gn0, gn1, s0, s1, W_inv(2,2), participants)
    or None when the node has no usable tangent neighbour.
    """
    p = X[P]
    e = X[fr["Q"]] - p  # current cross edge
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
    ca, ta = fr["cross_axis"], fr["tan_axis"]

    # Try each available tangent neighbour, keep the one giving det W > 0.
    for R in (fr["Rp"], fr["Rm"]):
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
        gn[ca], gn[ta] = fr["Q"], R
        W_inv = np.linalg.inv(W)
        part_node = np.array([P, fr["Q"], R], dtype=np.int64)
        part_role = np.array([0, 1 + ca, 1 + ta], dtype=np.int32)
        return (P, gn[0], gn[1], s[0], s[1], W_inv, part_node, part_role)
    return None


def interface_ortho_samples(
    grid, *, mode: str = "normal", weight: float = 1.0, topology=None
) -> InterfaceSamples:
    """Build the interface orthogonality/continuity samples for a 2D grid.

    Parameters
    ----------
    grid : MultiBlockGrid
    mode : {"normal", "continuous"}
    weight : float
        Per-sample weight of the composed term (soft; larger = more dominant).
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

    rows: list[tuple] = []
    for conn in topo.interface_connections:
        fa = _side_frames(grid, block_names, conn.face_a)
        fb = _side_frames(grid, block_names, conn.face_b)
        for P, frA in fa.items():
            frB = fb.get(P)
            if frB is None:
                continue  # non-conforming seam node; skip
            if mode == "normal":
                # Seam tangent from the seam polyline (central difference).
                tp = X[frA["Rp"]] if frA["Rp"] >= 0 else X[P]
                tm = X[frA["Rm"]] if frA["Rm"] >= 0 else X[P]
                tan = tp - tm
                n_hat = np.array([-tan[1], tan[0]])  # rotate +90
                cA = cB = n_hat
            else:  # continuous: straight crossing line through P
                cross = X[frA["Q"]] - X[frB["Q"]]
                cA, cB = cross, -cross
            for fr, c_hat in ((frA, cA), (frB, cB)):
                got = _sample_for(X, P, fr, c_hat)
                if got is not None:
                    rows.append(got)

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
    w = np.full(gc.shape[0], float(weight))
    return InterfaceSamples(gc, gn0, gn1, s0, s1, W_inv, w, part_node, part_role)
