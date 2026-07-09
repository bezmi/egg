# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Block-interface curvature-continuity (C2) term as composed line windows (2D).

The shape/shape_size metric leaves grid lines kinked where they cross a block
interface (the seam node is free, but no term rewards a smooth *curvature*
crossing). This module adds a high-order term over four-consecutive-node windows
of every grid line — assembled globally so a window spans the seam
transparently (:mod:`egg.smoothing.highorder`) — that penalises the change in
turning angle ``(theta_i - theta_{i+1})**2``. Zero on a straight line and on a
constant-curvature arc (so it neither fights smooth geometry nor clustering),
positive at a kink. Windows touching a seam node carry an ``iface_boost`` so the
term concentrates where the shape metric leaves waviness, without distorting
block interiors.

This is the NumPy definition the C++ core mirrors (``src/curvature.hpp`` /
``curvature_kernel``): the energy matches the prototype in
:func:`egg.smoothing.highorder._f_curv`; the gradient here is the closed form
(the prototype used local finite differences), and the per-node Hessian is the
Gauss-Newton (PSD) block ``2 w (dr/dp) (dr/dp)^T``. The term is added to the
sweep energy/grad/hess and gated by the unchanged det barrier, so it composes
and never inverts a cell.

A window is emitted once, with its four global DOF ids and a weight. The C++
structured path assigns each seam-crossing window to the block whose single
ghost already holds its lone across-seam node (see the port plan); no widened
halo is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .highorder import _curv_cols, grid_adjacency, grid_lines

__all__ = [
    "CurvatureWindows",
    "curvature_windows",
    "curvature_energy_grad_hess",
    "curvature_dof_table",
]


@dataclass
class CurvatureWindows:
    """Four-node curvature-continuity windows over the grid lines.

    ``nodes`` is ``(W, 4)`` int64 global DOF ids (consecutive along a grid line,
    seam-crossing where applicable); ``weight`` is ``(W,)`` float (base weight,
    with the interface boost already folded in).
    """

    nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.int64))
    weight: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __len__(self) -> int:
        return int(self.nodes.shape[0])


def _seam_nodes(grid) -> np.ndarray:
    """Global ids shared by more than one block map (interface seam nodes)."""
    seen: dict[int, int] = {}
    for m in grid.block_dof_maps:
        for nid in np.unique(np.asarray(m)):
            seen[int(nid)] = seen.get(int(nid), 0) + 1
    return np.array([n for n, c in seen.items() if c > 1], dtype=np.int64)


def curvature_windows(
    grid,
    *,
    weight: float = 1.0,
    iface_boost: float = 1.0,
    interface_only: bool = False,
    topology=None,
) -> CurvatureWindows:
    """Build the curvature-continuity windows for a 2D grid.

    Parameters
    ----------
    weight : float
        Base per-window weight of the composed term (soft; larger = more
        dominant).
    iface_boost : float
        Extra multiplier on windows that touch a block-interface seam node —
        concentrates the C2 constraint at the interfaces.
    interface_only : bool
        Drop windows that do not touch a seam (base weight applies only through
        the boost); the pure across-interface term.
    topology : BlockTopology, optional
        Defaults to ``grid.topology``.
    """
    topo = topology if topology is not None else grid.topology
    if topo.d != 2:
        raise NotImplementedError("curvature_windows is 2D-only")
    nbr, partner = grid_adjacency(grid)
    lines = grid_lines(grid, nbr, partner)
    cc = _curv_cols(lines)
    if cc[0].size == 0:
        return CurvatureWindows()
    nodes = np.stack(cc, axis=1).astype(np.int64)  # (W, 4)

    n_total = int(np.asarray(grid.global_nodes).shape[0])
    is_seam = np.zeros(n_total, bool)
    is_seam[_seam_nodes(grid)] = True
    touches = is_seam[nodes].any(axis=1)

    w = np.where(touches, weight * iface_boost, weight).astype(float)
    if interface_only:
        keep = touches
        nodes, w = nodes[keep], w[keep]
    return CurvatureWindows(nodes=np.ascontiguousarray(nodes), weight=w)


def curvature_dof_table(free_mask, win: CurvatureWindows):
    """Per-free-DOF fixed-K curvature-window table (wire layout).

    Aligns with the flat context's group DOF order (``np.flatnonzero(free_mask)``):
    for each free DOF, every window it participates in is listed with the DOF's
    slot (0..3) and the window's weight, padded to a common width ``K`` (slot
    ``-1`` marks an empty pad). The four node ids are GLOBAL — the structured
    remap rewrites them to owner node indices, and the packed X (read frozen per
    sweep) is global-readable, so no ghost slot is needed.

    Returns a dict of wire arrays: ``curv_k`` (int), ``curv_nodes``
    ``(ndof, K, 4)`` int32, ``curv_slot`` ``(ndof, K)`` int32,
    ``curv_weight`` ``(ndof, K)`` float64. Empty dict when there are no windows.
    """
    free = np.asarray(free_mask)
    dofs = np.flatnonzero(free)
    if len(win) == 0 or dofs.size == 0:
        return {}
    pos = np.full(free.shape[0], -1, dtype=np.int64)
    pos[dofs] = np.arange(dofs.size)

    lists: list[list[tuple]] = [[] for _ in range(dofs.size)]
    is_free = free[win.nodes]  # (W, 4)
    for wi in range(len(win)):
        row = win.nodes[wi]
        w = float(win.weight[wi])
        for slot in range(4):
            if is_free[wi, slot]:
                lists[int(pos[int(row[slot])])].append((row, slot, w))

    K = max((len(entry) for entry in lists), default=0)
    if K == 0:
        return {}
    curv_nodes = np.full((dofs.size, K, 4), -1, dtype=np.int32)
    curv_slot = np.full((dofs.size, K), -1, dtype=np.int32)
    curv_weight = np.zeros((dofs.size, K), dtype=np.float64)
    for i, entry in enumerate(lists):
        for j, (row, slot, w) in enumerate(entry):
            curv_nodes[i, j] = row
            curv_slot[i, j] = slot
            curv_weight[i, j] = w
    return {
        "curv_k": int(K),
        "curv_nodes": np.ascontiguousarray(curv_nodes),
        "curv_slot": np.ascontiguousarray(curv_slot),
        "curv_weight": np.ascontiguousarray(curv_weight),
    }


def _perp(v):
    """Rotate the batched 2-vectors ``v`` (…, 2) by +90 degrees."""
    out = np.empty_like(v)
    out[..., 0] = -v[..., 1]
    out[..., 1] = v[..., 0]
    return out


def _turn_grad(a, b, c):
    """Turning angle theta at b along a->b->c and d theta / d(a,b,c).

    Returns ``(theta (W,), grads (W, 3, 2))`` with grads ordered (a, b, c).
    theta = angle(e2) - angle(e1), e1 = b - a, e2 = c - b; the gradient of a
    vector's angle is ``perp(v)/|v|**2``."""
    e1, e2 = b - a, c - b
    l1 = (e1 * e1).sum(-1)[:, None]
    l2 = (e2 * e2).sum(-1)[:, None]
    l1 = np.where(l1 == 0.0, 1.0, l1)
    l2 = np.where(l2 == 0.0, 1.0, l2)
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    dot = (e1 * e2).sum(-1)
    theta = np.arctan2(cross, dot)
    g1 = _perp(e1) / l1  # d angle(e1) / db  (= -d/da)
    g2 = _perp(e2) / l2  # d angle(e2) / dc  (= -d/db)
    grads = np.stack([g1, -g2 - g1, g2], axis=1)  # d theta / d(a, b, c)
    return theta, grads


def curvature_energy_grad_hess(X, win: CurvatureWindows):
    """Total energy, gradient ``(N, 2)`` and per-window Gauss-Newton blocks.

    ``E = sum_w  weight * (theta1 - theta2)**2`` over each 4-node window, with
    theta1 = turn(p0,p1,p2), theta2 = turn(p1,p2,p3). Returns
    ``(E, grad (N, 2), gn (W, 4, 2, 2))`` — ``gn`` is the per-node
    Gauss-Newton Hessian block ``2 weight (dr/dp)(dr/dp)^T`` (PSD), the same
    the C++ ``curvature_kernel`` accumulates."""
    N = int(np.asarray(X).shape[0])
    if len(win) == 0:
        return 0.0, np.zeros((N, 2)), np.zeros((0, 4, 2, 2))
    X = np.asarray(X, dtype=float)
    p0, p1, p2, p3 = (X[win.nodes[:, k]] for k in range(4))
    w = win.weight

    th1, g1 = _turn_grad(p0, p1, p2)  # depends on p0, p1, p2
    th2, g2 = _turn_grad(p1, p2, p3)  # depends on p1, p2, p3
    r = th1 - th2  # (W,)

    # dr/dp_k over the four window slots.
    dr = np.zeros((len(win), 4, 2))
    dr[:, 0] = g1[:, 0]
    dr[:, 1] = g1[:, 1] - g2[:, 0]
    dr[:, 2] = g1[:, 2] - g2[:, 1]
    dr[:, 3] = -g2[:, 2]

    E = float((w * r * r).sum())
    contrib = (2.0 * w * r)[:, None, None] * dr  # (W, 4, 2) = dE/dp_slot
    grad = np.zeros((N, 2))
    for k in range(4):
        np.add.at(grad, win.nodes[:, k], contrib[:, k])
    gn = (2.0 * w)[:, None, None, None] * (dr[:, :, :, None] * dr[:, :, None, :])
    return E, grad, gn
