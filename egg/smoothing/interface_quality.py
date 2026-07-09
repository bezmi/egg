# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""High-order interface quality post-passes (2D).

Geometric passes over the working grid (like :mod:`egg.smoothing.respace`), run
between smoothing chunks, that make block interfaces *behave*:

* **along** — fit one smooth curve (a smoothing spline over the whole seam, a
  significant stencil) through each interface's seam nodes and slide the interior
  seam nodes onto it, keeping their spacing. C2-smooth block boundary.
* **across** — for each seam node, fit the crossing grid line (a significant
  stencil of nodes reaching into *both* blocks) to a smoothing spline and slide
  the near-seam nodes onto it. Curvature-continuous (C2) crossing through the
  seam, via the same safe projection the along pass uses (never inverts the thin
  near-wall cells).
* **stars** — at valence-N singular nodes, gently pull the N incident spokes
  toward the regular configuration (equiangular 360/N°, optionally equal length)
  so the cell fan is symmetric (the "equal pentagon" at a 5-way).

Orthogonality is left to the barrier-gated interface metric wired into the
sweep; these passes only smooth curvature and regularise the stars, so they
compose with it and with the base shape metric (which owns node distribution).
"""

from __future__ import annotations

import numpy as np

from .interface_ortho import _oriented_map

__all__ = ["smooth_interfaces"]


def _min_det(X, maps):
    """Min corner-cell det(A) over every block (positive = no inverted cells)."""
    md = np.inf
    for dm in maps:
        P = X[dm]
        u = P[1:, :-1] - P[:-1, :-1]
        v = P[:-1, 1:] - P[:-1, :-1]
        d = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        if d.size:
            md = min(md, float(d.min()))
    return md


def _oriented(grid, block_names, face):
    bi = block_names.index(face.block_name)
    return _oriented_map(np.asarray(grid.block_dof_maps[bi]), face.axis, face.side)


def _singular_nodes(grid, topology, block_names):
    """{global id -> (valence, [spoke ids])} for every valence-N (N!=4) node."""
    maps = [np.asarray(m) for m in grid.block_dof_maps]
    out: dict[int, tuple] = {}
    for sg in getattr(topology, "singularities", []):
        bi = block_names.index(sg.block_name)
        S = int(maps[bi][tuple(sg.logical_idx)])
        spokes: set[int] = set()
        for dm in maps:
            for i, j in np.argwhere(dm == S):
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < dm.shape[0] and 0 <= jj < dm.shape[1]:
                        spokes.add(int(dm[ii, jj]))
        out[S] = (int(sg.valence), sorted(spokes))
    return out


def _build_lines(grid, topology, block_names, singular):
    """Per interface: seam node sequence + per-seam-node crossing chains.

    Returns (seams, crossings). ``seams`` is a list of seam-node id lists (row 0
    of each interface, one per connection). ``crossings`` is a list of
    ``(P, chainA, chainB)``: the seam node and its outward crossing chains into
    each block (nearest-first, ending at a deep anchor). Singular seam nodes are
    skipped (handled by the star pass)."""
    seams = []
    crossings = []
    for conn in topology.interface_connections:
        mA = _oriented(grid, block_names, conn.face_a)
        mB = _oriented(grid, block_names, conn.face_b)
        seams.append([int(x) for x in mA[0, :]])
        posB = {int(mB[0, j]): j for j in range(mB.shape[1])}
        for jA in range(1, mA.shape[1] - 1):
            P = int(mA[0, jA])
            jB = posB.get(P)
            if jB is None or P in singular:
                continue
            chainA = [int(mA[k, jA]) for k in range(1, mA.shape[0])]
            chainB = [int(mB[k, jB]) for k in range(1, mB.shape[0])]
            crossings.append((P, chainA, chainB))
    return seams, crossings


def _project_spline(X, ids, movable, smoothing):
    """Fit a C2 smoothing spline through X[ids] and slide movable interior nodes
    onto it at their own chord-length parameter (preserves spacing). Endpoints
    are pinned (heavy weight)."""
    from scipy.interpolate import splev, splprep

    ids = list(ids)
    if len(ids) < 4:
        return
    P = X[np.asarray(ids)]
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    if seg.sum() == 0:
        return
    u = np.concatenate([[0.0], np.cumsum(seg)])
    u = u / u[-1]
    w = np.ones(len(ids))
    w[0] = w[-1] = 1.0e3
    try:
        tck, _ = splprep([P[:, 0], P[:, 1]], u=u, w=w, s=len(ids) * smoothing, k=3)
    except Exception:
        return
    Q = np.asarray(splev(u, tck)).T
    for i in range(1, len(ids) - 1):
        if movable[i]:
            X[ids[i]] = Q[i]


def _fit_crossing(X, P, chainA, chainB, movable, stencil, smoothing):
    """C2-smooth the crossing grid line through the seam node P.

    Builds the crossing polyline spanning both blocks (a significant stencil,
    ``stencil`` nodes each side) and slides the near-seam nodes onto a smoothing
    spline through it — the same safe projection the seam (along) pass uses, so
    it never inverts the thin near-wall cells. The seam node P is pinned (it is
    placed by the along pass); the deep ends anchor the fit."""
    a = chainA[:stencil]
    b = chainB[:stencil]
    ids = list(reversed(b)) + [P] + list(a)
    if len(ids) < 4:
        return
    kP = len(b)
    mv = [movable(n) for n in ids]
    mv[kP] = False  # keep the seam node on the seam
    _project_spline(X, ids, mv, smoothing)


def _regularize_stars(X, singular, movable, blend, equal_length):
    """Pull each valence-N node's N spokes toward equiangular (360/N) and equal
    length, blended by ``blend`` in [0, 1] (1 = snap)."""
    for S, (val, spokes) in singular.items():
        if len(spokes) != val or val < 3:
            continue
        s = X[S]
        d = X[np.asarray(spokes)] - s
        L = np.linalg.norm(d, axis=1)
        if np.any(L == 0):
            continue
        ang = np.arctan2(d[:, 1], d[:, 0])
        order = np.argsort(ang)
        step = 2.0 * np.pi / val
        # best-fit equiangular phase (circular mean of residuals)
        res = ang[order] - np.arange(val) * step
        phi0 = float(np.angle(np.mean(np.exp(1j * res))))
        Lm = float(np.mean(L))
        for k, idx in enumerate(order):
            n = spokes[idx]
            if not movable(n):
                continue
            phi = phi0 + k * step
            Ln = Lm if equal_length else L[idx]
            target = s + Ln * np.array([np.cos(phi), np.sin(phi)])
            X[n] = (1.0 - blend) * X[n] + blend * target


def smooth_interfaces(
    grid,
    topology=None,
    *,
    across_stencil: int = 4,
    seam_smoothing: float = 1.0e-5,
    across_smoothing: float = 1.0e-6,
    star_blend: float = 0.4,
    star_equal_length: bool = False,
):
    """One interface-quality pass over ``grid.global_nodes`` (in place).

    Delivers C2 continuity along and across every block interface plus regular
    singular fans. Orthogonality is left to the barrier-gated interface
    metric (:func:`egg.smoothing.interface_ortho.interface_ortho_samples`,
    wired into the sweep), which cannot invert cells; these geometric passes
    only smooth curvature and regularise the stars, so they compose with it.

    Parameters
    ----------
    across_stencil : int
        Crossing nodes each side of the seam the across fit spans (the
        significant stencil).
    seam_smoothing, across_smoothing : float
        Smoothing-spline factors for the along / across fits (larger = smoother).
    star_blend : float
        Per-pass fraction the singular spokes move toward the regular star.
    star_equal_length : bool
        Also equalise spoke lengths (overrides clustering at the node); off by
        default since it can distort a clustered fan.
    """
    topo = topology if topology is not None else grid.topology
    if topo.d != 2:
        raise NotImplementedError("smooth_interfaces is 2D-only")
    block_names = list(topo.block_specs.keys())
    X = np.asarray(grid.global_nodes)
    free = np.asarray(grid.free_mask)
    constrained = set(grid.dof_constraints.keys())
    singular = _singular_nodes(grid, topo, block_names)
    sing_ids = set(singular.keys())

    def movable(n):
        return bool(free[n]) and n not in constrained and n not in sing_ids

    maps = [np.asarray(m) for m in grid.block_dof_maps]
    md0 = _min_det(X, maps)
    X0 = X.copy()

    seams, crossings = _build_lines(grid, topo, block_names, sing_ids)

    # 1. along: C2 smoothing spline through each seam.
    for seq in seams:
        mv = [movable(n) for n in seq]
        _project_spline(X, seq, mv, seam_smoothing)

    # 2. across: C2 smoothing spline through each crossing (both blocks).
    for P, chainA, chainB in crossings:
        _fit_crossing(X, P, chainA, chainB, movable, across_stencil, across_smoothing)

    # 3. stars: regularise the valence-N fans.
    _regularize_stars(X, singular, movable, star_blend, star_equal_length)

    # Safeguard: these geometric passes are not barrier-gated, so back the whole
    # correction off toward the (positive, sweep-gated) pre-pass state until it no
    # longer inverts a cell or degrades the min-det floor. Keeps the pass safe
    # when stacked with the orthogonality metric.
    floor = 0.5 * md0 if md0 > 0 else 0.0
    Xn = X.copy()
    alpha = 1.0
    while alpha > 1.0e-3 and _min_det(X0 + alpha * (Xn - X0), maps) <= floor:
        alpha *= 0.5
    if alpha < 1.0:
        X[:] = X0 + (alpha if alpha > 1.0e-3 else 0.0) * (Xn - X0)

    # propagate to the per-block node arrays
    for bi, block in enumerate(grid.blocks):
        block.nodes[...] = grid.global_nodes[np.asarray(grid.block_dof_maps[bi])]
    return grid
