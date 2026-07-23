# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""High-order curvature + curve-orthogonality metric (2D NumPy prototype).

Reference implementation of the metric the user wants in the core: for *every*
node, take the two grid lines crossing it, and shape the energy so those lines
are (a) curvature-continuous (C2, no kinks — including across block interfaces)
and (b) orthogonal to each other. The two terms are added to the base TMOP
shape energy and the whole thing is minimised under the same shape barrier
(det A > 0), so it composes and never inverts a cell — the properties a
geometric post-pass could not guarantee.

The grid lines are assembled *globally* by walking a straightest-continuation
neighbour graph, so a line crosses a block interface transparently (the seam
node is shared) and only stops at a singular node (valence != 4). That makes
the curvature term span the seam with a significant stencil, which is exactly
the across-interface C2 the post-pass needed extra ghost layers for — here the
whole grid is resident, so it is free.

Energies (per window, all scale-free):

* **curvature** — over four consecutive nodes of a line, the squared change in
  turning angle ``(phi_i - phi_{i+1})**2``. Zero for a straight line *and* for a
  constant-curvature arc (so it does not fight smooth geometry or clustering),
  large at a kink.
* **orthogonality** — at each node, the squared cosine between the two crossing
  lines' tangents ``(t0 . t1)**2 / (|t0|**2 |t1|**2)``. Drives 90-degree
  crossings; wins at singularities where it matters most.

Gradients are taken by vectorised local central differences over the small,
fixed per-window stencil (this is the correctness gate for the closed-form core
kernel, not the fast path). ``highorder_smooth`` runs a backtracking
steepest-descent on ``E_shape + w_curv E_curv + w_orth E_orth`` with the
codebase's accept rule (finite, non-increasing, det A > 0).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .batch import _grad_T, _mu_batch, assemble_A

__all__ = [
    "grid_adjacency",
    "grid_lines",
    "line_diagnostics",
    "highorder_smooth",
]


def grid_adjacency(grid):
    """Global neighbour graph + straightest-continuation partner map.

    Returns ``(nbr, partner)``:

    * ``nbr[node]`` — set of global-id neighbours (union over every block, so
      an interface seam node lists neighbours in *both* blocks).
    * ``partner[node]`` — for a valence-4 node, ``{neighbour: opposite}`` pairing
      each incident edge with its straightest continuation (most anti-parallel
      direction). Absent for valence != 4 nodes (line endpoints / singularities).
    """
    X = np.asarray(grid.global_nodes, dtype=float)
    nbr: dict[int, set] = defaultdict(set)
    for dm in grid.block_dof_maps:
        m = np.asarray(dm)
        for a, b in ((m[:-1, :], m[1:, :]), (m[:, :-1], m[:, 1:])):
            for u, v in zip(a.reshape(-1), b.reshape(-1)):
                nbr[int(u)].add(int(v))
                nbr[int(v)].add(int(u))

    partner: dict[int, dict] = {}
    for node, ns in nbr.items():
        if len(ns) != 4:
            continue
        idx = list(ns)
        d = X[idx] - X[node]
        L = np.linalg.norm(d, axis=1)
        if np.any(L == 0):
            continue
        u = d / L[:, None]
        # three ways to split four dirs into two pairs; pick the one whose
        # paired dirs are most anti-parallel (sum of within-pair dots minimal).
        best, best_score = (0, 1, 2, 3), np.inf
        for p, q, r, t in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):
            score = float(u[p] @ u[q] + u[r] @ u[t])
            if score < best_score:
                best_score, best = score, (p, q, r, t)
        p, q, r, t = best
        partner[node] = {
            idx[p]: idx[q],
            idx[q]: idx[p],
            idx[r]: idx[t],
            idx[t]: idx[r],
        }
    return dict(nbr), partner


def grid_lines(grid, nbr=None, partner=None):
    """Maximal grid-line polylines (lists of global ids), interface-stitched.

    A line follows the straightest continuation through valence-4 nodes and
    terminates at a singular / boundary node (valence != 4); interface seam
    nodes are ordinary valence-4 interior nodes, so a line crosses the seam
    transparently. Closed rings (O-grid lines) are emitted once.
    """
    if nbr is None or partner is None:
        nbr, partner = grid_adjacency(grid)
    used: set = set()
    lines: list[list[int]] = []

    def walk(prev, cur):
        line = [prev, cur]
        used.add((prev, cur))
        while True:
            pm = partner.get(cur)
            if not pm:
                break
            nxt = pm.get(prev)
            if nxt is None or (cur, nxt) in used:
                break
            used.add((cur, nxt))
            line.append(nxt)
            prev, cur = cur, nxt
        return line

    def consume_reverse(line):
        for a, b in zip(line[1:], line[:-1]):
            used.add((a, b))

    endpoints = [n for n in nbr if len(partner.get(n, {})) != 4]
    for e in endpoints:
        for nb in nbr[e]:
            if (e, nb) in used:
                continue
            line = walk(e, nb)
            consume_reverse(line)
            lines.append(line)
    # remaining unused edges belong to closed rings
    for u in nbr:
        for v in nbr[u]:
            if (u, v) in used:
                continue
            line = walk(u, v)
            consume_reverse(line)
            lines.append(line)
    return lines


def _curv_cols(lines):
    """Stack all four-consecutive-node windows of every line: 4 int arrays."""
    w = [win for line in lines for win in zip(line, line[1:], line[2:], line[3:])]
    if not w:
        z = np.zeros(0, np.int64)
        return [z, z, z, z]
    a = np.asarray(w, dtype=np.int64)  # (W, 4)
    return [a[:, 0], a[:, 1], a[:, 2], a[:, 3]]


def _orth_cols(nbr, partner):
    """Per valence-4 node, the two crossing pairs (a,b) (c,d): 4 int arrays."""
    rows = []
    for node, pm in partner.items():
        if len(pm) != 4:
            continue
        seen, pairs = set(), []
        for a, b in pm.items():
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((a, b))
        if len(pairs) != 2:
            continue
        (a, b), (c, d) = pairs
        rows.append((a, b, c, d))
    if not rows:
        z = np.zeros(0, np.int64)
        return [z, z, z, z]
    r = np.asarray(rows, dtype=np.int64)
    return [r[:, 0], r[:, 1], r[:, 2], r[:, 3]]


def _turn(a, b, c):
    """Signed turning angle at b along a->b->c (batched, radians)."""
    e1, e2 = b - a, c - b
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    dot = e1[:, 0] * e2[:, 0] + e1[:, 1] * e2[:, 1]
    return np.arctan2(cross, dot)


def _f_curv(P):
    """Squared change in turning angle over a 4-node window (batched)."""
    p0, p1, p2, p3 = P
    return (_turn(p0, p1, p2) - _turn(p1, p2, p3)) ** 2


def _f_orth(P):
    """Squared cosine between the two crossing tangents (batched)."""
    pa, pb, pc, pd = P
    t1, t2 = pb - pa, pd - pc
    dot = (t1 * t2).sum(1)
    n1, n2 = (t1 * t1).sum(1), (t2 * t2).sum(1)
    return dot * dot / (n1 * n2 + 1e-30)


def _term_energy_grad(X, cols, efunc, eps, w=None):
    """Energy sum and gradient (N, 2) of a per-window term by local central FD.

    Each window depends on the few nodes in ``cols``; perturbing those handful
    of coordinates (vectorised over all windows) gives the exact gradient of
    this scale-free term without a hand-derived closed form (the prototype's
    correctness reference for the core kernel). ``w`` is an optional per-window
    weight (the interface / singularity boost) applied to energy and gradient
    alike."""
    pts = [X[c] for c in cols]
    e0 = efunc(pts)
    if w is not None:
        e0 = e0 * w
    E = float(e0.sum())
    grad = np.zeros_like(X)
    for s, c in enumerate(cols):
        for k in range(X.shape[1]):
            pp = [p.copy() for p in pts]
            pm = [p.copy() for p in pts]
            pp[s][:, k] += eps
            pm[s][:, k] -= eps
            d = (efunc(pp) - efunc(pm)) / (2.0 * eps)
            if w is not None:
                d = d * w
            np.add.at(grad, (c, k), d)
    return E, grad


def _shape_energy_grad(X, S, metric):
    """Base TMOP energy, gradient (N, 2), and min det(A) over all samples."""
    gc, gn0, gn1, s0, s1, W = S
    A = assemble_A(X, gc, gn0, gn1, s0, s1)
    detA = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    T = np.einsum("pij,pjk->pik", A, W)
    E = float(_mu_batch(T, metric).sum())
    dmu = _grad_T(T, metric)
    dA = np.einsum("pij,pkj->pik", dmu, W)  # dmu/dA = dmu/dT W^T
    col0, col1 = dA[:, :, 0], dA[:, :, 1]
    g = np.zeros_like(X)
    np.add.at(g, gc, -(s0[:, None] * col0 + s1[:, None] * col1))
    np.add.at(g, gn0, s0[:, None] * col0)
    np.add.at(g, gn1, s1[:, None] * col1)
    return E, g, float(detA.min()) if detA.size else np.inf


def line_diagnostics(grid, nbr=None, partner=None):
    """Waviness / orthogonality summary of the current grid.

    Returns a dict: ``turn_rms``/``turn_max`` (mean, max |change in turning
    angle| over line windows — the waviness measure) and ``obliq_rms`` (RMS
    |cos| between crossing grid lines — 0 is everywhere orthogonal)."""
    if nbr is None or partner is None:
        nbr, partner = grid_adjacency(grid)
    lines = grid_lines(grid, nbr, partner)
    X = np.asarray(grid.global_nodes, dtype=float)
    cc = _curv_cols(lines)
    oc = _orth_cols(nbr, partner)
    turn = np.sqrt(_f_curv([X[c] for c in cc])) if cc[0].size else np.zeros(0)
    obl = np.sqrt(_f_orth([X[c] for c in oc])) if oc[0].size else np.zeros(0)
    return {
        "turn_rms": float(np.sqrt((turn**2).mean())) if turn.size else 0.0,
        "turn_max": float(turn.max()) if turn.size else 0.0,
        "obliq_rms": float(np.sqrt((obl**2).mean())) if obl.size else 0.0,
        "n_lines": len(lines),
    }


def highorder_smooth(
    grid,
    target_fn,
    *,
    metric: str = "shape_size",
    w_curv: float = 0.05,
    w_orth: float = 0.05,
    iters: int = 60,
    step_frac: float = 0.4,
    momentum: float = 0.85,
    iface_boost: float = 1.0,
    eps: float = 1e-6,
    verbose: bool = False,
):
    """Minimise ``E_shape + w_curv E_curv + w_orth E_orth`` in place.

    Barrier-gated backtracking steepest descent (accept iff the trial energy is
    finite, non-increasing, and keeps det A > 0), over the movable DOFs (free,
    unconstrained). ``target_fn`` supplies the base shape target W (wrap it with
    :class:`~egg.smoothing.targets.DeclusterSingularities` to isotropise the
    singular fans). Returns ``grid``.
    """
    if grid.topology.d != 2:
        raise NotImplementedError("highorder_smooth is 2D-only (prototype)")
    from .solver import build_sweep_context

    ctx = build_sweep_context(grid, target_fn)
    es = ctx.energy_stencil
    S = (es["gc"], es["gn0"], es["gn1"], es["s0"], es["s1"], es["W_inv"])

    X = np.asarray(grid.global_nodes)
    movable = np.asarray(grid.free_mask).copy()
    for dof in grid.dof_constraints:
        movable[int(dof)] = False

    nbr, partner = grid_adjacency(grid)
    lines = grid_lines(grid, nbr, partner)
    cc = _curv_cols(lines)
    oc = _orth_cols(nbr, partner)

    # Interface boost: seam nodes are shared by more than one block map; a
    # curvature window touching one straddles a block interface, exactly where
    # shape smoothing leaves the kink. Weighting those windows up de-kinks the
    # seams (the user's target) without distorting block interiors — the same
    # per-sample-weight lever the core already carries for shape samples.
    cc_w = None
    if iface_boost != 1.0 and cc[0].size:
        seen: dict[int, int] = {}
        for m in grid.block_dof_maps:
            for nid in np.unique(np.asarray(m)):
                seen[int(nid)] = seen.get(int(nid), 0) + 1
        seam = np.array([n for n, c in seen.items() if c > 1], dtype=np.int64)
        is_seam = np.zeros(int(np.asarray(grid.global_nodes).shape[0]), bool)
        is_seam[seam] = True
        touches = is_seam[cc[0]] | is_seam[cc[1]] | is_seam[cc[2]] | is_seam[cc[3]]
        cc_w = np.where(touches, iface_boost, 1.0)

    # characteristic edge length sets the step scale
    edges = [(u, v) for u in nbr for v in nbr[u]]
    uu = np.array([e[0] for e in edges])
    vv = np.array([e[1] for e in edges])
    charlen = float(np.median(np.linalg.norm(X[vv] - X[uu], axis=1)))

    def total(Xt):
        Es, gs, md = _shape_energy_grad(Xt, S, metric)
        E = Es
        g = gs
        if w_curv and cc[0].size:
            Ec, gcv = _term_energy_grad(Xt, cc, _f_curv, eps, w=cc_w)
            E += w_curv * Ec
            g = g + w_curv * gcv
        if w_orth and oc[0].size:
            Eo, go = _term_energy_grad(Xt, oc, _f_orth, eps)
            E += w_orth * Eo
            g = g + w_orth * go
        g[~movable] = 0.0
        return E, g, md

    E, g, md = total(X)
    move_prev = np.zeros_like(X)
    if verbose:
        print(f"[hi] it 0  E={E:.6e} mindet={md:.3e}")
    for it in range(iters):
        mx = float(np.linalg.norm(g, axis=1).max())
        if mx < 1e-12:
            break
        steep = -g / mx * (step_frac * charlen)  # max node move = step_frac*h
        # heavy-ball: try the momentum-augmented direction first, fall back to
        # plain steepest descent (resetting the velocity) if it is rejected.
        accepted = False
        # Seed the trial state so it is always bound; the line search overwrites
        # it before any accepted step is used.
        Xt = X
        Et, gt, mdt = E, g, md
        for direction, keep in ((steep + momentum * move_prev, True), (steep, False)):
            alpha = 1.0
            while alpha > 1e-4:
                Xt = X + alpha * direction
                Et, gt, mdt = total(Xt)
                if np.isfinite(Et) and Et <= E + 1e-12 * abs(E) + 1e-14 and mdt > 0.0:
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                move_prev = alpha * direction if keep else np.zeros_like(X)
                break
        if not accepted:
            break
        X[:] = Xt
        E, g, md = Et, gt, mdt
        if verbose and (it + 1) % 20 == 0:
            print(f"[hi] it {it + 1}  E={E:.6e} mindet={md:.3e} alpha={alpha:.2e}")

    for bi in range(len(grid.blocks)):
        grid.blocks[bi].nodes[...] = grid.global_nodes[
            np.asarray(grid.block_dof_maps[bi])
        ]
    return grid
