# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Wall constraints for the control-net smoother: sliding + orthogonality.

NumPy reference (single 2D block) for boundary walls on CAD entities. The
grid-line direction leaving a wall is governed directly by the first control
leg ``L_j = P1_j - P0_j`` (clamped basis: ``dX/dn ~ sum_j N_j(t) L_j``), so
wall behaviour becomes a small set of geometric conditions on control legs
instead of oriented per-cell metric targets on fine nodes:

- **sliding**: a wall's boundary controls ``P0_j`` live ON the entity and move
  tangentially — one scalar DOF each against a tangent frame frozen per outer
  iteration, reprojected onto the entity after each accepted step. Corner
  controls (codim >= 2) stay fixed.
- **orthogonality, per wall** (``ortho``):
  - ``"penalty"``: energy ``(w_j/2) |t_j . L_j|^2 / l_j^2`` per leg (frame and
    normalisation ``l_j`` frozen per outer iteration) — quadratic rows folded
    into the reduced Gauss-Newton system AND into the line-search energy, so
    the accept rule gates the composed objective.
  - ``"hard"``: the first-interior control is driven, ``P1_j = P0_j + h_j n_j``
    with one scalar DOF ``h_j > 0`` (floored) — eliminated into the reduction
    matrix ``R``, no penalty, no KKT.
  - ``"off"``: sliding only.
- **exactness**: the Boolean-sum offset ``b`` keeps the fine wall nodes exactly
  on the entity — after each accepted step the spline boundary is projected
  onto the entity and ``b`` re-extended (frozen within an outer iteration).

Everything reduces to a per-iteration frozen frame: ``dC = R dq`` with ``R`` a
sparse gather (<= 2 terms per row), so the reduced system is
``A_q = R^T A R + penalty`` — the same shape the device path implements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from egg.geometry.control_net import tensor_map, tfi_operator
from egg.smoothing.batch import energy_and_mindet
from egg.smoothing.control_ref import (
    ENERGY_TOL,
    ControlRefContext,
    boundary_ctrl_mask,
    dense_M,
    fit_control_net,
    reduced_system,
)
from egg.smoothing.solver import build_sweep_context

__all__ = [
    "WallSpec",
    "ControlWallRef",
    "build_control_wall_ref",
    "run_control_wall_ref",
    "wall_angle_deviation",
]


@dataclass
class WallSpec:
    """One block face lying on a CAD entity, with its orthogonality dial.

    ``axis``/``side`` select the face (side 0 = low, 1 = high). ``ortho`` is
    the per-face dial: ``"hard"``, ``"penalty"``, or ``"off"``. ``weight``
    (penalty mode) is a scalar, or one value per non-corner tangential index
    (a taper along the face, low-index first) — zero entries disable the
    penalty for that leg, which is the capsule-style relaxation.
    """

    axis: int
    side: int
    entity: object
    ortho: str = "penalty"
    weight: float | np.ndarray = 1.0


@dataclass
class _WallMeta:
    """Per-wall control bookkeeping (flat control-lattice indices)."""

    spec: WallSpec
    p0: np.ndarray  # (m,) sliding boundary controls (corners excluded)
    p1: np.ndarray  # (m,) matching first-interior controls
    weight: np.ndarray  # (m,) per-leg penalty weight (penalty mode)
    node_sl: tuple  # index tuple selecting the wall's fine-node slab


@dataclass
class ControlWallRef:
    """The wall-augmented reference context: a plain :class:`ControlRefContext`
    (its free set includes the sliding wall controls) plus wall metadata and
    the per-iteration frozen frame."""

    cref: ControlRefContext
    walls: list[_WallMeta]
    X0_boundary: np.ndarray  # (node_shape + (d,)) initial exact boundary
    h_min: float = 0.0
    # Frozen frame (rebuilt each outer iteration by _build_frame):
    R: np.ndarray = field(default=None, repr=False)
    pen: list = field(default_factory=list, repr=False)  # (i0, i1, t_hat, w_eff)


def _face_slab(axis: int, side: int, shape: tuple[int, ...]) -> tuple:
    sl: list = [slice(None)] * len(shape)
    sl[axis] = 0 if side == 0 else shape[axis] - 1
    return tuple(sl)


def _wall_ctrl_rows(axis, side, ctrl_shape):
    """Flat control ids of a wall's boundary slab P0 and first-interior slab
    P1 at the non-corner tangential multi-indices (codim >= 2 lattice points
    excluded), row-major over the tangential axes."""
    from itertools import product

    d = len(ctrl_shape)
    r0 = 0 if side == 0 else ctrl_shape[axis] - 1
    r1 = 1 if side == 0 else ctrl_shape[axis] - 2
    tangential = [k for k in range(d) if k != axis]
    p0, p1 = [], []
    for tidx in product(*(range(1, ctrl_shape[k] - 1) for k in tangential)):
        idx0 = [0] * d
        idx0[axis] = r0
        for k, j in zip(tangential, tidx):
            idx0[k] = j
        idx1 = list(idx0)
        idx1[axis] = r1
        p0.append(np.ravel_multi_index(idx0, ctrl_shape))
        p1.append(np.ravel_multi_index(idx1, ctrl_shape))
    return np.array(p0, dtype=int), np.array(p1, dtype=int)


def build_control_wall_ref(
    grid,
    target_fn,
    ctrl_shape,
    walls: list[WallSpec],
    *,
    degree: int = 3,
    metric: str = "shape_2d",
) -> ControlWallRef:
    """Build the wall-augmented reference: single 2D block, walls sliding.

    The free-control set is the block interior plus each wall's non-corner
    boundary controls; wall boundary controls are projected onto their entity
    (they are shape handles near the wall — fine-node exactness on the CAD is
    owned by ``b``, recomputed against the projected spline boundary).
    """
    d = grid.topology.d
    if d != 2:
        raise NotImplementedError("the NumPy wall reference is 2D")
    if len(grid.blocks) != 1:
        raise NotImplementedError("the NumPy wall reference is single-block")
    block = grid.blocks[0]
    node_shape = tuple(block.logical_shape)
    ctrl_shape = tuple(int(s) for s in ctrl_shape)

    ctx = build_sweep_context(grid, target_fn)
    cmap = tensor_map(node_shape, ctrl_shape, degree=degree)
    X0 = np.asarray(block.nodes, dtype=float)
    C = fit_control_net(cmap, X0)

    free_ctrl = ~boundary_ctrl_mask(ctrl_shape)
    metas: list[_WallMeta] = []
    seen = set()
    for spec in walls:
        if spec.ortho not in ("hard", "penalty", "off"):
            raise ValueError(f"unknown ortho mode {spec.ortho!r}")
        key = (spec.axis, spec.side)
        if key in seen:
            raise ValueError("duplicate wall face")
        seen.add(key)
        p0, p1 = _wall_ctrl_rows(spec.axis, spec.side, ctrl_shape)
        w = np.broadcast_to(np.asarray(spec.weight, dtype=float), p0.shape).copy()
        metas.append(
            _WallMeta(
                spec=spec,
                p0=p0,
                p1=p1,
                weight=w,
                node_sl=_face_slab(spec.axis, spec.side, node_shape),
            )
        )
        # The wall's boundary controls become sliding DOFs, on the entity.
        free_flat = free_ctrl.reshape(-1)
        free_flat[p0] = True
        Cf = C.reshape(-1, d)
        Cf[p0] = spec.entity.project_many(Cf[p0])

    # Dense M in global-node rows, free columns only (mirrors control_ref).
    Md = dense_M(cmap)
    dof_map = grid.block_dof_maps[0].reshape(-1)
    Mg = np.zeros((grid.global_node_count, Md.shape[1]))
    Mg[dof_map] = Md
    Mf = Mg[:, free_ctrl.reshape(-1)]

    st = ctx.energy_stencil
    from egg.smoothing.batch import make_chain_J

    J = make_chain_J(st["s0"].astype(float), st["s1"].astype(float), st["W_inv"])
    cref = ControlRefContext(
        grid=grid,
        ctx=ctx,
        cmap=cmap,
        block_idx=0,
        C=C,
        free_ctrl=free_ctrl,
        Mf=Mf,
        b_field=np.zeros(node_shape + (d,)),
        metric=metric,
        _J=J,
    )
    # Leg scale for the hard-mode h floor: the shortest initial wall leg.
    Cf = C.reshape(-1, d)
    lmin = min(
        (
            float(np.linalg.norm(Cf[m.p1[j]] - Cf[m.p0[j]]))
            for m in metas
            for j in range(len(m.p0))
        ),
        default=1.0,
    )
    wref = ControlWallRef(
        cref=cref, walls=metas, X0_boundary=X0.copy(), h_min=1e-3 * lmin
    )
    _snap_hard_legs(wref)
    _recompute_b(wref)
    return wref


def _snap_hard_legs(wref: ControlWallRef) -> None:
    """Enforce the hard-mode state constraint P1 = P0 + h n (h floored) against
    the CURRENT frame — run at setup and after each reprojection.

    The entity's ``normal`` carries no inward/outward convention, so each leg
    orients it by its own current direction (the first-interior control is
    inside the domain by construction, and the h floor keeps it there — the
    orientation cannot flip mid-run)."""
    Cf = wref.cref.C.reshape(-1, 2)
    for m in wref.walls:
        if m.spec.ortho != "hard":
            continue
        for j in range(len(m.p0)):
            p0 = Cf[m.p0[j]]
            n = np.asarray(m.spec.entity.normal(p0), dtype=float)
            leg = Cf[m.p1[j]] - p0
            if float(n @ leg) < 0.0:
                n = -n
            h = float(n @ leg)
            Cf[m.p1[j]] = p0 + max(h, wref.h_min) * n


def _recompute_b(wref: ControlWallRef) -> None:
    """Re-extend the Boolean-sum offset: wall boundary nodes exact on their
    entity (projection of the spline boundary), other boundaries at their
    initial positions."""
    cref = wref.cref
    cmap = cref.cmap
    node_shape = cmap.node_shape
    Xs = cmap.prolong(cref.C)
    exact = wref.X0_boundary.copy()
    for m in wref.walls:
        exact[m.node_sl] = m.spec.entity.project_many(Xs[m.node_sl])
    delta = np.zeros(node_shape + (2,))
    bmask = boundary_ctrl_mask(node_shape)
    delta[bmask] = (exact - Xs)[bmask]
    T = tfi_operator(node_shape)
    n_total = int(np.prod(node_shape))
    cref.b_field = (T @ delta.reshape(n_total, 2)).reshape(node_shape + (2,))


def _build_frame(wref: ControlWallRef) -> None:
    """Freeze the per-iteration frame: reproject sliding controls, re-snap
    hard legs, rebuild ``R`` (reduced DOFs -> free-control displacements) and
    the penalty rows, and re-extend ``b``."""
    cref = wref.cref
    Cf = cref.C.reshape(-1, 2)
    free_idx = np.flatnonzero(cref.free_ctrl.reshape(-1))
    col_of = {int(g): i for i, g in enumerate(free_idx)}  # ctrl id -> free row
    F = len(free_idx)

    # Reproject sliding controls onto their entities, then restore hard legs.
    for m in wref.walls:
        Cf[m.p0] = m.spec.entity.project_many(Cf[m.p0])
    _snap_hard_legs(wref)
    _recompute_b(wref)

    driven: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}  # p1 -> (p0, t, n)
    frames: dict[int, np.ndarray] = {}  # sliding p0 -> t
    for m in wref.walls:
        for j in range(len(m.p0)):
            p0 = int(m.p0[j])
            t = np.asarray(m.spec.entity.tangent_space(Cf[p0]), dtype=float).reshape(2)
            t /= np.linalg.norm(t)
            frames[p0] = t
            if m.spec.ortho == "hard":
                # Post-snap the leg IS h·n with the inward orientation, so the
                # frame direction comes from the leg itself.
                leg = Cf[int(m.p1[j])] - Cf[p0]
                n = leg / np.linalg.norm(leg)
                driven[int(m.p1[j])] = (p0, t, n)

    # Column layout: interior controls get 2 columns, sliding P0 one (t),
    # each hard leg one (h along n). Driven P1 rows are filled from their
    # P0 column and their own h column — <= 2 terms per row.
    ncols = 0
    col_start: dict[int, int] = {}
    for g in free_idx:
        g = int(g)
        if g in driven:
            continue
        col_start[g] = ncols
        ncols += 1 if g in frames else 2
    h_col: dict[int, int] = {}
    for p1 in driven:
        h_col[p1] = ncols
        ncols += 1

    R = np.zeros((F * 2, ncols))
    for g in free_idx:
        g = int(g)
        r = 2 * col_of[g]
        if g in driven:
            p0, t, n = driven[g]
            R[r : r + 2, col_start[p0]] = t  # follows its foot
            R[r : r + 2, h_col[g]] = n
        elif g in frames:
            R[r : r + 2, col_start[g]] = frames[g]
        else:
            R[r, col_start[g]] = 1.0
            R[r + 1, col_start[g] + 1] = 1.0

    pen = []
    for m in wref.walls:
        if m.spec.ortho != "penalty":
            continue
        for j in range(len(m.p0)):
            if m.weight[j] == 0.0:
                continue
            p0, p1 = int(m.p0[j]), int(m.p1[j])
            t = frames[p0]
            leg = Cf[p1] - Cf[p0]
            l2 = float(leg @ leg)
            if l2 <= 0.0:
                continue
            pen.append((col_of[p0], col_of[p1], t, float(m.weight[j]) / l2))

    wref.R = R
    wref.pen = pen


def _db_map(wref: ControlWallRef) -> np.ndarray:
    """Effective component map ``d vec(X) / d vec(C_free)`` with the
    frozen-frame Boolean-sum linearization ``d b / d C`` composed in.

    Within a frame the wall-face nodes sit at ``proj(spline)``; a control
    motion moves the spline face by ``dS = (M dC)|face`` and the projected
    foot by ``t̂ t̂ᵀ dS``, so the face picks up ``db = -n̂ n̂ᵀ dS`` and the
    Boolean-sum extension carries that correction into the interior. The
    plain map sees a tangential wall-control slide as free fine-node motion;
    with ``db/dC`` the step model knows wall nodes only move along the CAD —
    the mismatch between those two is what feeds the frame-to-frame energy
    ratchet.

    Returns ``(M_global_nodes, d, F, d)``: the coordinate-mixing map consumed
    by :func:`egg.smoothing.control_ref.reduced_system` via ``cref.Meff``.
    """
    cref = wref.cref
    cmap = cref.cmap
    node_shape = cmap.node_shape
    d = cmap.d
    n_total = int(np.prod(node_shape))
    F = cref.n_free
    dof = np.asarray(cref.grid.block_dof_maps[cref.block_idx]).reshape(-1)
    Mb = cref.Mf[dof]  # spline map in block-node rows

    # Frozen -n̂ n̂ᵀ per wall-face node (feet of the current spline face).
    Xs = cmap.prolong(cref.C)
    Nproj = np.zeros((n_total, d, d))
    mask = np.zeros(n_total, dtype=bool)
    flat_ids = np.arange(n_total).reshape(node_shape)
    for m in wref.walls:
        ids = flat_ids[m.node_sl].reshape(-1)
        feet = m.spec.entity.project_many(Xs[m.node_sl].reshape(-1, d))
        for i, fi in enumerate(ids):
            t = np.asarray(m.spec.entity.tangent_space(feet[i]), float).reshape(d)
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            Nproj[fi] = -np.outer(n, n)
            mask[fi] = True
    idx = np.flatnonzero(mask)

    Meff = np.zeros((cref.Mf.shape[0], d, F, d))
    for x in range(d):
        Meff[:, x, :, x] = cref.Mf
    # L·M: every node's TFI response to the wall-face normal corrections
    # (T's face rows are identity, so the face nodes themselves get -n̂n̂ᵀ dS).
    T = tfi_operator(node_shape)
    contrib = np.einsum(
        "ij,jxy,jF->ixFy", T[:, idx], Nproj[idx], Mb[idx], optimize=True
    )
    Meff[dof] += contrib
    return Meff


def _pen_energy(wref: ControlWallRef, Cfree: np.ndarray) -> float:
    """Wall-orthogonality penalty energy at a free-control state (F, 2)."""
    e = 0.0
    for i0, i1, t, w_eff in wref.pen:
        leg = Cfree[i1] - Cfree[i0]
        e += 0.5 * w_eff * float(t @ leg) ** 2
    return e


def _pen_grad_hess(wref: ControlWallRef) -> tuple[np.ndarray, np.ndarray]:
    """Penalty gradient and GN Hessian in the FULL free-control space (F*2)."""
    cref = wref.cref
    F = cref.n_free
    free_idx = np.flatnonzero(cref.free_ctrl.reshape(-1))
    Cfree = cref.C.reshape(-1, 2)[free_idx]
    G = np.zeros(F * 2)
    A = np.zeros((F * 2, F * 2))
    for i0, i1, t, w_eff in wref.pen:
        leg = Cfree[i1] - Cfree[i0]
        v = float(t @ leg)
        row = np.zeros(F * 2)
        row[2 * i1 : 2 * i1 + 2] = t
        row[2 * i0 : 2 * i0 + 2] = -t
        G += w_eff * v * row
        A += w_eff * np.outer(row, row)
    return G, A


def wall_angle_deviation(
    wref: ControlWallRef, wall: int = 0, skip: int = 0
) -> np.ndarray:
    """Per-node angle (degrees) between the wall-normal derivative of the
    SPLINE map and the entity normal, along one wall.

    This is the quantity the wall machinery actually controls (and what an
    algebraic regrid inherits): ``dX/dn(t) ~ sum_j N_j(t) L_j`` analytically at
    the wall. The first-cell chord seen on the fine grid additionally carries
    cell-height truncation and the Boolean-sum gradient, which no leg
    condition can (or should) cancel. ``skip`` drops nodes at each end —
    corner controls are fixed (codim >= 2 exempt), and their basis support
    owns the angle within ~2 control intervals of each corner.
    """
    m = wref.walls[wall]
    axis, side = m.spec.axis, m.spec.side
    cmap = wref.cref.cmap
    C = wref.cref.C
    ta = 1 - axis
    row = 0 if side == 0 else cmap.node_shape[axis] - 1
    Bt = cmap.basis[ta]  # (n_t, nc_t) tangential value basis
    Bnd = cmap.basis_d1[axis][row]  # (nc_a,) wall-axis derivative row
    Bn = cmap.basis[axis][row]  # (nc_a,) wall-axis value row
    if axis == 0:
        dXdn = np.einsum("j,ik,jkc->ic", Bnd, Bt, C)
        Xw = np.einsum("j,ik,jkc->ic", Bn, Bt, C)
    else:
        dXdn = np.einsum("ij,k,jkc->ic", Bt, Bnd, C)
        Xw = np.einsum("ij,k,jkc->ic", Bt, Bn, C)
    n = dXdn.shape[0]
    devs = []
    for i in range(skip, n - skip):
        t = np.asarray(m.spec.entity.tangent_space(Xw[i]), dtype=float).reshape(2)
        s = abs(float(t @ dXdn[i])) / float(np.linalg.norm(dXdn[i]))
        devs.append(np.degrees(np.arcsin(min(s, 1.0))))
    return np.asarray(devs)


def run_control_wall_ref(
    wref: ControlWallRef,
    *,
    max_outer: int = 30,
    grad_tol: float = 1e-8,
    alpha_min: float = 1e-4,
    lm0: float = 0.0,
    model_db: bool = False,
) -> dict:
    """Damped reduced GN over the frozen-frame reduced DOFs.

    Same loop shape as :func:`egg.smoothing.control_ref.run_control_ref`, with
    three wall additions per outer iteration: the frame rebuild (reprojection,
    hard-leg snap, ``b`` re-extension — the state and baseline energy are
    re-evaluated after it, so the accept rule always compares like with like),
    the reduction ``dC = R dq``, and the penalty rows composed into both the
    GN system and the line-search energy. ``model_db`` composes the
    frozen-frame Boolean-sum linearization :func:`_db_map` into the GN system
    and the step direction — the step model then knows wall nodes only move
    along the CAD, which removes the model error that feeds the
    frame-to-frame ratchet.
    """
    cref = wref.cref
    st = cref.ctx.energy_stencil
    free_flat = cref.free_ctrl.reshape(-1)
    lam = float(lm0)

    def fine_energy_mindet(Xg):
        return energy_and_mindet(
            Xg,
            st["gc"],
            st["gn0"],
            st["gn1"],
            st["s0"],
            st["s1"],
            st["W_inv"],
            metric=cref.metric,
        )

    energies, mindets, alphas, frame_jumps = [], [], [], []
    converged = False
    iters = 0

    for _ in range(max_outer):
        # Frozen frame for this iteration (moves the state slightly:
        # reprojection + hard snap + b — so re-evaluate the baseline).
        _build_frame(wref)
        cref.Meff = _db_map(wref) if model_db else None
        Xg = cref.prolong_global()
        e_fine, md0 = fine_energy_mindet(Xg)
        e0 = e_fine + _pen_energy(wref, cref.C.reshape(-1, 2)[free_flat])
        if not energies:
            energies.append(e0)
            mindets.append(md0)
        else:
            # Energy injected by the frame rebuild itself (reprojection + b
            # re-extension) — the ratchet's raw signal.
            frame_jumps.append(e0 - energies[-1])

        G_full, A_full = reduced_system(Xg, cref)
        Gp, Ap = _pen_grad_hess(wref)
        R = wref.R
        Gq = R.T @ (G_full.reshape(-1) + Gp)
        Aq = R.T @ (A_full + Ap) @ R
        nq = Gq.shape[0]
        if np.abs(Gq).max() < grad_tol:
            converged = True
            break

        accepted = False
        for _try in range(8):  # Levenberg ladder
            Areg = Aq + lam * np.eye(nq) if lam > 0.0 else Aq
            try:
                dq = np.linalg.solve(Areg, -Gq)
            except np.linalg.LinAlgError:
                lam = max(10.0 * lam, 1e-8 * float(np.trace(Aq)) / nq)
                continue
            dC = (R @ dq).reshape(-1, 2)  # free-control displacement
            corr = (
                np.einsum("ixFy,Fy->ix", cref.Meff, dC)
                if cref.Meff is not None
                else cref.Mf @ dC
            )
            Cfree0 = cref.C.reshape(-1, 2)[free_flat]
            alpha = 1.0
            while alpha >= alpha_min:
                e_f, md = fine_energy_mindet(Xg + alpha * corr)
                e = e_f + _pen_energy(wref, Cfree0 + alpha * dC)
                if np.isfinite(e) and e <= e0 + ENERGY_TOL and md > 0.0:
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                Cflat = cref.C.reshape(-1, 2)
                Cflat[free_flat] += alpha * dC
                energies.append(e)
                mindets.append(md)
                alphas.append(alpha)
                iters += 1
                lam = lam / 3.0 if alpha == 1.0 else lam
                break
            lam = max(10.0 * lam, 1e-8 * float(np.trace(Aq)) / nq)
        if not accepted:
            break

    # Final consistency pass: reproject + b so the reported state is exact.
    _build_frame(wref)
    cref.write_to_grid()
    Xg = cref.prolong_global()
    e_f, md = fine_energy_mindet(Xg)
    return {
        "energies": energies,
        "mindets": mindets,
        "alphas": alphas,
        "frame_jumps": frame_jumps,
        "final_fine_energy": e_f,
        "final_mindet": md,
        "iters": iters,
        "converged": converged,
    }
