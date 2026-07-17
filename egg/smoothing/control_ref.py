# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""NumPy reference for the control-net reduced Gauss-Newton smoother.

The sequential parity target for the control-point mode: the fine grid is the
fixed linear map ``X = M C + b`` (:mod:`egg.geometry.control_net`) and the
TMOP energy is minimised over the *control points* by a damped Gauss-Newton
loop —

    G = M_f^T g,      A = M_f^T H M_f + lambda I,      A dC = -G

with ``g``/``H`` the per-node gradient and contracted per-node d x d Hessian
of the same all-corner-sample energy the node smoother uses, and ``M_f`` the
columns of ``M`` belonging to free controls. Every candidate step passes the
codebase's global accept rule (finite, monotone energy to ``tol``, min det A
positive) through a backtracking line search on the *fine* energy; rejection
bumps the Levenberg damping instead of freezing.

Explicit dense operators throughout — this is the small-n debug/parity path,
never the production one. The metric layer it drives
(:mod:`egg.smoothing.batch`) is currently 2D; the control machinery itself is
dimension-general and the device path validates D = 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from egg.geometry.control_net import (
    ControlNetMap,
    facets,
    tensor_map,
    tfi_operator,
)
from egg.smoothing.batch import (
    _grad_T,
    _hess_T,
    assemble_A,
    energy_and_mindet,
    make_chain_J,
)
from egg.smoothing.solver import SweepContext, build_sweep_context

__all__ = [
    "ControlRefContext",
    "boundary_ctrl_mask",
    "build_control_ref",
    "dense_M",
    "fit_control_net",
    "reduced_system",
    "run_control_ref",
]

#: fp64 line-search accept slack / min-det rule, mirroring src/core.hpp tol::energy
#: and the shape-phase Objective::accept_mindet (see src/metric.hpp).
ENERGY_TOL = 1e-12


def dense_M(cmap: ControlNetMap) -> np.ndarray:
    """Materialise ``M`` as a dense ``(n_nodes, n_ctrl)`` matrix (small n only)."""
    M = cmap.basis[0]
    for B in cmap.basis[1:]:
        M = np.kron(M, B)
    return M


def boundary_ctrl_mask(ctrl_shape: tuple[int, ...]) -> np.ndarray:
    """Boolean mask of control lattice points on the block boundary."""
    mask = np.zeros(ctrl_shape, dtype=bool)
    for k in range(len(ctrl_shape)):
        sl: list = [slice(None)] * len(ctrl_shape)
        sl[k] = 0
        mask[tuple(sl)] = True
        sl[k] = ctrl_shape[k] - 1
        mask[tuple(sl)] = True
    return mask


def _facet_slices(offset: tuple[int, ...], shape: tuple[int, ...]) -> tuple:
    """Index tuple selecting a facet's slab of a lattice with the given shape."""
    out: list = []
    for o, n in zip(offset, shape):
        if o == -1:
            out.append(0)
        elif o == 1:
            out.append(n - 1)
        else:
            out.append(slice(None))
    return tuple(out)


def _mode_apply(mats: list[np.ndarray], T: np.ndarray) -> np.ndarray:
    """Apply one matrix per leading axis of ``T`` (mode products)."""
    for k, A in enumerate(mats):
        T = np.moveaxis(np.tensordot(A, T, axes=([1], [k])), 0, k)
    return T


def fit_control_net(cmap: ControlNetMap, X0: np.ndarray) -> np.ndarray:
    """Facet-ordered hierarchical least-squares fit of a control net to ``X0``.

    Corners are taken exactly, then each facet's interior controls are fitted
    to that facet's node data with the facet's own boundary (already-set,
    lower-idim) controls frozen and their contribution subtracted, ending with
    the block interior. Clamped bases localise each facet's nodes to its own
    control slab, so the fit is consistent facet-by-facet and dimension-general.

    Each facet solve is SEPARABLE: the free controls of a slab are exactly the
    tensor product of the per-axis interior ranges (its boundary lattice was
    set by lower-idim facets), and the Moore-Penrose inverse distributes over
    Kronecker products — so per-axis pseudo-inverse mode products give the
    same least-squares solution as the joint dense solve at O(N) instead of
    a dense lstsq per facet (the at-scale build cost).
    """
    d = cmap.d
    ctrl_shape = cmap.ctrl_shape
    C = np.zeros(ctrl_shape + (d,))
    is_set = np.zeros(ctrl_shape, dtype=bool)
    pinv_int: dict[int, np.ndarray] = {}  # axis -> pinv of interior columns

    # Facets low idim first (corners, then edges, then faces...), interior last.
    fs = sorted(facets(d), key=lambda f: f.idim)
    all_slabs = [f.offset for f in fs] + [(0,) * d]
    for offset in all_slabs:
        span = [k for k in range(d) if offset[k] == 0]
        csl = _facet_slices(offset, ctrl_shape)
        nsl = _facet_slices(offset, cmap.node_shape)
        if not span:  # corner: interpolate exactly (clamped basis passes through)
            C[csl] = X0[nsl]
            is_set[csl] = True
            continue
        # Frozen contribution: evaluate the slab with the free controls
        # zeroed (the free set is the slab interior — every already-set
        # point sits on the slab boundary).
        Cslab = np.array(C[csl])
        interior = tuple(slice(1, -1) for _ in span)
        Cslab[interior] = 0.0
        rhs = X0[nsl] - _mode_apply([cmap.basis[k] for k in span], Cslab)
        for k in span:
            if k not in pinv_int:
                pinv_int[k] = np.linalg.pinv(cmap.basis[k][:, 1:-1])
        sol = _mode_apply([pinv_int[k] for k in span], rhs)
        Cslab = np.array(C[csl])
        Cslab[interior] = sol
        C[csl] = Cslab
        is_set[csl] = True
    return C


def boolean_sum_b(cmap: ControlNetMap, C: np.ndarray, X0: np.ndarray) -> np.ndarray:
    """Boolean-sum boundary correction ``b``: TFI extension of ``X0 - MC`` on the boundary.

    With ``b`` added, boundary nodes are exact (``X0``) and the interior blends
    the mismatch away; ``b`` depends only on boundary controls, so it is
    constant while those are frozen.
    """
    node_shape = cmap.node_shape
    d = cmap.d
    Xs = cmap.prolong(C)
    delta = np.zeros(node_shape + (d,))
    bmask = boundary_ctrl_mask(node_shape)
    delta[bmask] = (X0 - Xs)[bmask]
    T = tfi_operator(node_shape)
    n_total = int(np.prod(node_shape))
    return (T @ delta.reshape(n_total, d)).reshape(node_shape + (d,))


@dataclass
class ControlRefContext:
    """Everything the reference reduced-GN loop needs, built once."""

    grid: object
    ctx: SweepContext
    cmap: ControlNetMap
    block_idx: int
    C: np.ndarray  # (ctrl_shape + (d,)) current control net
    free_ctrl: np.ndarray  # bool (ctrl_shape,)
    Mf: np.ndarray  # (M_global_nodes, F) free-control columns in global-node rows
    b_field: np.ndarray  # (node_shape + (d,)) Boolean-sum offset
    metric: str = "shape_2d"
    # Optional coordinate-mixing effective map (M_global_nodes, d, F, d):
    # set per frozen frame when the Boolean-sum linearization d b/d C is
    # modelled (wall runners); None keeps the coordinate-diagonal Mf chain.
    Meff: np.ndarray | None = None
    # Position-independent per-sample chain Jacobian for the Hessian assembly.
    _J: np.ndarray = field(repr=False, default=None)

    @property
    def n_free(self) -> int:
        return int(self.free_ctrl.sum())

    def prolong_global(self, C: np.ndarray | None = None) -> np.ndarray:
        """Evaluate the net into global-node order (M, d)."""
        Cc = self.C if C is None else C
        X = self.cmap.prolong(Cc) + self.b_field
        grid = self.grid
        Xg = np.array(grid.global_nodes)
        dof_map = grid.block_dof_maps[self.block_idx]
        Xg[dof_map.reshape(-1)] = X.reshape(-1, self.cmap.d)
        return Xg

    def write_to_grid(self) -> None:
        """Push the current net's fine grid into the MultiBlockGrid state."""
        grid = self.grid
        grid.global_nodes[:] = self.prolong_global()
        for bi, block in enumerate(grid.blocks):
            block.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]


def build_control_ref(
    grid,
    target_fn,
    ctrl_shape,
    *,
    degree: int = 3,
    metric: str = "shape_2d",
) -> ControlRefContext:
    """Build the reference context: single block, fixed boundary controls."""
    d = grid.topology.d
    if d != 2:
        raise NotImplementedError(
            "the NumPy metric layer (egg.smoothing.batch) is 2D; use the device "
            "path for D=3"
        )
    if len(grid.blocks) != 1:
        raise NotImplementedError("the NumPy reference is single-block")
    block = grid.blocks[0]
    node_shape = tuple(block.logical_shape)

    ctx = build_sweep_context(grid, target_fn)
    cmap = tensor_map(node_shape, ctrl_shape, degree=degree)
    X0 = np.asarray(block.nodes, dtype=float)
    C = fit_control_net(cmap, X0)
    b_field = boolean_sum_b(cmap, C, X0)

    free_ctrl = ~boundary_ctrl_mask(tuple(ctrl_shape))
    # Dense M in global-node rows, free columns only.
    Md = dense_M(cmap)  # (n_block_nodes, n_ctrl)
    dof_map = grid.block_dof_maps[0].reshape(-1)
    Mg = np.zeros((grid.global_node_count, Md.shape[1]))
    Mg[dof_map] = Md
    Mf = Mg[:, free_ctrl.reshape(-1)]

    st = ctx.energy_stencil
    J = make_chain_J(st["s0"].astype(float), st["s1"].astype(float), st["W_inv"])
    return ControlRefContext(
        grid=grid,
        ctx=ctx,
        cmap=cmap,
        block_idx=0,
        C=C,
        free_ctrl=free_ctrl,
        Mf=Mf,
        b_field=b_field,
        metric=metric,
        _J=J,
    )


def node_grad_hess_field(
    Xg: np.ndarray, ctx: SweepContext, J: np.ndarray, metric: str = "shape_2d"
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node gradient (M, d) and contracted per-node Hessian (M, d, d).

    The NumPy analogue of the device ``metric_kernel`` buffers: every
    (cell, corner) sample scatters its gradient columns and its role-restricted
    ``J^T H_T J`` block onto each participating node.
    """
    st = ctx.energy_stencil
    mem = ctx._stencil
    gc, gn0, gn1 = st["gc"], st["gn0"], st["gn1"]
    s0, s1, W_inv = st["s0"], st["s1"], st["W_inv"]
    M = ctx._grid.global_node_count

    A = assemble_A(Xg, gc, gn0, gn1, s0, s1)
    T = np.einsum("pij,pjk->pik", A, W_inv)
    dmu_dT = _grad_T(T, metric)
    dmu_dA = np.einsum("pij,pkj->pik", dmu_dT, W_inv)
    col0, col1 = dmu_dA[:, :, 0], dmu_dA[:, :, 1]
    H_T = _hess_T(T, metric)

    m_node = mem["m_node"]
    m_sid = mem["m_sid"]
    m_role = mem["m_role"].astype(np.intp)

    # Gradient: role-selected column contribution per membership entry.
    contrib = np.zeros((m_node.shape[0], 2))
    is_c, is_0, is_1 = m_role == 0, m_role == 1, m_role == 2
    sid_c, sid_0, sid_1 = m_sid[is_c], m_sid[is_0], m_sid[is_1]
    contrib[is_c] = -(s0[sid_c, None] * col0[sid_c] + s1[sid_c, None] * col1[sid_c])
    contrib[is_0] = s0[sid_0, None] * col0[sid_0]
    contrib[is_1] = s1[sid_1, None] * col1[sid_1]
    g = np.zeros((M, 2))
    np.add.at(g, m_node, contrib)

    # Hessian: J^T H_T J restricted to the entry's role columns. Role -1 marks
    # a non-participating slot (mirrors batch.dof_grad_hess's `valid` guard).
    valid = m_role >= 0
    role_safe = np.maximum(m_role, 0)
    cols = (2 * role_safe)[:, None] + np.array([0, 1])[None]
    Jb = np.take_along_axis(J[m_sid], cols[:, None, :], axis=2)  # (Pm, 4, 2)
    Jb[~valid] = 0.0
    blk = np.einsum("pai,pab,pbj->pij", Jb, H_T[m_sid], Jb)
    H = np.zeros((M, 2, 2))
    np.add.at(H, m_node, blk)
    return g, H


def reduced_system(
    Xg: np.ndarray, cref: ControlRefContext
) -> tuple[np.ndarray, np.ndarray]:
    """Exact reduced gradient (F, d) and full GN Hessian (F*d, F*d).

    Per sample, the chain to the free controls is ``W_p = J_p · Mf[nodes_p]``
    (d vec(T) / d C-hat), so ``A = sum_p W_p^T H_T,p W_p`` keeps every
    cross-node coupling a sample induces between controls. The per-node
    *contracted* Hessian (the node smoother's buffer) drops those couplings —
    correct for one-node-at-a-time Jacobi, but far too stiff as a curvature
    model for a globally coupled step (it halves the step the quadratic model
    wants and convergence degrades to slow gradient descent).
    """
    st = cref.ctx.energy_stencil
    gc, gn0, gn1 = st["gc"], st["gn0"], st["gn1"]
    s0, s1, W_inv = st["s0"], st["s1"], st["W_inv"]
    J = cref._J
    P = gc.shape[0]
    F = cref.n_free

    A_geom = assemble_A(Xg, gc, gn0, gn1, s0, s1)
    T = np.einsum("pij,pjk->pik", A_geom, W_inv)
    dmu_dT = _grad_T(T, cref.metric)
    H_T = _hess_T(T, cref.metric)

    # coords layout of J is [corner_x, corner_y, nbr0_x, nbr0_y, nbr1_x, nbr1_y]
    Jr = J.reshape(P, 4, 3, 2)
    if getattr(cref, "Meff", None) is not None:
        # Coordinate-mixing effective map (node, d, F, d): the frozen-frame
        # Boolean-sum linearization d b/d C couples coordinates through the
        # wall-normal projector, so the chain runs at component level.
        M3 = np.stack(
            [cref.Meff[gc], cref.Meff[gn0], cref.Meff[gn1]], axis=1
        )  # (P, 3, d, F, d)
        W = np.einsum("pacx,pcxFy->paFy", Jr, M3, optimize=True).reshape(P, 4, F * 2)
    else:
        Mf3 = np.stack([cref.Mf[gc], cref.Mf[gn0], cref.Mf[gn1]], axis=1)  # (P,3,F)
        W = np.einsum("pacd,pcF->paFd", Jr, Mf3, optimize=True).reshape(P, 4, F * 2)
    G = np.einsum("pa,paI->I", dmu_dT.reshape(P, 4), W, optimize=True).reshape(F, 2)
    A_red = np.einsum("paI,pab,pbJ->IJ", W, H_T, W, optimize=True)
    return G, A_red


def run_control_ref(
    cref: ControlRefContext,
    *,
    max_outer: int = 30,
    grad_tol: float = 1e-8,
    alpha_min: float = 1e-4,
    lm0: float = 0.0,
) -> dict:
    """Damped reduced Gauss-Newton with a global fine-energy line search.

    Returns a report dict: per-iteration ``energies`` / ``mindets`` /
    ``alphas``, the accepted iteration count ``iters``, and ``converged``.
    The final accepted state is left in ``cref.C`` and pushed to the grid.
    """
    st = cref.ctx.energy_stencil
    d = cref.cmap.d
    F = cref.n_free
    free_flat = cref.free_ctrl.reshape(-1)
    lam = float(lm0)

    def energy_mindet(Xg):
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

    Xg = cref.prolong_global()
    e0, md0 = energy_mindet(Xg)
    energies, mindets, alphas = [e0], [md0], []
    converged = False
    iters = 0

    for _ in range(max_outer):
        G, A_red = reduced_system(Xg, cref)
        if np.abs(G).max() < grad_tol:
            converged = True
            break

        accepted = False
        for _try in range(8):  # Levenberg ladder
            Areg = A_red + lam * np.eye(F * d) if lam > 0.0 else A_red
            try:
                dC = np.linalg.solve(Areg, -G.reshape(F * d))
            except np.linalg.LinAlgError:
                lam = max(10.0 * lam, 1e-8 * float(np.trace(A_red)) / (F * d))
                continue
            corr = cref.Mf @ dC.reshape(F, d)  # (M, d) node-space correction
            alpha = 1.0
            while alpha >= alpha_min:
                e, md = energy_mindet(Xg + alpha * corr)
                if np.isfinite(e) and e <= e0 + ENERGY_TOL and md > 0.0:
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                Cflat = cref.C.reshape(-1, d)
                Cflat[free_flat] += alpha * dC.reshape(F, d)
                Xg = Xg + alpha * corr
                e0, md0 = e, md
                energies.append(e0)
                mindets.append(md0)
                alphas.append(alpha)
                iters += 1
                lam = lam / 3.0 if alpha == 1.0 else lam
                break
            lam = max(10.0 * lam, 1e-8 * float(np.trace(A_red)) / (F * d))
        if not accepted:
            break  # damping exhausted; report what we have

    cref.write_to_grid()
    return {
        "energies": energies,
        "mindets": mindets,
        "alphas": alphas,
        "iters": iters,
        "converged": converged,
    }
