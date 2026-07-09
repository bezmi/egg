# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Interface C2 curvature windows: functional, closed-form grad, seam weighting."""

import numpy as np

from egg.smoothing import highorder as H
from egg.smoothing.interface_c2 import (
    CurvatureWindows,
    _seam_nodes,
    _turn_grad,
    curvature_dof_table,
    curvature_energy_grad_hess,
    curvature_windows,
)
from egg.topology.builder import TopologyBuilder


def _lr_grid(res=(7, 7)):
    b = TopologyBuilder(d=2)
    for name, pos in [
        ("A", (0.0, 0.0)),
        ("D", (0.0, 2.0)),
        ("B", (2.0, 0.0)),
        ("C", (2.0, 2.0)),
        ("E", (4.0, 0.0)),
        ("F", (4.0, 2.0)),
    ]:
        b.add_corner(name, pos, fixed=True)
    b.add_block("L", ("A", "D", "B", "C"), res)
    b.add_block("R", ("B", "C", "E", "F"), res)
    b.connect("L", 0, 1, "R", 0, 0)
    return b.build().initialize_grid()


def _win():
    return CurvatureWindows(
        nodes=np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64),
        weight=np.array([1.3, 0.7]),
    )


def test_energy_matches_prototype():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 2))
    win = _win()
    E, _, _ = curvature_energy_grad_hess(X, win)
    e_proto = (win.weight * H._f_curv([X[win.nodes[:, k]] for k in range(4)])).sum()
    assert abs(E - e_proto) < 1e-12


def test_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 2))
    win = _win()
    _, g, _ = curvature_energy_grad_hess(X, win)
    eps = 1e-6
    fd = np.zeros_like(X)
    for i in range(X.shape[0]):
        for k in range(2):
            Xp, Xm = X.copy(), X.copy()
            Xp[i, k] += eps
            Xm[i, k] -= eps
            fd[i, k] = (
                curvature_energy_grad_hess(Xp, win)[0]
                - curvature_energy_grad_hess(Xm, win)[0]
            ) / (2 * eps)
    assert np.allclose(g, fd, atol=1e-6)


def test_gauss_newton_hessian_is_psd():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(6, 2))
    _, _, gn = curvature_energy_grad_hess(X, _win())
    for blk in gn.reshape(-1, 2, 2):
        assert np.linalg.eigvalsh(blk).min() >= -1e-12


def test_zero_on_straight_positive_on_kink():
    win = _win()
    Xs = np.array([[0, 0], [1, 0], [2.5, 0], [4, 0], [5, 0.0]])
    assert curvature_energy_grad_hess(Xs, win)[0] == 0.0
    Xk = Xs.copy()
    Xk[2, 1] = 0.5
    assert curvature_energy_grad_hess(Xk, win)[0] > 1e-3


def test_windows_cross_seam_and_carry_boost():
    grid = _lr_grid()
    seam = set(int(n) for n in _seam_nodes(grid))
    win = curvature_windows(grid, weight=1.0, iface_boost=5.0)
    assert len(win) > 0
    touches = np.array([bool(seam & set(int(x) for x in row)) for row in win.nodes])
    assert touches.any(), "no window crosses the seam"
    # seam-touching windows carry the boosted weight, interior ones the base.
    assert np.allclose(win.weight[touches], 5.0)
    assert np.allclose(win.weight[~touches], 1.0)


def test_interface_only_keeps_seam_windows():
    grid = _lr_grid()
    full = curvature_windows(grid, weight=1.0, iface_boost=3.0)
    only = curvature_windows(grid, weight=1.0, iface_boost=3.0, interface_only=True)
    assert 0 < len(only) < len(full)
    assert np.allclose(only.weight, 3.0)


def test_dof_table_reconstructs_global_gradient():
    """Each free DOF's fixed-K windows sum to its global-gradient entry — the
    consistency the C++ metric/line-search kernels rely on."""
    grid = _lr_grid()
    X = np.asarray(grid.global_nodes).copy()
    free = np.asarray(grid.free_mask)
    rng = np.random.default_rng(0)
    X[free] += 0.03 * rng.normal(size=X[free].shape)
    win = curvature_windows(grid, weight=1.0, iface_boost=3.0)
    tab = curvature_dof_table(free, win)
    _, gglob, _ = curvature_energy_grad_hess(X, win)

    dofs = np.flatnonzero(free)
    gtab = np.zeros_like(X)
    for i, dof in enumerate(dofs):
        for j in range(tab["curv_k"]):
            slot = int(tab["curv_slot"][i, j])
            if slot < 0:
                continue
            nodes = tab["curv_nodes"][i, j]
            w = tab["curv_weight"][i, j]
            p = [X[nodes[k]][None] for k in range(4)]
            th1, g1 = _turn_grad(p[0], p[1], p[2])
            th2, g2 = _turn_grad(p[1], p[2], p[3])
            r = float((th1 - th2)[0])
            dr = [g1[0, 0], g1[0, 1] - g2[0, 0], g1[0, 2] - g2[0, 1], -g2[0, 2]]
            gtab[dof] += 2.0 * w * r * dr[slot]
    assert np.abs(gtab[dofs] - gglob[dofs]).max() < 1e-10
