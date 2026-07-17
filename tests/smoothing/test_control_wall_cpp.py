# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Device wall constraints: parity with the NumPy wall reference + criteria.

The session's wall mode (control wire with ``walls=``) must reproduce
:mod:`egg.smoothing.control_wall` — same frozen-frame cadence (reprojection,
hard-leg snap, Boolean-sum re-extension per outer iteration), same composed
accept rule — and independently meet the wall acceptance criteria: nodes on
the entity, hard-mode angle <= 1 degree, penalty monotone in the weight.
"""

from __future__ import annotations

import numpy as np
import pytest

from egg.geometry.control_net import tensor_map
from egg.smoothing.control_ref import (
    boolean_sum_b,
    boundary_ctrl_mask,
    fit_control_net,
)
from egg.smoothing.control_wall import (
    WallSpec,
    _wall_ctrl_rows,
    build_control_wall_ref,
    run_control_wall_ref,
)
from egg.smoothing.solver import build_sweep_context
from egg.smoothing.targets import IdentityTarget
from tests.smoothing.test_control_wall import (
    _SKIP,
    _make_arc_grid,
    _make_sheared_arc_grid,
)


def _has_cpp() -> bool:
    try:
        from egg._cpp import cpp_core  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cpp(),
    reason="egg._cpp.cpp_core not built (requires cmake build)",
)


def _make_wall_session(grid, arc, ortho, weight=1.0, ctrl=(8, 8)):
    from egg.smoothing.control_backend import build_control_wire
    from egg.smoothing.cpp_backend import (
        CppStructuredSweepSession,
        build_block_structured_context,
    )

    block = grid.blocks[0]
    node_shape = tuple(block.logical_shape)
    cmap = tensor_map(node_shape, ctrl, degree=3)
    X0 = np.asarray(block.nodes, dtype=float)
    C0 = fit_control_net(cmap, X0)
    b_field = boolean_sum_b(cmap, C0, X0)
    free_ctrl = ~boundary_ctrl_mask(tuple(ctrl))
    spec = WallSpec(axis=1, side=0, entity=arc, ortho=ortho, weight=weight)
    p0, _ = _wall_ctrl_rows(spec.axis, spec.side, tuple(ctrl))
    free_ctrl.reshape(-1)[p0] = True

    wire = build_control_wire(cmap, C0, free_ctrl, b_field, walls=[spec], X0=X0)
    ctx = build_sweep_context(grid, IdentityTarget(d=2))
    bsc = build_block_structured_context(grid)
    sess = CppStructuredSweepSession(
        ctx, bsc, np.asarray(grid.global_nodes), device="cpu", control=wire
    )
    return sess, cmap


def _analytic_dev(sess, cmap, arc, skip=_SKIP):
    """Angle (degrees) between the spline wall-normal derivative and the
    entity normal along the bottom wall, from the device's control lattice."""
    C = sess.get_C().reshape(cmap.ctrl_shape + (2,))
    Bt = cmap.basis[0]
    Bnd = cmap.basis_d1[1][0]
    Bn = cmap.basis[1][0]
    dXdn = np.einsum("ij,k,jkc->ic", Bt, Bnd, C)
    Xw = np.einsum("ij,k,jkc->ic", Bt, Bn, C)
    devs = []
    for i in range(skip, dXdn.shape[0] - skip):
        t = np.asarray(arc.tangent_space(Xw[i]), dtype=float).reshape(2)
        s = abs(float(t @ dXdn[i])) / float(np.linalg.norm(dXdn[i]))
        devs.append(np.degrees(np.arcsin(min(s, 1.0))))
    return np.asarray(devs)


def test_wall_run_keeps_nodes_on_entity():
    grid, arc = _make_arc_grid()
    sess, _ = _make_wall_session(grid, arc, "off")
    from tests.real_tol import real_tol

    rep = sess.run_control(20)
    assert rep["iters"] >= 1
    assert all(np.isfinite(rep["energies"]))
    assert all(md > 0.0 for md in rep["mindets"])
    X = sess.get_X()
    X_wall = X[grid.block_dof_maps[0][:, 0]]
    feet = arc.project_many(X_wall)
    np.testing.assert_allclose(X_wall, feet, atol=real_tol(1e-9))


def test_wall_parity_with_numpy_reference():
    from tests.real_tol import real_tol

    for ortho, weight in (("off", 0.0), ("penalty", 10.0), ("hard", 0.0)):
        grid_ref, arc_ref = _make_sheared_arc_grid()
        wref = build_control_wall_ref(
            grid_ref,
            IdentityTarget(d=2),
            (8, 8),
            [WallSpec(axis=1, side=0, entity=arc_ref, ortho=ortho, weight=weight)],
        )
        rep_ref = run_control_wall_ref(wref, max_outer=12)

        grid_dev, arc_dev = _make_sheared_arc_grid()
        sess, _ = _make_wall_session(grid_dev, arc_dev, ortho, weight=weight)
        rep_dev = sess.run_control(
            12, pcg_max_iter=500, pcg_rtol=1e-12, pcg_forcing=False
        )

        e_ref = np.asarray(rep_ref["energies"])
        e_dev = np.asarray(rep_dev["energies"])
        assert all(md > 0.0 for md in rep_dev["mindets"])
        n = min(e_ref.size, e_dev.size)
        # never assert exact energies — the trajectories involve per-frame
        # projections; PCG vs dense drift compounds mildly across iterations.
        np.testing.assert_allclose(
            e_dev[:n], e_ref[:n], rtol=real_tol(1e-5), err_msg=f"ortho={ortho}"
        )


def test_wall_hard_ortho_angle_within_one_degree():
    grid, arc = _make_sheared_arc_grid()
    sess, cmap = _make_wall_session(grid, arc, "hard")
    rep = sess.run_control(25)
    assert all(md > 0.0 for md in rep["mindets"])
    assert _analytic_dev(sess, cmap, arc).max() <= 1.0


def _make_sphere_cube_grid(n=9, shear=0.5, perturb=0.01, seed=3):
    """Unit cube, bottom face on a sphere patch through the face corners,
    sheared in x with height (the 3D twin of the sheared-arc fixture)."""
    from egg.geometry.analytic3d import Sphere
    from egg.topology.builder import TopologyBuilder

    h = 0.1  # bulge height at the face centre
    zc = (0.5 - h * h) / (2.0 * h)
    r = zc + h
    sphere = Sphere((0.5, 0.5, -zc), r, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    tb = TopologyBuilder(d=3)
    for name, pos in {
        "c000": (0, 0, 0),
        "c001": (0, 0, 1),
        "c010": (0, 1, 0),
        "c011": (0, 1, 1),
        "c100": (1, 0, 0),
        "c101": (1, 0, 1),
        "c110": (1, 1, 0),
        "c111": (1, 1, 1),
    }.items():
        tb.add_corner(name, pos, fixed=True)
    tb.add_block(
        "main",
        corners=("c000", "c001", "c010", "c011", "c100", "c101", "c110", "c111"),
        resolutions=(n, n, n),
    )
    grid = tb.build().initialize_grid()
    block = grid.blocks[0]
    ns = block.nodes.shape[0]

    # Rebuild the node field analytically: bottom on the sphere (vertical
    # lift), blended out to a planar top, then sheared and perturbed.
    u = np.linspace(0.0, 1.0, ns)
    X, Y, Z = np.meshgrid(u, u, u, indexing="ij")
    lift = -zc + np.sqrt(r * r - (X - 0.5) ** 2 - (Y - 0.5) ** 2)
    z = lift + Z * (1.0 - lift)
    nodes = np.stack([X + shear * z, Y, z], axis=-1)
    rng = np.random.default_rng(seed)
    nodes[1:-1, 1:-1, 1:-1] += perturb * rng.standard_normal(
        nodes[1:-1, 1:-1, 1:-1].shape
    )
    block.nodes[...] = nodes
    grid.global_nodes[grid.block_dof_maps[0].reshape(-1)] = nodes.reshape(-1, 3)
    return grid, sphere


def _make_wall_session_3d(grid, sphere, ortho, ctrl=(5, 5, 5)):
    from egg.smoothing.control_backend import build_control_wire
    from egg.smoothing.cpp_backend import (
        CppStructuredSweepSession,
        build_block_structured_context,
    )

    block = grid.blocks[0]
    node_shape = tuple(block.logical_shape)
    cmap = tensor_map(node_shape, ctrl, degree=3)
    X0 = np.asarray(block.nodes, dtype=float)
    C0 = fit_control_net(cmap, X0)
    b_field = boolean_sum_b(cmap, C0, X0)
    free_ctrl = ~boundary_ctrl_mask(tuple(ctrl))
    spec = WallSpec(axis=2, side=0, entity=sphere, ortho=ortho)
    p0, _ = _wall_ctrl_rows(spec.axis, spec.side, tuple(ctrl))
    free_ctrl.reshape(-1)[p0] = True

    wire = build_control_wire(cmap, C0, free_ctrl, b_field, walls=[spec], X0=X0)
    ctx = build_sweep_context(grid, IdentityTarget(d=3))
    bsc = build_block_structured_context(grid)
    sess = CppStructuredSweepSession(
        ctx, bsc, np.asarray(grid.global_nodes), device="cpu", control=wire
    )
    return sess, cmap


def _analytic_dev_3d(sess, cmap, sphere, skip=2):
    """Angle (degrees) between the spline wall-normal derivative and the
    sphere's radial normal over the bottom face's interior window."""
    C = sess.get_C().reshape(cmap.ctrl_shape + (3,))
    B0, B1 = cmap.basis[0], cmap.basis[1]
    B2d = cmap.basis_d1[2][0]
    B2 = cmap.basis[2][0]
    dXdn = np.einsum("ia,jb,c,abcx->ijx", B0, B1, B2d, C)
    Xw = np.einsum("ia,jb,c,abcx->ijx", B0, B1, B2, C)
    devs = []
    n0, n1 = dXdn.shape[:2]
    for i in range(skip, n0 - skip):
        for j in range(skip, n1 - skip):
            nrm = np.asarray(sphere.normal(Xw[i, j]), dtype=float).reshape(3)
            d = dXdn[i, j]
            c = abs(float(nrm @ d)) / float(np.linalg.norm(d))
            devs.append(np.degrees(np.arccos(min(c, 1.0))))
    return np.asarray(devs)


def test_wall_3d_sphere_hard_ortho():
    # The full 3D wall path: sliding face controls on a sphere (2 tangential
    # DOFs each), hard legs along the radial normal, 3D Boolean-sum
    # re-extension. Hard mode must beat the unconstrained wall run and land
    # near-orthogonal despite the shear.
    grid_a, sphere_a = _make_sphere_cube_grid()
    sess_a, cmap_a = _make_wall_session_3d(grid_a, sphere_a, "off")
    rep_off = sess_a.run_control(10)
    assert all(md > 0.0 for md in rep_off["mindets"])

    grid_b, sphere_b = _make_sphere_cube_grid()
    sess_b, cmap_b = _make_wall_session_3d(grid_b, sphere_b, "hard")
    rep = sess_b.run_control(10)
    assert rep["iters"] >= 1
    assert all(np.isfinite(rep["energies"]))
    assert all(md > 0.0 for md in rep["mindets"])

    # The constraint itself is exact: every wall control leg is normal.
    from egg.smoothing.control_wall import _wall_ctrl_rows as _rows

    C = sess_b.get_C().reshape(cmap_b.ctrl_shape + (3,)).reshape(-1, 3)
    p0, p1 = _rows(2, 0, cmap_b.ctrl_shape)
    for j in range(len(p0)):
        leg = C[p1[j]] - C[p0[j]]
        nrm = np.asarray(sphere_b.normal(C[p0[j]]), dtype=float).reshape(3)
        cosang = abs(float(nrm @ leg)) / float(np.linalg.norm(leg))
        # arccos loses ~sqrt(eps) resolution near 1; 1e-4 deg is still exact.
        assert np.degrees(np.arccos(min(cosang, 1.0))) <= 1e-4

    # The fine-map derivative: the fixed corner/edge ring's cubic support
    # owns ~2 control intervals (~3 of 9 face nodes at this coarse net); the
    # window inside it must beat the unconstrained run and be near-normal.
    dev_hard = _analytic_dev_3d(sess_b, cmap_b, sphere_b, skip=3).max()
    assert dev_hard < _analytic_dev_3d(sess_a, cmap_a, sphere_a, skip=3).max()
    assert dev_hard <= 4.0

    # b owns exactness in 3D too: every bottom-face node is on the sphere.
    from tests.real_tol import real_tol

    X = sess_b.get_X()
    wall_ids = grid_b.block_dof_maps[0][:, :, 0].reshape(-1)
    X_wall = X[wall_ids]
    feet = sphere_b.project_many(X_wall)
    np.testing.assert_allclose(X_wall, feet, atol=real_tol(1e-9))


def test_wall_penalty_monotone_in_weight():
    maxdev = []
    for w in (0.0, 1.0, 100.0):
        grid, arc = _make_sheared_arc_grid()
        sess, cmap = _make_wall_session(
            grid, arc, "penalty" if w > 0 else "off", weight=w
        )
        rep = sess.run_control(25)
        assert all(md > 0.0 for md in rep["mindets"])
        maxdev.append(_analytic_dev(sess, cmap, arc).max())
    assert maxdev[1] < maxdev[0]
    assert maxdev[2] < maxdev[1]
