# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Device multi-block control session: parity with the topology reference.

The multi-block device path (per-block nets over one stacked control vector,
the reduction CSR from ``control_topology``, generic penalty rows, halo
refresh after every net eval, alias-reduced pullback) must reproduce the
NumPy reference ``run_control_topo_ref`` on the same problems with exact PCG
solves. The 3D runs exercise the same machinery where the NumPy reference
cannot follow (its metric layer is 2D).
"""

from __future__ import annotations

import numpy as np
import pytest

from egg.smoothing.control_topology import (
    run_control_topo_ref,
    seam_angle_deviation,
    seam_c1_jump,
    seam_curvature_mismatch,
    watertight_mismatch,
)
from egg.smoothing.targets import IdentityTarget
from tests.smoothing.test_control_topology import (
    _fan_grid,
    _lr_grid,
    _perturb,
    _quad_grid,
    _slab3_grid,
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

# Exact-solve options: the reference solves densely, so parity runs disable
# the inexact-Newton forcing and solve PCG to a tight residual.
_EXACT = dict(pcg_forcing=False, pcg_rtol=1e-12, pcg_max_iter=2000)


def _run_pair(make_grid, **kw):
    """Reference and device runs on identical grids; returns both reports."""
    from egg.smoothing.control_backend import run_control_topo

    grid_ref = make_grid()
    _perturb(grid_ref)
    topo_ref, rep_ref = run_control_topo_ref(grid_ref, IdentityTarget(2), **kw)

    grid_dev = make_grid()
    _perturb(grid_dev)
    topo_dev, rep_dev = run_control_topo(
        grid_dev, IdentityTarget(2), device="cpu", **_EXACT, **kw
    )
    return (grid_ref, topo_ref, rep_ref), (grid_dev, topo_dev, rep_dev)


def _assert_parity(ref, dev, rtol=1e-7, x_atol=1e-6, monotone=True):
    from tests.real_tol import real_tol

    (grid_ref, _topo_ref, rep_ref), (grid_dev, _topo_dev, rep_dev) = ref, dev
    e_ref = np.asarray(rep_ref["energies"])
    e_dev = np.asarray(rep_dev["energies"])
    assert np.all(np.isfinite(e_dev))
    if monotone:
        # With per-iteration frozen-frame rebuilds (seam ortho) the composed
        # energy is monotone only within a frame, not across rebuilds.
        assert np.all(np.diff(e_dev) <= 1e-12)
    assert all(md > 0.0 for md in rep_dev["mindets"])
    assert abs(rep_dev["iters"] - rep_ref["iters"]) <= 2
    n = min(e_ref.size, e_dev.size)
    # atol floors the comparison where the energy itself converges to ~0
    # (never assert exact energies — reduction order).
    np.testing.assert_allclose(
        e_dev[:n], e_ref[:n], rtol=real_tol(rtol), atol=real_tol(1e-12)
    )
    np.testing.assert_allclose(
        rep_dev["final_fine_energy"],
        rep_ref["final_fine_energy"],
        rtol=real_tol(rtol),
        atol=real_tol(1e-12),
    )
    np.testing.assert_allclose(
        np.asarray(grid_dev.global_nodes),
        np.asarray(grid_ref.global_nodes),
        atol=real_tol(x_atol),
    )


def test_parity_two_block_seam():
    ref, dev = _run_pair(lambda: _lr_grid(wave=0.3))
    _assert_parity(ref, dev)
    _grid, topo, _rep = dev
    assert watertight_mismatch(topo) < 1e-12
    for si in range(len(topo.seams)):
        assert seam_c1_jump(topo, si) < 1e-12


def test_parity_crossing():
    ref, dev = _run_pair(_quad_grid)
    _assert_parity(ref, dev)
    _grid, topo, _rep = dev
    for si in range(len(topo.seams)):
        assert seam_c1_jump(topo, si) < 1e-12


def test_parity_c2_penalty():
    ref, dev = _run_pair(lambda: _lr_grid(wave=0.3), c2_weight=100.0)
    _assert_parity(ref, dev)
    # The penalty must act on the device exactly as in the reference: the
    # device curvature mismatch matches the reference's.
    (_g, topo_ref, _r), (_g2, topo_dev, _r2) = ref, dev
    m_ref = max(
        seam_curvature_mismatch(topo_ref, si) for si in range(len(topo_ref.seams))
    )
    m_dev = max(
        seam_curvature_mismatch(topo_dev, si) for si in range(len(topo_dev.seams))
    )
    np.testing.assert_allclose(m_dev, m_ref, rtol=1e-4)


def test_parity_fan_fallback():
    ref, dev = _run_pair(_fan_grid)
    _assert_parity(ref, dev)
    _grid, topo, _rep = dev
    assert watertight_mismatch(topo) < 1e-12


def test_parity_seam_ortho():
    angles = {}
    for mode, w in (("off", 0.0), ("penalty", 100.0), ("hard", 0.0)):
        ref, dev = _run_pair(
            lambda: _lr_grid(res=(9, 13), shear=1.0), ortho=mode, ortho_weight=w
        )
        _assert_parity(ref, dev, rtol=1e-6, x_atol=1e-5, monotone=(mode == "off"))
        _grid, topo, _rep = dev
        angles[mode] = max(
            seam_angle_deviation(topo, si, n=17, skip=5).max()
            for si in range(len(topo.seams))
        )
    assert angles["penalty"] < angles["off"]
    assert angles["hard"] <= 1.0


def test_device_sliding_wall_roots():
    """Sliding wall roots: boundary controls ride their entity through the
    run (frozen tangential frames + per-iteration b re-extension), the wall
    fine nodes stay on the entity exactly, and seams stay exactly C1."""
    from egg.geometry.analytic2d import LineSegment
    from egg.smoothing.control_backend import run_control_topo
    from egg.smoothing.control_topology import build_control_topology

    def make():
        grid = _lr_grid(wave=0.3)
        bottom = LineSegment((0.0, 0.0), (4.0, 0.0))
        for bi in range(2):
            dm = grid.block_dof_maps[bi]
            for g in dm[1:-1, 0]:
                grid.dof_constraints[int(g)] = bottom
        _perturb(grid)
        return grid

    grid = make()
    topo = build_control_topology(grid, walls=True)
    assert topo.root_entity  # sliding roots detected
    topo, rep = run_control_topo(grid, IdentityTarget(2), topo=topo, device="cpu")
    assert rep["iters"] > 0
    assert rep["final_mindet"] > 0.0
    assert watertight_mismatch(topo) < 1e-12
    for si in range(len(topo.seams)):
        assert seam_c1_jump(topo, si) < 1e-12
    # Wall fine nodes are exactly on the entity (y = 0 on the segment).
    Xg = np.asarray(grid.global_nodes)
    for bi in range(2):
        for g in grid.block_dof_maps[bi][1:-1, 0]:
            assert abs(Xg[int(g)][1]) < 1e-12

    # Sliding must reach at least the fixed-boundary optimum.
    grid_f = make()
    _t, rep_f = run_control_topo(grid_f, IdentityTarget(2), device="cpu")
    assert rep["final_fine_energy"] <= rep_f["final_fine_energy"] * 1.001 + 1e-12


def test_single_block_segmented_wall():
    """Segmented (B-spline) wall entities run through the multi-block driver
    even on a single-block grid — the frozen frames project host-side, so
    the single-block device wall mode's analytic-only restriction does not
    apply. (The old path still rejects them with a pointer here.)"""
    from egg.geometry.curves2d import BSplineCurve
    from egg.smoothing.control_backend import run_control_topo
    from egg.smoothing.control_topology import build_control_topology
    from egg.topology.builder import TopologyBuilder

    # A gently wavy cubic B-spline as the bottom wall.
    xs = np.linspace(0.0, 4.0, 7)
    ctrl = np.stack([xs, 0.15 * np.sin(np.pi * xs / 4.0)], axis=1)
    n_ctrl, degree = len(ctrl), 3
    knots = np.concatenate(
        [
            np.zeros(degree + 1),
            np.linspace(0, 1, n_ctrl - degree + 1)[1:-1],
            np.ones(degree + 1),
        ]
    )
    wall = BSplineCurve(degree, knots, ctrl)

    b = TopologyBuilder(d=2)
    for name, pos in [
        ("A", tuple(wall.eval(0.0))),
        ("B", tuple(wall.eval(1.0))),
        ("C", (0.0, 2.0)),
        ("D", (4.0, 2.0)),
    ]:
        b.add_corner(name, pos, fixed=True)
    b.add_block("blk", ("A", "C", "B", "D"), (9, 9))
    grid = b.build().initialize_grid()
    dm = grid.block_dof_maps[0]
    for g in dm[1:-1, 0]:
        grid.dof_constraints[int(g)] = wall
    # Put the boundary nodes on the wall before fitting.
    X = np.array(grid.global_nodes)
    for g in dm[1:-1, 0]:
        X[int(g)] = wall.project(X[int(g)])
    grid.global_nodes = X
    for bi, blk in enumerate(grid.blocks):
        blk.nodes[...] = X[grid.block_dof_maps[bi]]
    _perturb(grid)

    topo = build_control_topology(grid, walls=True)
    assert topo.root_entity  # spline wall slides
    topo, rep = run_control_topo(grid, IdentityTarget(2), topo=topo, device="cpu")
    assert rep["iters"] > 0
    assert rep["final_mindet"] > 0.0
    Xg = np.asarray(grid.global_nodes)
    for g in dm[1:-1, 0]:
        p = Xg[int(g)]
        assert np.linalg.norm(wall.project(p) - p) < 1e-9


def test_resample_block_regrid():
    """Algebraic regrid on the stored net: refine 2x and cluster toward the
    wall without a re-solve — valid cells, wall nodes back on the entity,
    and the shared seam stays watertight across independently resampled
    blocks (matching seam parameters)."""
    from egg.geometry.analytic2d import LineSegment
    from egg.smoothing.control_backend import run_control_topo
    from egg.smoothing.control_topology import (
        build_control_topology,
        resample_block,
    )

    grid = _lr_grid(wave=0.3)
    bottom = LineSegment((0.0, 0.0), (4.0, 0.0))
    for bi in range(2):
        dm = grid.block_dof_maps[bi]
        for g in dm[1:-1, 0]:
            grid.dof_constraints[int(g)] = bottom
    _perturb(grid)
    topo = build_control_topology(grid, walls=True)
    topo, _rep = run_control_topo(grid, IdentityTarget(2), topo=topo, device="cpu")

    n0, n1 = grid.blocks[0].logical_shape
    fine_shape = ((n0 - 1) * 2 + 1, (n1 - 1) * 2 + 1)
    # Clustered normal-axis parameters toward the wall (axis 1, side 0).
    t = np.linspace(0.0, 1.0, fine_shape[1])
    cluster = t**1.7
    spacing = [None, cluster]
    Xs = [
        resample_block(topo, bi, node_shape=fine_shape, spacing=spacing)
        for bi in range(2)
    ]
    for Xf in Xs:
        du = np.diff(Xf, axis=0)[:, :-1]
        dv = np.diff(Xf, axis=1)[:-1, :]
        det = du[..., 0] * dv[..., 1] - du[..., 1] * dv[..., 0]
        assert float(det.min()) > 0.0
        # Wall boundary back on the entity at the new sampling.
        assert np.abs(Xf[:, 0, 1]).max() < 1e-9
    # Watertight across the seam: block 0's x=2 face equals block 1's,
    # sampled at the same parameters (seam faces stay pure spline).
    np.testing.assert_allclose(Xs[0][-1], Xs[1][0], atol=1e-12)


def test_good_topo_energy_competitive():
    """The composed run on the circle-in-rectangle example topology (12
    blocks, valence-3/5 fans, sliding circle + channel-wall entities): the
    control solver's fine energy lands within 1.05x of the node smoother,
    watertight, boundary nodes exactly on their entities."""
    import os
    import sys

    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(__file__), "..", "..", "examples", "2D", "circles"
        ),
    )
    from topologies import build_circle_in_rectangle

    from egg.smoothing.control_backend import run_control_topo
    from egg.smoothing.control_topology import (
        build_control_topology,
        default_ctrl_shapes,
    )
    from egg.smoothing.cpp_backend import (
        CppStructuredSweepSession,
        build_block_structured_context,
    )
    from egg.smoothing.solver import build_sweep_context

    def presmoothed_grid(sweeps=100):
        topo_b, _ents = build_circle_in_rectangle()
        grid = topo_b.initialize_grid()
        sess = CppStructuredSweepSession(
            build_sweep_context(grid, IdentityTarget(2)),
            build_block_structured_context(grid),
            np.asarray(grid.global_nodes),
            device="cpu",
        )
        sess.run(sweeps, phase="barrier", omega=0.8)
        grid.global_nodes[:] = sess.get_X()
        for bi, blk in enumerate(grid.blocks):
            blk.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]
        return grid, sess

    # Node baseline: run to convergence on the same start.
    grid_n, sess_n = presmoothed_grid()
    e_prev = None
    for _ in range(30):
        energies, _md = sess_n.run(100, phase="barrier", omega=0.8)
        e_node = float(energies[-1])
        if e_prev is not None and abs(e_prev - e_node) < 1e-10 * abs(e_node):
            break
        e_prev = e_node

    # Control run, net fitted to the presmoothed grid. The default r=4 net
    # is too coarse for the O-ring's boundary layer here (the sliding-wall
    # b feedback needs the spline sag below the fit scale); r=2 resolves it.
    grid_c, _s = presmoothed_grid()
    topo = build_control_topology(grid_c, default_ctrl_shapes(grid_c, r=2), walls=True)
    assert topo.root_entity  # circle + channel walls slide
    assert any(s.fan.any() for s in topo.seams)  # fans exercised
    topo, rep = run_control_topo(grid_c, IdentityTarget(2), topo=topo, device="cpu")
    assert rep["iters"] > 0
    assert rep["final_mindet"] > 0.0
    assert rep["final_fine_energy"] <= 1.05 * e_node + 1e-12
    assert watertight_mismatch(topo) < 1e-12
    # Every constrained boundary node sits exactly on its entity (the final
    # frame rebuild reprojects; the spline sag is absorbed by b).
    Xg = np.asarray(grid_c.global_nodes)
    for g, ent in grid_c.dof_constraints.items():
        p = Xg[int(g)]
        assert np.linalg.norm(ent.project(p) - p) < 1e-9


def test_cubed_sphere_edge_fans_and_energy():
    """6-block cubed-sphere shell: the valence-3 radial edges at the cube
    corners are 3D edge fans (elimination drops to the penalty on the
    adjacent lattice bands — without this the signed union-find rejects the
    topology outright), sphere/plane walls slide, and the control solver
    reaches the node smoother's converged energy. The stored net regrids
    validly at 2x."""
    import importlib.util
    import os

    from egg.geometry.control_net import tensor_map
    from egg.smoothing.control_backend import run_control_topo
    from egg.smoothing.control_topology import build_control_topology
    from egg.smoothing.cpp_backend import (
        CppStructuredSweepSession,
        build_block_structured_context,
    )
    from egg.smoothing.solver import build_sweep_context

    ex = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "examples",
        "3D",
        "cubed_sphere",
        "cubed_sphere.py",
    )
    spec = importlib.util.spec_from_file_location("cubed_sphere_ex", ex)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    grid = mod.cubed_sphere(4, 8, r0=0.5).build().initialize_grid()

    # Node baseline to convergence.
    sess = CppStructuredSweepSession(
        build_sweep_context(grid, IdentityTarget(3)),
        build_block_structured_context(grid),
        np.asarray(grid.global_nodes),
        device="cpu",
    )
    e_prev = None
    for _ in range(30):
        energies, _md = sess.run(200, phase="barrier", omega=0.8)
        e_node = float(energies[-1])
        if e_prev is not None and abs(e_prev - e_node) < 1e-9 * abs(e_node):
            break
        e_prev = e_node
    grid.global_nodes[:] = sess.get_X()
    for bi, blk in enumerate(grid.blocks):
        blk.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]

    topo = build_control_topology(grid, walls=True)
    # Edge fans flagged: every seam face has bands next to its two radial
    # fan edges, and interior eliminations survive between them.
    assert all(s.fan.any() for s in topo.seams)
    assert any(s.elim.any() for s in topo.seams)
    assert topo.root_entity  # sphere + cube-face walls slide

    topo, rep = run_control_topo(grid, IdentityTarget(3), topo=topo, device="cpu")
    assert rep["iters"] > 0
    assert rep["final_mindet"] > 0.0
    assert rep["final_fine_energy"] <= 1.05 * e_node + 1e-12
    assert watertight_mismatch(topo) < 1e-12
    Xg = np.asarray(grid.global_nodes)
    for g, ent in grid.dof_constraints.items():
        p = Xg[int(g)]
        assert np.linalg.norm(ent.project(p) - p) < 1e-9

    # Algebraic 2x regrid stays valid (per-block, on the stored net + b=0
    # interior check via cell triple products).
    for bi, cmap in enumerate(topo.cmaps):
        fine_shape = tuple((s - 1) * 2 + 1 for s in cmap.node_shape)
        fine = tensor_map(fine_shape, cmap.ctrl_shape, degree=3)
        Xf = fine.prolong(topo.block_C(bi))
        du = np.diff(Xf, axis=0)[:, :-1, :-1]
        dv = np.diff(Xf, axis=1)[:-1, :, :-1]
        dw = np.diff(Xf, axis=2)[:-1, :-1, :]
        det = np.einsum("...i,...i->...", du, np.cross(dv, dw))
        assert float(det.min()) > 0.0


def test_device_3d_two_block_run():
    """The device path is D-general: a rotated/flipped 3D seam relaxes with
    exact C1 held through the run (no NumPy reference exists in 3D)."""
    from egg.smoothing.control_backend import run_control_topo

    grid = _slab3_grid(rotated=True)
    _perturb(grid)
    topo, rep = run_control_topo(grid, IdentityTarget(3), device="cpu")
    assert rep["iters"] > 0
    e = np.asarray(rep["energies"])
    assert np.all(np.isfinite(e))
    assert np.all(np.diff(e) <= 1e-12)
    assert rep["final_mindet"] > 0.0
    assert watertight_mismatch(topo) < 1e-12
    for si in range(len(topo.seams)):
        assert seam_c1_jump(topo, si) < 1e-12


def test_device_3d_seam_ortho_hard():
    """Hard seam orthogonality in 3D: the sheared slab's crossing lines are
    driven to the seam normal (both sides at once through the shared T)."""
    from egg.smoothing.control_backend import run_control_topo

    def make():
        # Enough tangential resolution that the interior Greville window is
        # governed by snapped (free) T's — the seam-endpoint T's on the outer
        # boundary are fixed by design, so samples near the face edge always
        # carry their direction (same reason the 2D criterion skips the
        # fixed-corner window).
        grid = _slab3_grid(res=(4, 9, 9))
        gn = np.array(grid.global_nodes)
        gn[:, 0] += 0.8 * gn[:, 1]  # shear: the shape optimum is non-orthogonal
        grid.global_nodes = gn
        for bi, blk in enumerate(grid.blocks):
            blk.nodes[...] = gn[grid.block_dof_maps[bi]]
        return grid

    grid_off = make()
    topo_off, _ = run_control_topo(grid_off, IdentityTarget(3), device="cpu")
    a_off = seam_angle_deviation(topo_off, 0, n=13, skip=5).max()

    grid_hard = make()
    topo_hard, rep = run_control_topo(
        grid_hard, IdentityTarget(3), device="cpu", ortho="hard"
    )
    assert rep["final_mindet"] > 0.0
    a_hard = seam_angle_deviation(topo_hard, 0, n=13, skip=5).max()
    assert a_hard <= 1.0
    assert a_hard < a_off
    for si in range(len(topo_hard.seams)):
        assert seam_c1_jump(topo_hard, si) < 1e-12


def test_nurbs_surface_wall_runs_cxx_frames():
    """A NURBS-surface wall (rational, arena payload) runs the in-session C++
    frame loop and agrees with the Python-frame fallback: valid result, wall
    nodes exactly on the surface, energies close (the two paths differ only
    in whose Newton projects the frames)."""
    import egg.smoothing.control_backend as cb
    from egg.geometry.surfaces3d import BSplineSurface
    from egg.init.tfi import tfi_fill_interior
    from egg.smoothing.control_backend import run_control_topo
    from egg.smoothing.control_topology import build_control_topology
    from egg.topology.builder import TopologyBuilder

    def bump_surface():
        n = 5
        knots = np.concatenate([np.zeros(4), [0.5], np.ones(4)])
        xs = np.linspace(0.0, 4.0, n)
        cx, cy = np.meshgrid(xs, xs, indexing="ij")
        cz = 0.5 * np.sin(np.pi * cx / 4.0) * np.sin(np.pi * cy / 4.0)
        ctrl = np.stack([cx, cy, cz], axis=-1)
        w = np.ones((n, n))
        w[2, 2] = 1.2  # rational: exercises the weights arena field
        # A near-full rectangular UV trim loop: exercises the trim-polygon
        # arena fields of the frame-wire blob (feet clamp identically on the
        # Python and C++ sides).
        m = 0.01
        trim = [np.array([[m, m], [1.0 - m, m], [1.0 - m, 1.0 - m], [m, 1.0 - m]])]
        return BSplineSurface(3, 3, knots, knots, ctrl, weights=w, trim=trim)

    def make():
        surf = bump_surface()
        b = TopologyBuilder(d=3)
        for i, x in enumerate((0.0, 4.0)):
            for j, y in enumerate((0.0, 4.0)):
                for k, z in enumerate((0.0, 3.0)):
                    b.add_corner(f"c{i}{j}{k}", (x, y, z), fixed=True)
        b.add_block(
            "A",
            ("c000", "c001", "c010", "c011", "c100", "c101", "c110", "c111"),
            (7, 7, 6),
        )
        grid = b.build().initialize_grid()
        blk = grid.blocks[0]
        X = np.asarray(blk.nodes, dtype=float)
        # Bottom face (axis 2, side 0) onto the surface, TFI the interior.
        feet = np.array([surf.project(p) for p in X[:, :, 0].reshape(-1, 3)])
        X[:, :, 0] = feet.reshape(X[:, :, 0].shape)
        interior = np.full_like(X, np.nan)
        for ax in range(3):
            sl = [slice(None)] * 3
            for side in (0, -1):
                sl[ax] = side
                interior[tuple(sl)] = X[tuple(sl)]
            sl[ax] = slice(None)
        blk.nodes[...] = interior
        tfi_fill_interior(blk)
        grid.global_nodes[grid.block_dof_maps[0].reshape(-1)] = blk.nodes.reshape(-1, 3)
        dm = grid.block_dof_maps[0]
        for g in dm[1:-1, 1:-1, 0].ravel():
            grid.dof_constraints[int(g)] = surf
        _perturb(grid, scale=0.01)
        return grid, surf

    reps = {}
    grids = {}
    for mode in ("cxx", "python"):
        grid, surf = make()
        topo = build_control_topology(grid, walls=True)
        assert topo.root_entity  # sliding surface roots detected
        if mode == "python":
            orig = cb._attach_slide_wire

            def raise_ni(*a, **k):
                raise NotImplementedError

            cb._attach_slide_wire = raise_ni
        try:
            topo, rep = run_control_topo(
                grid, IdentityTarget(3), topo=topo, device="cpu"
            )
        finally:
            if mode == "python":
                cb._attach_slide_wire = orig
        assert rep["final_mindet"] > 0.0
        assert rep["iters"] > 0
        # The C++ path must actually be the in-session frame loop.
        if mode == "cxx":
            assert rep["session"]._ctrl_cxx_frames
        else:
            assert not rep["session"]._ctrl_cxx_frames
        # Wall fine nodes end exactly on the surface.
        Xg = np.asarray(grid.global_nodes)
        dm = grid.block_dof_maps[0]
        for g in dm[1:-1, 1:-1, 0].ravel()[::7]:
            p = Xg[int(g)]
            assert np.linalg.norm(surf.project(p) - p) < 1e-8
        reps[mode] = rep
        grids[mode] = grid

    # The in-session loop models db/dC and keeps one continuous GN run, so
    # it must do at least as well as the per-iteration Python-frame cadence
    # (it does noticeably better on this curved-surface fixture).
    e_cxx = reps["cxx"]["final_fine_energy"]
    e_py = reps["python"]["final_fine_energy"]
    assert e_cxx <= e_py * 1.001 + 1e-12
