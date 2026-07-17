# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Multi-block control topology: the acceptance criteria for seams.

- watertight: shared face nodes bitwise-identical (one owner per node) and
  both sides' spline evaluations equal to machine precision;
- exact C1: zero cross-seam jump of the per-global-cell normal derivative at
  eliminated indices, including through a regular 2x2 crossing (twist DOFs);
- soft C2: curvature mismatch drops >= 10x with the penalty on;
- seam orthogonality: <= 1 degree from normal in hard mode, monotone in the
  weight in penalty mode (away from the fixed-corner window);
- fans: valence != 4 corners fall back to shared controls + C1 penalty;
- energy: within 1.10x of the sequential node smoother on the same grid.
"""

import numpy as np

from egg.init.tfi import tfi_fill_interior
from egg.smoothing import batch as _batch
from egg.smoothing.control_topology import (
    _t_row,
    build_control_topology,
    run_control_topo_ref,
    seam_angle_deviation,
    seam_c1_jump,
    seam_curvature_mismatch,
    watertight_mismatch,
)
from egg.smoothing.objective import assemble_energy_vec
from egg.smoothing.solver import build_sweep_context
from egg.smoothing.targets import IdentityTarget
from egg.topology.builder import TopologyBuilder

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _sync(grid):
    d = grid.blocks[0].nodes.shape[-1]
    for bi, blk in enumerate(grid.blocks):
        grid.global_nodes[grid.block_dof_maps[bi].reshape(-1)] = blk.nodes.reshape(
            -1, d
        )


def _fix_outer_boundary(grid):
    """Freeze fine nodes on outer (non-interface) faces.

    The control solver always holds the outer boundary exactly (fixed
    boundary controls + the TFI correction b); without this the node
    baseline would smooth the boundary shape away and the energy
    comparison would be between different DOF sets.
    """
    d = grid.blocks[0].nodes.ndim - 1
    face_count = {}
    for bi in range(len(grid.blocks)):
        dm = grid.block_dof_maps[bi]
        for axis in range(d):
            for side in (0, -1):
                sl = [slice(None)] * d
                sl[axis] = side
                for g in np.unique(dm[tuple(sl)]):
                    face_count[int(g)] = face_count.get(int(g), set()) | {bi}
    free = np.array(grid.free_mask)
    for g, owners in face_count.items():
        if len(owners) < 2:
            free[g] = False
    grid.free_mask = free


def _perturb(grid, seed=3, scale=0.02):
    rng = np.random.default_rng(seed)
    free = np.array(grid.free_mask)
    gn = np.array(grid.global_nodes)
    d = gn.shape[1]
    for i in range(grid.global_node_count):
        if free[i] and i not in grid.dof_constraints:
            gn[i] += rng.normal(0, scale, d)
    grid.global_nodes = gn
    for bi, blk in enumerate(grid.blocks):
        blk.nodes[...] = gn[grid.block_dof_maps[bi]]


def _lr_grid(res=(9, 9), wave=0.0, shear=0.0, flip_r=False):
    """Two blocks sharing the x=2 seam; optional wavy top edge (a nontrivial
    curved optimum for the C2 tests) and x-shear (a TMOP optimum that is
    deliberately non-orthogonal at the seam, for the ortho tests)."""
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
    if flip_r:
        # Reversed tangential axis on the R side of the seam.
        b.add_block("R", ("C", "B", "F", "E"), res)
    else:
        b.add_block("R", ("B", "C", "E", "F"), res)
    grid = b.build().initialize_grid()

    if wave:
        for bi, blk in enumerate(grid.blocks):
            nodes = blk.nodes
            top = nodes[:, -1] if not (flip_r and bi == 1) else nodes[:, 0]
            top[:, 1] = 2.0 + wave * np.sin(np.pi * top[:, 0] / 4.0)
            # Redistribute each boundary face linearly is unnecessary; only
            # the interior needs re-filling from the new boundary.
            interior = np.full_like(nodes, np.nan)
            interior[0, :] = nodes[0, :]
            interior[-1, :] = nodes[-1, :]
            interior[:, 0] = nodes[:, 0]
            interior[:, -1] = nodes[:, -1]
            blk.nodes[...] = interior
            tfi_fill_interior(blk)
        _sync(grid)
    if shear:
        gn = np.array(grid.global_nodes)
        gn[:, 0] += shear * gn[:, 1]
        grid.global_nodes = gn
        for bi, blk in enumerate(grid.blocks):
            blk.nodes[...] = gn[grid.block_dof_maps[bi]]
    _fix_outer_boundary(grid)
    return grid


def _quad_grid(res=(7, 7)):
    """2x2 blocks meeting at one interior crossing (regular valence 4)."""
    b = TopologyBuilder(d=2)
    for i, x in enumerate((0.0, 2.0, 4.0)):
        for j, y in enumerate((0.0, 2.0, 4.0)):
            b.add_corner(f"P{i}{j}", (x, y), fixed=True)
    b.add_block("BL", ("P00", "P01", "P10", "P11"), res)
    b.add_block("BR", ("P10", "P11", "P20", "P21"), res)
    b.add_block("TL", ("P01", "P02", "P11", "P12"), res)
    b.add_block("TR", ("P11", "P12", "P21", "P22"), res)
    grid = b.build().initialize_grid()
    _fix_outer_boundary(grid)
    return grid


def _fan_grid(res=(7, 7)):
    """Three blocks around one interior corner (valence 3: a singular fan)."""
    b = TopologyBuilder(d=2)
    b.add_corner("O", (0.0, 0.0), fixed=True)
    r_v, r_m = 2.0, 1.4
    for i, ang in enumerate((90.0, 210.0, 330.0)):
        a = np.radians(ang)
        b.add_corner(f"V{i}", (r_v * np.cos(a), r_v * np.sin(a)), fixed=True)
    for i, ang in enumerate((150.0, 270.0, 30.0)):
        a = np.radians(ang)
        # M{i} sits between V{i} and V{i+1}.
        b.add_corner(f"M{i}", (r_m * np.cos(a), r_m * np.sin(a)), fixed=True)
    for i in range(3):
        b.add_block(f"blk{i}", ("O", f"M{i}", f"M{(i - 1) % 3}", f"V{i}"), res)
    grid = b.build().initialize_grid()
    _fix_outer_boundary(grid)
    return grid


def _slab3_grid(res=(4, 5, 6), rotated=False):
    """Two 3D blocks sharing the x=2 face. ``rotated`` attaches B with its
    axis 1 as the seam normal, axis 0 along A's y, and axis 2 reversed in z
    (a nontrivial tangential permutation + flip)."""
    b = TopologyBuilder(d=3)
    for x, tag in ((0.0, "a"), (2.0, "s"), (4.0, "b")):
        for j in (0, 1):
            for k in (0, 1):
                b.add_corner(f"{tag}{j}{k}", (x, 2.0 * j, 2.0 * k), fixed=True)
    b.add_block(
        "A",
        ("a00", "a01", "a10", "a11", "s00", "s01", "s10", "s11"),
        res,
    )
    if rotated:
        b.add_block(
            "B",
            ("s01", "s00", "b01", "b00", "s11", "s10", "b11", "b10"),
            (res[1], res[0], res[2]),
        )
    else:
        b.add_block(
            "B",
            ("s00", "s01", "s10", "s11", "b00", "b01", "b10", "b11"),
            res,
        )
    grid = b.build().initialize_grid()
    _fix_outer_boundary(grid)
    return grid


def _oct_grid(res=(4, 4, 4)):
    """2x2x2 blocks meeting at one interior corner (regular valence 8):
    seam crossings in every axis pair plus the triple corner."""
    b = TopologyBuilder(d=3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                b.add_corner(f"P{i}{j}{k}", (2.0 * i, 2.0 * j, 2.0 * k), fixed=True)
    for bi in (0, 1):
        for bj in (0, 1):
            for bk in (0, 1):
                corners = tuple(
                    f"P{bi + ci}{bj + cj}{bk + ck}"
                    for ci in (0, 1)
                    for cj in (0, 1)
                    for ck in (0, 1)
                )
                b.add_block(f"B{bi}{bj}{bk}", corners, res)
    grid = b.build().initialize_grid()
    _fix_outer_boundary(grid)
    return grid


def _node_baseline(grid, target_fn, sweeps=200, metric="shape_2d"):
    """Sequential per-DOF Newton over every free unconstrained node."""
    ctx = build_sweep_context(grid, target_fn)
    free = [
        i
        for i in range(grid.global_node_count)
        if grid.free_mask[i] and i not in grid.dof_constraints
    ]
    X = grid.global_nodes
    for _ in range(sweeps):
        moved = 0.0
        for dof in free:
            p = ctx.dof_patches[int(dof)]
            g, H = _batch.dof_grad_hess(
                X,
                p["gc"],
                p["gn0"],
                p["gn1"],
                p["s0"],
                p["s1"],
                p["W_inv"],
                p["role"],
                p["J"],
                metric=metric,
            )
            try:
                step = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                continue
            e0, _ = _batch.energy_and_mindet(
                X,
                p["gc"],
                p["gn0"],
                p["gn1"],
                p["s0"],
                p["s1"],
                p["W_inv"],
                metric=metric,
            )
            alpha = 1.0
            while alpha >= 1e-4:
                X_try = X.copy()
                X_try[dof] += alpha * step
                e, md = _batch.energy_and_mindet(
                    X_try,
                    p["gc"],
                    p["gn0"],
                    p["gn1"],
                    p["s0"],
                    p["s1"],
                    p["W_inv"],
                    metric=metric,
                )
                if np.isfinite(e) and e <= e0 + 1e-12 and md > 0.0:
                    X[dof] += alpha * step
                    moved += alpha * float(np.linalg.norm(step))
                    break
                alpha *= 0.5
        if moved < 1e-12:
            break
    for bi, block in enumerate(grid.blocks):
        block.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]
    return ctx


def _global_energy(grid, ctx, metric="shape_2d"):
    st = ctx.energy_stencil
    return assemble_energy_vec(
        grid.global_nodes,
        st["gc"],
        st["gn0"],
        st["gn1"],
        st["s0"],
        st["s1"],
        st["W_inv"],
        metric=metric,
    )


# --------------------------------------------------------------------------- #
# Structure: discovery, orientation, and the reduction null-space property
# --------------------------------------------------------------------------- #


def test_seam_discovery_and_orientation():
    topo = build_control_topology(_lr_grid())
    assert len(topo.seams) == 1
    s = topo.seams[0]
    assert (s.axis_a, s.side_a, s.axis_b, s.side_b) == (0, 1, 0, 0)
    assert s.flips == (False,)

    topo_f = build_control_topology(_lr_grid(flip_r=True))
    assert len(topo_f.seams) == 1
    assert topo_f.seams[0].flips == (True,)


def _c0_c1_rows(topo):
    """Explicit constraint rows over the stacked controls: shared-face control
    equality (C0) and the first-derivative matching condition (C1)."""
    rows = []
    for s in topo.seams:
        csa, csb = topo.ctrl_shapes[s.a], topo.ctrl_shapes[s.b]
        fa = 0 if s.side_a == 0 else csa[s.axis_a] - 1
        fb = 0 if s.side_b == 0 else csb[s.axis_b] - 1
        qb = 1 if s.side_b == 0 else csb[s.axis_b] - 2
        taxes_a = [k for k in range(topo.d) if k != s.axis_a]
        taxes_b = [k for k in range(topo.d) if k != s.axis_b]

        def flat(bi, axis, taxes, normal_i, tg, cs):
            idx = [0] * topo.d
            idx[axis] = normal_i
            for m, k in enumerate(taxes):
                idx[k] = tg[m]
            return topo.block_offsets[bi] + int(np.ravel_multi_index(idx, cs))

        from itertools import product as _product

        for tang in _product(*(range(n) for n in s.tang_shape_a)):
            tb = s.map_tang(tang)
            row = np.zeros(topo.R.shape[0])
            row[flat(s.a, s.axis_a, taxes_a, fa, tang, csa)] += 1.0
            row[flat(s.b, s.axis_b, taxes_b, fb, tb, csb)] -= 1.0
            rows.append(row)  # C0
            if s.elim[tang]:
                row = _t_row(topo, s, tang).copy()
                row[flat(s.b, s.axis_b, taxes_b, fb, tb, csb)] += s.mu_b
                row[flat(s.b, s.axis_b, taxes_b, qb, tb, csb)] -= s.mu_b
                rows.append(row)  # C1
    return np.asarray(rows)


def test_reduction_annihilates_constraints():
    """R parameterizes the C0+C1 manifold exactly: every constraint row lies
    in the left null space of R (any reduced state satisfies them)."""
    for grid in (_lr_grid(), _lr_grid(flip_r=True), _quad_grid()):
        topo = build_control_topology(grid)
        rows = _c0_c1_rows(topo)
        assert np.abs(topo.R.T @ rows.T).max() < 1e-12


def test_initial_state_watertight_and_c1():
    _seed_all = 5
    grid = _quad_grid()
    _perturb(grid, seed=_seed_all)
    topo = build_control_topology(grid)
    assert watertight_mismatch(topo) < 1e-12
    for si in range(len(topo.seams)):
        assert seam_c1_jump(topo, si) < 1e-12


def test_wall_detection_and_sliding_roots():
    """Outer faces riding one entity classify their plain-P roots as sliding;
    everything else stays fixed. (The runner-side tangential frames are
    pending — both runners refuse a sliding topology loudly.)"""
    import pytest

    from egg.geometry.analytic2d import LineSegment

    grid = _lr_grid()
    bottom = LineSegment((0.0, 0.0), (4.0, 0.0))
    d = 2
    for bi in range(2):
        dm = grid.block_dof_maps[bi]
        for g in dm[1:-1, 0]:  # bottom-face interior fine nodes
            grid.dof_constraints[int(g)] = bottom

    topo = build_control_topology(grid, walls=True)
    # Both blocks' bottom faces detected on the one entity.
    assert set(topo.wall_faces) == {(0, 1, 0), (1, 1, 0)}
    assert all(e is bottom for e in topo.wall_faces.values())
    # Sliding roots exist, started on the entity, and are not free.
    assert topo.root_entity
    for col, ent in topo.root_entity.items():
        assert ent is bottom
        assert not topo.root_free[col]
        assert abs(topo.q[col][1]) < 1e-12  # projected onto y = 0
    # A topology without walls has none.
    assert not build_control_topology(_lr_grid()).root_entity

    with pytest.raises(NotImplementedError, match="sliding wall roots"):
        run_control_topo_ref(grid, IdentityTarget(d), topo=topo)


# --------------------------------------------------------------------------- #
# 3D structure: same criteria on 3D seams (the metric-free part of the
# machinery is dimension-general; the NumPy runner itself is 2D-only).
# --------------------------------------------------------------------------- #


def test_seam_discovery_and_orientation_3d():
    topo = build_control_topology(_slab3_grid())
    assert len(topo.seams) == 1
    s = topo.seams[0]
    assert (s.axis_a, s.side_a, s.axis_b, s.side_b) == (0, 1, 0, 0)
    assert tuple(s.perm) == (0, 1)
    assert tuple(s.flips) == (False, False)

    topo_r = build_control_topology(_slab3_grid(rotated=True))
    assert len(topo_r.seams) == 1
    s = topo_r.seams[0]
    assert (s.axis_a, s.side_a, s.axis_b, s.side_b) == (0, 1, 1, 0)
    assert tuple(s.perm) == (0, 1)
    assert tuple(s.flips) == (False, True)


def test_reduction_annihilates_constraints_3d():
    for grid in (_slab3_grid(), _slab3_grid(rotated=True), _oct_grid()):
        topo = build_control_topology(grid)
        rows = _c0_c1_rows(topo)
        assert np.abs(topo.R.T @ rows.T).max() < 1e-12


def test_initial_state_watertight_and_c1_3d():
    for grid in (_slab3_grid(rotated=True), _oct_grid()):
        _perturb(grid, seed=5)
        topo = build_control_topology(grid)
        assert watertight_mismatch(topo) < 1e-12
        for si in range(len(topo.seams)):
            assert seam_c1_jump(topo, si) < 1e-12


# --------------------------------------------------------------------------- #
# Convergence + exact C1 through a regular crossing
# --------------------------------------------------------------------------- #


def test_crossing_run_keeps_exact_c1():
    grid = _quad_grid()
    _perturb(grid)
    topo, rep = run_control_topo_ref(grid, IdentityTarget(2))
    assert rep["iters"] > 0
    assert all(np.isfinite(rep["energies"]))
    assert np.all(np.diff(rep["energies"]) <= 1e-12)
    assert all(md > 0.0 for md in rep["mindets"])
    assert watertight_mismatch(topo) < 1e-12
    for si in range(len(topo.seams)):
        assert seam_c1_jump(topo, si) < 1e-12


def test_energy_competitive_with_node_smoother():
    target = IdentityTarget(2)
    grid_node = _lr_grid(wave=0.3)
    _perturb(grid_node)
    ctx_node = _node_baseline(grid_node, target)
    e_node = _global_energy(grid_node, ctx_node)

    grid_ctrl = _lr_grid(wave=0.3)
    _perturb(grid_ctrl)
    topo, rep = run_control_topo_ref(grid_ctrl, target)
    e_ctrl = rep["final_fine_energy"]
    assert rep["final_mindet"] > 0.0
    assert e_ctrl <= 1.10 * e_node + 1e-12


# --------------------------------------------------------------------------- #
# Soft C2
# --------------------------------------------------------------------------- #


def test_c2_penalty_reduces_curvature_mismatch():
    target = IdentityTarget(2)
    mismatches = {}
    for w in (0.0, 100.0):
        grid = _lr_grid(wave=0.3)
        _perturb(grid)
        topo, _rep = run_control_topo_ref(grid, target, c2_weight=w)
        mismatches[w] = max(
            seam_curvature_mismatch(topo, si) for si in range(len(topo.seams))
        )
        # C1 must stay exact regardless of the C2 dial.
        for si in range(len(topo.seams)):
            assert seam_c1_jump(topo, si) < 1e-12
    assert mismatches[100.0] <= mismatches[0.0] / 10.0


# --------------------------------------------------------------------------- #
# Seam orthogonality
# --------------------------------------------------------------------------- #


def _seam_angles(topo, skip=5):
    return max(
        seam_angle_deviation(topo, si, n=17, skip=skip).max()
        for si in range(len(topo.seams))
    )


def test_seam_ortho_penalty_monotone_and_hard():
    target = IdentityTarget(2)
    angles = {}
    for mode, w in (("off", 0.0), ("penalty", 1.0), ("penalty", 100.0), ("hard", 0.0)):
        grid = _lr_grid(res=(9, 13), shear=1.0)
        _perturb(grid)
        topo, rep = run_control_topo_ref(
            grid, target, ortho=mode if mode != "off" else "off", ortho_weight=w
        )
        assert rep["final_mindet"] > 0.0
        angles[(mode, w)] = _seam_angles(topo)
    # The sheared optimum is non-orthogonal; the dial must fight it monotonely.
    assert angles[("penalty", 1.0)] < angles[("off", 0.0)]
    assert angles[("penalty", 100.0)] < angles[("penalty", 1.0)]
    assert angles[("hard", 0.0)] <= 1.0


# --------------------------------------------------------------------------- #
# Fan fallback (valence-3 interior corner)
# --------------------------------------------------------------------------- #


def test_fan_fallback_c0_and_penalty():
    grid = _fan_grid()
    _perturb(grid)
    topo = build_control_topology(grid)
    assert len(topo.seams) == 3
    # Every seam dropped the elimination on the window next to the fan corner
    # (corner index + its neighbour) and nowhere else.
    for s in topo.seams:
        assert s.fan.sum() == 2
        assert not s.elim[np.nonzero(s.fan)].any()
        assert s.elim[~s.fan].all()

    topo, rep = run_control_topo_ref(grid, IdentityTarget(2), topo=topo)
    assert rep["iters"] > 0
    assert np.all(np.diff(rep["energies"]) <= 1e-12)
    assert rep["final_mindet"] > 0.0
    assert watertight_mismatch(topo) < 1e-12


# --------------------------------------------------------------------------- #
# Fan-local net refinement (knot insertion at fan-touching axis ends)
# --------------------------------------------------------------------------- #


def test_fan_refine_bumps_chains_and_stays_valid():
    from egg.smoothing.control_topology import _corner_mindet

    grid = _fan_grid(res=(13, 13))
    t0 = build_control_topology(grid)
    t2 = build_control_topology(grid, fan_refine=2)
    # the fan-touching axis chains carry more controls...
    assert sum(sum(s) for s in t2.ctrl_shapes) > sum(sum(s) for s in t0.ctrl_shapes)
    # ...knot vectors match the counts, seams conform (builder validates),
    # the net stays watertight, and the insertion-based initial fit is valid.
    for bi, cs in enumerate(t2.ctrl_shapes):
        for k in range(2):
            assert len(t2.cmaps[bi].knots[k]) == cs[k] + 4
    assert watertight_mismatch(t2) < 1e-9
    d = t2.d
    C = np.asarray(t2.R @ t2.q).reshape(-1, d)
    for bi in range(len(t2.ctrl_shapes)):
        sl = slice(t2.block_offsets[bi], t2.block_offsets[bi + 1])
        X = t2.cmaps[bi].prolong(C[sl].reshape(t2.ctrl_shapes[bi] + (d,)))
        assert _corner_mindet(X + t2.b_fields[bi]) > 0.0


def test_fan_refine_energy_stays_competitive():
    """Refinement changes the stationary-point structure (GN lands in the
    nearest basin), so per-fixture fine energy can move a little either way;
    it must stay competitive and valid. The quality gains are asserted at
    the pipeline level on the real fixtures."""
    target = IdentityTarget(2)
    finals = {}
    for fr in (0, 2):
        grid = _fan_grid(res=(13, 13))
        _perturb(grid)
        topo = build_control_topology(grid, fan_refine=fr)
        _t, rep = run_control_topo_ref(grid, target, topo=topo)
        assert rep["final_mindet"] > 0.0
        assert rep["converged"]
        finals[fr] = rep["final_fine_energy"]
    assert finals[2] <= finals[0] * 1.05 + 1e-12


# --------------------------------------------------------------------------- #
# Interior line-straightness rows
# --------------------------------------------------------------------------- #


def test_smooth_rows_annihilate_straight_uneven_nets():
    from egg.smoothing.control_topology import _smooth_rows

    grid = _quad_grid(res=(9, 9))
    # straight grid lines at nonuniform spacing: squash x (lines stay straight)
    for blk in grid.blocks:
        X = np.asarray(blk.nodes, dtype=float)
        X[..., 0] = (X[..., 0] / 4.0) ** 2 * 4.0
        blk.nodes[...] = X
    _sync(grid)
    topo = build_control_topology(grid, fit_spacing="chord")
    qflat = topo.q.reshape(-1)
    vals = [abs(float(rowv @ qflat)) for rowv in _smooth_rows(topo)]
    assert vals and max(vals) < 1e-6

    # bowing one interior control produces a nonzero straightness residual
    free = np.flatnonzero(topo.root_free)
    topo.q[free[len(free) // 2]] += (0.05, 0.05)
    qflat = topo.q.reshape(-1)
    vals_bowed = [abs(float(rowv @ qflat)) for rowv in _smooth_rows(topo)]
    assert max(vals_bowed) > 1e-3


def test_smooth_weight_straightens_interior_lines():
    from egg.smoothing.control_topology import _smooth_rows

    target = IdentityTarget(2)
    bows = {}
    for w in (0.0, 10.0):
        grid = _lr_grid(res=(13, 13))
        _perturb(grid, scale=0.05)
        topo, rep = run_control_topo_ref(grid, target, smooth_weight=w)
        assert rep["final_mindet"] > 0.0
        qflat = topo.q.reshape(-1)
        bows[w] = max(abs(float(rowv @ qflat)) for rowv in _smooth_rows(topo))
    assert bows[10.0] < bows[0.0]
