# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Control-net wall constraints: sliding, exactness, and orthogonality.

Acceptance criteria for the wall machinery (NumPy reference, curved wall):
  (i)   wall boundary nodes lie on the CAD entity to projection tolerance;
  (ii)  hard mode: wall-normal spline derivative within 1 degree of the wall
        normal (away from corners), even on a sheared domain whose shape
        optimum is deliberately non-orthogonal;
  (iii) penalty mode: the angle error improves monotonically with the weight;
  (iv)  the per-leg taper relaxes the constraint locally (the capsule dial);
  (v)   energy decreases and min det A stays positive throughout.

Angles are measured on the analytic spline derivative dX/dn at the wall
(:func:`egg.smoothing.control_wall.wall_angle_deviation`) — the quantity the
leg conditions control and an algebraic regrid inherits; the fine first-cell
chord additionally carries cell-height truncation and the Boolean-sum
gradient.
"""

from __future__ import annotations

import numpy as np

from egg.geometry.curves2d import CircleArc
from egg.init.tfi import tfi_fill_interior
from egg.smoothing.control_wall import (
    WallSpec,
    build_control_wall_ref,
    run_control_wall_ref,
    wall_angle_deviation,
)
from egg.smoothing.targets import IdentityTarget
from egg.topology.builder import TopologyBuilder

# --------------------------------------------------------------------------- #
# Fixture: a square with a gently bulging circular-arc bottom wall
# --------------------------------------------------------------------------- #


def _make_arc():
    # Chord (0,0)-(4,0), sagitta 0.35 bulging upward (into the domain).
    sag = 0.35
    r = (2.0**2 + sag**2) / (2.0 * sag)
    center = np.array([2.0, sag - r])
    t0 = float(np.arctan2(0.0 - center[1], 0.0 - center[0]))
    t1 = float(np.arctan2(0.0 - center[1], 4.0 - center[0]))
    return CircleArc(center, r, t0, t1)


def _make_arc_grid(n=17, perturb=0.03, seed=7):
    """Unit-ish square, bottom edge on the arc, TFI interior, perturbed."""
    arc = _make_arc()
    builder = TopologyBuilder(d=2)
    for name, pos in [
        ("A", (0.0, 0.0)),
        ("B", (4.0, 0.0)),
        ("C", (4.0, 4.0)),
        ("D", (0.0, 4.0)),
    ]:
        builder.add_corner(name, pos, fixed=True)
    builder.add_block("main", ("A", "D", "B", "C"), (n, n))
    topo = builder.build()
    topo.initialize_grid()
    grid = topo.grid
    block = grid.blocks[0]

    # Bottom edge (axis 1, side 0) on the arc at uniform angles, then re-TFI.
    nodes = block.nodes
    ts = np.linspace(arc.t0, arc.t1, nodes.shape[0])
    nodes[:, 0] = arc.eval_many(ts)
    interior = np.full_like(nodes, np.nan)
    interior[0, :] = nodes[0, :]
    interior[-1, :] = nodes[-1, :]
    interior[:, 0] = nodes[:, 0]
    interior[:, -1] = nodes[:, -1]
    block.nodes[...] = interior
    tfi_fill_interior(block)

    rng = np.random.default_rng(seed)
    block.nodes[1:-1, 1:-1] += perturb * rng.standard_normal(
        block.nodes[1:-1, 1:-1].shape
    )
    grid.global_nodes[grid.block_dof_maps[0].reshape(-1)] = block.nodes.reshape(-1, 2)
    return grid, arc


def _make_sheared_arc_grid(shear=1.5, **kw):
    """The arc grid sheared in x with height: the shape optimum near the wall
    is then deliberately NON-orthogonal, so the ortho dial has real work."""
    grid, arc = _make_arc_grid(**kw)
    block = grid.blocks[0]
    block.nodes[..., 0] += shear * block.nodes[..., 1] / 4.0
    grid.global_nodes[grid.block_dof_maps[0].reshape(-1)] = block.nodes.reshape(-1, 2)
    return grid, arc


# Corner controls are fixed (codim >= 2 exempt); their cubic support owns the
# angle within ~2 control intervals (~5 fine nodes at this resolution).
_SKIP = 5


def _run(grid, arc, ortho, weight=1.0, ctrl=(8, 8), max_outer=25):
    wref = build_control_wall_ref(
        grid,
        IdentityTarget(d=2),
        ctrl,
        [WallSpec(axis=1, side=0, entity=arc, ortho=ortho, weight=weight)],
    )
    report = run_control_wall_ref(wref, max_outer=max_outer)
    return wref, report


# --------------------------------------------------------------------------- #
# Sliding + exactness
# --------------------------------------------------------------------------- #


def test_sliding_keeps_wall_nodes_on_entity():
    grid, arc = _make_arc_grid()
    _, report = _run(grid, arc, "off")
    assert report["iters"] >= 1
    assert all(np.isfinite(report["energies"]))
    assert all(md > 0.0 for md in report["mindets"])
    assert report["final_mindet"] > 0.0
    # b owns exactness: every bottom-wall node is on the arc.
    X_wall = grid.blocks[0].nodes[:, 0]
    feet = arc.project_many(X_wall)
    np.testing.assert_allclose(X_wall, feet, atol=1e-10)


def test_sliding_beats_fixed_boundary_controls():
    # Freeing the wall controls must not do worse than the plain fixed-
    # boundary reference on the same grid (same net, same budget).
    from egg.smoothing.control_ref import build_control_ref, run_control_ref

    grid_fixed, _ = _make_arc_grid()
    cref = build_control_ref(grid_fixed, IdentityTarget(d=2), (8, 8))
    rep_fixed = run_control_ref(cref, max_outer=25)

    grid_slide, arc = _make_arc_grid()
    _, rep_slide = _run(grid_slide, arc, "off")
    assert rep_slide["final_fine_energy"] <= rep_fixed["energies"][-1] * 1.02


# --------------------------------------------------------------------------- #
# Orthogonality
# --------------------------------------------------------------------------- #


def test_hard_ortho_angle_within_one_degree():
    grid, arc = _make_sheared_arc_grid()
    wref, report = _run(grid, arc, "hard")
    assert report["final_mindet"] > 0.0
    assert wall_angle_deviation(wref, skip=_SKIP).max() <= 1.0


def test_penalty_ortho_improves_monotonically_with_weight():
    maxdev = []
    for w in (0.0, 1.0, 100.0):
        grid, arc = _make_sheared_arc_grid()
        wref, report = _run(grid, arc, "penalty" if w > 0 else "off", weight=w)
        assert report["final_mindet"] > 0.0
        maxdev.append(wall_angle_deviation(wref, skip=_SKIP).max())
    assert maxdev[1] < maxdev[0]
    assert maxdev[2] < maxdev[1]


def test_penalty_taper_relaxes_locally():
    # Weight only the low-x half of the wall: the weighted half must end up
    # markedly more orthogonal than the free half (the capsule-style dial).
    grid, arc = _make_sheared_arc_grid()
    ctrl = (8, 8)
    m = ctrl[0] - 2  # non-corner tangential legs
    w = np.zeros(m)
    w[: m // 2] = 100.0
    wref, report = _run(grid, arc, "penalty", weight=w)
    assert report["final_mindet"] > 0.0
    devs = wall_angle_deviation(wref, skip=_SKIP)
    half = len(devs) // 2
    assert devs[:half].mean() < 0.5 * devs[half:].mean()


def test_hard_ortho_keeps_energy_reasonable():
    # The hard constraint trades energy for orthogonality, but must stay in
    # the same regime as the unconstrained wall run (no blow-up, valid mesh).
    grid_a, arc_a = _make_arc_grid()
    _, rep_free = _run(grid_a, arc_a, "off")
    grid_b, arc_b = _make_arc_grid()
    _, rep_hard = _run(grid_b, arc_b, "hard")
    assert rep_hard["final_mindet"] > 0.0
    assert rep_hard["final_fine_energy"] <= rep_free["final_fine_energy"] * 1.25


# --------------------------------------------------------------------------- #
# Frozen-frame db/dC model (the sliding-frame ratchet's root cause)
# --------------------------------------------------------------------------- #


def test_db_model_removes_frame_energy_injection():
    """On a tangentially coarse net the plain-map GN step bunches sliding
    controls (it treats a tangential slide as free normal fine-node motion)
    and every frame's b re-extension injects energy — the ratchet. With
    db/dC composed into the GN system the injections collapse by orders of
    magnitude and the final energy is no worse."""
    reps = {}
    for mdb in (False, True):
        grid, arc = _make_arc_grid()
        wref = build_control_wall_ref(
            grid,
            IdentityTarget(d=2),
            (4, 6),
            [WallSpec(axis=1, side=0, entity=arc, ortho="off", weight=1.0)],
        )
        reps[mdb] = run_control_wall_ref(wref, max_outer=40, model_db=mdb)
        assert reps[mdb]["final_mindet"] > 0.0

    def injected(rep):
        return sum(j for j in rep["frame_jumps"] if j > 0.0)

    assert injected(reps[True]) < injected(reps[False]) / 100.0
    assert reps[True]["final_fine_energy"] <= reps[False]["final_fine_energy"] + 1e-9
