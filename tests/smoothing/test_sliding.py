"""Sliding gate: boundary-constrained DOFs move *along* their entity (tangential),
never off it.

Self-contained: the demo grid builder and clumping helpers were previously imported
from ``examples/phase4_sliding_demo.py`` (since removed). The pieces the gate needs
are inlined here so the test does not depend on the examples package.
"""


import numpy as np

from egg.geometry.analytic2d import Circle, LineSegment
from egg.smoothing.solver import build_sweep_context, local_relaxation_sweep
from egg.smoothing.targets import IdentityTarget
from egg.topology.builder import TopologyBuilder


# --- demo grid (Phase-4 O-grid around a circle in a square) ---------------------
def build_demo_grid():
    """The Phase-4 O-grid topology; returns (grid, topology, target, entities)."""
    circle = Circle(center=(2.0, 2.0), radius=0.8)
    bottom = LineSegment(start=(0.0, 0.0), end=(4.0, 0.0))
    right = LineSegment(start=(4.0, 0.0), end=(4.0, 4.0))
    top = LineSegment(start=(4.0, 4.0), end=(0.0, 4.0))
    left = LineSegment(start=(0.0, 0.0), end=(0.0, 4.0))

    R = 5  # radial cell count of the outer edge/corner blocks

    builder = TopologyBuilder(d=2)
    for n, p in [("sw", (0, 0)), ("se", (4, 0)), ("ne", (4, 4)), ("nw", (0, 4))]:
        builder.add_corner(n, p, fixed=True)
    for n, p in [("msw", (1, 1)), ("mse", (3, 1)), ("mne", (3, 3)), ("mnw", (1, 3))]:
        builder.add_corner(n, p, fixed=False)
    for n, p in [("isw", (1.3, 1.3)), ("ise", (2.7, 1.3)), ("ine", (2.7, 2.7)), ("inw", (1.3, 2.7))]:
        builder.add_corner(n, p, fixed=False)
    for n, p in [
        ("bsw", (1, 0)), ("bse", (3, 0)),  # bottom wall
        ("rse", (4, 1)), ("rne", (4, 3)),  # right wall
        ("tne", (3, 4)), ("tnw", (1, 4)),  # top wall
        ("lnw", (0, 3)), ("lsw", (0, 1)),  # left wall
    ]:
        builder.add_corner(n, p, fixed=False)

    for nm, sw, nw, se, ne in [
        ("o_s", "msw", "isw", "mse", "ise"),
        ("o_e", "mse", "ise", "mne", "ine"),
        ("o_n", "mne", "ine", "mnw", "inw"),
        ("o_w", "mnw", "inw", "msw", "isw"),
    ]:
        builder.add_block(nm, (sw, nw, se, ne), (10, 4))
    for nm, sw, nw, se, ne in [
        ("e_s", "bsw", "msw", "bse", "mse"),
        ("e_e", "rse", "mse", "rne", "mne"),
        ("e_n", "tne", "mne", "tnw", "mnw"),
        ("e_w", "lnw", "mnw", "lsw", "msw"),
    ]:
        builder.add_block(nm, (sw, nw, se, ne), (10, R))
    for nm, sw, nw, se, ne in [
        ("c_sw", "sw", "lsw", "bsw", "msw"),
        ("c_se", "se", "bse", "rse", "mse"),
        ("c_ne", "ne", "rne", "tne", "mne"),
        ("c_nw", "nw", "tnw", "lnw", "mnw"),
    ]:
        builder.add_block(nm, (sw, nw, se, ne), (R, R))

    for a, b_ in [("o_s", "o_e"), ("o_e", "o_n"), ("o_n", "o_w"), ("o_w", "o_s")]:
        builder.connect(a, 0, 1, b_, 0, 0)
    for e, o in [("e_s", "o_s"), ("e_e", "o_e"), ("e_n", "o_n"), ("e_w", "o_w")]:
        builder.connect(e, 1, 1, o, 1, 0)
    for cb, ca, cs, eb, ea, es in [
        ("c_sw", 0, 1, "e_s", 0, 0), ("c_sw", 1, 1, "e_w", 0, 1),
        ("c_se", 0, 1, "e_e", 0, 0), ("c_se", 1, 1, "e_s", 0, 1),
        ("c_ne", 0, 1, "e_n", 0, 0), ("c_ne", 1, 1, "e_e", 0, 1),
        ("c_nw", 0, 1, "e_w", 0, 0), ("c_nw", 1, 1, "e_n", 0, 1),
    ]:
        builder.connect(cb, ca, cs, eb, ea, es)

    for blk in ("o_s", "o_e", "o_n", "o_w"):
        builder.associate(blk, 1, 1, circle)
    for blk, ent in [("e_s", bottom), ("e_e", right), ("e_n", top), ("e_w", left)]:
        builder.associate(blk, 1, 0, ent)
    for blk, a0_ent, a1_ent in [
        ("c_sw", left, bottom), ("c_se", bottom, right),
        ("c_ne", right, top), ("c_nw", top, left),
    ]:
        builder.associate(blk, 0, 0, a0_ent)
        builder.associate(blk, 1, 0, a1_ent)

    topology = builder.build()
    topology.initialize_grid()
    grid = topology.grid
    target = IdentityTarget(d=2)
    return grid, topology, target, [circle, bottom, right, top, left]


# --- clumping / measurement helpers --------------------------------------------
def _propagate(grid):
    for bi, block in enumerate(grid.blocks):
        block.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]


def _circle_angle(ent, p):
    d = np.asarray(p) - ent.center
    return float(np.arctan2(d[1], d[0]))


def _line_param(ent, p):
    ab = ent.end - ent.start
    return float(np.clip(np.dot(np.asarray(p) - ent.start, ab) / np.dot(ab, ab), 0, 1))


def clump_boundary(grid, amp):
    """Tangentially warp constrained DOFs into an uneven (clumped) distribution,
    re-projecting so each point stays exactly on its entity."""
    for dof, ent in grid.dof_constraints.items():
        p = grid.global_nodes[dof]
        if isinstance(ent, Circle):
            th = _circle_angle(ent, p)
            th2 = th + amp * np.sin(2.0 * th)
            grid.global_nodes[dof] = ent.center + ent.radius * np.array([np.cos(th2), np.sin(th2)])
        elif isinstance(ent, LineSegment):
            t = _line_param(ent, p)
            t2 = np.clip(t + amp * 0.25 * np.sin(np.pi * t), 0.0, 1.0)
            grid.global_nodes[dof] = ent.start + t2 * (ent.end - ent.start)
    _propagate(grid)


def constrained_positions(grid):
    return {dof: (ent, grid.global_nodes[dof].copy())
            for dof, ent in grid.dof_constraints.items()}


# --- gates ----------------------------------------------------------------------
def test_constrained_dofs_slide_along_entity():
    grid, _, target, _ = build_demo_grid()

    # Clump the boundary into an uneven distribution, record it.
    clump_boundary(grid, amp=0.6)
    initial = constrained_positions(grid)

    # Relax — constrained DOFs may only slide along their entity.
    ctx = build_sweep_context(grid, target)
    for _ in range(60):
        local_relaxation_sweep(grid, target, "shape_2d", ctx)
    final = constrained_positions(grid)

    # (1) Every constrained point is still exactly on its entity.
    max_resid = max(
        float(np.linalg.norm(p - np.asarray(ent.project(p))))
        for ent, p in final.values()
    )
    assert max_resid < 1e-8, f"a constrained DOF left its entity: resid={max_resid:.2e}"

    # (2) Points actually slid tangentially (non-trivial along-boundary shift).
    shifts = []
    for dof, (ent, p1) in final.items():
        p0 = initial[dof][1]
        if isinstance(ent, Circle):
            dth = (_circle_angle(ent, p1) - _circle_angle(ent, p0) + np.pi) % (2 * np.pi) - np.pi
            shifts.append(abs(ent.radius * dth))
        else:
            shifts.append(abs(_line_param(ent, p1) - _line_param(ent, p0))
                          * float(np.linalg.norm(ent.end - ent.start)))
    assert np.mean(shifts) > 1e-3, "constrained DOFs did not slide tangentially"


def test_clumped_circle_spacing_becomes_more_even():
    """Relaxation reduces the variance of circle-node angular spacing."""
    grid, _, target, _ = build_demo_grid()
    clump_boundary(grid, amp=0.6)

    def circle_gap_std(g):
        angs = sorted(_circle_angle(e, g.global_nodes[d])
                      for d, e in g.dof_constraints.items() if isinstance(e, Circle))
        return float(np.std(np.diff(angs)))

    before = circle_gap_std(grid)
    ctx = build_sweep_context(grid, target)
    for _ in range(60):
        local_relaxation_sweep(grid, target, "shape_2d", ctx)
    after = circle_gap_std(grid)

    assert after < before, f"spacing did not even out: std {before:.4f} -> {after:.4f}"
