"""Sliding gate: boundary-constrained DOFs move *along* their entity (tangential),
never off it. Reuses the clumping helper from the sliding demo.
"""

import numpy as np

from egg.geometry.analytic2d import Circle, LineSegment
from egg.smoothing.solver import build_sweep_context, local_relaxation_sweep
from examples.phase4_sliding_demo import (
    build_demo_grid, clump_boundary, constrained_positions, _circle_angle, _line_param,
)


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
