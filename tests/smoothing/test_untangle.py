# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Tests for the δ-continuation untangler.

The untangle driver delegates to the C++ backend via
:func:`egg.smoothing.cpp_backend.cpp_untangle`. These tests exercise the
host-side schedule logic and the binding to the grid.
"""

import os
import sys

import numpy as np
import pytest

from egg.smoothing.solver import build_sweep_context
from egg.smoothing.targets import IdentityTarget
from egg.smoothing.untangle import untangle, grid_min_det
from egg.topology.builder import TopologyBuilder

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "2D", "circles")
)
from topologies import build_circle_in_rectangle  # noqa: E402


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


def _folded_demo_grid(seed=1, scale=0.35):
    """Valid circle-in-rectangle grid with free interior DOFs perturbed to fold."""
    topo, _ = build_circle_in_rectangle(rough=False)
    grid = topo.initialize_grid()
    rng = np.random.default_rng(seed)
    free = np.array(grid.free_mask)
    constrained = set(grid.dof_constraints.keys())
    gn = np.array(grid.global_nodes)
    for i in range(grid.global_node_count):
        if free[i] and i not in constrained:
            gn[i] += rng.normal(0, scale, 2)
    grid.global_nodes = gn
    for bi, blk in enumerate(grid.blocks):
        blk.nodes[...] = gn[grid.block_dof_maps[bi]]
    ctx = build_sweep_context(grid, IdentityTarget(2))
    return grid, ctx


def test_2d_recovery_from_folded():
    """A folded multiblock grid reaches min det A > margin within the schedule."""
    grid, ctx = _folded_demo_grid()
    md0 = grid_min_det(grid.global_nodes, ctx.energy_stencil)
    assert md0 < 0  # genuinely folded
    res = untangle(grid, ctx, margin=1e-9)
    assert res.converged
    assert res.min_det > 1e-9
    assert grid_min_det(grid.global_nodes, ctx.energy_stencil) > 1e-9


def test_no_op_on_valid_grid():
    """Untangling an already-valid grid is a no-op and does not degrade it."""
    topo, _ = build_circle_in_rectangle(rough=False)
    grid = topo.initialize_grid()
    ctx = build_sweep_context(grid, IdentityTarget(2))
    before = grid.global_nodes.copy()
    res = untangle(grid, ctx, margin=1e-9)
    assert res.no_op
    assert np.allclose(grid.global_nodes, before)


def test_stall_reports_bounded():
    """An infeasible (bowtie, fixed boundary) topology stalls, bounded, no hang."""
    bld = TopologyBuilder(d=2)
    bld.add_corner("sw", (0, 0), fixed=True)
    bld.add_corner("nw", (0, 1), fixed=True)
    bld.add_corner("se", (1, 0), fixed=True)
    bld.add_corner("ne", (1, 1), fixed=True)
    bld.add_block("blk", ("sw", "ne", "se", "nw"), (6, 6))
    topo = bld.build()
    grid = topo.initialize_grid()
    ctx = build_sweep_context(grid, IdentityTarget(2))
    res = untangle(grid, ctx, margin=1e-9, max_outer=15)
    assert not res.converged
    assert res.outer_iters <= 15
