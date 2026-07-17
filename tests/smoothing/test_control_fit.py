# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Fit a control net to an existing grid (egg.smoothing.fit_control_net).

The fit reads the grid, never writes it, always returns a fold-free net (or
declines with ControlFitError), and records how closely the net matches the
grid.
"""

import numpy as np
import pytest

from egg.smoothing import ControlFitError, fit_control_net
from egg.smoothing.control_fit import _net_min_det
from egg.topology.builder import TopologyBuilder


def _two_block_grid(res=(9, 9)):
    """Two blocks sharing the x=2 seam (a plain rectangle split in half)."""
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
    return b.build().initialize_grid()


def _perturb_interior(grid, seed=0, scale=0.03):
    """Nudge free interior nodes so the fit target is not exactly a net."""
    rng = np.random.default_rng(seed)
    gn = np.array(grid.global_nodes)
    free = np.asarray(grid.free_mask)
    for i in range(grid.global_node_count):
        if free[i] and i not in grid.dof_constraints:
            gn[i] += rng.normal(0.0, scale, gn.shape[1])
    grid.global_nodes = gn
    for bi, blk in enumerate(grid.blocks):
        blk.nodes[...] = gn[grid.block_dof_maps[bi]]


def test_fit_is_read_only_and_valid():
    grid = _two_block_grid()
    before = np.array(grid.global_nodes)
    topo = fit_control_net(grid, ratio=2, walls=False, fan_refine=0)
    # The grid is untouched: the fit stores a net alongside it, it does not
    # move nodes.
    np.testing.assert_array_equal(grid.global_nodes, before)
    assert _net_min_det(topo) > 0.0
    # A flat rectangle interior is exactly a bilinear map, which a cubic net
    # represents with no error.
    assert topo.fit_residual < 1e-8


def test_fit_approximates_a_perturbed_grid_and_stays_valid():
    grid = _two_block_grid()
    _perturb_interior(grid)
    topo = fit_control_net(grid, ratio=2, walls=False, fit_spacing="chord")
    assert _net_min_det(topo) > 0.0
    # The net cannot follow the random nudges exactly, so the fit is loose.
    assert topo.fit_residual > 0.0


def test_fit_declines_when_the_net_always_folds(monkeypatch):
    grid = _two_block_grid()
    import egg.smoothing.control_fit as cf

    monkeypatch.setattr(cf, "_net_min_det", lambda topo: -1.0)
    with pytest.raises(ControlFitError):
        cf.fit_control_net(grid, ratio=2, walls=False, fan_refine=1)
