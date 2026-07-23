# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Re-tabulating loader: put a saved net onto a new-resolution grid.

The net's control state is restored unchanged (resolution-independent), b is
re-extended at the new sampling, the evaluated grid is valid and watertight,
and a topology change (not just a resolution change) is refused.
"""

import numpy as np

from egg.io import (
    load_control_net,
    retabulate_control_net,
    save_control_net,
)
from egg.smoothing import fit_control_net
from egg.smoothing.control_fit import _net_min_det
from egg.smoothing.control_topology import watertight_mismatch
from egg.topology.builder import TopologyBuilder


def _two_block_grid(res=(9, 9)):
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


def _perturb_interior(grid, seed=0, scale=0.05):
    rng = np.random.default_rng(seed)
    gn = np.array(grid.global_nodes)
    free = np.asarray(grid.free_mask)
    for i in range(grid.global_node_count):
        if free[i] and i not in grid.dof_constraints:
            gn[i] += rng.normal(0.0, scale, gn.shape[1])
    grid.global_nodes = gn
    for bi, blk in enumerate(grid.blocks):
        blk.nodes[...] = gn[grid.block_dof_maps[bi]]


def test_retabulate_restores_state_and_stays_valid(tmp_path):
    path = str(tmp_path / "net.npz")
    topo = fit_control_net(_two_block_grid((9, 9)), ratio=2, walls=False)
    save_control_net(topo, path)

    # A finer grid, same topology.
    fine = _two_block_grid((17, 17))
    loaded = retabulate_control_net(fine, path)
    assert loaded is not None
    # Same control state (the net does not depend on resolution).
    np.testing.assert_allclose(loaded.q, topo.q, atol=1e-12)
    assert _net_min_det(loaded) > 0.0
    # Both blocks agree on the shared seam nodes.
    assert watertight_mismatch(loaded) < 1e-9


def test_retabulate_matches_a_same_resolution_load(tmp_path):
    path = str(tmp_path / "net.npz")
    grid = _two_block_grid((9, 9))
    topo = fit_control_net(grid, ratio=2, walls=False)
    save_control_net(topo, path)
    # At the SAME resolution the re-evaluated grid reproduces the fitted net.
    same = retabulate_control_net(_two_block_grid((9, 9)), path)
    assert same is not None
    np.testing.assert_allclose(same.prolong_global(), topo.prolong_global(), atol=1e-9)


def test_retabulate_refuses_a_topology_change(tmp_path):
    path = str(tmp_path / "net.npz")
    topo = fit_control_net(_two_block_grid((9, 9)), ratio=2, walls=False)
    save_control_net(topo, path)

    # A single-block grid has a different reduced layout: refuse, do not
    # restore a mismatched state.
    b = TopologyBuilder(d=2)
    for name, pos in [
        ("A", (0.0, 0.0)),
        ("D", (0.0, 2.0)),
        ("B", (4.0, 0.0)),
        ("C", (4.0, 2.0)),
    ]:
        b.add_corner(name, pos, fixed=True)
    b.add_block("S", ("A", "D", "B", "C"), (9, 9))
    other = b.build().initialize_grid()
    assert retabulate_control_net(other, path) is None


def test_residual_reproduces_the_grid_bit_exact(tmp_path):
    path = str(tmp_path / "net.npz")
    grid = _two_block_grid((9, 9))
    _perturb_interior(grid)
    solved = np.array(grid.global_nodes)
    topo = fit_control_net(grid, ratio=2, walls=False)
    # The fit is lossy: the net alone does not equal the perturbed grid.
    assert np.abs(topo.prolong_global() - solved).max() > 1e-6
    save_control_net(topo, path, residual=True)

    # A fresh same-resolution grid: net + residual reproduces the grid exactly.
    grid2 = _two_block_grid((9, 9))
    loaded = load_control_net(grid2, path)
    assert loaded.residual is not None
    loaded.write_to_grid(exact=True)
    np.testing.assert_allclose(grid2.global_nodes, solved, atol=1e-9)


def test_no_residual_layer_by_default(tmp_path):
    path = str(tmp_path / "net.npz")
    grid = _two_block_grid((9, 9))
    _perturb_interior(grid)
    save_control_net(fit_control_net(grid, ratio=2, walls=False), path)
    loaded = load_control_net(_two_block_grid((9, 9)), path)
    assert loaded.residual is None


def test_retabulate_interpolates_a_valid_residual(tmp_path):
    path = str(tmp_path / "net.npz")
    grid = _two_block_grid((9, 9))
    _perturb_interior(grid, scale=0.02)
    topo = fit_control_net(grid, ratio=2, walls=False)
    save_control_net(topo, path, residual=True)

    # A finer grid: the residual is interpolated and kept when it stays valid.
    fine = retabulate_control_net(_two_block_grid((17, 17)), path)
    assert fine is not None
    assert fine.residual is not None
    fine.write_to_grid(exact=True)
    from egg.smoothing.control_topology import _corner_mindet

    for bi in range(len(fine.grid.blocks)):
        blk = np.asarray(fine.grid.blocks[bi].nodes, dtype=float)
        assert _corner_mindet(blk) > 0.0
