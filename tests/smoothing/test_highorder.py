# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""High-order curvature/orthogonality prototype: energies, lines, safe descent."""

import numpy as np

from egg.smoothing import highorder as H
from egg.smoothing.batch import energy_and_mindet
from egg.smoothing.solver import build_sweep_context
from egg.smoothing.targets import IdentityTarget
from egg.topology.builder import TopologyBuilder


def _lr_grid(res=(7, 7)):
    """Two unit blocks sharing the x=2 edge (axis-0 seam)."""
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


def _mindet(grid, target):
    es = build_sweep_context(grid, target).energy_stencil
    return energy_and_mindet(
        grid.global_nodes,
        es["gc"],
        es["gn0"],
        es["gn1"],
        es["s0"],
        es["s1"],
        es["W_inv"],
        metric="shape_2d",
    )[1]


def test_curvature_zero_straight_positive_kinked():
    # four collinear (even non-uniformly spaced) points -> no turning change.
    p = [np.array([[x, 0.0]]) for x in (0.0, 1.0, 2.3, 4.0)]
    assert float(H._f_curv(p)[0]) < 1e-12
    # a kink at the middle raises the turning-angle change.
    p2 = [p[0], p[1], np.array([[2.0, 0.6]]), p[3]]
    assert float(H._f_curv(p2)[0]) > 1e-3


def test_orthogonality_zero_perp_positive_skew():
    pa, pb = np.array([[-1.0, 0.0]]), np.array([[1.0, 0.0]])  # tangent 0: +x
    pc, pd = np.array([[0.0, -1.0]]), np.array([[0.0, 1.0]])  # tangent 1: +y
    assert float(H._f_orth([pa, pb, pc, pd])[0]) < 1e-12
    pd2 = np.array([[0.7, 1.0]])  # skew the crossing
    assert float(H._f_orth([pa, pb, pc, pd2])[0]) > 1e-2


def test_grid_lines_cross_the_seam():
    grid = _lr_grid()
    nbr, partner = H.grid_adjacency(grid)
    lines = H.grid_lines(grid, nbr, partner)
    seen = {}
    for m in grid.block_dof_maps:
        for nid in np.unique(np.asarray(m)):
            seen[int(nid)] = seen.get(int(nid), 0) + 1
    seam = {n for n, c in seen.items() if c > 1}
    # at least one assembled line threads through a seam node with interior
    # neighbours on both sides (a genuine across-interface crossing line).
    crossing = [
        ln
        for ln in lines
        if any(ln[k] in seam and 0 < k < len(ln) - 1 for k in range(len(ln)))
    ]
    assert crossing, "no grid line crosses the block seam"


def test_smoother_reduces_kink_and_stays_valid():
    grid = _lr_grid()
    X = np.asarray(grid.global_nodes)
    free = np.asarray(grid.free_mask)
    rng = np.random.default_rng(1)
    X[free] += 0.025 * rng.normal(size=X[free].shape)  # mild waviness, stays valid
    for bi in range(len(grid.blocks)):
        grid.blocks[bi].nodes[...] = grid.global_nodes[
            np.asarray(grid.block_dof_maps[bi])
        ]

    tgt = IdentityTarget(2)
    md0 = _mindet(grid, tgt)
    assert md0 > 0.0
    d0 = H.line_diagnostics(grid)
    H.highorder_smooth(grid, tgt, metric="shape_2d", w_curv=0.3, w_orth=0.1, iters=120)
    d1 = H.line_diagnostics(grid)
    assert _mindet(grid, tgt) > 0.0  # barrier never crossed
    assert d1["turn_rms"] < d0["turn_rms"]  # waviness reduced


def test_iface_boost_targets_the_seam():
    grid = _lr_grid()
    X = np.asarray(grid.global_nodes)
    free = np.asarray(grid.free_mask)
    rng = np.random.default_rng(2)
    X[free] += 0.025 * rng.normal(size=X[free].shape)
    for bi in range(len(grid.blocks)):
        grid.blocks[bi].nodes[...] = grid.global_nodes[
            np.asarray(grid.block_dof_maps[bi])
        ]
    tgt = IdentityTarget(2)
    # a boost != 1 is accepted and keeps the grid valid (targets seam windows).
    H.highorder_smooth(
        grid,
        tgt,
        metric="shape_2d",
        w_curv=0.3,
        w_orth=0.0,
        iface_boost=10.0,
        iters=120,
    )
    assert _mindet(grid, tgt) > 0.0
