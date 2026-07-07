# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Exact wall-normal respacing tests."""

import os
import sys

import numpy as np
import pytest

from egg.smoothing.respace import (
    _respace_line,
    _solve_stretch_ratio,
    enforce_boundary_layer_spacing,
)

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "2D", "circles")
)
from topologies import build_circle_in_rectangle  # noqa: E402


def test_stretch_ratio_solves_sum():
    s0, n, length = 0.01, 10, 0.5
    r = _solve_stretch_ratio(s0, n, length)
    assert sum(s0 * r**k for k in range(1, n + 1)) == pytest.approx(length)


def test_respace_line_exact_on_straight_line():
    pts = np.column_stack([np.linspace(0.0, 1.0, 21), np.zeros(21)])
    new = _respace_line(
        pts, first_height=0.005, growth=1.3, n_layers=6, max_height=None
    )
    sp = np.diff(new[:, 0])
    assert sp[0] == pytest.approx(0.005, rel=1e-12)
    assert sp[1:6] / sp[:5] == pytest.approx(np.full(5, 1.3), rel=1e-12)
    # Endpoints untouched; tail monotone and fills the rest.
    assert new[0, 0] == 0.0 and new[-1, 0] == 1.0
    assert np.all(np.diff(sp[5:]) > -1e-15)


def test_respace_line_graded_tail_when_layers_overshoot():
    """Growth past the mean remainder spacing grades the tail smoothly.

    30 cells over 0.5 with the eighth layer at 0.005·1.3^7 ≈ 0.031, above
    the 0.017 mean of the remainder: the single-ratio tail comes out below
    1 and the cells grade monotonically finer towards the far boundary —
    smooth double-sided clustering, never a step in the spacing profile.
    """
    pts = np.column_stack([np.linspace(0.0, 0.5, 31), np.zeros(31)])
    new = _respace_line(
        pts, first_height=0.005, growth=1.3, n_layers=8, max_height=None
    )
    sp = np.diff(new[:, 0])
    assert sp[0] == pytest.approx(0.005, rel=1e-12)
    assert sp[1:8] / sp[:7] == pytest.approx(np.full(7, 1.3), rel=1e-12)
    # Tail: single ratio < 1, monotone decreasing, no jump at the joint.
    tail_ratios = sp[8:] / sp[7:-1]
    assert np.all(tail_ratios < 1.0)
    assert tail_ratios == pytest.approx(
        np.full(len(tail_ratios), tail_ratios[0]), rel=1e-6
    )
    assert new[-1, 0] == 0.5


def test_respace_line_too_thin_raises():
    pts = np.column_stack([np.linspace(0.0, 0.01, 11), np.zeros(11)])
    with pytest.raises(ValueError, match="not fit"):
        _respace_line(pts, first_height=0.005, growth=1.5, n_layers=8, max_height=None)


def test_enforce_on_grid_first_height_and_growth():
    topo, ents = build_circle_in_rectangle(rough=False)
    topo.boundary_layer_specs = {
        id(ents["circle"]): dict(
            first_height=0.004,
            growth=1.25,
            n_layers=3,
            max_height=None,
            tangential_spacing=None,
        )
    }
    grid = topo.initialize_grid()
    enforce_boundary_layer_spacing(grid)

    names = list(topo.block_specs.keys())
    for blk in ("o_s", "o_e", "o_n", "o_w"):
        bi = names.index(blk)
        nodes = grid.global_nodes[grid.block_dof_maps[bi]]
        # Wall is axis 1 high side.
        first = np.linalg.norm(nodes[:, -1] - nodes[:, -2], axis=1)
        second = np.linalg.norm(nodes[:, -2] - nodes[:, -3], axis=1)
        # Chord lengths track arc spacing to ~curvature error on these lines.
        assert first == pytest.approx(np.full_like(first, 0.004), rel=1e-3)
        assert second / first == pytest.approx(np.full_like(first, 1.25), rel=1e-3)


def test_enforce_preserves_interfaces_and_boundaries():
    topo, ents = build_circle_in_rectangle(rough=False)
    topo.boundary_layer_specs = {
        id(ents["circle"]): dict(
            first_height=0.004,
            growth=1.25,
            n_layers=3,
            max_height=None,
            tangential_spacing=None,
        )
    }
    grid = topo.initialize_grid()
    before = grid.global_nodes.copy()
    enforce_boundary_layer_spacing(grid)

    names = list(topo.block_specs.keys())
    for ring, behind in (
        ("o_s", "e_s"),
        ("o_e", "e_e"),
        ("o_n", "e_n"),
        ("o_w", "e_w"),
    ):
        # Lines extend through the e_* blocks; their far face (the e_*/wall
        # boundary of the rectangle) does not move, and the shared o/e face
        # stays conforming because both blocks reference the same DOFs.
        dm_e = grid.block_dof_maps[names.index(behind)]
        assert np.allclose(grid.global_nodes[dm_e[:, 0]], before[dm_e[:, 0]])
        # Wall nodes stay on the circle.
        dm = grid.block_dof_maps[names.index(ring)]
        wall = grid.global_nodes[dm[:, -1]]
        radii = np.linalg.norm(wall - ents["circle"].center, axis=1)
        assert radii == pytest.approx(
            np.full_like(radii, ents["circle"].radius), rel=1e-6
        )


def test_enforce_without_extension_pins_wall_block_face():
    topo, ents = build_circle_in_rectangle(rough=False)
    topo.boundary_layer_specs = {
        id(ents["circle"]): dict(
            first_height=0.004,
            growth=1.25,
            n_layers=3,
            max_height=None,
            tangential_spacing=None,
        )
    }
    grid = topo.initialize_grid()
    before = grid.global_nodes.copy()
    enforce_boundary_layer_spacing(grid, extend_through_neighbours=False)

    names = list(topo.block_specs.keys())
    for blk in ("o_s", "o_e", "o_n", "o_w"):
        dm = grid.block_dof_maps[names.index(blk)]
        assert np.allclose(grid.global_nodes[dm[:, 0]], before[dm[:, 0]])


def test_enforce_noop_without_specs():
    topo, ents = build_circle_in_rectangle(rough=False)
    grid = topo.initialize_grid()
    before = grid.global_nodes.copy()
    enforce_boundary_layer_spacing(grid)
    assert np.array_equal(grid.global_nodes, before)


def test_respace_line_oblique_uses_normal_distance():
    """On a 45-degree line off a flat wall, layer heights are perpendicular."""
    from egg.geometry.analytic2d import LineSegment

    wall = LineSegment(np.array([-5.0, 0.0]), np.array([5.0, 0.0]))
    t = np.linspace(0.0, 1.0, 31)
    pts = np.column_stack([t, t])  # 45-degree straight line from the wall
    new = _respace_line(
        pts, first_height=0.01, growth=1.2, n_layers=5, max_height=None, entity=wall
    )
    heights = new[:, 1]  # perpendicular distance to the wall is just y
    sp = np.diff(heights)
    assert sp[0] == pytest.approx(0.01, rel=1e-9)
    assert sp[1:5] / sp[:4] == pytest.approx(np.full(4, 1.2), rel=1e-9)


def test_enforce_straightens_columns_onto_wall_normal():
    """Within the clustered layers, columns leave the wall orthogonally."""
    topo, ents = build_circle_in_rectangle(rough=False)
    topo.boundary_layer_specs = {
        id(ents["circle"]): dict(
            first_height=0.004,
            growth=1.25,
            n_layers=3,
            max_height=None,
            tangential_spacing=None,
        )
    }
    grid = topo.initialize_grid()
    enforce_boundary_layer_spacing(grid)

    circle = ents["circle"]
    names = list(topo.block_specs.keys())
    for blk in ("o_s", "o_e", "o_n", "o_w"):
        nodes = grid.global_nodes[grid.block_dof_maps[names.index(blk)]]
        for col in range(nodes.shape[0]):
            foot = nodes[col, -1]
            n_hat = np.asarray(circle.normal(circle.project(foot)))
            v = nodes[col, -4] - foot  # node at layer 3
            v = v / np.linalg.norm(v)
            assert abs(v[0] * n_hat[1] - v[1] * n_hat[0]) < 1e-9


def test_straightening_keeps_far_boundary_row_on_geometry():
    """On coarse grids where 2*n_layers reaches the line's end, the column
    straightening redistributes the far-boundary nodes tangentially (so the
    line endpoints follow the wall-normal fan instead of bunching) but must
    never pull them off their own entity — the rectangle walls here."""
    topo, ents = build_circle_in_rectangle(rough=False)
    # n_layers=8: extended circle->outer-wall lines are 9 cells, so the
    # straightening band (2*8) overruns the line end.
    topo.boundary_layer_specs = {
        id(ents["circle"]): dict(
            first_height=0.02,
            growth=1.3,
            n_layers=8,
            max_height=None,
            tangential_spacing=None,
        )
    }
    grid = topo.initialize_grid()
    enforce_boundary_layer_spacing(grid)

    names = list(topo.block_specs.keys())
    # The far ends of the extended wall-normal lines are the e_* blocks'
    # outer faces (axis 1 low side), lying on the rectangle walls.
    for blk, ent in (
        ("e_s", "bottom"),
        ("e_e", "right"),
        ("e_n", "top"),
        ("e_w", "left"),
    ):
        bi = names.index(blk)
        dofs = grid.block_dof_maps[bi][:, 0]
        pos = grid.global_nodes[dofs]
        off = np.linalg.norm(
            pos - np.array([ents[ent].project(p) for p in pos]), axis=1
        )
        assert off.max() == pytest.approx(0.0, abs=1e-12), (blk, off.max())
