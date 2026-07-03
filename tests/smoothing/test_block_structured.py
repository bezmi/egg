# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Tests for build_block_structured_context.

Validates the halo-padded structured tables against a known two-block grid:
the interior scatter map reproduces the block DOF maps, and the ghost-fill
entries make a block's ghost layer carry the neighbour block's first interior
layer — exactly the copy the C++ halo_exchange kernel (1.2) will perform.
"""

from __future__ import annotations

import numpy as np

from egg.smoothing.cpp_backend import build_block_structured_context
from egg.topology.builder import TopologyBuilder


def _lr_grid(res=(4, 4)):
    """Two unit blocks L (x:0..2) and R (x:2..4), sharing the x=2 edge.

    Each block has ``res`` cells. L's high-x face (axis 0, side 1) connects to
    R's low-x face (axis 0, side 0).
    """
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
    topo = b.build()
    grid = topo.initialize_grid()
    return grid


def test_interior_shapes_and_scatter_map():
    grid = _lr_grid()
    sc = build_block_structured_context(grid)

    assert sc.d == 2
    assert sc.num_blocks == 2
    assert sc.interior_shapes == [(5, 5), (5, 5)]

    # block_global_dof must reproduce block_dof_maps in row-major order.
    for bi in range(2):
        expected = grid.block_dof_maps[bi].reshape(-1).astype(np.int32)
        np.testing.assert_array_equal(sc.block_global_dof[bi], expected)


def test_no_singularities_on_cartesian_pair():
    grid = _lr_grid()
    sc = build_block_structured_context(grid)
    assert sc.sing_block.shape == (0,)
    assert sc.sing_logical.shape == (0, 2)
    assert sc.sing_valence.shape == (0,)


def test_halo_entries_count_and_ghost_validity():
    grid = _lr_grid()
    sc = build_block_structured_context(grid)

    # 5 shared face nodes, two directions (L<-R, R<-L) -> 10 ghost-fill entries.
    assert sc.num_halo_entries == 10

    for e in range(sc.num_halo_entries):
        db = int(sc.halo_dst_block[e])
        sb = int(sc.halo_src_block[e])
        shape_d = sc.interior_shapes[db]
        shape_s = sc.interior_shapes[sb]
        dst = sc.halo_dst_padded[e]
        src = sc.halo_src_padded[e]

        # dst is a ghost: exactly one axis is in the ghost shell (0 or n+1), the
        # rest are interior padded indices (1..n).
        ghost_axes = [k for k in range(2) if dst[k] in (0, shape_d[k] + 1)]
        assert len(ghost_axes) == 1
        for k in range(2):
            if k not in ghost_axes:
                assert 1 <= dst[k] <= shape_d[k]

        # src is interior (padded index 1..n on every axis).
        for k in range(2):
            assert 1 <= src[k] <= shape_s[k]


def _padded_coords_with_halo(grid, sc, bi):
    """Reconstruct block bi's padded coordinate array, ghosts filled via tables.

    Mirrors what the C++ halo_exchange does: scatter the global coordinates into
    the interior, then copy each named source interior node into its ghost slot.
    """
    d = sc.d
    shape = sc.interior_shapes[bi]
    padded = np.full(tuple(n + 2 for n in shape) + (d,), np.nan)

    gn = grid.global_nodes
    # Interior: padded[1:-1, 1:-1, :] = global coords in row-major logical order.
    interior = gn[sc.block_global_dof[bi]].reshape(shape + (d,))
    padded[tuple(slice(1, n + 1) for n in shape)] = interior

    for e in range(sc.num_halo_entries):
        if int(sc.halo_dst_block[e]) != bi:
            continue
        sb = int(sc.halo_src_block[e])
        src_logical = tuple(
            int(c) - 1 for c in sc.halo_src_padded[e]
        )  # padded -> interior
        flat = int(np.ravel_multi_index(src_logical, sc.interior_shapes[sb]))
        coord = gn[sc.block_global_dof[sb][flat]]
        padded[tuple(int(c) for c in sc.halo_dst_padded[e])] = coord
    return padded


def test_ghost_layer_carries_neighbour_interior():
    """L's ghost column just past x=2 must equal R's first interior layer (x=2.5)."""
    grid = _lr_grid()
    sc = build_block_structured_context(grid)

    padded_L = _padded_coords_with_halo(grid, sc, 0)  # block L
    # L padded axis-0 index 6 is the ghost shell beyond the high-x (shared) face.
    ghost_col = padded_L[6, 1:6, :]  # the 5 face-aligned ghost nodes
    assert not np.any(np.isnan(ghost_col)), "every face ghost must be filled"
    # R's first interior layer sits at x = 2 + 2/4 = 2.5.
    np.testing.assert_allclose(ghost_col[:, 0], 2.5)
    # y runs 0..2 matching the shared edge.
    np.testing.assert_allclose(ghost_col[:, 1], np.linspace(0.0, 2.0, 5))

    # Symmetric: R's ghost just below x=2 equals L's last interior layer (x=1.5).
    padded_R = _padded_coords_with_halo(grid, sc, 1)
    ghost_col_R = padded_R[0, 1:6, :]  # ghost shell below R's low-x face
    assert not np.any(np.isnan(ghost_col_R))
    np.testing.assert_allclose(ghost_col_R[:, 0], 1.5)
