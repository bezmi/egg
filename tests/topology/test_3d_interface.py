# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""3D multiblock interfaces with non-trivial (rotated/reflected) shared faces.

Two hex blocks share the x=1 plane. Block B lists its corners so its logical
axes on the shared face are permuted relative to block A, exercising the general
face-orientation map in ``_build_dof_map`` / ``_detect_singularities``. A wrong
map merges geometrically distinct nodes, so the initialized block arrays no
longer match their own trilinear cubes.
"""

from itertools import product

import numpy as np

from egg.topology.builder import TopologyBuilder

# Corner positions on the [0,2] x [0,1] x [0,1] slab.
CORNERS = {
    "a00": (0, 0, 0),
    "a01": (0, 0, 1),
    "a10": (0, 1, 0),
    "a11": (0, 1, 1),
    "s00": (1, 0, 0),
    "s01": (1, 0, 1),
    "s10": (1, 1, 0),
    "s11": (1, 1, 1),
    "b00": (2, 0, 0),
    "b01": (2, 0, 1),
    "b10": (2, 1, 0),
    "b11": (2, 1, 1),
}


def _trilinear(corner_names, shape):
    """Trilinear cube from 8 corners (product order) on a (nx,ny,nz) node grid."""
    c = {
        idx: np.asarray(CORNERS[nm], float)
        for idx, nm in zip(product((0, 1), repeat=3), corner_names)
    }
    out = np.empty(shape + (3,))
    for a, b, d in product(*[range(s) for s in shape]):
        u, v, w = a / (shape[0] - 1), b / (shape[1] - 1), d / (shape[2] - 1)
        p = np.zeros(3)
        for (i0, i1, i2), cc in c.items():
            wt = (u if i0 else 1 - u) * (v if i1 else 1 - v) * (w if i2 else 1 - w)
            p += wt * cc
        out[a, b, d] = p
    return out


def _build(b_corners, b_res):
    tb = TopologyBuilder(d=3)
    for name, pos in CORNERS.items():
        tb.add_corner(name, pos)
    tb.add_block(
        "A",
        corners=("a00", "a01", "a10", "a11", "s00", "s01", "s10", "s11"),
        resolutions=(2, 3, 4),
    )
    tb.add_block("B", corners=b_corners, resolutions=b_res)
    return tb.build()


def _check_grid(topo, b_corners, b_res):
    grid = topo.initialize_grid()
    assert not np.any(np.isnan(grid.global_nodes))
    for bi, name in enumerate(topo.block_specs):
        block = grid.blocks[bi]
        if name == "A":
            expect = _trilinear(
                ("a00", "a01", "a10", "a11", "s00", "s01", "s10", "s11"), (3, 4, 5)
            )
        else:
            expect = _trilinear(b_corners, tuple(r + 1 for r in b_res))
        np.testing.assert_allclose(block.nodes, expect, atol=1e-9)
    return grid


def test_3d_interface_aligned():
    """Trivial orientation: B's shared face aligns axis-for-axis with A."""
    b_corners = ("s00", "s01", "s10", "s11", "b00", "b01", "b10", "b11")
    topo = _build(b_corners, (2, 3, 4))
    grid = _check_grid(topo, b_corners, (2, 3, 4))
    # 60 + 60 nodes, minus the shared 4x5 face.
    assert grid.global_node_count == 60 + 60 - 20


def test_3d_interface_rotated():
    """B's logical y/z axes are swapped on the shared face (a face rotation)."""
    b_corners = ("s00", "s10", "s01", "s11", "b00", "b10", "b01", "b11")
    topo = _build(
        b_corners, (2, 4, 3)
    )  # axis1 rides physical z (4), axis2 physical y (3)
    grid = _check_grid(topo, b_corners, (2, 4, 3))
    assert grid.global_node_count == 60 + 60 - 20
    assert topo.singularities == []
