# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Boundary-layer clustering target tests."""

import os
import sys

import numpy as np

from egg.smoothing.targets import (
    BoundaryLayerTarget,
    MultiBlockTarget,
    IdentityTarget,
    AnisotropicTarget,
    build_topology_target,
)

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "2D", "circles")
)
from topologies import build_circle_in_rectangle  # noqa: E402


def _ring_block(spec_kwargs):
    topo, ents = build_circle_in_rectangle(rough=False)
    grid = topo.initialize_grid()
    topo.boundary_layer_specs = {id(ents["circle"]): spec_kwargs}
    tgt = build_topology_target(topo, interior_spacing=0.2)
    bi = list(topo.block_specs.keys()).index("o_s")
    return topo, grid, tgt, bi, ents


def test_metric_selects_far_field_default():
    """metric= picks the far-field default: Identity for shape, mean-size for
    shape_size — so shape_size needs no manual mean_size_target plumbing."""
    import pytest

    from egg.smoothing.targets import mean_size_target

    topo, ents = build_circle_in_rectangle(rough=False)
    grid = topo.initialize_grid()
    topo.boundary_layer_specs = {
        id(ents["circle"]): dict(
            first_height=0.02,
            growth=1.3,
            n_layers=4,
            max_height=None,
            tangential_spacing=None,
        )
    }
    far = list(topo.block_specs.keys()).index("c_sw")  # a block with no BL spec
    block = grid.blocks[far]
    cb = next(iter(block.iter_cells()))

    # plain shape (default): far-field default is IdentityTarget -> det W == 1
    tgt_shape = build_topology_target(topo, grid, metric="shape")
    assert abs(np.linalg.det(tgt_shape(far, block, cb, (0, 0))) - 1.0) < 1e-12

    # shape_size: far-field default carries physical scale (mean cell size)
    tgt_size = build_topology_target(topo, grid, metric="shape_size")
    ref = mean_size_target(grid)
    W = tgt_size(far, block, cb, (0, 0))
    assert np.allclose(W, ref(far, block, cb, (0, 0)))
    assert abs(np.linalg.det(W) - 1.0) > 1e-6  # not identity

    # an explicit default always wins over metric
    idt = IdentityTarget(2)
    tgt_override = build_topology_target(topo, grid, default=idt, metric="shape_size")
    assert abs(np.linalg.det(tgt_override(far, block, cb, (0, 0))) - 1.0) < 1e-12

    # shape_size without a grid is a clear error, not a silent Identity fallback
    with pytest.raises(ValueError, match="needs"):
        build_topology_target(topo, metric="shape_size")


def test_det_W_positive_everywhere():
    topo, grid, tgt, bi, _ = _ring_block(
        dict(
            first_height=0.02,
            growth=1.3,
            n_layers=4,
            max_height=None,
            tangential_spacing=None,
        )
    )
    block = grid.blocks[bi]
    for cb in block.iter_cells():
        W = tgt(bi, block, cb, (0, 0))
        assert np.linalg.det(W) > 0


def test_geometric_growth_law_preblend():
    _, _, tgt, bi, _ = _ring_block(
        dict(
            first_height=0.02,
            growth=1.4,
            n_layers=5,
            max_height=None,
            tangential_spacing=None,
        )
    )
    blt = tgt.per_block[bi]
    sn = [blt.normal_spacing(k) for k in range(5)]
    for k in range(4):
        assert abs(sn[k + 1] / sn[k] - 1.4) < 1e-12


def test_max_height_clamp():
    blt = BoundaryLayerTarget(
        IdentityTarget,
        first_height=0.1,
        growth=2.0,
        wall_axis=1,
        wall_side=1,
        n_layers=10,
        interior_spacing=1.0,
        max_height=0.3,
    )
    # 0.1, 0.2, 0.4->clamped 0.3, ...
    assert blt.normal_spacing(0) == 0.1
    assert abs(blt.normal_spacing(1) - 0.2) < 1e-12
    assert blt.normal_spacing(2) <= 0.3 + 1e-12


def test_orientation_normal_parallel_to_entity_normal():
    topo, grid, tgt, bi, ents = _ring_block(
        dict(
            first_height=0.02,
            growth=1.3,
            n_layers=4,
            max_height=None,
            tangential_spacing=None,
        )
    )
    blt = tgt.per_block[bi]
    block = grid.blocks[bi]
    n1 = block.logical_shape[1] - 1
    for cb in block.iter_cells():
        if int(cb[1]) == n1 - 1:  # wall-adjacent cell (high side)
            W = blt(bi, block, cb, (0, 0))
            anchor = blt._wall_anchor(block, cb)
            q = ents["circle"].project(anchor)
            nrm = np.asarray(ents["circle"].normal(q))
            coln = W[:, blt.wall_axis]
            coln = coln / np.linalg.norm(coln)
            cross = abs(coln[0] * nrm[1] - coln[1] * nrm[0])
            assert cross < 1e-12


def test_multiblock_dispatch():
    topo, grid, tgt, bi, _ = _ring_block(
        dict(
            first_height=0.02,
            growth=1.3,
            n_layers=4,
            max_height=None,
            tangential_spacing=None,
        )
    )
    assert isinstance(tgt, MultiBlockTarget)
    # ring block dispatches to a BL target, a fill block to the identity default.
    block = grid.blocks[bi]
    W_ring = tgt(bi, block, (0, 0), (0, 0))
    assert not np.allclose(W_ring, np.eye(2))  # anisotropic
    fill_bi = list(topo.block_specs.keys()).index("c_sw")
    W_fill = tgt(fill_bi, grid.blocks[fill_bi], (0, 0), (0, 0))
    assert np.allclose(W_fill, np.eye(2))


def test_backward_compat_existing_targets():
    """IdentityTarget/AnisotropicTarget still work under the 4-arg call."""
    idt = IdentityTarget(2)
    assert np.allclose(idt(0, None, (0, 0), (0, 0)), np.eye(2))
    ani = AnisotropicTarget((0.1, 0.5))
    assert np.allclose(ani(0, None, (0, 0), (0, 0)), np.diag([0.1, 0.5]))


def test_normal_spacing_capped_and_monotone():
    """Growth is capped at interior_spacing — no overshoot band."""
    blt = BoundaryLayerTarget(
        None,
        first_height=0.1,
        growth=2.0,
        wall_axis=1,
        wall_side=1,
        interior_spacing=0.5,
    )
    sn = [blt.normal_spacing(k) for k in range(12)]
    assert sn[:3] == [0.1, 0.2, 0.4]
    assert all(s <= 0.5 + 1e-15 for s in sn)
    assert all(b >= a for a, b in zip(sn, sn[1:]))


def test_far_field_isotropic_by_default():
    """With no explicit interior cap, far layers ask for W with s_n = s_t."""
    blt = BoundaryLayerTarget(
        None,
        first_height=0.01,
        growth=1.5,
        wall_axis=1,
        wall_side=1,
        tangential_spacing=0.3,
    )
    assert abs(blt.normal_spacing(50) - 0.3) < 1e-15


def test_per_face_tangential_spacing():
    """Wall blocks derive s_t from their own face length / cell count."""
    topo, ents = build_circle_in_rectangle(rough=False)
    topo.boundary_layer_specs = {
        id(ents["circle"]): dict(
            first_height=0.002,
            growth=1.2,
            n_layers=4,
            max_height=None,
            tangential_spacing=None,
        )
    }
    tgt = build_topology_target(topo)
    names = list(topo.block_specs.keys())
    bi = names.index("o_s")
    spec = topo.block_specs["o_s"]
    cnames = spec.face_corner_names(1, 1, 2)
    p0 = topo.corners[cnames[0]].position
    p1 = topo.corners[cnames[1]].position
    expected = np.linalg.norm(p1 - p0) / spec.resolutions[0]
    assert abs(tgt.per_block[bi].tangential_spacing - expected) < 1e-12


def test_neighbour_blend_continues_profile():
    """A slow-growing profile spills into the block behind the wall block."""
    topo, ents = build_circle_in_rectangle(rough=False)
    topo.boundary_layer_specs = {
        id(ents["circle"]): dict(
            first_height=0.001,
            growth=1.05,
            n_layers=4,
            max_height=None,
            tangential_spacing=None,
        )
    }
    tgt = build_topology_target(topo)
    names = list(topo.block_specs.keys())
    ring, behind = names.index("o_s"), names.index("e_s")
    assert behind in tgt.per_block
    rblt, nblt = tgt.per_block[ring], tgt.per_block[behind]
    wall_cells = topo.block_specs["o_s"].resolutions[1]
    assert nblt.k_offset == wall_cells
    # Spacing is continuous across the shared interface.
    assert (
        abs(nblt.normal_spacing(wall_cells) - rblt.normal_spacing(wall_cells)) < 1e-15
    )
    # The neighbour's wall side faces the ring block (its high axis-1 face).
    assert (nblt.wall_axis, nblt.wall_side) == (1, 1)


def test_no_neighbour_blend_once_isotropic():
    """If the profile reaches the cap inside the wall block, no spill-over."""
    topo, ents = build_circle_in_rectangle(rough=False)
    topo.boundary_layer_specs = {
        id(ents["circle"]): dict(
            first_height=0.05,
            growth=2.0,
            n_layers=4,
            max_height=None,
            tangential_spacing=None,
        )
    }
    tgt = build_topology_target(topo)
    names = list(topo.block_specs.keys())
    ring_names = {"o_s", "o_e", "o_n", "o_w"}
    assert {names[bi] for bi in tgt.per_block} == ring_names


def test_blend_neighbours_off():
    topo, ents = build_circle_in_rectangle(rough=False)
    topo.boundary_layer_specs = {
        id(ents["circle"]): dict(
            first_height=0.001,
            growth=1.05,
            n_layers=4,
            max_height=None,
            tangential_spacing=None,
        )
    }
    tgt = build_topology_target(topo, blend_neighbours=False)
    names = list(topo.block_specs.keys())
    assert {names[bi] for bi in tgt.per_block} == {"o_s", "o_e", "o_n", "o_w"}
