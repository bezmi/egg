"""Boundary-layer clustering target tests (M5+)."""

import os
import sys

import numpy as np

from egg.smoothing.targets import (
    BoundaryLayerTarget,
    MultiBlockTarget,
    IdentityTarget,
    AnisotropicTarget,
    build_boundary_layer_target,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "circles"))
from topologies import build_circle_in_rectangle  # noqa: E402


def _ring_block(spec_kwargs):
    topo, ents = build_circle_in_rectangle(rough=False)
    grid = topo.initialize_grid()
    topo.boundary_layer_specs = {id(ents["circle"]): spec_kwargs}
    tgt = build_boundary_layer_target(topo, interior_spacing=0.2)
    bi = list(topo.block_specs.keys()).index("o_s")
    return topo, grid, tgt, bi, ents


def test_det_W_positive_everywhere():
    topo, grid, tgt, bi, _ = _ring_block(
        dict(first_height=0.02, growth=1.3, n_layers=4,
             max_height=None, tangential_spacing=None))
    block = grid.blocks[bi]
    for cb in block.iter_cells():
        W = tgt(bi, block, cb, (0, 0))
        assert np.linalg.det(W) > 0


def test_geometric_growth_law_preblend():
    _, _, tgt, bi, _ = _ring_block(
        dict(first_height=0.02, growth=1.4, n_layers=5,
             max_height=None, tangential_spacing=None))
    blt = tgt.per_block[bi]
    sn = [blt.normal_spacing(k) for k in range(5)]
    for k in range(4):
        assert abs(sn[k + 1] / sn[k] - 1.4) < 1e-12


def test_max_height_clamp():
    blt = BoundaryLayerTarget(
        IdentityTarget, first_height=0.1, growth=2.0, wall_axis=1, wall_side=1,
        n_layers=10, interior_spacing=1.0, max_height=0.3)
    # 0.1, 0.2, 0.4->clamped 0.3, ...
    assert blt.normal_spacing(0) == 0.1
    assert abs(blt.normal_spacing(1) - 0.2) < 1e-12
    assert blt.normal_spacing(2) <= 0.3 + 1e-12


def test_orientation_normal_parallel_to_entity_normal():
    topo, grid, tgt, bi, ents = _ring_block(
        dict(first_height=0.02, growth=1.3, n_layers=4,
             max_height=None, tangential_spacing=None))
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
        dict(first_height=0.02, growth=1.3, n_layers=4,
             max_height=None, tangential_spacing=None))
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
