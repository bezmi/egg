"""Tests for objective assembly — gradient gate via check_grad."""

import numpy as np
from scipy.optimize import check_grad

from egg.smoothing.objective import (
    assemble_energy,
    assemble_gradient,
    pack_x,
    unpack_x,
)
from egg.smoothing.targets import IdentityTarget
from egg.topology.builder import TopologyBuilder


def _make_test_grid_2d():
    """Build a simple 1-block valid grid for testing."""
    builder = TopologyBuilder(d=2)
    for name, pos in [("A", (0., 0.)), ("B", (4., 0.)),
                      ("C", (4., 4.)), ("D", (0., 4.))]:
        builder.add_corner(name, pos, fixed=True)
    builder.add_block("main", ("A", "D", "B", "C"), (8, 8))
    topo = builder.build()
    topo.initialize_grid()
    return topo.grid


def _make_test_grid_3d():
    """Build a simple 1-block valid 3D grid for testing."""
    builder = TopologyBuilder(d=3)
    for name, pos in [("swn", (0., 0., 0.)), ("sew", (2., 0., 0.)),
                      ("nes", (2., 1., 0.)), ("nwn", (0., 1., 0.)),
                      ("sws", (0., 0., 1.)), ("see", (2., 0., 1.)),
                      ("nee", (2., 1., 1.)), ("nww", (0., 1., 1.))]:
        builder.add_corner(name, pos, fixed=True)
    builder.add_block(
        "main",
        ("swn", "nwn", "sew", "nes"),
        (4, 2),
    )
    # Actually 3D needs 8 corners properly ordered...
    return None  # 3D test deferred until builder supports d=3


class TestGradientGate:
    def test_check_grad_2d(self):
        """Gradient gate: analytic ∇F ≈ finite-difference ∇F for 2D grid."""
        grid = _make_test_grid_2d()
        target = IdentityTarget(d=2)

        x0 = pack_x(grid)

        # Move away from optimum so gradient is non-zero
        x0 = x0 + 0.01 * np.random.default_rng(42).normal(size=len(x0))

        def f(x):
            unpack_x(x, grid)
            return assemble_energy(grid, target, "shape_2d")

        def grad(x):
            unpack_x(x, grid)
            G = assemble_gradient(grid, target, "shape_2d")
            return G[grid.free_mask].ravel()

        err = check_grad(f, grad, x0)
        assert err < 1e-3, f"check_grad error = {err}"

    def test_energy_positive(self):
        """Energy is non-negative for valid grid."""
        grid = _make_test_grid_2d()
        target = IdentityTarget(d=2)
        F = assemble_energy(grid, target, "shape")
        assert F >= -1e-14

    def test_energy_zero_for_uniform_identity(self):
        """Uniform grid with W=I gives zero energy."""
        ni, nj = 5, 5
        nodes = np.zeros((ni, nj, 2))
        for i in range(ni):
            for j in range(nj):
                nodes[i, j, 0] = float(i)
                nodes[i, j, 1] = float(j)

        builder = TopologyBuilder(d=2)
        for name, pos in [("A", (0., 0.)), ("B", (4., 0.)),
                          ("C", (4., 4.)), ("D", (0., 4.))]:
            builder.add_corner(name, pos, fixed=True)
        builder.add_block("main", ("A", "D", "B", "C"), (4, 4))
        topo = builder.build()

        # Directly set nodes to uniform grid
        topo.grid.blocks[0].nodes = nodes
        topo.grid.global_nodes = np.full((topo.grid.global_node_count, 2), np.nan)
        dof_map = topo.grid.block_dof_maps[0]
        for i in range(ni):
            for j in range(nj):
                topo.grid.global_nodes[int(dof_map[i, j])] = nodes[i, j]

        target = IdentityTarget(d=2)
        F = assemble_energy(topo.grid, target, "shape")
        assert F < 1e-10, f"Expected zero energy, got {F}"

    def test_pack_unpack_roundtrip(self):
        """pack_x then unpack_x preserves the grid."""
        grid = _make_test_grid_2d()
        x0 = pack_x(grid)
        unpack_x(x0, grid)
        x1 = pack_x(grid)
        np.testing.assert_allclose(x0, x1)
