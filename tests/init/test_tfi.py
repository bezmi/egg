"""Tests for Transfinite Interpolation."""

import numpy as np

from egg.core.types import Block
from egg.init.tfi import tfi_fill_interior
from egg.topology.builder import TopologyBuilder


class TestTFIExactness:
    def test_affine_exact_2d(self):
        """TFI reproduces an affine grid exactly."""
        ni, nj = 5, 7
        nodes = np.zeros((ni, nj, 2))
        for i in range(ni):
            for j in range(nj):
                nodes[i, j, 0] = 2.0 * i + 0.5 * j
                nodes[i, j, 1] = 0.3 * i + 1.5 * j
        orig = nodes.copy()
        block = Block(nodes)
        tfi_fill_interior(block)
        max_err = np.max(np.abs(block.nodes - orig))
        assert max_err < 1e-14

    def test_affine_exact_3d(self):
        """TFI reproduces an affine 3D grid exactly."""
        ni, nj, nk = 4, 5, 6
        nodes = np.zeros((ni, nj, nk, 3))
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    nodes[i, j, k, 0] = 2.0 * i + 0.5 * j - 0.2 * k
                    nodes[i, j, k, 1] = 0.3 * i + 1.5 * j + 0.1 * k
                    nodes[i, j, k, 2] = -0.1 * i + 0.2 * j + 2.0 * k
        orig = nodes.copy()
        block = Block(nodes)
        tfi_fill_interior(block)
        max_err = np.max(np.abs(block.nodes - orig))
        assert max_err < 1e-14


class TestTFIBoundary:
    def test_boundary_preserved_2d(self):
        """TFI does not modify boundary nodes."""
        ni, nj = 6, 8
        nodes = np.random.randn(ni, nj, 2)
        block = Block(nodes)
        orig = nodes.copy()
        tfi_fill_interior(block)
        for i in range(ni):
            for j in range(nj):
                if i == 0 or i == ni - 1 or j == 0 or j == nj - 1:
                    assert np.allclose(block.nodes[i, j], orig[i, j])

    def test_boundary_preserved_3d(self):
        """TFI does not modify boundary nodes in 3D."""
        ni, nj, nk = 4, 5, 6
        nodes = np.random.randn(ni, nj, nk, 3)
        block = Block(nodes)
        orig = nodes.copy()
        tfi_fill_interior(block)
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    on_boundary = (
                        i == 0 or i == ni - 1
                        or j == 0 or j == nj - 1
                        or k == 0 or k == nk - 1
                    )
                    if on_boundary:
                        assert np.allclose(block.nodes[i, j, k], orig[i, j, k])


class TestTFIIdempotence:
    def test_idempotent_2d(self):
        """TFI of an already-TFI grid returns unchanged."""
        ni, nj = 6, 8
        nodes = np.random.randn(ni, nj, 2)
        block = Block(nodes)
        tfi_fill_interior(block)
        after_first = block.nodes.copy()
        tfi_fill_interior(block)
        after_second = block.nodes.copy()
        max_diff = np.max(np.abs(after_first - after_second))
        assert max_diff < 1e-14

    def test_idempotent_3d(self):
        """TFI of an already-TFI 3D grid returns unchanged."""
        ni, nj, nk = 4, 5, 6
        nodes = np.random.randn(ni, nj, nk, 3)
        block = Block(nodes)
        tfi_fill_interior(block)
        after_first = block.nodes.copy()
        tfi_fill_interior(block)
        after_second = block.nodes.copy()
        max_diff = np.max(np.abs(after_first - after_second))
        assert max_diff < 1e-14


class TestTFITangling:
    @staticmethod
    def _cell_det_A(nodes: np.ndarray, cell_base: tuple) -> float:
        """Compute det(A) for a 2D cell using central differences."""
        i0, j0 = cell_base
        # Jacobian columns: (x_i, y_i) and (x_j, y_j)
        # FD along i: (node(i0+1,j0) - node(i0,j0) + node(i0+1,j0+1) - node(i0,j0+1)) / 2
        # Simpler: just use the four corners
        # A = [X(i+1,j) - X(i,j), X(i,j+1) - X(i,j)]
        #     [Y(i+1,j) - Y(i,j), Y(i,j+1) - Y(i,j)]
        A = np.zeros((2, 2))
        A[:, 0] = nodes[i0 + 1, j0] - nodes[i0, j0]
        A[:, 1] = nodes[i0, j0 + 1] - nodes[i0, j0]
        return float(np.linalg.det(A))

    def test_tangling_diagnostic_fires(self):
        """A deliberately crossed topology yields det A <= 0."""
        builder = TopologyBuilder(d=2)
        # Crossed corners: swap the top two corners to create a bowtie
        builder.add_corner("sw", (0.0, 0.0))
        builder.add_corner("se", (4.0, 0.0))
        builder.add_corner("nw", (0.0, 4.0))
        builder.add_corner("ne", (4.0, 4.0))
        # Deliberately swap nw and ne in the corner list to cross edges
        builder.add_block("crossed", ("sw", "ne", "se", "nw"), (4, 4))
        topo = builder.build()
        topo.initialize_grid()
        nodes = topo.grid.blocks[0].nodes

        min_det = float("inf")
        for i in range(4):
            for j in range(4):
                det_val = self._cell_det_A(nodes, (i, j))
                if det_val < min_det:
                    min_det = det_val
        assert min_det < 0, f"Expected tangled grid (min det < 0), got min det = {min_det}"
