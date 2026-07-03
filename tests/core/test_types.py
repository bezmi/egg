"""Tests for Block and MultiBlockGrid data model."""

import numpy as np

from egg.core.types import Block
from egg.topology.builder import TopologyBuilder


class TestBlock:
    def test_d_derived_2d(self):
        b = Block(np.zeros((10, 20, 2)))
        assert b.d == 2
        assert b.logical_shape == (10, 20)

    def test_d_derived_3d(self):
        b = Block(np.zeros((5, 6, 7, 3)))
        assert b.d == 3
        assert b.logical_shape == (5, 6, 7)

    def test_cell_count_2d(self):
        b = Block(np.zeros((10, 20, 2)))
        assert b.cell_count() == (9, 19)

    def test_cell_count_3d(self):
        b = Block(np.zeros((5, 6, 7, 3)))
        assert b.cell_count() == (4, 5, 6)

    def test_iter_cells_count_2d(self):
        b = Block(np.zeros((10, 20, 2)))
        assert len(list(b.iter_cells())) == 9 * 19

    def test_iter_cells_count_3d(self):
        b = Block(np.zeros((5, 6, 7, 3)))
        assert len(list(b.iter_cells())) == 4 * 5 * 6

    def test_corner_indices_count_2d(self):
        b = Block(np.zeros((5, 5, 2)))
        assert len(b.corner_indices((0, 0))) == 4  # 2**2

    def test_corner_indices_count_3d(self):
        b = Block(np.zeros((3, 3, 3, 3)))
        assert len(b.corner_indices((0, 0, 0))) == 8  # 2**3

    def test_corner_indices_values_2d(self):
        b = Block(np.zeros((5, 5, 2)))
        corners = b.corner_indices((2, 3))
        assert set(corners) == {(2, 3), (2, 4), (3, 3), (3, 4)}

    def test_boundary_faces_count_2d(self):
        b = Block(np.zeros((5, 7, 2)))
        assert len(list(b.iter_boundary_faces())) == 4  # 2*d

    def test_boundary_faces_count_3d(self):
        b = Block(np.zeros((3, 4, 5, 3)))
        assert len(list(b.iter_boundary_faces())) == 6  # 2*d

    def test_is_interior_2d(self):
        b = Block(np.zeros((5, 5, 2)))
        assert b.is_interior((2, 2))
        assert not b.is_interior((0, 2))
        assert not b.is_interior((4, 2))
        assert not b.is_interior((2, 0))
        assert not b.is_interior((2, 4))

    def test_node_patch_count(self):
        """Interior node in 2D has 4 incident cells."""
        b = Block(np.zeros((5, 5, 2)))
        assert len(b.node_patch((2, 2))) == 4

    def test_node_patch_boundary(self):
        """Corner node has 1 incident cell."""
        b = Block(np.zeros((5, 5, 2)))
        assert len(b.node_patch((0, 0))) == 1

    def test_face_node_slice_axis0_low(self):
        b = Block(np.zeros((5, 7, 2)))
        idx = b.face_node_slice(0, 0)
        assert idx[0] == 0
        assert idx[1] == slice(None)

    def test_face_node_slice_axis1_high(self):
        b = Block(np.zeros((5, 7, 2)))
        idx = b.face_node_slice(1, 1)
        assert idx[0] == slice(None)
        assert idx[1] == 6  # 7-1


class TestMultiBlockGrid:
    def test_interface_map_shared_dofs(self):
        """Two blocks sharing an edge: shared nodes have one global DOF each."""
        builder = TopologyBuilder(d=2)
        (
            builder.add_corner("A", (0.0, 0.0), fixed=False)
            .add_corner("B", (2.0, 0.0), fixed=False)
            .add_corner("C", (2.0, 2.0), fixed=False)
            .add_corner("D", (0.0, 2.0), fixed=False)
            .add_corner("E", (4.0, 0.0), fixed=False)
            .add_corner("F", (4.0, 2.0), fixed=False)
            .add_block("L", ("A", "D", "B", "C"), (4, 4))
            .add_block("R", ("B", "C", "E", "F"), (4, 4))
            .connect("L", 0, 1, "R", 0, 0)
        )
        topo = builder.build()
        grid = topo.grid
        # Each block: 5*5 = 25, shared edge has 5 nodes
        # Unique: 25 + 25 - 5 = 45
        assert grid.global_node_count == 45

        # Verify shared DOFs match
        dof_L = grid.block_dof_maps[0]
        dof_R = grid.block_dof_maps[1]
        shared_L = dof_L[-1, :]  # i=hi in L
        shared_R = dof_R[0, :]  # i=lo in R
        assert np.array_equal(shared_L, shared_R)

    def test_free_mask_corners_fixed(self):
        """Corners with fixed=True are NOT in free_mask."""
        builder = TopologyBuilder(d=2)
        (
            builder.add_corner("A", (0.0, 0.0), fixed=True)
            .add_corner("B", (2.0, 0.0), fixed=True)
            .add_corner("C", (2.0, 2.0), fixed=True)
            .add_corner("D", (0.0, 2.0), fixed=True)
            .add_block("main", ("A", "D", "B", "C"), (4, 4))
        )
        topo = builder.build()
        grid = topo.grid
        # 5*5 = 25 nodes, 4 fixed corners
        assert grid.global_node_count == 25
        assert np.sum(grid.free_mask) == 21
        assert np.sum(~grid.free_mask) == 4

    def test_free_mask_corners_free(self):
        """Corners with fixed=False ARE in free_mask."""
        builder = TopologyBuilder(d=2)
        (
            builder.add_corner("A", (0.0, 0.0), fixed=False)
            .add_corner("B", (2.0, 0.0), fixed=False)
            .add_corner("C", (2.0, 2.0), fixed=False)
            .add_corner("D", (0.0, 2.0), fixed=False)
            .add_block("main", ("A", "D", "B", "C"), (4, 4))
        )
        topo = builder.build()
        grid = topo.grid
        assert np.sum(grid.free_mask) == 25
