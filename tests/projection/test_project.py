# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Tests for tangential sliding and the DOF constraint hierarchy."""

import numpy as np

from egg.geometry.analytic2d import Circle, LineSegment
from egg.projection.associate import build_dof_constraints
from egg.projection.project import tangential_slide, tangent_projector
from egg.topology.builder import TopologyBuilder


class TestTangentialSlide:
    def test_circle_node_stays_on_circle(self):
        c = Circle(center=(2.0, 2.0), radius=0.8)
        q = c.project(np.array([1.4, 1.5]))
        moved = tangential_slide(q, np.array([0.7, -0.3]), c)
        assert np.isclose(np.linalg.norm(moved - c.center), 0.8, atol=1e-12)

    def test_circle_motion_is_tangential(self):
        c = Circle(center=(2.0, 2.0), radius=0.8)
        q = c.project(np.array([2.9, 2.1]))
        moved = tangential_slide(q, np.array([0.05, 0.0]), c)
        # displacement should be (nearly) orthogonal to the radial normal
        n = c.normal(q)
        assert abs(np.dot(moved - q, n)) < 1e-3
        assert np.linalg.norm(moved - q) > 1e-4  # it actually moved

    def test_wall_node_clamps_at_corner(self):
        seg = LineSegment(start=(0.0, 0.0), end=(4.0, 0.0))
        near = np.array([0.1, 0.0])
        clamped = tangential_slide(near, np.array([-1.0, 0.0]), seg)
        np.testing.assert_allclose(clamped, [0.0, 0.0], atol=1e-12)

    def test_wall_node_slides_along_edge(self):
        seg = LineSegment(start=(0.0, 0.0), end=(4.0, 0.0))
        p = np.array([1.0, 0.0])
        moved = tangential_slide(p, np.array([0.5, 0.9]), seg)  # normal part dropped
        np.testing.assert_allclose(moved, [1.5, 0.0], atol=1e-12)

    def test_tangent_projector_idempotent(self):
        c = Circle(center=(0.0, 0.0), radius=1.0)
        P = tangent_projector(c, np.array([1.0, 0.0]))
        np.testing.assert_allclose(P @ P, P, atol=1e-12)


class TestConstraintHierarchy:
    def _two_wall_corner_topology(self):
        """One block with two adjacent faces on two different line segments."""
        bottom = LineSegment(start=(0.0, 0.0), end=(1.0, 0.0))
        left = LineSegment(start=(0.0, 0.0), end=(0.0, 1.0))
        b = TopologyBuilder(d=2)
        for n, p in [("sw", (0, 0)), ("se", (1, 0)), ("ne", (1, 1)), ("nw", (0, 1))]:
            b.add_corner(n, p, fixed=False)
        b.add_block("blk", ("sw", "nw", "se", "ne"), (3, 3))
        b.associate("blk", 1, 0, bottom)  # axis1 side0 -> bottom (sw-se)
        b.associate("blk", 0, 0, left)  # axis0 side0 -> left (sw-nw)
        topo = b.build()
        topo.initialize_grid()
        return topo, bottom, left

    def test_single_entity_slides_two_entities_fixed(self):
        topo, bottom, left = self._two_wall_corner_topology()
        grid = topo.grid
        dof_constraints, fixed_dofs = build_dof_constraints(topo, grid)

        # The shared corner sw=(0,0) lies on both faces -> fixed
        sw_gid = int(np.argmin(np.linalg.norm(grid.global_nodes, axis=1)))
        assert sw_gid in fixed_dofs
        assert sw_gid not in dof_constraints

        # An interior-of-edge node on only the bottom face -> sliding on bottom
        bottom_mid = int(
            np.argmin(np.linalg.norm(grid.global_nodes - np.array([0.5, 0.0]), axis=1))
        )
        assert dof_constraints.get(bottom_mid) is bottom

    def test_constraints_written_to_grid(self):
        topo, bottom, left = self._two_wall_corner_topology()
        # initialize_grid should have populated grid.dof_constraints and locked
        # the shared corner only.
        assert len(topo.grid.dof_constraints) > 0
        sw_gid = int(np.argmin(np.linalg.norm(topo.grid.global_nodes, axis=1)))
        assert not topo.grid.free_mask[sw_gid]
