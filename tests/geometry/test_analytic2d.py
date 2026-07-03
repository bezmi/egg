"""Tests for 2D analytic geometry entities: Circle, LineSegment."""

import numpy as np
import pytest

from egg.geometry.analytic2d import Circle, Ellipse, LineSegment


class TestCircle:
    @pytest.fixture
    def circle(self):
        return Circle(center=(0.0, 0.0), radius=1.0)

    def test_project_on_surface(self, circle):
        q = circle.project(np.array([0.0, 2.0]))
        assert abs(np.linalg.norm(q) - 1.0) < 1e-14

    def test_project_center_returns_on_circle(self, circle):
        q = circle.project(np.array([0.0, 0.0]))
        assert abs(np.linalg.norm(q) - 1.0) < 1e-14

    def test_project_idempotent(self, circle):
        q_orig = np.array([1.0, 0.0])
        q_proj = circle.project(q_orig)
        np.testing.assert_allclose(q_proj, q_orig, atol=1e-14)

    def test_project_non_origin_center(self):
        c = Circle(center=(2.0, 3.0), radius=5.0)
        p = np.array([2.0, 13.0])
        q = c.project(p)
        assert abs(np.linalg.norm(q - c.center) - 5.0) < 1e-14

    def test_tangent_space_shape(self, circle):
        T = circle.tangent_space(np.array([1.0, 0.0]))
        assert T.shape == (2, 1)

    def test_tangent_is_unit(self, circle):
        for angle in [0.0, np.pi / 4, np.pi / 2, np.pi]:
            q = np.array([np.cos(angle), np.sin(angle)])
            T = circle.tangent_space(q)
            assert abs(np.linalg.norm(T) - 1.0) < 1e-14

    def test_tangent_orthogonal_to_normal(self, circle):
        q = np.array([0.6, 0.8])
        T = circle.tangent_space(q)
        n = circle.normal(q)
        assert abs(np.dot(T[:, 0], n)) < 1e-14

    def test_normal_is_unit(self, circle):
        n = circle.normal(np.array([0.0, 1.0]))
        assert abs(np.linalg.norm(n) - 1.0) < 1e-14

    def test_normal_points_outward(self, circle):
        n = circle.normal(np.array([1.0, 0.0]))
        assert n[0] > 0

    def test_tangent_direction_at_axis(self, circle):
        T = circle.tangent_space(np.array([1.0, 0.0]))
        assert T[1, 0] > 0
        assert abs(T[0, 0]) < 1e-14


class TestLineSegment:
    @pytest.fixture
    def segment(self):
        return LineSegment(start=(0.0, 0.0), end=(2.0, 0.0))

    def test_project_midpoint_above(self, segment):
        q = segment.project(np.array([1.0, 1.0]))
        np.testing.assert_allclose(q, [1.0, 0.0])

    def test_project_clamp_before_start(self, segment):
        q = segment.project(np.array([-1.0, 0.0]))
        np.testing.assert_allclose(q, [0.0, 0.0])

    def test_project_clamp_after_end(self, segment):
        q = segment.project(np.array([3.0, 0.0]))
        np.testing.assert_allclose(q, [2.0, 0.0])

    def test_project_on_segment_returns_self(self, segment):
        q = segment.project(np.array([1.0, 0.0]))
        np.testing.assert_allclose(q, [1.0, 0.0])

    def test_project_non_axis_aligned(self):
        seg = LineSegment(start=(0.0, 0.0), end=(3.0, 4.0))
        midpoint = np.array([1.5, 2.0])
        normal = np.array([-0.8, 0.6])
        q = seg.project(midpoint + normal)
        np.testing.assert_allclose(q, midpoint, atol=1e-14)

    def test_tangent_direction(self, segment):
        T = segment.tangent_space(np.array([1.0, 0.0]))
        np.testing.assert_allclose(T[:, 0], [1.0, 0.0], atol=1e-14)

    def test_tangent_is_unit(self, segment):
        T = segment.tangent_space(np.array([0.5, 0.0]))
        assert abs(np.linalg.norm(T) - 1.0) < 1e-14

    def test_normal_is_perpendicular(self, segment):
        T = segment.tangent_space(np.array([1.0, 0.0]))
        n = segment.normal(np.array([1.0, 0.0]))
        assert abs(np.dot(T[:, 0], n)) < 1e-14


class TestEllipse:
    @pytest.fixture
    def ellipse(self):
        return Ellipse(center=(0.0, 0.0), rx=2.0, ry=1.0)

    def test_project_on_ellipse(self, ellipse):
        q = ellipse.project(np.array([4.0, 0.0]))
        np.testing.assert_allclose(q, [2.0, 0.0], atol=1e-14)

    def test_project_above(self, ellipse):
        q = ellipse.project(np.array([0.0, 3.0]))
        np.testing.assert_allclose(q, [0.0, 1.0], atol=1e-14)

    def test_tangent_space_shape(self, ellipse):
        T = ellipse.tangent_space(np.array([2.0, 0.0]))
        assert T.shape == (2, 1)

    def test_tangent_is_unit(self, ellipse):
        T = ellipse.tangent_space(np.array([2.0, 0.0]))
        assert abs(np.linalg.norm(T) - 1.0) < 1e-14

    def test_tangent_orthogonal_to_normal(self, ellipse):
        q = np.array([0.0, 1.0])
        T = ellipse.tangent_space(q)
        n = ellipse.normal(q)
        assert abs(np.dot(T[:, 0], n)) < 1e-14

    def test_normal_is_unit(self, ellipse):
        n = ellipse.normal(np.array([2.0, 0.0]))
        assert abs(np.linalg.norm(n) - 1.0) < 1e-14

    def test_dim_is_one(self, ellipse):
        assert ellipse.dim == 1

    def test_degenerate_segment(self):
        seg = LineSegment(start=(1.0, 2.0), end=(1.0, 2.0))
        q = seg.project(np.array([3.0, 4.0]))
        np.testing.assert_allclose(q, [1.0, 2.0])
        T = seg.tangent_space(np.array([3.0, 4.0]))
        assert abs(np.linalg.norm(T) - 1.0) < 1e-14

    def test_ellipse_dim_is_one(self, ellipse):
        assert ellipse.dim == 1

    def test_circle_dim_is_one(self):
        c = Circle(center=(0.0, 0.0), radius=1.0)
        assert c.dim == 1
