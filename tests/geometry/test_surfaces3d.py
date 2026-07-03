# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Tests for 3D analytic and NURBS surface entities."""

import numpy as np
import pytest

from egg.geometry.analytic3d import Cylinder, Line3, Plane, Sphere
from egg.geometry.surfaces3d import BSplineSurface


class TestPlane:
    @pytest.fixture
    def plane(self):
        return Plane(origin=(0.0, 0.0, 0.0), ax=(1.0, 0.0, 0.0), ay=(0.0, 1.0, 0.0))

    def test_project_drops_onto_plane(self, plane):
        p = plane.project(np.array([1.0, 2.0, 5.0]))
        np.testing.assert_allclose(p, [1.0, 2.0, 0.0], atol=1e-14)

    def test_tangent_space_is_constant(self, plane):
        ts = plane.tangent_space(np.array([0.0, 0.0, 0.0]))
        assert ts.shape == (3, 2)
        np.testing.assert_allclose(ts.T @ ts, np.eye(2), atol=1e-14)

    def test_normal_is_az(self, plane):
        np.testing.assert_allclose(
            plane.normal(np.array([1.0, 1.0, 0.0])), [0.0, 0.0, 1.0], atol=1e-14
        )

    def test_arbitrary_frame_is_orthonormalized(self):
        pl = Plane((1, 1, 1), (1, 1, 0), (0, 1, 1))
        np.testing.assert_allclose(pl.ax @ pl.ay, 0.0, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(pl.ax), 1.0)
        np.testing.assert_allclose(np.linalg.norm(pl.ay), 1.0)


class TestSphere:
    @pytest.fixture
    def sphere(self):
        return Sphere(
            center=(0.0, 0.0, 0.0), radius=1.0, ax=(1.0, 0.0, 0.0), ay=(0.0, 1.0, 0.0)
        )

    def test_project_is_radial(self, sphere):
        pr = sphere.project(np.array([4.0, 4.0, 5.0]))
        assert abs(np.linalg.norm(pr) - 1.0) < 1e-14

    def test_project_center_returns_on_sphere(self, sphere):
        pr = sphere.project(np.array([0.0, 0.0, 0.0]))
        assert abs(np.linalg.norm(pr) - 1.0) < 1e-14

    def test_tangent_space_orthonormal(self, sphere):
        ts = sphere.tangent_space(np.array([4.0, 4.0, 5.0]))
        assert ts.shape == (3, 2)
        np.testing.assert_allclose(ts.T @ ts, np.eye(2), atol=1e-12)

    def test_normal_is_radial(self, sphere):
        n = sphere.normal(np.array([4.0, 4.0, 5.0]))
        expected = np.array([4.0, 4.0, 5.0]) / np.linalg.norm([4.0, 4.0, 5.0])
        np.testing.assert_allclose(n, expected, atol=1e-14)


class TestCylinder:
    @pytest.fixture
    def cylinder(self):
        return Cylinder(
            origin=(0.0, 0.0, 0.0), ax=(1.0, 0.0, 0.0), ay=(0.0, 1.0, 0.0), radius=1.0
        )

    def test_project_onto_surface(self, cylinder):
        pr = cylinder.project(np.array([4.0, 0.0, 5.0]))
        np.testing.assert_allclose(pr, [1.0, 0.0, 5.0], atol=1e-14)

    def test_tangent_space_orthonormal(self, cylinder):
        ts = cylinder.tangent_space(np.array([4.0, 0.0, 5.0]))
        assert ts.shape == (3, 2)
        np.testing.assert_allclose(ts.T @ ts, np.eye(2), atol=1e-12)

    def test_normal_is_cross_section_radial(self, cylinder):
        n = cylinder.normal(np.array([4.0, 0.0, 5.0]))
        np.testing.assert_allclose(n, [1.0, 0.0, 0.0], atol=1e-14)


class TestLine3:
    @pytest.fixture
    def line(self):
        return Line3(p0=(0.0, 0.0, 0.0), p1=(1.0, 1.0, 0.0))

    def test_project_onto_line(self, line):
        pr = line.project(np.array([0.5, 0.5, 5.0]))
        np.testing.assert_allclose(pr, [0.5, 0.5, 0.0], atol=1e-14)

    def test_clamps_to_endpoint(self, line):
        pr = line.project(np.array([3.0, 3.0, 0.0]))
        np.testing.assert_allclose(pr, [1.0, 1.0, 0.0], atol=1e-14)

    def test_tangent_space(self, line):
        ts = line.tangent_space(np.array([0.0, 0.0, 0.0]))
        assert ts.shape == (3, 1)
        np.testing.assert_allclose(
            ts, [[1 / np.sqrt(2)], [1 / np.sqrt(2)], [0.0]], atol=1e-14
        )


class TestBSplineSurfacePolynomial:
    @pytest.fixture
    def bilinear(self):
        ku = np.array([0, 0, 1, 1])
        kv = np.array([0, 0, 1, 1])
        ctrl = np.array(
            [
                [[0, 0, 0], [0, 1, 0]],
                [[2, 0, 0], [2, 1, 0]],
            ]
        )
        return BSplineSurface(1, 1, ku, kv, ctrl)

    def test_eval_bilinear(self, bilinear):
        p = bilinear.eval(0.5, 0.5)
        np.testing.assert_allclose(p, [1.0, 0.5, 0.0], atol=1e-14)

    def test_project_onto_surface(self, bilinear):
        pr = bilinear.project(np.array([1.0, 0.5, 5.0]))
        np.testing.assert_allclose(pr, [1.0, 0.5, 0.0], atol=1e-9)

    def test_tangent_space_orthonormal(self, bilinear):
        ts = bilinear.tangent_space(np.array([1.0, 0.5, 5.0]))
        assert ts.shape == (3, 2)
        np.testing.assert_allclose(ts.T @ ts, np.eye(2), atol=1e-12)

    def test_derivatives_match_finite_differences(self, bilinear):
        h = 1e-6
        for u, v in [(0.3, 0.5), (0.5, 0.7)]:
            Su_fd = (bilinear.eval(u + h, v) - bilinear.eval(u - h, v)) / (2 * h)
            Sv_fd = (bilinear.eval(u, v + h) - bilinear.eval(u, v - h)) / (2 * h)
            Su, Sv = bilinear.frame(u, v)
            np.testing.assert_allclose(Su, Su_fd, atol=1e-5)
            np.testing.assert_allclose(Sv, Sv_fd, atol=1e-5)


class TestBSplineSurfaceNURBS:
    @pytest.fixture
    def quarter_cylinder(self):
        """Rational quadratic NURBS quarter-cylinder: radius 2, axis z, 0≤z≤1."""
        ku = np.array([0, 0, 0, 1, 1, 1])
        kv = np.array([0, 0, 1, 1])
        ctrl = np.array(
            [
                [[2, 0, 0], [2, 0, 1]],
                [[2, 2, 0], [2, 2, 1]],
                [[0, 2, 0], [0, 2, 1]],
            ]
        )
        w = np.array([[1, 1], [1 / np.sqrt(2), 1 / np.sqrt(2)], [1, 1]])
        return BSplineSurface(2, 1, ku, kv, ctrl, weights=w)

    def test_lies_exactly_on_the_cylinder(self, quarter_cylinder):
        for v in [0.0, 0.5, 1.0]:
            p = quarter_cylinder.eval(0.5, v)
            r = np.hypot(p[0], p[1])
            assert abs(r - 2.0) < 1e-12, f"radius {r} at v={v}"

    def test_clamped_ends_interpolate_control_points(self, quarter_cylinder):
        np.testing.assert_allclose(quarter_cylinder.eval(0.0, 0.0), [2, 0, 0])
        np.testing.assert_allclose(quarter_cylinder.eval(1.0, 0.0), [0, 2, 0])
        np.testing.assert_allclose(
            quarter_cylinder.eval(0.5, 1.0), [np.sqrt(2), np.sqrt(2), 1.0]
        )

    def test_projection_onto_surface(self, quarter_cylinder):
        q = np.array([np.sqrt(2), np.sqrt(2), 0.5])
        pr = quarter_cylinder.project(q)
        np.testing.assert_allclose(pr, q, atol=1e-9)

    def test_tangent_space_orthonormal(self, quarter_cylinder):
        q = np.array([np.sqrt(2), np.sqrt(2), 0.5])
        ts = quarter_cylinder.tangent_space(q)
        assert ts.shape == (3, 2)
        np.testing.assert_allclose(ts.T @ ts, np.eye(2), atol=1e-10)

    def test_rational_derivatives_match_finite_differences(self, quarter_cylinder):
        h = 1e-6
        for u, v in [(0.3, 0.5), (0.5, 0.2)]:
            Su_fd = (
                quarter_cylinder.eval(u + h, v) - quarter_cylinder.eval(u - h, v)
            ) / (2 * h)
            Sv_fd = (
                quarter_cylinder.eval(u, v + h) - quarter_cylinder.eval(u, v - h)
            ) / (2 * h)
            Su, Sv = quarter_cylinder.frame(u, v)
            np.testing.assert_allclose(Su, Su_fd, atol=1e-5)
            np.testing.assert_allclose(Sv, Sv_fd, atol=1e-5)

    def test_all_ones_weights_equal_polynomial(self):
        ku = np.array([0, 0, 0, 1, 1, 1])
        kv = np.array([0, 0, 1, 1])
        ctrl = np.array(
            [
                [[0, 0, 0], [0, 0, 1]],
                [[1, 0, 0], [1, 0, 1]],
                [[2, 0, 0], [2, 0, 1]],
            ]
        )
        poly = BSplineSurface(2, 1, ku, kv, ctrl)
        rat = BSplineSurface(2, 1, ku, kv, ctrl, weights=np.ones((3, 2)))
        for u in [0.3, 0.5, 0.7]:
            for v in [0.2, 0.8]:
                np.testing.assert_allclose(poly.eval(u, v), rat.eval(u, v), atol=1e-13)
