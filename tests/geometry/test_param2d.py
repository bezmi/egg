"""Standalone-Python parametric interface (eval/deriv/t0/t1) on 2D entities."""

import numpy as np
import pytest

from egg.geometry.analytic2d import Circle, Ellipse, LineSegment
from egg.geometry.curves2d import CircleArc, CompositePath


def finite_diff(entity, t, h=1e-6):
    return (entity.eval(t + h) - entity.eval(t - h)) / (2.0 * h)


class TestEvalFrac:
    def test_line_frac_equals_native(self):
        seg = LineSegment((1.0, 2.0), (3.0, 6.0))
        np.testing.assert_allclose(seg.eval_frac(0.5), seg.eval(0.5))

    def test_circle_frac_maps_to_angle(self):
        c = Circle(center=(0.0, 0.0), radius=1.0)
        np.testing.assert_allclose(c.eval_frac(0.5), c.eval(np.pi),
                                   atol=1e-15)
        np.testing.assert_allclose(c.eval_frac(0.25), [0.0, 1.0], atol=1e-15)

    def test_arc_frac_maps_onto_trim(self):
        arc = CircleArc((0.0, 0.0), 2.0, 0.0, np.pi / 2)
        np.testing.assert_allclose(arc.eval_frac(0.5), arc.eval(np.pi / 4))

    def test_frac_clamps(self):
        seg = LineSegment((0.0, 0.0), (1.0, 0.0))
        np.testing.assert_allclose(seg.eval_frac(-1.0), [0.0, 0.0])
        np.testing.assert_allclose(seg.eval_frac(2.0), [1.0, 0.0])


class TestLineSegmentParam:
    def test_endpoints_and_midpoint(self):
        seg = LineSegment((1.0, 2.0), (3.0, 6.0))
        assert seg.t0 == 0.0 and seg.t1 == 1.0 and not seg.closed
        np.testing.assert_allclose(seg.eval(0.0), [1.0, 2.0])
        np.testing.assert_allclose(seg.eval(1.0), [3.0, 6.0])
        np.testing.assert_allclose(seg.eval(0.5), [2.0, 4.0])

    def test_deriv(self):
        seg = LineSegment((1.0, 2.0), (3.0, 6.0))
        np.testing.assert_allclose(seg.deriv(0.3), finite_diff(seg, 0.3),
                                   atol=1e-8)


class TestCircleParam:
    def test_eval_on_circle(self):
        c = Circle(center=(2.0, -1.0), radius=0.5)
        assert c.closed
        for t in np.linspace(c.t0, c.t1, 9):
            p = c.eval(t)
            assert np.linalg.norm(p - c.center) == pytest.approx(0.5)
        np.testing.assert_allclose(c.eval(0.0), [2.5, -1.0])
        np.testing.assert_allclose(c.eval(np.pi / 2), [2.0, -0.5], atol=1e-15)

    def test_deriv(self):
        c = Circle(center=(2.0, -1.0), radius=0.5)
        np.testing.assert_allclose(c.deriv(1.1), finite_diff(c, 1.1),
                                   atol=1e-8)


class TestEllipseParam:
    def test_eval_and_deriv(self):
        e = Ellipse(center=(0.0, 0.0), rx=2.0, ry=1.0)
        assert e.closed
        np.testing.assert_allclose(e.eval(0.0), [2.0, 0.0])
        np.testing.assert_allclose(e.eval(np.pi / 2), [0.0, 1.0], atol=1e-15)
        np.testing.assert_allclose(e.deriv(0.7), finite_diff(e, 0.7),
                                   atol=1e-8)


class TestCompositePathParam:
    def test_arc_length_proportional_breaks(self):
        # 3-long then 1-long segment: the joint sits at t = 0.75.
        path = CompositePath([
            LineSegment((0.0, 0.0), (3.0, 0.0)),
            LineSegment((3.0, 0.0), (3.0, 1.0)),
        ])
        np.testing.assert_allclose(path.eval(0.0), [0.0, 0.0])
        np.testing.assert_allclose(path.eval(0.75), [3.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(path.eval(1.0), [3.0, 1.0])
        # Uniform speed within each straight segment.
        np.testing.assert_allclose(path.eval(0.25), [1.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(path.eval(0.875), [3.0, 0.5], atol=1e-12)

    def test_eval_clamps_out_of_range(self):
        path = CompositePath([LineSegment((0.0, 0.0), (1.0, 0.0))])
        np.testing.assert_allclose(path.eval(-0.5), [0.0, 0.0])
        np.testing.assert_allclose(path.eval(1.5), [1.0, 0.0])

    def test_deriv_chain_rule(self):
        path = CompositePath([
            LineSegment((0.0, 0.0), (3.0, 0.0)),
            CircleArc((3.0, 1.0), 1.0, -np.pi / 2, 0.0),
        ])
        for t in (0.3, 0.9):
            np.testing.assert_allclose(path.deriv(t), finite_diff(path, t),
                                       atol=1e-6)

    def test_deriv_respects_segment_trim(self):
        # A trimmed arc: local parameter spans [t0, t1] != [0, 1].
        arc = CircleArc((0.0, 0.0), 2.0, 0.0, np.pi)
        path = CompositePath([arc])
        np.testing.assert_allclose(path.eval(0.5), [0.0, 2.0], atol=1e-12)
        np.testing.assert_allclose(path.deriv(0.5), finite_diff(path, 0.5),
                                   atol=1e-6)
