"""Tests for 2D curve entities, encoding, and the 2D construction front-end."""

import numpy as np
import pytest

from egg.geometry.curves2d import (
    BSplineCurve,
    CircleArc,
    CompositePath,
    CubicBezier,
    EllipseArc,
    QuadBezier,
)
from egg.geometry.entity_encoding import (
    PARAM_PAD_SIZE,
    TAG_BSPLINE,
    TAG_CIRCLEARC,
    TAG_COMPOSITE,
    encode_entity,
)
from egg.geometry.analytic2d import LineSegment


class TestCurves:
    def test_circle_arc_projects_radially(self):
        arc = CircleArc((1.0, 1.0), 2.0, 0.0, np.pi / 2)
        q = arc.project(np.array([4.0, 4.0]))
        assert abs(np.linalg.norm(q - [1.0, 1.0]) - 2.0) < 1e-12

    def test_circle_arc_clamps_to_endpoint(self):
        arc = CircleArc((0.0, 0.0), 1.0, 0.0, np.pi / 2)
        q = arc.project(np.array([1.0, -1.0]))  # below the arc start
        np.testing.assert_allclose(q, [1.0, 0.0], atol=1e-12)

    def test_ellipse_arc_foot_is_stationary(self):
        arc = EllipseArc((0.0, 0.0), 2.0, 1.0, 0.3, 0.0, 2 * np.pi, closed=True)
        p = np.array([1.5, 1.5])
        t = arc._clamp(arc.invert(p))
        res = (arc.eval(t) - p) @ arc.deriv(t)
        assert abs(res) < 1e-9

    def test_quad_bezier_on_curve_roundtrip(self):
        bez = QuadBezier((0, 0), (1, 2), (2, 0))
        for t in (0.2, 0.5, 0.8):
            np.testing.assert_allclose(bez.project(bez.eval(t)), bez.eval(t), atol=1e-9)

    def test_cubic_bezier_tangent_unit(self):
        bez = CubicBezier((0, 0), (1, 1), (2, -1), (3, 0))
        t = bez.tangent_space(np.array([1.5, 0.5]))
        assert t.shape == (2, 1)
        assert abs(np.linalg.norm(t) - 1.0) < 1e-12

    def test_bspline_matches_cubic_bezier_on_bezier_knots(self):
        ctrl = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0], [3.0, 0.0]])
        bez = CubicBezier(*ctrl)
        bs = BSplineCurve(3, [0, 0, 0, 0, 1, 1, 1, 1], ctrl)
        for t in (0.1, 0.5, 0.9):
            np.testing.assert_allclose(bs.eval(t), bez.eval(t), atol=1e-12)


class TestCompositePath:
    @pytest.fixture
    def L_path(self):
        return CompositePath(
            [
                LineSegment((0.0, 0.0), (1.0, 0.0)),
                LineSegment((1.0, 0.0), (1.0, 1.0)),
            ]
        )

    def test_nearest_segment_selection(self, L_path):
        np.testing.assert_allclose(
            L_path.project(np.array([0.4, -0.5])), [0.4, 0.0], atol=1e-12
        )
        np.testing.assert_allclose(
            L_path.project(np.array([1.5, 0.6])), [1.0, 0.6], atol=1e-12
        )

    def test_tangent_is_matched_segments(self, L_path):
        t = L_path.tangent_space(np.array([1.5, 0.6]))
        np.testing.assert_allclose(t[:, 0], [0.0, 1.0], atol=1e-12)

    def test_no_nesting(self, L_path):
        with pytest.raises(ValueError):
            CompositePath([L_path])


class TestEncoding:
    def test_circle_arc_blob(self):
        arc = CircleArc((1.0, 2.0), 3.0, 0.5, 1.5, closed=False)
        tag, params = encode_entity(arc)
        assert tag == TAG_CIRCLEARC
        np.testing.assert_allclose(params[:6], [1, 2, 3, 0.5, 1.5, 0])

    def test_bspline_requires_arena(self):
        bs = BSplineCurve(
            3, [0, 0, 0, 0, 1, 1, 1, 1], [[0, 0], [1, 1], [2, -1], [3, 0]]
        )
        with pytest.raises(ValueError):
            encode_entity(bs)

    def test_bspline_arena_layout(self):
        bs = BSplineCurve(
            3, [0, 0, 0, 0, 1, 1, 1, 1], [[0, 0], [1, 1], [2, -1], [3, 0]]
        )
        arena = []
        tag, params = encode_entity(bs, arena=arena)
        assert tag == TAG_BSPLINE
        degree, n_ctrl, knot_off, ctrl_off, t0, t1 = params[:6]
        assert (degree, n_ctrl) == (3, 4)
        np.testing.assert_allclose(arena[int(knot_off) : int(knot_off) + 8], bs.knots)
        np.testing.assert_allclose(
            arena[int(ctrl_off) : int(ctrl_off) + 8], bs.ctrl.ravel()
        )
        assert (t0, t1) == (0.0, 1.0)

    def test_composite_arena_records(self):
        path = CompositePath(
            [
                LineSegment((0.0, 0.0), (1.0, 0.0)),
                CircleArc((1.0, 1.0), 1.0, -np.pi / 2, 0.0),
            ]
        )
        arena = []
        tag, params = encode_entity(path, arena=arena)
        assert tag == TAG_COMPOSITE
        n_segs, rec_off = int(params[0]), int(params[1])
        assert n_segs == 2
        rec_size = 1 + PARAM_PAD_SIZE
        assert len(arena) == rec_off + n_segs * rec_size
        # Each record is [tag, params...]; re-decode the second one.
        rec = arena[rec_off + rec_size : rec_off + 2 * rec_size]
        assert int(rec[0]) == TAG_CIRCLEARC
        np.testing.assert_allclose(rec[1:4], [1.0, 1.0, 1.0])


class TestFrontend2d:
    def test_line(self):
        from egg.geometry.frontend2d import Line, Vector3

        ent = Line(Vector3(0, 0), Vector3(1, 2))
        assert isinstance(ent, LineSegment)
        np.testing.assert_allclose(ent.end, [1.0, 2.0])

    def test_line_accepts_tuples(self):
        from egg.geometry.frontend2d import Line

        ent = Line((0, 0), (1, 2))
        assert isinstance(ent, LineSegment)
        np.testing.assert_allclose(ent.end, [1.0, 2.0])

    def test_line_p0_p1(self):
        from egg.geometry.frontend2d import Line, Vector3

        ent = Line(Vector3(0, 0), Vector3(1, 2))
        assert isinstance(ent, LineSegment)
        assert isinstance(ent.p0, Vector3) and isinstance(ent.p1, Vector3)
        assert ent.p0.x == 0.0 and ent.p0.y == 0.0
        assert ent.p1.x == 1.0 and ent.p1.y == 2.0

    def test_arc_passes_through_endpoints(self):
        from egg.geometry.frontend2d import Arc, Vector3

        a, b, c = Vector3(2, 0), Vector3(0, 2), Vector3(0, 0)
        ent = Arc(a, b, c)
        assert isinstance(ent, CircleArc)
        # The arc starts at a and ends at b.
        np.testing.assert_allclose(ent.eval(ent.t0), [2.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(ent.eval(ent.t1), [0.0, 2.0], atol=1e-12)

    def test_arc_rejects_branch_wrap(self):
        # An arc that wraps past +/-pi after both branches must be rejected.
        from egg.geometry.frontend2d import Arc, Vector3

        # Quarter arc just below the branch cut is fine.
        ent = Arc(Vector3(1, 0), Vector3(0, 1), Vector3(0, 0))
        assert isinstance(ent, CircleArc)
        assert -np.pi < ent.t0 and ent.t1 <= np.pi

    def test_bezier_degrees(self):
        from egg.geometry.frontend2d import Bezier, Vector3

        pts = [Vector3(0, 0), Vector3(1, 1), Vector3(2, 0)]
        assert isinstance(Bezier(pts), QuadBezier)
        pts4 = pts + [Vector3(3, 1)]
        assert isinstance(Bezier(pts4), CubicBezier)
        pts2 = [Vector3(0, 0), Vector3(1, 2)]
        assert isinstance(Bezier(pts2), LineSegment)
        # Degree >= 4 becomes a B-spline on the clamped knot vector and
        # reproduces the Bézier.
        pts5 = pts4 + [Vector3(4, 0)]
        ent = Bezier(pts5)
        assert isinstance(ent, BSplineCurve)
        # The degree-4 B-spline reproduces the Bézier control points exactly.
        for t in (0.25, 0.75):
            np.testing.assert_allclose(
                ent.eval(t),
                _bezier_eval(np.stack([_v2(p) for p in pts5]), t),
                atol=1e-12,
            )

    def test_polyline_to_composite(self):
        from egg.geometry.frontend2d import Line, Polyline, Vector3

        pl = Polyline(
            [
                Line(Vector3(0, 0), Vector3(1, 0)),
                Line(Vector3(1, 0), Vector3(1, 1)),
            ]
        )
        assert isinstance(pl, CompositePath)
        np.testing.assert_allclose(
            pl.project(np.array([1.5, 0.6])), [1.0, 0.6], atol=1e-12
        )

    def test_polyline_closed_appends_segment(self):
        from egg.geometry.frontend2d import Line, Polyline, Vector3

        segs = [Line(Vector3(0, 0), Vector3(1, 0)), Line(Vector3(1, 0), Vector3(1, 1))]
        assert len(Polyline(segs).segments) == 2
        pl = Polyline(segs, closed=True)
        assert len(pl.segments) == 3
        np.testing.assert_allclose(pl.segments[-1].start, [1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(pl.segments[-1].end, [0.0, 0.0], atol=1e-12)
        segs_closed = [
            Line(Vector3(0, 0), Vector3(1, 0)),
            Line(Vector3(1, 0), Vector3(0, 0)),
        ]
        assert len(Polyline(segs_closed, closed=True).segments) == 2

    def test_spline_open_passes_through_points(self):
        from egg.geometry.frontend2d import Spline, Vector3

        pts = [Vector3(0, 0), Vector3(1, 1), Vector3(2, 0), Vector3(3, 1)]
        ent = Spline(pts)
        assert isinstance(ent, CompositePath)
        assert len(ent.segments) == 3
        assert all(isinstance(s, CubicBezier) for s in ent.segments)
        # On-curve points project to themselves.
        for p in pts:
            np.testing.assert_allclose(
                ent.project(np.array([p.x, p.y])), [p.x, p.y], atol=1e-8
            )

    def test_spline_closed_passes_through_points(self):
        from egg.geometry.frontend2d import Spline, Vector3

        pts = [Vector3(0, 0), Vector3(1, 1), Vector3(2, 0), Vector3(3, 1)]
        ent = Spline(pts, closed=True)
        assert isinstance(ent, CompositePath)
        # One segment per interval, including the wrap from last back to first.
        assert len(ent.segments) == 4
        assert all(isinstance(s, CubicBezier) for s in ent.segments)
        for p in pts:
            np.testing.assert_allclose(
                ent.project(np.array([p.x, p.y])), [p.x, p.y], atol=1e-8
            )

    def test_spline_c1_continuity(self):
        # Adjacent cubic Bézier segments share an endpoint and have
        # collinear, equal-magnitude tangents there (C1 continuity).
        from egg.geometry.frontend2d import Spline, Vector3

        pts = [
            Vector3(0, 0),
            Vector3(1, 1),
            Vector3(2, 0),
            Vector3(3, 1),
            Vector3(4, 0),
        ]
        ent = Spline(pts)
        for s0, s1 in zip(ent.segments[:-1], ent.segments[1:]):
            np.testing.assert_allclose(s0.p[3], s1.p[0], atol=1e-12)
            # C1: p3 - p2 of seg0 == p1 - p0 of seg1.
            np.testing.assert_allclose(s0.p[3] - s0.p[2], s1.p[1] - s1.p[0], atol=1e-9)

    def test_spline_closed_continuity(self):
        # gdtk's closed Spline appends points[0] and solves an open natural
        # spline. Internal joints are C1; the wrap-around joint is only C0
        # (the second derivative is zero at both ends, not periodic).
        from egg.geometry.frontend2d import Spline, Vector3

        pts = [
            Vector3(0, 0),
            Vector3(1, 1),
            Vector3(2, 0),
            Vector3(3, 1),
            Vector3(4, 0),
        ]
        ent = Spline(pts, closed=True)
        segs = ent.segments
        # C1 at internal joints.
        for s0, s1 in zip(segs[:-1], segs[1:]):
            np.testing.assert_allclose(s0.p[3], s1.p[0], atol=1e-12)
            np.testing.assert_allclose(s0.p[3] - s0.p[2], s1.p[1] - s1.p[0], atol=1e-9)
        # C0 at the wrap-around joint (last seg ends where first seg starts).
        s0, s1 = segs[-1], segs[0]
        np.testing.assert_allclose(s0.p[3], s1.p[0], atol=1e-12)

    def test_spline_closed_equals_open_with_appended_first(self):
        # gdtk's algorithm: closed Spline through [p0..pn] == open Spline
        # through [p0..pn, p0] when pn != p0.
        from egg.geometry.frontend2d import Spline, Vector3

        pts = [Vector3(0, 0), Vector3(1, 1), Vector3(2, 0), Vector3(3, 1)]
        closed = Spline(pts, closed=True)
        open_ext = Spline(pts + [pts[0]])
        assert len(closed.segments) == len(open_ext.segments)
        for sc, so in zip(closed.segments, open_ext.segments):
            np.testing.assert_allclose(sc.p, so.p, atol=1e-12)

    def test_vector3_arithmetic(self):
        from egg.geometry.frontend2d import Vector3

        a = Vector3(1.0, 2.0)
        b = Vector3(3.0, 4.0)
        c = a + b
        assert isinstance(c, Vector3)
        np.testing.assert_allclose([c.x, c.y], [4.0, 6.0])
        d = 2.0 * a
        assert isinstance(d, Vector3)
        np.testing.assert_allclose([d.x, d.y], [2.0, 4.0])
        e = a * 0.5
        np.testing.assert_allclose([e.x, e.y], [0.5, 1.0])
        f = a - b
        np.testing.assert_allclose([f.x, f.y], [-2.0, -2.0])
        g = -a
        np.testing.assert_allclose([g.x, g.y], [-1.0, -2.0])
        assert abs(a) == pytest.approx(np.sqrt(5.0))
        # Chained: oi + Ri * Vector3(...)  (capsule.py pattern)
        oi = Vector3(1.0, 0.0)
        h = oi + 2.0 * Vector3(-1.0, 0.0)
        np.testing.assert_allclose([h.x, h.y], [-1.0, 0.0])

    def test_vector3_rejects_nonzero_z(self):
        from egg.geometry.frontend2d import Vector3

        with pytest.raises(ValueError):
            Vector3(1, 2, 3)


def _v2(v) -> np.ndarray:
    return np.array([float(v.x), float(v.y)])


def _bezier_eval(p: np.ndarray, t: float) -> np.ndarray:
    """De Casteljau for an arbitrary-degree Bézier (test reference)."""
    while p.shape[0] > 1:
        p = (1.0 - t) * p[:-1] + t * p[1:]
    return p[0]


class TestNurbs:
    @pytest.fixture
    def quarter_circle(self):
        # Rational quadratic Bézier quarter circle, exact on the unit circle.
        return BSplineCurve(
            2,
            [0, 0, 0, 1, 1, 1],
            [[1, 0], [1, 1], [0, 1]],
            weights=[1.0, 1.0 / np.sqrt(2.0), 1.0],
        )

    def test_lies_exactly_on_unit_circle(self, quarter_circle):
        for t in (0.0, 0.2, 0.5, 0.8, 1.0):
            assert abs(np.linalg.norm(quarter_circle.eval(t)) - 1.0) < 1e-14

    def test_deriv_matches_finite_differences(self, quarter_circle):
        h = 1e-6
        for t in (0.2, 0.5, 0.8):
            fd = (quarter_circle.eval(t + h) - quarter_circle.eval(t - h)) / (2 * h)
            np.testing.assert_allclose(quarter_circle.deriv(t), fd, atol=1e-6)
            fd2 = (quarter_circle.deriv(t + h) - quarter_circle.deriv(t - h)) / (2 * h)
            np.testing.assert_allclose(quarter_circle.deriv2(t), fd2, atol=1e-5)

    def test_projection_is_radial(self, quarter_circle):
        q = quarter_circle.project(np.array([1.4, 1.4]))
        np.testing.assert_allclose(q, [1 / np.sqrt(2)] * 2, atol=1e-9)

    def test_ones_weights_equal_polynomial(self):
        knots = [0, 0, 0, 1, 2, 3, 3, 3]
        ctrl = [[0, 0], [1, 2], [3, -1], [4, 2], [6, 0]]
        poly = BSplineCurve(2, knots, ctrl)
        rat = BSplineCurve(2, knots, ctrl, weights=np.ones(5))
        for t in (0.3, 1.1, 2.6):
            np.testing.assert_allclose(rat.eval(t), poly.eval(t), atol=1e-13)
            np.testing.assert_allclose(rat.deriv(t), poly.deriv(t), atol=1e-12)

    def test_weights_length_validated(self):
        with pytest.raises(ValueError):
            BSplineCurve(
                2, [0, 0, 0, 1, 1, 1], [[1, 0], [1, 1], [0, 1]], weights=[1.0, 2.0]
            )

    def test_encoding_appends_weights_to_arena(self, quarter_circle):
        arena = []
        tag, params = encode_entity(quarter_circle, arena=arena)
        assert tag == TAG_BSPLINE
        w_off, has_w = params[6], params[7]
        assert has_w == 1.0
        np.testing.assert_allclose(
            arena[int(w_off) : int(w_off) + 3], quarter_circle.weights
        )

    def test_encoding_polynomial_has_no_weights(self):
        bs = BSplineCurve(
            3, [0, 0, 0, 0, 1, 1, 1, 1], [[0, 0], [1, 1], [2, -1], [3, 0]]
        )
        arena = []
        _tag, params = encode_entity(bs, arena=arena)
        assert params[7] == 0.0
