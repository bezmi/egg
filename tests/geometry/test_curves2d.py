"""Tests for 2D curve entities, encoding, and the gdtk adapter."""

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
            np.testing.assert_allclose(bez.project(bez.eval(t)), bez.eval(t),
                                       atol=1e-9)

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
        return CompositePath([
            LineSegment((0.0, 0.0), (1.0, 0.0)),
            LineSegment((1.0, 0.0), (1.0, 1.0)),
        ])

    def test_nearest_segment_selection(self, L_path):
        np.testing.assert_allclose(L_path.project(np.array([0.4, -0.5])),
                                   [0.4, 0.0], atol=1e-12)
        np.testing.assert_allclose(L_path.project(np.array([1.5, 0.6])),
                                   [1.0, 0.6], atol=1e-12)

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
        bs = BSplineCurve(3, [0, 0, 0, 0, 1, 1, 1, 1],
                          [[0, 0], [1, 1], [2, -1], [3, 0]])
        with pytest.raises(ValueError):
            encode_entity(bs)

    def test_bspline_arena_layout(self):
        bs = BSplineCurve(3, [0, 0, 0, 0, 1, 1, 1, 1],
                          [[0, 0], [1, 1], [2, -1], [3, 0]])
        arena = []
        tag, params = encode_entity(bs, arena=arena)
        assert tag == TAG_BSPLINE
        degree, n_ctrl, knot_off, ctrl_off, t0, t1 = params[:6]
        assert (degree, n_ctrl) == (3, 4)
        np.testing.assert_allclose(arena[int(knot_off):int(knot_off) + 8],
                                   bs.knots)
        np.testing.assert_allclose(arena[int(ctrl_off):int(ctrl_off) + 8],
                                   bs.ctrl.ravel())
        assert (t0, t1) == (0.0, 1.0)

    def test_composite_arena_records(self):
        path = CompositePath([
            LineSegment((0.0, 0.0), (1.0, 0.0)),
            CircleArc((1.0, 1.0), 1.0, -np.pi / 2, 0.0),
        ])
        arena = []
        tag, params = encode_entity(path, arena=arena)
        assert tag == TAG_COMPOSITE
        n_segs, rec_off = int(params[0]), int(params[1])
        assert n_segs == 2
        rec_size = 1 + PARAM_PAD_SIZE
        assert len(arena) == rec_off + n_segs * rec_size
        # Each record is [tag, params...]; re-decode the second one.
        rec = arena[rec_off + rec_size:rec_off + 2 * rec_size]
        assert int(rec[0]) == TAG_CIRCLEARC
        np.testing.assert_allclose(rec[1:4], [1.0, 1.0, 1.0])


class TestGdtkAdapter:
    def test_line(self):
        from gdtk.geom.path import Line
        from gdtk.geom.vector3 import Vector3
        from egg.geometry.gdtk_adapter import from_gdtk

        ent = from_gdtk(Line(Vector3(0, 0), Vector3(1, 2)))
        assert isinstance(ent, LineSegment)
        np.testing.assert_allclose(ent.end, [1.0, 2.0])

    def test_arc_matches_gdtk_eval(self):
        from gdtk.geom.path import Arc
        from gdtk.geom.vector3 import Vector3
        from egg.geometry.gdtk_adapter import from_gdtk

        a, b, c = Vector3(2, 0), Vector3(0, 2), Vector3(0, 0)
        arc = Arc(a, b, c)
        ent = from_gdtk(arc)
        assert isinstance(ent, CircleArc)
        for t in (0.0, 0.3, 1.0):
            p = arc(t)
            proj = ent.project(np.array([p.x, p.y]))
            np.testing.assert_allclose(proj, [p.x, p.y], atol=1e-9)

    def test_bezier_degrees(self):
        from gdtk.geom.path import Bezier
        from gdtk.geom.vector3 import Vector3
        from egg.geometry.gdtk_adapter import from_gdtk

        pts = [Vector3(0, 0), Vector3(1, 1), Vector3(2, 0)]
        assert isinstance(from_gdtk(Bezier(pts)), QuadBezier)
        pts4 = pts + [Vector3(3, 1)]
        assert isinstance(from_gdtk(Bezier(pts4)), CubicBezier)
        pts5 = pts4 + [Vector3(4, 0)]
        ent = from_gdtk(Bezier(pts5))
        assert isinstance(ent, BSplineCurve)
        # The degree-4 B-spline reproduces the Bézier.
        bez = Bezier(pts5)
        for t in (0.25, 0.75):
            p = bez(t)
            np.testing.assert_allclose(ent.eval(t), [p.x, p.y], atol=1e-12)

    def test_polyline_to_composite(self):
        from gdtk.geom.path import Line, Polyline
        from gdtk.geom.vector3 import Vector3
        from egg.geometry.gdtk_adapter import from_gdtk

        pl = Polyline([
            Line(Vector3(0, 0), Vector3(1, 0)),
            Line(Vector3(1, 0), Vector3(1, 1)),
        ])
        ent = from_gdtk(pl)
        assert isinstance(ent, CompositePath)
        np.testing.assert_allclose(ent.project(np.array([1.5, 0.6])),
                                   [1.0, 0.6], atol=1e-12)

    def test_spline_to_composite(self):
        from gdtk.geom.path import Spline
        from gdtk.geom.vector3 import Vector3
        from egg.geometry.gdtk_adapter import from_gdtk

        sp = Spline([Vector3(0, 0), Vector3(1, 1), Vector3(2, 0), Vector3(3, 1)])
        ent = from_gdtk(sp)
        assert isinstance(ent, CompositePath)
        # On-curve points project to themselves.
        for t in (0.2, 0.6):
            p = sp(t)
            np.testing.assert_allclose(ent.project(np.array([p.x, p.y])),
                                       [p.x, p.y], atol=1e-8)
