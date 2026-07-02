"""Edge/Node parametric node placement and Lua-style constructor parity."""

import numpy as np
import pytest

from egg.geometry import (
    Arc,
    Bezier,
    Circle,
    CompositePath,
    Edge,
    Line,
    Node,
    Polyline,
    Spline,
    Vector3,
)


class TestLuaStyleConstructors:
    def test_vector3_keywords(self):
        v = Vector3(x=1.0, y=2.0)
        assert (v.x, v.y, v.z) == (1.0, 2.0, 0.0)

    def test_line_keywords(self):
        ln = Line(p0=Vector3(0, 0), p1=Vector3(4, 0))
        np.testing.assert_allclose(ln.eval(0.5), [2.0, 0.0])

    def test_arc_keywords(self):
        a = Arc(p0=Vector3(1, 0), p1=Vector3(0, 1), centre=Vector3(0, 0))
        np.testing.assert_allclose(a.eval(np.pi / 4),
                                   [np.sqrt(0.5), np.sqrt(0.5)])

    def test_bezier_keywords(self):
        bz = Bezier(points=[Vector3(0, 0), Vector3(1, 1), Vector3(2, 0)])
        np.testing.assert_allclose(bz.eval(0.5), [1.0, 0.5])

    def test_arc_preserves_direction(self):
        """eval_frac runs p0 -> p1 regardless of the sweep's sense."""
        ccw = Arc(p0=Vector3(1, 0), p1=Vector3(0, 1), centre=Vector3(0, 0))
        np.testing.assert_allclose(ccw.eval_frac(0.0), [1.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(ccw.eval_frac(1.0), [0.0, 1.0], atol=1e-14)
        cw = Arc(p0=Vector3(0, 1), p1=Vector3(1, 0), centre=Vector3(0, 0))
        assert cw.t1 < cw.t0
        np.testing.assert_allclose(cw.eval_frac(0.0), [0.0, 1.0], atol=1e-14)
        np.testing.assert_allclose(cw.eval_frac(1.0), [1.0, 0.0], atol=1e-14)

    def test_reversed_arc_projection_unchanged(self):
        cw = Arc(p0=Vector3(0, 1), p1=Vector3(1, 0), centre=Vector3(0, 0))
        q = cw.project(np.array([2.0, 2.0]))
        np.testing.assert_allclose(q, [np.sqrt(0.5), np.sqrt(0.5)])
        # Below the angular range: clamps to the t=0 end of the interval.
        q = cw.project(np.array([0.0, -1.0]))
        np.testing.assert_allclose(q, [1.0, 0.0], atol=1e-14)

    def test_polyline_traverses_clockwise_arcs(self):
        """A wall polyline whose arcs sweep clockwise stays in order."""
        centre = Vector3(1.0, 0.0)
        a, b = Vector3(0.0, 0.0), Vector3(1.0, 1.0)
        c = Vector3(2.0, 1.0)
        wire = Edge(Polyline([Arc(a, b, centre), Line(b, c)]),
                    arc_length=True)
        np.testing.assert_allclose(tuple(wire.point_at(0.0))[:2], (0.0, 0.0),
                                   atol=1e-12)
        np.testing.assert_allclose(tuple(wire.point_at(1.0))[:2], (2.0, 1.0),
                                   atol=1e-12)
        # Arc quarter = pi/2, line = 1: midpoint of total length sits on
        # the arc near its far end, monotonically ordered.
        pts = np.array([tuple(wire.point_at(t))[:2]
                        for t in np.linspace(0, 1, 41)])
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert np.all(steps < 0.12)  # no jumps: continuous forward walk

    def test_reversed_arc_soa_interval_sorted(self):
        from egg.geometry.entity_soa import encode_entity_soa

        cw = Arc(p0=Vector3(0, 1), p1=Vector3(1, 0), centre=Vector3(0, 0))
        _tag, _k, row, _seg = encode_entity_soa(cw)
        assert row[3] <= row[4]
        assert row[3] == pytest.approx(0.0) and row[4] == pytest.approx(np.pi / 2)


class TestEdge:
    def test_point_on_line(self):
        e = Edge(Line(Vector3(0, 0), Vector3(4, 0)))
        p = e.point_at(0.25)
        assert isinstance(p, Vector3)
        assert (p.x, p.y) == (1.0, 0.0)
        q = e.place_node(0.75)
        assert (q.x, q.y) == (3.0, 0.0)

    def test_point_on_circle(self):
        e = Edge(Circle(center=(2.0, 2.0), radius=0.8))
        p = e.point_at(5.0 / 8.0)  # angle 225 deg
        np.testing.assert_allclose(
            [p.x, p.y], 2.0 + 0.8 * np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
        )

    def test_point_on_arc(self):
        e = Edge(Arc(p0=Vector3(1, 0), p1=Vector3(0, 1), centre=Vector3(0, 0)))
        p = e.point_at(0.5)
        np.testing.assert_allclose([p.x, p.y], [np.sqrt(0.5), np.sqrt(0.5)])

    def test_node_records_edge_and_t(self):
        e = Edge(Line(Vector3(0, 0), Vector3(4, 0)))
        n = e.place_node(0.25)
        assert isinstance(n, Node) and isinstance(n, Vector3)
        assert n.edge is e and n.t == 0.25
        assert (n.x, n.y) == (1.0, 0.0)

    def test_native_parametrisation(self):
        # Quarter arc, native parameter = angle in radians on [0, pi/2].
        arc = Arc(p0=Vector3(1, 0), p1=Vector3(0, 1), centre=Vector3(0, 0))
        e = Edge(arc)
        p = e.point_at(np.pi / 3, param="native")
        np.testing.assert_allclose([p.x, p.y], [0.5, np.sqrt(3) / 2])
        n = e.place_node(np.pi / 3, param="native")
        assert n.t == pytest.approx((np.pi / 3) / (np.pi / 2))  # stored as frac
        np.testing.assert_allclose([n.x, n.y], [0.5, np.sqrt(3) / 2])

    def test_native_param_with_arc_length_table(self):
        bz = Bezier([Vector3(0, 0), Vector3(0, 2), Vector3(4, 2)])
        e = Edge(bz, arc_length=True)
        p = e.point_at(0.5, param="native")  # native t, bypasses the table
        np.testing.assert_allclose([p.x, p.y], bz.eval(0.5))
        n = e.place_node(0.5, param="native")
        np.testing.assert_allclose([n.x, n.y], bz.eval(0.5), atol=1e-12)

    def test_native_convenience_methods(self):
        arc = Arc(p0=Vector3(1, 0), p1=Vector3(0, 1), centre=Vector3(0, 0))
        e = Edge(arc)
        p = e.point_at_native(np.pi / 3)
        np.testing.assert_allclose([p.x, p.y], [0.5, np.sqrt(3) / 2])
        n = e.place_node_native(np.pi / 3)
        assert n.t == pytest.approx((np.pi / 3) / (np.pi / 2))
        np.testing.assert_allclose([n.x, n.y], [0.5, np.sqrt(3) / 2])

    def test_rejects_unknown_param(self):
        e = Edge(Line(Vector3(0, 0), Vector3(4, 0)))
        with pytest.raises(ValueError):
            e.point_at(0.5, param="arclen")
        with pytest.raises(ValueError):
            e.place_node(0.5, param="arclen")

    def test_clamps_parameter(self):
        e = Edge(Line(Vector3(0, 0), Vector3(4, 0)))
        assert e.point_at(-1.0).x == 0.0
        assert e.point_at(2.0).x == 4.0

    def test_rejects_nonparametric_entity(self):
        with pytest.raises(TypeError):
            Edge(object())

    def test_delegates_entity_queries(self):
        c = Circle(center=(0.0, 0.0), radius=1.0)
        e = Edge(c)
        np.testing.assert_allclose(e.project((2.0, 0.0)), [1.0, 0.0])
        np.testing.assert_allclose(e.normal((2.0, 0.0)), [1.0, 0.0])
        assert e.dim == 1

    def test_arc_length_reparameterization(self):
        # A composite of a 3-long and a 1-long line: native (uniform-in-t)
        # placement and arc-length placement agree because CompositePath's
        # parameter is already arc-length proportional.
        wire = Polyline([Line(Vector3(0, 0), Vector3(3, 0)),
                         Line(Vector3(3, 0), Vector3(3, 1))])
        e = Edge(wire, arc_length=True)
        p = e.point_at(0.75)
        np.testing.assert_allclose([p.x, p.y], [3.0, 0.0], atol=1e-6)
        # On a quadratic Bezier with non-uniform speed, arc-length midpoint
        # bisects the curve's total length.
        bz = Bezier([Vector3(0, 0), Vector3(0, 2), Vector3(4, 2)])
        ea = Edge(bz, arc_length=True, samples=2048)
        mid = ea.point_at(0.5)
        ts = np.linspace(0.0, 1.0, 4097)
        pts = np.stack([bz.eval(t) for t in ts])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        t_half = np.interp(0.5 * s[-1], s, ts)
        np.testing.assert_allclose([mid.x, mid.y], bz.eval(t_half), atol=1e-4)


class TestEdgeComposites:
    def test_edge_on_polyline(self):
        wire = Polyline([Line(Vector3(0, 0), Vector3(2, 0)),
                         Line(Vector3(2, 0), Vector3(2, 2))])
        e = Edge(wire)
        assert isinstance(wire, CompositePath)
        p = e.point_at(0.5)
        np.testing.assert_allclose([p.x, p.y], [2.0, 0.0], atol=1e-12)

    def test_edge_on_closed_spline(self):
        theta = np.linspace(0.0, 2.0 * np.pi, 13)[:-1]
        ring = [Vector3(np.cos(t), np.sin(t)) for t in theta]
        blob = Spline(ring, closed=True)
        e = Edge(blob)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            p = e.point_at(t)
            # Spline through unit-circle points stays near the circle.
            assert np.hypot(p.x, p.y) == pytest.approx(1.0, abs=5e-3)
        start, end = e.point_at(0.0), e.point_at(1.0)
        np.testing.assert_allclose([start.x, start.y], [end.x, end.y],
                                   atol=1e-12)


class TestBuilderIntegration:
    def _square_edges(self):
        a, b = Vector3(0, 0), Vector3(4, 0)
        c, d = Vector3(4, 4), Vector3(0, 4)
        return {
            "bottom": Edge(Line(p0=a, p1=b)),
            "right": Edge(Line(p0=b, p1=c)),
            "top": Edge(Line(p0=c, p1=d)),
            "left": Edge(Line(p0=d, p1=a)),
        }

    def test_corner_accepts_vector3_and_node(self):
        from egg.topology.builder import TopologyBuilder

        edges = self._square_edges()
        b = TopologyBuilder(d=2)
        b.add_corner("sw", Vector3(0, 0), fixed=True)
        b.add_corner("bsw", edges["bottom"].place_node(0.25), fixed=False)
        np.testing.assert_allclose(b._corners["sw"].position, [0.0, 0.0])
        np.testing.assert_allclose(b._corners["bsw"].position, [1.0, 0.0])

    def test_associate_unwraps_edge(self):
        from egg.geometry.analytic2d import LineSegment
        from egg.topology.builder import TopologyBuilder

        edges = self._square_edges()
        b = TopologyBuilder(d=2)
        for n, p in [("sw", (0, 0)), ("se", (4, 0)), ("ne", (4, 4)),
                     ("nw", (0, 4))]:
            b.add_corner(n, p)
        b.add_block("blk", ("sw", "nw", "se", "ne"), (4, 4))
        b.associate("blk", 1, 0, edges["bottom"])
        ent = b._associations[0].entity
        assert not isinstance(ent, Edge)
        assert isinstance(ent, LineSegment)

    def test_set_boundary_layer_unwraps_edge(self):
        from egg.topology.builder import TopologyBuilder

        edges = self._square_edges()
        b = TopologyBuilder(d=2)
        b.set_boundary_layer(edges["bottom"], first_height=0.01, growth=1.2)
        assert id(edges["bottom"].entity) in b._boundary_layer_specs

    def test_grid_from_edge_authored_topology(self):
        """A single block authored purely via Edge nodes initializes cleanly."""
        from egg.topology.builder import TopologyBuilder

        edges = self._square_edges()
        b = TopologyBuilder(d=2)
        b.add_corner("sw", edges["bottom"].place_node(0.0), fixed=True)
        b.add_corner("se", edges["bottom"].place_node(1.0), fixed=True)
        b.add_corner("ne", edges["top"].place_node(0.0), fixed=True)
        b.add_corner("nw", edges["top"].place_node(1.0), fixed=True)
        b.add_block("blk", ("sw", "nw", "se", "ne"), (4, 4))
        for axis, side, e in [(1, 0, edges["bottom"]), (1, 1, edges["top"]),
                              (0, 0, edges["left"]), (0, 1, edges["right"])]:
            b.associate("blk", axis, side, e)
        topo = b.build()
        grid = topo.initialize_grid()
        assert not np.any(np.isnan(grid.global_nodes))
