# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""SVG import: path grammar, transforms, arcs, labels, and the egg.svg golden."""

import math
from pathlib import Path

import numpy as np
import pytest

from egg.geometry import Spline, Vector3, svg_import
from egg.geometry.analytic2d import Circle, LineSegment
from egg.geometry.curves2d import (
    CircleArc,
    CompositePath,
    CubicBezier,
    EllipseArc,
    QuadBezier,
)
from egg.geometry.entity_encoding import encode_entity

EGG_SVG = (
    Path(__file__).resolve().parents[2] / "examples" / "2D" / "egg-svg" / "egg.svg"
)


def _svg(body: str, viewbox: str = "0 0 10 10") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        f'viewBox="{viewbox}">{body}</svg>'
    )


def _segments(entity):
    return entity.segments if isinstance(entity, CompositePath) else [entity]


def _assert_on(entity, points, tol=1e-8):
    """Every point projects onto the entity at ~zero distance.

    The tolerance is the C++ seeded-Newton projection's finishing precision
    (fixed seed/iteration counts, shared with the sweep kernels) — still
    orders below any curve-approximation error the tests guard against.
    """
    for p in points:
        assert np.linalg.norm(entity.project(np.asarray(p, float)) - p) < tol, p


# --------------------------------------------------------------------------
# Path grammar


def test_path_lines_and_close():
    dom = svg_import(_svg('<path id="p" d="M 1,1 L 4,1 v 2 h -3 Z"/>'), y_up=False)
    p = dom["p"]
    segs = _segments(p)
    assert len(segs) == 4  # Z appends the closing segment
    assert all(isinstance(s, LineSegment) for s in segs)
    assert np.allclose(p.eval_frac(0.0), [1, 1])
    assert np.allclose(p.eval_frac(1.0), [1, 1])
    _assert_on(p, [(2.5, 1), (4, 2), (2.5, 3), (1, 2)])


def test_path_cubic_and_smooth_reflection():
    dom = svg_import(
        _svg('<path id="p" d="M 0,0 C 0,1 1,1 1,0 S 2,-1 2,0"/>'), y_up=False
    )
    segs = _segments(dom["p"])
    assert [type(s) for s in segs] == [CubicBezier, CubicBezier]
    # S reflects the previous C's second control point about the join.
    ref = CubicBezier([1, 0], [1, -1], [2, -1], [2, 0])
    for t in np.linspace(0, 1, 9):
        assert np.allclose(segs[1].eval(t), ref.eval(t))


def test_path_quad_and_smooth_reflection():
    dom = svg_import(_svg('<path id="p" d="M 0,0 Q 1,2 2,0 T 4,0"/>'), y_up=False)
    segs = _segments(dom["p"])
    assert [type(s) for s in segs] == [QuadBezier, QuadBezier]
    ref = QuadBezier([2, 0], [3, -2], [4, 0])
    for t in np.linspace(0, 1, 9):
        assert np.allclose(segs[1].eval(t), ref.eval(t))


def test_path_relative_commands_and_compact_numbers():
    # ".5.25" and packed arc flags ("01") are legal SVG path data.
    dom = svg_import(_svg('<path id="p" d="M.5.25l1 0a1 1 0 011 1"/>'), y_up=False)
    segs = _segments(dom["p"])
    assert isinstance(segs[0], LineSegment)
    assert np.allclose(segs[0].start, [0.5, 0.25])
    assert np.allclose(segs[0].end, [1.5, 0.25])
    assert np.allclose(_segments(dom["p"])[-1].eval(segs[-1].t1), [2.5, 1.25])


def test_path_multiple_subpaths_rejected():
    with pytest.raises(ValueError, match="[Bb]reak"):
        svg_import(_svg('<path id="p" d="M 0,0 L 1,0 M 2,0 L 3,0"/>'))


# --------------------------------------------------------------------------
# Arcs


def test_arc_quarter_circle():
    dom = svg_import(_svg('<path id="p" d="M 5,0 A 5,5 0 0,1 0,5"/>'), y_up=False)
    segs = _segments(dom["p"])
    assert all(isinstance(s, CircleArc) for s in segs)
    pts = [
        5.0 * np.array([math.cos(t), math.sin(t)])
        for t in np.linspace(0, np.pi / 2, 17)
    ]
    _assert_on(dom["p"], pts)


def test_arc_large_sweep_flags():
    # large-arc, negative direction: 3/4 turn the long way round.
    dom = svg_import(_svg('<path id="p" d="M 5,0 A 5,5 0 1,0 0,5"/>'), y_up=False)
    pts = [
        5.0 * np.array([math.cos(t), math.sin(t)])
        for t in np.linspace(0, -1.5 * np.pi, 33)
    ]
    _assert_on(dom["p"], pts)


def test_arc_trims_are_branch_representable():
    # A long arc must be split so every CircleArc trim sits inside the
    # atan2 branch (-pi, pi] and every EllipseArc trim inside [0, 2*pi]
    # (what the C++ projection kernels can invert).
    body = (
        '<path id="c" d="M 5,0 A 5,5 0 1,0 0,5"/>'
        '<path id="e" d="M 3,0 A 3,1 0 1,1 -3,0"/>'
    )
    dom = svg_import(_svg(body), y_up=False)
    for label in ("c", "e"):
        for s in _segments(dom[label]):
            lo, hi = sorted((s.t0, s.t1))
            assert hi - lo <= np.pi / 2 + 1e-9
            if isinstance(s, CircleArc):
                assert -np.pi <= lo and hi <= np.pi
            else:
                assert 0.0 <= lo and hi <= 2.0 * np.pi


def test_arc_under_affine_transform_is_exact():
    dom = svg_import(
        _svg(
            '<g transform="translate(2,1) rotate(30) scale(1.5,0.75)">'
            '<path id="arc" d="M 1,0 A 1,1 0 0,1 -1,0"/></g>'
        ),
        y_up=False,
    )
    A = np.array(
        [
            [math.cos(math.radians(30)), -math.sin(math.radians(30))],
            [math.sin(math.radians(30)), math.cos(math.radians(30))],
        ]
    ) @ np.diag([1.5, 0.75])
    pts = [
        A @ np.array([math.cos(t), math.sin(t)]) + np.array([2.0, 1.0])
        for t in np.linspace(0, np.pi, 33)
    ]
    _assert_on(dom["arc"], pts)


def test_arc_composite_is_continuous():
    dom = svg_import(
        _svg('<path id="p" d="M 5,0 A 5,5 0 1,0 0,5 L 0,0 Z"/>'), y_up=False
    )
    segs = _segments(dom["p"])
    for a, b in zip(segs, segs[1:]):
        assert np.linalg.norm(a.eval(a.t1) - b.eval(b.t0)) < 1e-9


def test_arc_degenerate_transform_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        svg_import(
            _svg(
                '<g transform="scale(1,0)"><path id="p" d="M 1,0 A 1,1 0 0,1 -1,0"/></g>'
            ),
            y_up=False,
        )


def test_arc_zero_radius_is_a_line():
    dom = svg_import(_svg('<path id="p" d="M 0,0 A 0,5 0 0,1 3,4"/>'), y_up=False)
    assert isinstance(dom["p"], LineSegment)


# --------------------------------------------------------------------------
# Shapes


def test_rect_and_line_and_polygon():
    body = (
        '<rect id="r" x="1" y="1" width="2" height="3"/>'
        '<line id="l" x1="0" y1="0" x2="3" y2="4"/>'
        '<polygon id="pg" points="0,0 2,0 2,2"/>'
        '<polyline id="pl" points="0,0 1,1 2,0"/>'
    )
    dom = svg_import(_svg(body), y_up=False)
    _assert_on(dom["r"], [(1, 2), (3, 2), (2, 1), (2, 4)])
    assert np.allclose(dom["l"].eval(1.0), [3, 4])
    assert len(_segments(dom["pg"])) == 3  # closed
    assert len(_segments(dom["pl"])) == 2  # open


def test_rounded_rect_rejected():
    with pytest.raises(ValueError, match="rounded"):
        svg_import(_svg('<rect id="r" x="0" y="0" width="2" height="2" rx="0.5"/>'))


def test_circle_element_stays_exact_circle():
    dom = svg_import(
        _svg(
            '<g transform="rotate(37) translate(1,2)"><circle id="c" cx="5" cy="5" r="2"/></g>'
        ),
        y_up=False,
    )
    c = dom["c"]
    assert isinstance(c, Circle)
    assert abs(c.radius - 2.0) < 1e-12


def test_scaled_circle_becomes_closed_ellipse():
    dom = svg_import(
        _svg('<g transform="scale(2,1)"><circle id="c" cx="0" cy="0" r="1"/></g>'),
        y_up=False,
    )
    c = dom["c"]
    assert isinstance(c, EllipseArc) and c.closed
    assert abs(c.a - 2.0) < 1e-12 and abs(c.b - 1.0) < 1e-12
    pts = [
        np.array([2 * math.cos(t), math.sin(t)]) for t in np.linspace(0, 2 * np.pi, 33)
    ]
    _assert_on(c, pts, tol=1e-7)


def test_ellipse_element():
    dom = svg_import(_svg('<ellipse id="e" cx="1" cy="1" rx="3" ry="2"/>'), y_up=False)
    e = dom["e"]
    assert isinstance(e, EllipseArc) and e.closed
    pts = [
        np.array([1 + 3 * math.cos(t), 1 + 2 * math.sin(t)])
        for t in np.linspace(0, 2 * np.pi, 33)
    ]
    _assert_on(e, pts, tol=1e-7)


# --------------------------------------------------------------------------
# Transforms, y-flip, scale


def test_transform_stack_composition():
    dom = svg_import(
        _svg(
            '<g transform="translate(1,0)"><g transform="matrix(0,1,-1,0,0,0)">'
            '<line id="l" x1="1" y1="0" x2="2" y2="0"/></g></g>'
        ),
        y_up=False,
    )
    # matrix rotates +90 deg: (1,0)->(0,1), (2,0)->(0,2); then translate x+1.
    assert np.allclose(dom["l"].eval(0.0), [1, 1])
    assert np.allclose(dom["l"].eval(1.0), [1, 2])


def test_skew_transform():
    dom = svg_import(
        _svg('<g transform="skewX(45)"><line id="l" x1="0" y1="1" x2="1" y2="1"/></g>'),
        y_up=False,
    )
    assert np.allclose(dom["l"].eval(0.0), [1, 1])  # x + y*tan45
    assert np.allclose(dom["l"].eval(1.0), [2, 1])


def test_y_flip_about_viewbox_and_scale():
    svg = _svg('<line id="l" x1="0" y1="10" x2="4" y2="10"/>')
    dom = svg_import(svg)  # y_up: svg bottom (y=10) -> model y=0
    assert np.allclose(dom["l"].eval(0.0), [0, 0])
    dom2 = svg_import(svg, scale=0.5)
    assert np.allclose(dom2["l"].eval(1.0), [2, 0])
    dom3 = svg_import(svg, y_up=False)
    assert np.allclose(dom3["l"].eval(0.0), [0, 10])


def test_y_flip_reverses_arc_side():
    # A sweep=1 arc bulges one way in SVG coords; after the y-flip the
    # model-space arc must bulge the mirrored way (through (5, 10-3)).
    dom = svg_import(_svg('<path id="p" d="M 2,5 A 3,3 0 0,1 8,5"/>'))
    mid = dom["p"].eval_frac(0.5)
    assert np.allclose(mid, [5.0, 10.0 - 2.0], atol=1e-9)


# --------------------------------------------------------------------------
# Labels, layers, visibility, warnings


def test_labels_layers_hidden_and_duplicates():
    svg = _svg(
        '<g inkscape:groupmode="layer" inkscape:label="domain">'
        '<path inkscape:label="inflow" id="path7" d="M 0,4 L 0,0"/>'
        '<path inkscape:label="wall" id="w1" d="M 0,0 L 4,0"/>'
        '<path inkscape:label="wall" id="w2" d="M 0,4 L 4,4"/>'
        "</g>"
        '<g inkscape:groupmode="layer" inkscape:label="notes" style="display:none">'
        '<path inkscape:label="ghost" id="g1" d="M 0,0 L 1,1"/></g>'
        "<text id='t1'>hi</text>",
        viewbox="0 0 4 4",
    )
    dom = svg_import(svg)
    assert dom.labels == ["inflow", "wall"]
    assert "ghost" not in dom
    assert len(dom.all("wall")) == 2
    assert next(iter(dom)).layer == "domain"
    with pytest.raises(KeyError, match="2 SVG objects"):
        dom["wall"]
    with pytest.raises(KeyError, match="available"):
        dom["nope"]
    assert dom.get("nope") is None
    assert any("<text>" in w for w in dom.warnings)


def test_id_fallback_label():
    dom = svg_import(_svg('<line id="just-an-id" x1="0" y1="0" x2="1" y2="0"/>'))
    assert dom.labels == ["just-an-id"]


def test_edge_wrapping_and_node_placement():
    dom = svg_import(_svg('<path inkscape:label="w" id="p" d="M 0,10 L 10,10"/>'))
    e = dom.edge("w")
    n = e.place_node(0.25)
    assert np.allclose([n.x, n.y], [2.5, 0.0])
    assert n.edge is e


# --------------------------------------------------------------------------
# The shipped egg.svg


def test_egg_svg_matches_code_built_spline():
    dom = svg_import(EGG_SVG)
    # The geometry lives on the domain layer; the block wireframe is a
    # separate "topology" group consumed by svg_topology, not geometry.
    geom = {it.label for it in dom if it.group != "topology"}
    assert geom == {"inflow", "outflow", "wall_bottom", "wall_top", "egg"}

    theta = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
    ring = [
        Vector3(2.0 + (0.66 - 0.15 * np.sin(t)) * np.cos(t), 2.0 + 0.85 * np.sin(t))
        for t in theta
    ]
    ref = Spline(ring, closed=True)
    egg = dom["egg"]
    assert len(egg.segments) == len(ref.segments)
    for t in np.linspace(0.0, 1.0, 129):
        assert np.linalg.norm(egg.eval_frac(t) - ref.eval_frac(t)) < 1e-9

    # Wall directions are part of the contract (node fractions rely on them).
    for lbl, s, e in [
        ("inflow", (0, 0), (0, 4)),
        ("outflow", (4, 0), (4, 4)),
        ("wall_bottom", (0, 0), (4, 0)),
        ("wall_top", (4, 4), (0, 4)),
    ]:
        ent = dom[lbl]
        assert np.allclose(ent.eval(ent.t0), s)
        assert np.allclose(ent.eval(ent.t1), e)


def test_egg_svg_entities_encode_for_cpp():
    arena: list = []
    for it in svg_import(EGG_SVG):
        encode_entity(it.entity, d=2, arena=arena)
