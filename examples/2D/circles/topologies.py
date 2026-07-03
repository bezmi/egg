"""Circle-in-channel topologies, authored with the egg 2D front-end.

Geometry is declared gdtk/Eilmer-style — :class:`~egg.geometry.Vector3`
points and :class:`~egg.geometry.Line` walls, with wall corners placed
parametrically along :class:`~egg.geometry.Edge` wrappers
(``edge.place_node(t)``). Blocks take their corner objects directly
(compass keywords); block-to-block connectivity is inferred from shared
corner objects, and wall-face associations are inferred from node
provenance. Only the O-ring→circle and corner-block associations remain
explicit (their corners are rough points / points shared by two walls).
"""

from __future__ import annotations

from egg.geometry import Circle, Edge, Line, Vector3
from egg.topology.builder import TopologyBuilder

__all__ = ["build_circle_in_rectangle", "build_twin_circle"]


# Proper vs. rough (folded) inner O-ring corner placements.
_INNER_PROPER = [(1.3, 1.3), (2.7, 1.3), (2.7, 2.7), (1.3, 2.7)]
_INNER_ROUGH = [(2.5, 2.4), (1.6, 2.5), (1.5, 1.6), (2.4, 1.5)]


def build_circle_in_rectangle(rough: bool = False, R: int = 1, bl=None):
    """Circle-in-rectangle O-grid topology.

    ``rough`` → folded TFI start (exercises the δ-continuation untangler).
    ``bl`` (dict) → ``set_boundary_layer`` spec for the circle, e.g.
    ``dict(first_height=0.01, growth=1.3)``.
    ``R`` → resolution multiplier.
    """
    circle = Circle(center=(2.0, 2.0), radius=0.8)

    sw = Vector3(x=0.0, y=0.0, fixed=True)
    se = Vector3(x=4.0, y=0.0, fixed=True)
    ne = Vector3(x=4.0, y=4.0, fixed=True)
    nw = Vector3(x=0.0, y=4.0, fixed=True)
    bottom = Edge(Line(p0=sw, p1=se))
    right = Edge(Line(p0=se, p1=ne))
    top = Edge(Line(p0=ne, p1=nw))
    left = Edge(Line(p0=sw, p1=nw))

    # O-ring corners: mid square and (rough or proper) inner square.
    msw, mse, mne, mnw = (Vector3(*p) for p in [(1, 1), (3, 1), (3, 3), (1, 3)])
    isw, ise, ine, inw = (
        Vector3(*p) for p in (_INNER_ROUGH if rough else _INNER_PROPER)
    )
    # Wall corners placed along the wall edges in parametric space.
    bsw, bse = bottom.place_node(0.25), bottom.place_node(0.75)
    rse, rne = right.place_node(0.25), right.place_node(0.75)
    tne, tnw = top.place_node(0.25), top.place_node(0.75)
    lnw, lsw = left.place_node(0.75), left.place_node(0.25)

    b = TopologyBuilder(d=2)
    for nm, c_sw, c_nw, c_se, c_ne in [
        ("o_s", msw, isw, mse, ise),
        ("o_e", mse, ise, mne, ine),
        ("o_n", mne, ine, mnw, inw),
        ("o_w", mnw, inw, msw, isw),
    ]:
        b.add_block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=(10 * R, 4 * R))
    for nm, c_sw, c_nw, c_se, c_ne in [
        ("e_s", bsw, msw, bse, mse),
        ("e_e", rse, mse, rne, mne),
        ("e_n", tne, mne, tnw, mnw),
        ("e_w", lnw, mnw, lsw, msw),
    ]:
        b.add_block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=(10 * R, 5 * R))
    for nm, c_sw, c_nw, c_se, c_ne in [
        ("c_sw", sw, lsw, bsw, msw),
        ("c_se", se, bse, rse, mse),
        ("c_ne", ne, rne, tne, mne),
        ("c_nw", nw, tnw, lnw, mnw),
    ]:
        b.add_block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=(5 * R, 5 * R))

    # Connectivity and wall associations are inferred; the O-ring faces on
    # the circle (rough corners) and the corner-block faces (corners shared
    # by two walls) stay explicit.
    for blk in ("o_s", "o_e", "o_n", "o_w"):
        b.associate(blk, 1, 1, circle)
    for blk, a0, a1 in [
        ("c_sw", left, bottom),
        ("c_se", bottom, right),
        ("c_ne", right, top),
        ("c_nw", top, left),
    ]:
        b.associate(blk, 0, 0, a0)
        b.associate(blk, 1, 0, a1)

    if bl is not None:
        b.set_boundary_layer(circle, **bl)

    topology = b.build()
    entities = {
        "circle": circle,
        "bottom": bottom.entity,
        "right": right.entity,
        "top": top.entity,
        "left": left.entity,
    }
    return topology, entities


_PROPER_TWIN = {
    "c1": [(1.3, 1.3), (2.7, 1.3), (2.7, 2.7), (1.3, 2.7)],
    "c2": [(4.3, 1.3), (5.7, 1.3), (5.7, 2.7), (4.3, 2.7)],
}

_ROUGH_TWIN = {
    "c1": [(2.5, 2.4), (1.6, 2.5), (1.5, 1.6), (2.4, 1.5)],
    "c2": [(5.5, 2.4), (4.6, 2.5), (4.5, 1.6), (5.4, 1.5)],
}


def build_twin_circle(rough: bool = False, bl=None, R: int = 1):
    """Twin-circle O-grid topology in a 7×4 channel with a shared bridge block.

    ``rough`` → folded TFI start (exercises δ-continuation untangler).
    ``bl`` (dict) → set_boundary_layer on each circle via ``bl["circle"]`` and
    ``bl["circle2"]`` dicts, e.g. ``dict(first_height=0.05, growth=1.5)``.
    ``R`` → corner-block resolution multiplier.
    """
    circle = Circle(center=(2.0, 2.0), radius=0.8)
    circle2 = Circle(center=(5.0, 2.0), radius=0.8)

    sw = Vector3(x=0.0, y=0.0, fixed=True)
    s2e = Vector3(x=7.0, y=0.0, fixed=True)
    n2e = Vector3(x=7.0, y=4.0, fixed=True)
    nw = Vector3(x=0.0, y=4.0, fixed=True)
    bottom = Edge(Line(p0=sw, p1=s2e))
    right = Edge(Line(p0=s2e, p1=n2e))
    top = Edge(Line(p0=n2e, p1=nw))
    left = Edge(Line(p0=sw, p1=nw))

    # O-ring corners around each circle.
    msw, mse, mne, mnw = (Vector3(*p) for p in [(1, 1), (3, 1), (3, 3), (1, 3)])
    m2sw, m2se, m2ne, m2nw = (Vector3(*p) for p in [(4, 1), (6, 1), (6, 3), (4, 3)])
    inner = _ROUGH_TWIN if rough else _PROPER_TWIN
    isw, ise, ine, inw = (Vector3(*p) for p in inner["c1"])
    i2sw, i2se, i2ne, i2nw = (Vector3(*p) for p in inner["c2"])
    # Wall corners placed along the wall edges in parametric space.
    bsw, bse = bottom.place_node(1 / 7), bottom.place_node(3 / 7)
    b2sw, b2se = bottom.place_node(4 / 7), bottom.place_node(6 / 7)
    r2se, r2ne = right.place_node(0.25), right.place_node(0.75)
    tne, tnw = top.place_node(4 / 7), top.place_node(6 / 7)
    t2ne, t2nw = top.place_node(1 / 7), top.place_node(3 / 7)
    lnw, lsw = left.place_node(0.75), left.place_node(0.25)

    b = TopologyBuilder(d=2)
    for nm, c_sw, c_nw, c_se, c_ne in [
        ("o_s", msw, isw, mse, ise),
        ("o_e", mse, ise, mne, ine),
        ("o_n", mne, ine, mnw, inw),
        ("o_w", mnw, inw, msw, isw),
        ("o2_s", m2sw, i2sw, m2se, i2se),
        ("o2_e", m2se, i2se, m2ne, i2ne),
        ("o2_n", m2ne, i2ne, m2nw, i2nw),
        ("o2_w", m2nw, i2nw, m2sw, i2sw),
    ]:
        b.add_block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=(10 * R, 4 * R))
    for nm, c_sw, c_nw, c_se, c_ne in [
        ("e_s", bsw, msw, bse, mse),
        ("e_e", m2sw, mse, m2nw, mne),
        ("e_n", tne, mne, tnw, mnw),
        ("e_w", lnw, mnw, lsw, msw),
        ("e2_s", b2sw, m2sw, b2se, m2se),
        ("e2_e", r2se, m2se, r2ne, m2ne),
        ("e2_n", t2ne, m2ne, t2nw, m2nw),
    ]:
        b.add_block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=(10 * R, 5 * R))
    for nm, c_sw, c_nw, c_se, c_ne in [
        ("c_sw", sw, lsw, bsw, msw),
        ("c_se", b2sw, bse, m2sw, mse),
        ("c_ne", t2nw, m2nw, tne, mne),
        ("c_nw", nw, tnw, lnw, mnw),
        ("c2_se", s2e, b2se, r2se, m2se),
        ("c2_ne", n2e, r2ne, t2ne, m2ne),
    ]:
        b.add_block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=(5 * R, 5 * R))

    # Connectivity and wall associations are inferred; the O-ring faces on
    # the circles (rough corners) and the corner-block faces (corners shared
    # by two walls) stay explicit.
    for blk in ("o_s", "o_e", "o_n", "o_w"):
        b.associate(blk, 1, 1, circle)
    for blk in ("o2_s", "o2_e", "o2_n", "o2_w"):
        b.associate(blk, 1, 1, circle2)
    for blk, a0, a1 in [
        ("c_sw", left, bottom),
        ("c_nw", top, left),
        ("c2_ne", right, top),
        ("c2_se", bottom, right),
    ]:
        b.associate(blk, 0, 0, a0)
        b.associate(blk, 1, 0, a1)

    if bl is not None:
        b.set_boundary_layer(circle, **bl["circle"])
        b.set_boundary_layer(circle2, **bl["circle2"])

    topology = b.build()
    entities = {
        "circle": circle,
        "circle2": circle2,
        "bottom": bottom.entity,
        "right": right.entity,
        "top": top.entity,
        "left": left.entity,
    }
    return topology, entities
