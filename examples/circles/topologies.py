from __future__ import annotations

import numpy as np

from egg.geometry.analytic2d import Circle, LineSegment
from egg.topology.builder import TopologyBuilder

__all__ = ["build_circle_in_rectangle", "build_twin_circle"]


# Proper vs. rough (folded) inner O-ring corner placements.
_INNER_PROPER = [
    ("isw", (1.3, 1.3)),
    ("ise", (2.7, 1.3)),
    ("ine", (2.7, 2.7)),
    ("inw", (1.3, 2.7)),
]
_INNER_ROUGH = [
    ("isw", (2.5, 2.4)),
    ("ise", (1.6, 2.5)),
    ("ine", (1.5, 1.6)),
    ("inw", (2.4, 1.5)),
]


def build_circle_in_rectangle(rough: bool = False, R: int = 1):
    circle = Circle(center=(2.0, 2.0), radius=0.8)
    bottom = LineSegment(start=(0.0, 0.0), end=(4.0, 0.0))
    right = LineSegment(start=(4.0, 0.0), end=(4.0, 4.0))
    top = LineSegment(start=(4.0, 4.0), end=(0.0, 4.0))
    left = LineSegment(start=(0.0, 0.0), end=(0.0, 4.0))

    b = TopologyBuilder(d=2)
    for n, p in [("sw", (0, 0)), ("se", (4, 0)), ("ne", (4, 4)), ("nw", (0, 4))]:
        b.add_corner(n, p, fixed=True)
    for n, p in [("msw", (1, 1)), ("mse", (3, 1)), ("mne", (3, 3)), ("mnw", (1, 3))]:
        b.add_corner(n, p, fixed=False)
    for n, p in _INNER_ROUGH if rough else _INNER_PROPER:
        b.add_corner(n, p, fixed=False)
    for n, p in [
        ("bsw", (1, 0)),
        ("bse", (3, 0)),
        ("rse", (4, 1)),
        ("rne", (4, 3)),
        ("tne", (3, 4)),
        ("tnw", (1, 4)),
        ("lnw", (0, 3)),
        ("lsw", (0, 1)),
    ]:
        b.add_corner(n, p, fixed=False)

    for nm, sw, nw, se, ne in [
        ("o_s", "msw", "isw", "mse", "ise"),
        ("o_e", "mse", "ise", "mne", "ine"),
        ("o_n", "mne", "ine", "mnw", "inw"),
        ("o_w", "mnw", "inw", "msw", "isw"),
    ]:
        b.add_block(nm, (sw, nw, se, ne), (10 * R, 4 * R))
    for nm, sw, nw, se, ne in [
        ("e_s", "bsw", "msw", "bse", "mse"),
        ("e_e", "rse", "mse", "rne", "mne"),
        ("e_n", "tne", "mne", "tnw", "mnw"),
        ("e_w", "lnw", "mnw", "lsw", "msw"),
    ]:
        b.add_block(nm, (sw, nw, se, ne), (10 * R, 5 * R))
    for nm, sw, nw, se, ne in [
        ("c_sw", "sw", "lsw", "bsw", "msw"),
        ("c_se", "se", "bse", "rse", "mse"),
        ("c_ne", "ne", "rne", "tne", "mne"),
        ("c_nw", "nw", "tnw", "lnw", "mnw"),
    ]:
        b.add_block(nm, (sw, nw, se, ne), (5 * R, 5 * R))

    for a, b_ in [("o_s", "o_e"), ("o_e", "o_n"), ("o_n", "o_w"), ("o_w", "o_s")]:
        b.connect(a, 0, 1, b_, 0, 0)
    for e, o in [("e_s", "o_s"), ("e_e", "o_e"), ("e_n", "o_n"), ("e_w", "o_w")]:
        b.connect(e, 1, 1, o, 1, 0)
    for cb, ca, cs, eb, ea, es in [
        ("c_sw", 0, 1, "e_s", 0, 0),
        ("c_sw", 1, 1, "e_w", 0, 1),
        ("c_se", 0, 1, "e_e", 0, 0),
        ("c_se", 1, 1, "e_s", 0, 1),
        ("c_ne", 0, 1, "e_n", 0, 0),
        ("c_ne", 1, 1, "e_e", 0, 1),
        ("c_nw", 0, 1, "e_w", 0, 0),
        ("c_nw", 1, 1, "e_n", 0, 1),
    ]:
        b.connect(cb, ca, cs, eb, ea, es)

    for blk in ("o_s", "o_e", "o_n", "o_w"):
        b.associate(blk, 1, 1, circle)
    for blk, ent in [("e_s", bottom), ("e_e", right), ("e_n", top), ("e_w", left)]:
        b.associate(blk, 1, 0, ent)
    for blk, a0, a1 in [
        ("c_sw", left, bottom),
        ("c_se", bottom, right),
        ("c_ne", right, top),
        ("c_nw", top, left),
    ]:
        b.associate(blk, 0, 0, a0)
        b.associate(blk, 1, 0, a1)

    topology = b.build()
    entities = {
        "circle": circle,
        "bottom": bottom,
        "right": right,
        "top": top,
        "left": left,
    }
    return topology, entities


_PROPER_TWIN = {
    "c1": [
        ("isw", (1.3, 1.3)),
        ("ise", (2.7, 1.3)),
        ("ine", (2.7, 2.7)),
        ("inw", (1.3, 2.7)),
    ],
    "c2": [
        ("i2sw", (4.3, 1.3)),
        ("i2se", (5.7, 1.3)),
        ("i2ne", (5.7, 2.7)),
        ("i2nw", (4.3, 2.7)),
    ],
}

_ROUGH_TWIN = {
    "c1": [
        ("isw", (2.5, 2.4)),
        ("ise", (1.6, 2.5)),
        ("ine", (1.5, 1.6)),
        ("inw", (2.4, 1.5)),
    ],
    "c2": [
        ("i2sw", (5.5, 2.4)),
        ("i2se", (4.6, 2.5)),
        ("i2ne", (4.5, 1.6)),
        ("i2nw", (5.4, 1.5)),
    ],
}


def build_twin_circle(rough: bool = False, bl=None, R: int = 1):
    """Twin-circle O-grid topology in a 7×4 channel with a shared bridge block.

    ``rough`` → folded TFI start (exercises δ-continuation untangler).
    ``bl`` (dict) → set_boundary_layer on each circle via ``bl["circle"]`` and
    ``bl["circle2"]`` dicts of ``{first_height, growth, n_layers}``.
    ``R`` → corner-block resolution multiplier.
    """
    circle = Circle(center=(2.0, 2.0), radius=0.8)
    circle2 = Circle(center=(5.0, 2.0), radius=0.8)
    bottom = LineSegment((0.0, 0.0), (7.0, 0.0))
    right = LineSegment((7.0, 0.0), (7.0, 4.0))
    top = LineSegment((7.0, 4.0), (0.0, 4.0))
    left = LineSegment((0.0, 0.0), (0.0, 4.0))

    b = TopologyBuilder(d=2)
    for n, p in [("sw", (0, 0)), ("nw", (0, 4)), ("n2e", (7, 4)), ("s2e", (7, 0))]:
        b.add_corner(n, p, fixed=True)
    for n, p in [("msw", (1, 1)), ("mse", (3, 1)), ("mne", (3, 3)), ("mnw", (1, 3))]:
        b.add_corner(n, p, fixed=False)
    for n, p in [
        ("m2sw", (4, 1)),
        ("m2se", (6, 1)),
        ("m2ne", (6, 3)),
        ("m2nw", (4, 3)),
    ]:
        b.add_corner(n, p, fixed=False)
    inner = _ROUGH_TWIN if rough else _PROPER_TWIN
    for n, p in inner["c1"] + inner["c2"]:
        b.add_corner(n, p, fixed=False)
    for n, p in [
        ("bsw", (1, 0)),
        ("bse", (3, 0)),
        ("b2sw", (4, 0)),
        ("b2se", (6, 0)),
        ("r2se", (7, 1)),
        ("r2ne", (7, 3)),
        ("tne", (3, 4)),
        ("tnw", (1, 4)),
        ("t2ne", (6, 4)),
        ("t2nw", (4, 4)),
        ("lnw", (0, 3)),
        ("lsw", (0, 1)),
    ]:
        b.add_corner(n, p, fixed=False)

    for nm, sw, nw, se, ne in [
        ("o_s", "msw", "isw", "mse", "ise"),
        ("o_e", "mse", "ise", "mne", "ine"),
        ("o_n", "mne", "ine", "mnw", "inw"),
        ("o_w", "mnw", "inw", "msw", "isw"),
        ("o2_s", "m2sw", "i2sw", "m2se", "i2se"),
        ("o2_e", "m2se", "i2se", "m2ne", "i2ne"),
        ("o2_n", "m2ne", "i2ne", "m2nw", "i2nw"),
        ("o2_w", "m2nw", "i2nw", "m2sw", "i2sw"),
    ]:
        b.add_block(nm, (sw, nw, se, ne), (10 * R, 4 * R))
    for nm, sw, nw, se, ne in [
        ("e_s", "bsw", "msw", "bse", "mse"),
        ("e_e", "m2sw", "mse", "m2nw", "mne"),
        ("e_n", "tne", "mne", "tnw", "mnw"),
        ("e_w", "lnw", "mnw", "lsw", "msw"),
        ("e2_s", "b2sw", "m2sw", "b2se", "m2se"),
        ("e2_e", "r2se", "m2se", "r2ne", "m2ne"),
        ("e2_n", "t2ne", "m2ne", "t2nw", "m2nw"),
    ]:
        b.add_block(nm, (sw, nw, se, ne), (10 * R, 5 * R))
    for nm, sw, nw, se, ne in [
        ("c_sw", "sw", "lsw", "bsw", "msw"),
        ("c_se", "b2sw", "bse", "m2sw", "mse"),
        ("c_ne", "t2nw", "m2nw", "tne", "mne"),
        ("c_nw", "nw", "tnw", "lnw", "mnw"),
        ("c2_se", "s2e", "b2se", "r2se", "m2se"),
        ("c2_ne", "n2e", "r2ne", "t2ne", "m2ne"),
    ]:
        b.add_block(nm, (sw, nw, se, ne), (5 * R, 5 * R))

    for a, b_ in [
        ("o_s", "o_e"),
        ("o_e", "o_n"),
        ("o_n", "o_w"),
        ("o_w", "o_s"),
        ("o2_s", "o2_e"),
        ("o2_e", "o2_n"),
        ("o2_n", "o2_w"),
        ("o2_w", "o2_s"),
    ]:
        b.connect(a, 0, 1, b_, 0, 0)
    for e, o in [
        ("e_s", "o_s"),
        ("e_e", "o_e"),
        ("e_n", "o_n"),
        ("e_w", "o_w"),
        ("e2_s", "o2_s"),
        ("e2_e", "o2_e"),
        ("e2_n", "o2_n"),
    ]:
        b.connect(e, 1, 1, o, 1, 0)
    b.connect("e_e", 1, 0, "o2_w", 1, 0)
    for cb, ca, cs, eb, ea, es in [
        ("c_sw", 0, 1, "e_s", 0, 0),
        ("c_sw", 1, 1, "e_w", 0, 1),
        ("c_se", 0, 1, "e_e", 0, 0),
        ("c_se", 1, 1, "e_s", 0, 1),
        ("c_se", 1, 0, "e2_s", 0, 0),
        ("c2_se", 0, 1, "e2_e", 0, 0),
        ("c2_se", 1, 1, "e2_s", 0, 1),
        ("c_ne", 0, 1, "e_n", 0, 0),
        ("c_ne", 1, 1, "e_e", 0, 1),
        ("c_ne", 0, 0, "e2_n", 0, 1),
        ("c2_ne", 0, 1, "e2_n", 0, 0),
        ("c2_ne", 1, 1, "e2_e", 0, 1),
        ("c_nw", 0, 1, "e_w", 0, 0),
        ("c_nw", 1, 1, "e_n", 0, 1),
    ]:
        b.connect(cb, ca, cs, eb, ea, es)

    for blk in ("o_s", "o_e", "o_n", "o_w"):
        b.associate(blk, 1, 1, circle)
    for blk in ("o2_s", "o2_e", "o2_n", "o2_w"):
        b.associate(blk, 1, 1, circle2)
    for blk, ent in [
        ("e_s", bottom),
        ("e2_s", bottom),
        ("e_n", top),
        ("e2_n", top),
        ("e_w", left),
        ("e2_e", right),
        ("c_ne", top),
    ]:
        b.associate(blk, 1, 0, ent)
    b.associate("c_se", 0, 0, bottom)
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
        "bottom": bottom,
        "right": right,
        "top": top,
        "left": left,
    }
    return topology, entities
