# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""ExplicitTopology: blocking schema, flatten, diagnostics, geometry binding."""

from copy import deepcopy

import pytest

from egg.geometry import Circle
from egg.topology import ExplicitTopology, TopologyBuilder, editable

# A 2x1 wireframe: six corners, one shared interior edge -> two quad blocks.
TWO_QUADS = {
    "nodes": {
        "n0": {"xy": [0, 0]},
        "n1": {"xy": [1, 0]},
        "n2": {"xy": [2, 0]},
        "n3": {"xy": [0, 2]},
        "n4": {"xy": [1, 2]},
        "n5": {"xy": [2, 2]},
    },
    "edges": [
        {"a": "n0", "b": "n1"},
        {"a": "n1", "b": "n2"},
        {"a": "n3", "b": "n4"},
        {"a": "n4", "b": "n5"},
        {"a": "n0", "b": "n3"},
        {"a": "n1", "b": "n4"},
        {"a": "n2", "b": "n5"},
    ],
    "res": 4,
}


def test_editable_marker_is_identity():
    d = {"nodes": {}}
    assert editable(d) is d
    assert editable(3, choices=[1, 2, 3], label="rings") == 3


def test_wrapped_and_unwrapped_flatten_identically():
    plain = ExplicitTopology(connectivity=TWO_QUADS)
    wrapped = ExplicitTopology(connectivity=editable(TWO_QUADS))
    ta, da = plain.flatten()
    tb, db = wrapped.flatten()
    assert not da and not db
    assert len(ta.block_specs) == len(tb.block_specs) == 2


def test_blocking_flattens_to_two_conforming_blocks():
    et = ExplicitTopology(connectivity=TWO_QUADS)
    topo, diags = et.flatten()
    assert diags == []
    assert len(topo.block_specs) == 2
    assert len(topo.corners) == 6
    assert len(topo.singularities) == 0
    grid = et.initialize_grid()  # conforming shared edge, builds + inits cleanly
    assert grid.global_node_count > 0


def test_blocking_matches_programmatic():
    """The drawn blocking and the hand-written builder agree block-for-block."""
    b = TopologyBuilder(d=2)
    for name, spec in TWO_QUADS["nodes"].items():
        b.add_corner(name, spec["xy"], fixed=False)
    b.add_block("blk0", sw="n0", se="n1", nw="n3", ne="n4", res=(4, 4))
    b.add_block("blk1", sw="n1", se="n2", nw="n4", ne="n5", res=(4, 4))
    prog = b.build()

    drawn, diags = ExplicitTopology(connectivity=TWO_QUADS).flatten()
    assert diags == []
    assert len(drawn.block_specs) == len(prog.block_specs)
    assert len(drawn.corners) == len(prog.corners)
    assert drawn.grid.global_node_count == prog.grid.global_node_count


def test_declarative_binding_needs_no_coincidence():
    """An edge bound to a curve it is not drawn on top of is associated."""
    circle = Circle(center=(0.5, 8.0), radius=1.0)  # far from the unit square
    conn = {
        "nodes": {
            "sw": {"xy": [0, 0]},
            "se": {"xy": [1, 0]},
            "ne": {"xy": [1, 1]},
            "nw": {"xy": [0, 1]},
        },
        "edges": [
            {"a": "sw", "b": "se"},
            {"a": "se", "b": "ne"},
            {"a": "ne", "b": "nw", "bind": "top"},
            {"a": "nw", "b": "sw"},
        ],
        "res": 3,
    }
    topo, diags = ExplicitTopology(
        geometry={"top": circle}, connectivity=conn
    ).flatten()
    assert diags == []
    assert any(a.entity is circle for a in topo.associations)


def test_invalid_face_returns_diagnostic_not_raise():
    tri = {
        "nodes": {"a": {"xy": [0, 0]}, "b": {"xy": [1, 0]}, "c": {"xy": [0.5, 1]}},
        "edges": [
            {"a": "a", "b": "b"},
            {"a": "b", "b": "c"},
            {"a": "c", "b": "a"},
        ],
    }
    et = ExplicitTopology(connectivity=tri)
    topo, diags = et.flatten()
    assert topo is None
    assert any(d.kind in ("non_quad_face", "no_blocks") for d in diags)
    with pytest.raises(ValueError):
        et.build()


def test_stale_edge_reference_is_flagged():
    conn = {"nodes": {"a": {"xy": [0, 0]}}, "edges": [{"a": "a", "b": "ghost"}]}
    topo, diags = ExplicitTopology(connectivity=conn).flatten()
    assert topo is None
    assert any(d.kind == "stale_ref" for d in diags)


def test_unknown_geometry_binding_is_flagged():
    conn = dict(TWO_QUADS)
    conn = {
        **TWO_QUADS,
        "edges": [{**TWO_QUADS["edges"][0], "bind": "nope"}, *TWO_QUADS["edges"][1:]],
    }
    topo, diags = ExplicitTopology(geometry={}, connectivity=conn).flatten()
    assert any(d.kind == "unknown_geometry" for d in diags)


def _two_block_base(res_a=(3, 5), res_b=(7, 5)):  # shared vertical edge: axis1 == 5
    b = TopologyBuilder(d=2)
    for name, spec in TWO_QUADS["nodes"].items():
        b.add_corner(name, spec["xy"], fixed=False)
    b.add_block("blkA", sw="n0", se="n1", nw="n3", ne="n4", res=res_a)
    b.add_block("blkB", sw="n1", se="n2", nw="n4", ne="n5", res=res_b)
    return b


def test_base_only_flatten_preserves_base_verbatim():
    """base= with no blocking edits reproduces the base build, res and all."""
    base = _two_block_base()
    prog = base.build()
    topo, diags = ExplicitTopology(base=base).flatten()
    assert diags == []
    assert len(topo.block_specs) == len(prog.block_specs) == 2
    assert topo.grid.global_node_count == prog.grid.global_node_count
    assert sorted(s.resolutions for s in topo.block_specs.values()) == sorted(
        s.resolutions for s in prog.block_specs.values()
    )


def test_base_not_mutated_by_flatten():
    base = _two_block_base()
    before = len(base._block_specs)
    ExplicitTopology(base=base).flatten()
    ExplicitTopology(base=base).flatten()  # idempotent across repeat renders
    assert len(base._block_specs) == before == 2


def test_blocking_appends_block_adjacent_to_base():
    """A blocking edge naming a base corner grows a new block off the base."""
    base = TopologyBuilder(d=2)
    for name, xy in [("s0", (0, 0)), ("s1", (1, 0)), ("s2", (0, 1)), ("s3", (1, 1))]:
        base.add_corner(name, xy, fixed=False)
    # base axis1 (the shared edge) == 4 so the uniform new block conforms to it
    base.add_block("base0", sw="s0", se="s1", nw="s2", ne="s3", res=(3, 4))
    conn = {
        "nodes": {"e0": {"xy": [2, 0]}, "e1": {"xy": [2, 1]}},
        "edges": [
            {"a": "e0", "b": "e1"},
            {"a": "s1", "b": "e0"},  # attaches to a base corner by name
            {"a": "s3", "b": "e1"},
        ],
        "res": 4,
    }
    topo, diags = ExplicitTopology(base=base, connectivity=conn).flatten()
    assert diags == []
    assert len(topo.block_specs) == 2
    resolutions = sorted(s.resolutions for s in topo.block_specs.values())
    assert resolutions == [(3, 4), (4, 4)]  # base res kept, new block default
    assert topo.initialize_grid().global_node_count > 0


def test_base_with_a_hole_is_not_filled():
    """A base ring leaves its centre empty; flatten must not add a hole block."""
    b = TopologyBuilder(d=2)
    pts = {
        "osw": (0, 0),
        "ose": (3, 0),
        "one": (3, 3),
        "onw": (0, 3),
        "isw": (1, 1),
        "ise": (2, 1),
        "ine": (2, 2),
        "inw": (1, 2),
    }
    for nm, xy in pts.items():
        b.add_corner(nm, xy, fixed=False)
    # four blocks ringing the inner square (which stays a hole)
    b.add_block("rs", sw="osw", se="ose", nw="isw", ne="ise", res=(4, 3))
    b.add_block("re", sw="ose", se="one", nw="ise", ne="ine", res=(4, 3))
    b.add_block("rn", sw="one", se="onw", nw="ine", ne="inw", res=(4, 3))
    b.add_block("rw", sw="onw", se="osw", nw="inw", ne="isw", res=(4, 3))
    assert len(b.build().block_specs) == 4

    topo, diags = ExplicitTopology(base=b).flatten()
    assert diags == []
    assert len(topo.block_specs) == 4  # the hole is left empty, not filled


def _unit_base(fixed=("s0",)):
    b = TopologyBuilder(d=2)
    for nm, xy in [("s0", (0, 0)), ("s1", (1, 0)), ("s2", (0, 1)), ("s3", (1, 1))]:
        b.add_corner(nm, xy, fixed=(nm in fixed))
    b.add_block("b0", sw="s0", se="s1", nw="s2", ne="s3", res=(3, 3))
    return b


def test_base_graph_exposes_corners_and_edges():
    g = ExplicitTopology(base=_unit_base(fixed=("s0",))).base_graph()
    assert set(g["nodes"]) == {"s0", "s1", "s2", "s3"}
    assert g["nodes"]["s0"]["fixed"] is True
    assert g["nodes"]["s1"]["fixed"] is False
    assert g["nodes"]["s3"]["xy"] == [1.0, 1.0]
    assert len(g["edges"]) == 4  # a single quad's four boundary edges


def test_base_corner_moves_via_blocking_override_without_mutating_base():
    base = _unit_base(fixed=())
    conn = {"nodes": {"s3": {"xy": [2, 2]}}, "edges": []}  # move s3
    topo, diags = ExplicitTopology(base=base, connectivity=conn).flatten()
    assert diags == []
    assert list(topo.corners["s3"].position[:2]) == [2.0, 2.0]
    assert list(base._corners["s3"].position[:2]) == [1.0, 1.0]  # original intact


def test_moved_base_boundary_corner_keeps_its_curve():
    """A base corner associated to a curve keeps that binding (projection +
    marker) even when the overlay moves it off the curve and the block is
    re-traced — a regression against silently dropping a wall association on a
    nudged sliding node."""
    from egg.geometry import Line, Vector3

    wall = Line(p0=Vector3(0, 0), p1=Vector3(1, 0)).named("wall")
    base = TopologyBuilder(d=2)
    for nm, xy in [("s0", (0, 0)), ("s1", (1, 0)), ("s2", (0, 1)), ("s3", (1, 1))]:
        base.add_corner(nm, xy, fixed=False)
    base.add_block("b0", sw="s0", se="s1", nw="s2", ne="s3", res=(3, 3))
    base.associate("b0", 1, 0, wall)  # south face lies on the wall
    # Overlay: split the south + north edges (retires b0, re-traces its region)
    # AND move the south-west corner off the wall.
    conn = {
        "nodes": {
            "s0": {"xy": [0.0, 0.4]},  # moved OFF the y=0 wall
            "M": {"split": ["s0", "s1"], "t": 0.5},
            "N": {"split": ["s2", "s3"], "t": 0.5},
        },
        "edges": [{"a": "M", "b": "N"}],
    }
    topo, diags = ExplicitTopology(
        base=base, geometry={"wall": wall}, connectivity=conn
    ).flatten()
    assert not [d for d in diags if not d.kind.startswith("warn")]
    # the wall association (and thus its 'wall' marker) survives the move
    assert any(getattr(a.entity, "name", None) == "wall" for a in topo.associations)
    assert "wall" in topo.boundary_tags


def test_node_inside_closed_curve_warns_but_builds():
    """A node drawn inside a closed curve (a hole the mesh wraps around) is a
    tangled-start footgun — flagged as an advisory ``warn_*`` diagnostic that
    does not block the build."""
    circle = Circle(center=(0.5, 0.5), radius=0.3).named("hole")
    conn = {  # a quad with corner 'a' at the circle's centre (inside it)
        "nodes": {
            "a": {"xy": [0.5, 0.5]},
            "b": {"xy": [2, 0]},
            "c": {"xy": [2, 2]},
            "d": {"xy": [0, 2]},
        },
        "edges": [
            {"a": "a", "b": "b"},
            {"a": "b", "b": "c"},
            {"a": "c", "b": "d"},
            {"a": "d", "b": "a"},
        ],
    }
    topo, diags = ExplicitTopology(
        geometry={"hole": circle}, connectivity=conn
    ).flatten()
    warns = [d for d in diags if d.kind == "warn_inside_closed_curve"]
    assert any("'a'" in w.msg and "hole" in w.msg for w in warns)
    assert topo is not None  # advisory only — still builds


def test_fixed_flag_pins_a_new_node():
    conn = {
        "nodes": {
            "a": {"xy": [0, 0]},
            "b": {"xy": [1, 0]},
            "c": {"xy": [1, 1], "fixed": True},
            "d": {"xy": [0, 1]},
        },
        "edges": [
            {"a": "a", "b": "b"},
            {"a": "b", "b": "c"},
            {"a": "c", "b": "d"},
            {"a": "d", "b": "a"},
        ],
    }
    topo, diags = ExplicitTopology(connectivity=conn).flatten()
    assert diags == []
    assert topo.corners["c"].fixed is True
    assert topo.corners["a"].fixed is False


def test_base_corner_pinned_via_fixed_override_without_mutating_base():
    base = _unit_base(fixed=())  # all free
    topo, diags = ExplicitTopology(
        base=base, connectivity={"nodes": {"s3": {"fixed": True}}, "edges": []}
    ).flatten()
    assert diags == []
    assert topo.corners["s3"].fixed is True
    assert base._corners["s3"].fixed is False  # base untouched


def test_base_edge_split_reblocks_the_base_block():
    """Splitting a base block's opposite edges and connecting them retires the
    block and re-traces its region into two conforming sub-quads."""
    base = _unit_base(fixed=())  # block b0 over s0,s1,s2,s3
    conn = {
        "nodes": {
            "M": {"split": ["s0", "s1"], "t": 0.5},  # on the south edge
            "N": {"split": ["s2", "s3"], "t": 0.5},  # on the north edge
        },
        "edges": [{"a": "M", "b": "N"}],  # the dividing cut
    }
    topo, diags = ExplicitTopology(base=base, connectivity=conn).flatten()
    assert diags == []
    assert len(topo.block_specs) == 2  # b0 retired, replaced by two sub-quads
    assert "b0" not in topo.block_specs
    assert list(base._block_specs) == ["b0"]  # the original base is untouched
    assert topo.initialize_grid().global_node_count > 0


def test_moved_bifurcation_point_uses_its_xy():
    """A split node dragged off its edge keeps the bifurcation but sits at its
    own xy, not the on-edge interpolation."""
    base = _unit_base(fixed=())
    conn = {
        "nodes": {
            "M": {"split": ["s0", "s1"], "xy": [0.5, -0.3]},  # dragged below south
            "N": {"split": ["s2", "s3"], "t": 0.5},
        },
        "edges": [{"a": "M", "b": "N"}],
    }
    topo, diags = ExplicitTopology(base=base, connectivity=conn).flatten()
    assert diags == []
    assert len(topo.block_specs) == 2
    assert list(topo.corners["M"].position[:2]) == [0.5, -0.3]


def test_reblocking_preserves_base_block_resolution():
    """A re-blocked base block's sub-quads inherit its per-axis cell counts, not
    the blocking's uniform default."""
    base = TopologyBuilder(d=2)
    for nm, xy in [("s0", (0, 0)), ("s1", (1, 0)), ("s2", (0, 1)), ("s3", (1, 1))]:
        base.add_corner(nm, xy)
    base.add_block("b0", sw="s0", se="s1", nw="s2", ne="s3", res=(6, 20))
    conn = {
        "nodes": {
            "M": {"split": ["s0", "s1"], "t": 0.5},
            "N": {"split": ["s2", "s3"], "t": 0.5},
        },
        "edges": [{"a": "M", "b": "N"}],
        "res": 10,  # the blocking default — must NOT win over the base's (6, 20)
    }
    topo, diags = ExplicitTopology(base=base, connectivity=conn).flatten()
    assert diags == []
    assert len(topo.block_specs) == 2
    for spec in topo.block_specs.values():
        assert set(spec.resolutions) == {6, 20}  # both directions preserved


def test_cut_base_edge_reblocks_via_blocking_subedges():
    """A base edge subdivided into editable sub-edges (each tagged with its base
    edge) retires the block and re-blocks from the sub-edges — no split node."""
    base = _unit_base(fixed=())
    conn = {
        "nodes": {"M": {"xy": [0.5, 0]}, "N": {"xy": [0.5, 1]}},
        "edges": [
            {"a": "s0", "b": "M", "base": ["s0", "s1"]},
            {"a": "M", "b": "s1", "base": ["s0", "s1"]},
            {"a": "s2", "b": "N", "base": ["s2", "s3"]},
            {"a": "N", "b": "s3", "base": ["s2", "s3"]},
            {"a": "M", "b": "N"},
        ],
    }
    topo, diags = ExplicitTopology(base=base, connectivity=conn).flatten()
    assert diags == []
    assert "b0" not in topo.block_specs
    assert len(topo.block_specs) == 2


def test_base_edge_cut_multiple_times_reblocks():
    """The same base edge can be cut by more than one node (three sub-edges)."""
    base = _unit_base(fixed=())
    conn = {
        "nodes": {
            "M1": {"xy": [0.33, 0]},
            "M2": {"xy": [0.66, 0]},
            "N1": {"xy": [0.33, 1]},
            "N2": {"xy": [0.66, 1]},
        },
        "edges": [
            {"a": "s0", "b": "M1", "base": ["s0", "s1"]},
            {"a": "M1", "b": "M2", "base": ["s0", "s1"]},
            {"a": "M2", "b": "s1", "base": ["s0", "s1"]},
            {"a": "s2", "b": "N1", "base": ["s2", "s3"]},
            {"a": "N1", "b": "N2", "base": ["s2", "s3"]},
            {"a": "N2", "b": "s3", "base": ["s2", "s3"]},
            {"a": "M1", "b": "N1"},
            {"a": "M2", "b": "N2"},
        ],
    }
    topo, diags = ExplicitTopology(base=base, connectivity=conn).flatten()
    assert diags == []
    assert len(topo.block_specs) == 3


def test_cut_into_strips_parallel_to_high_res_axis_stays_conforming():
    """Cutting a non-square block into strips parallel to its high-resolution
    axis keeps the cut edges conforming — the strip touching a base edge and the
    one that doesn't must agree on the shared cut edge (else the grid build
    crashes with an index-out-of-bounds on the mismatched interface)."""
    base = TopologyBuilder(d=2)
    for nm, xy in [("bl", (0, 0)), ("br", (1, 0)), ("tl", (0, 2)), ("tr", (1, 2))]:
        base.add_corner(nm, xy)
    # left/right (vertical) edges res 20, top/bottom (horizontal) res 10
    base.add_block("b0", sw="bl", se="br", nw="tl", ne="tr", res=(10, 20))
    conn = {
        "nodes": {
            "M1": {"xy": [0.33, 0]},
            "M2": {"xy": [0.66, 0]},
            "N1": {"xy": [0.33, 2]},
            "N2": {"xy": [0.66, 2]},
        },
        "edges": [
            {"a": "bl", "b": "M1", "base": ["bl", "br"]},
            {"a": "M1", "b": "M2", "base": ["bl", "br"]},
            {"a": "M2", "b": "br", "base": ["bl", "br"]},
            {"a": "tl", "b": "N1", "base": ["tl", "tr"]},
            {"a": "N1", "b": "N2", "base": ["tl", "tr"]},
            {"a": "N2", "b": "tr", "base": ["tl", "tr"]},
            {"a": "M1", "b": "N1"},  # vertical cuts -> must inherit res 20
            {"a": "M2", "b": "N2"},
        ],
    }
    topo, diags = ExplicitTopology(base=base, connectivity=conn).flatten()
    assert diags == []
    assert len(topo.block_specs) == 3
    topo.initialize_grid()  # must not raise (was an index-out-of-bounds crash)


def test_reblocking_preserves_a_non_geometry_marker():
    """Re-blocking carries a base block's boundary marker onto the sub-faces
    that lie on the marked edge, even when no geometry is named for it."""
    base = TopologyBuilder(d=2)
    for nm, xy in [("s0", (0, 0)), ("s1", (1, 0)), ("s2", (0, 1)), ("s3", (1, 1))]:
        base.add_corner(nm, xy)
    base.add_block("b0", sw="s0", se="s1", nw="s2", ne="s3", res=(4, 4))
    base.tag_boundary("wall", "b0", 1, 0)  # south face 'wall'; no geometry 'wall'
    conn = {
        "nodes": {"M": {"xy": [0.5, 0]}, "N": {"xy": [0.5, 1]}},
        "edges": [
            {"a": "s0", "b": "M", "base": ["s0", "s1"]},
            {"a": "M", "b": "s1", "base": ["s0", "s1"]},
            {"a": "s2", "b": "N", "base": ["s2", "s3"]},
            {"a": "N", "b": "s3", "base": ["s2", "s3"]},
            {"a": "M", "b": "N"},
        ],
    }
    topo, diags = ExplicitTopology(base=base, connectivity=conn).flatten()
    assert diags == []  # no warning — the marker is carried, not dropped
    assert len(topo.block_specs) == 2
    assert len(topo.boundary_tags.get("wall", [])) == 2  # both south sub-faces


def test_reblocked_subblocks_inherit_the_base_block_name():
    base = TopologyBuilder(d=2)
    for nm, xy in [("s0", (0, 0)), ("s1", (1, 0)), ("s2", (0, 1)), ("s3", (1, 1))]:
        base.add_corner(nm, xy)
    base.add_block("myblock", sw="s0", se="s1", nw="s2", ne="s3", res=(4, 4))
    conn = {
        "nodes": {
            "M": {"split": ["s0", "s1"], "t": 0.5},
            "N": {"split": ["s2", "s3"], "t": 0.5},
        },
        "edges": [{"a": "M", "b": "N"}],
    }
    topo, diags = ExplicitTopology(base=base, connectivity=conn).flatten()
    assert diags == []
    assert "myblock" not in topo.block_specs  # retired
    assert set(topo.block_specs) == {"myblock_0", "myblock_1"}  # keep the name


def test_base_edge_split_without_cut_is_flagged_not_crashed():
    """A lone base-edge split (no completing cut) is a hanging node -> diagnostic,
    not a crash."""
    base = _unit_base(fixed=())
    conn = {"nodes": {"M": {"split": ["s0", "s1"], "t": 0.5}}, "edges": []}
    topo, diags = ExplicitTopology(base=base, connectivity=conn).flatten()
    assert topo is None and diags  # invalid (non-quad region), reported


def _split_quad_into_two(t_south=0.5, t_north=0.5, on=None, edge_bind=None):
    """A rectangle split top and bottom, cut down the middle -> two quads."""
    sw = {"xy": [0, 0]}
    se = {"xy": [2, 0]}
    if on:
        sw = {"xy": [0, 0], "on": on}
        se = {"xy": [2, 0], "on": on}
    return {
        "nodes": {
            "sw": sw,
            "se": se,
            "ne": {"xy": [2, 1]},
            "nw": {"xy": [0, 1]},
            "M": {"split": ["sw", "se"], "t": t_south},
            "N": {"split": ["nw", "ne"], "t": t_north},
        },
        "edges": [
            {"a": "sw", "b": "M", **({"bind": edge_bind} if edge_bind else {})},
            {"a": "M", "b": "se"},
            {"a": "nw", "b": "N"},
            {"a": "N", "b": "ne"},
            {"a": "M", "b": "N"},
            {"a": "sw", "b": "nw"},
            {"a": "se", "b": "ne"},
        ],
        "res": 3,
    }


def test_split_reblocks_a_quad_into_two():
    topo, diags = ExplicitTopology(connectivity=_split_quad_into_two()).flatten()
    assert diags == []
    assert len(topo.block_specs) == 2
    assert (
        ExplicitTopology(connectivity=_split_quad_into_two())
        .initialize_grid()
        .global_node_count
        > 0
    )


def test_split_child_rides_relative_t():
    """The split node's position is derived from t along the parent edge."""
    et = ExplicitTopology(connectivity=_split_quad_into_two(t_south=0.25))
    topo, diags = et.flatten()
    assert diags == []
    m = topo.corners["M"]
    assert m.position[0] == pytest.approx(0.5)  # 0.25 of the way from x=0 to x=2


def test_binding_distributes_to_both_split_halves():
    """Splitting a curve-bound edge leaves both halves bound to that curve."""
    circle = Circle(center=(1.0, -6.0), radius=1.0)
    conn = _split_quad_into_two(on=["ground"])
    topo, diags = ExplicitTopology(
        geometry={"ground": circle}, connectivity=conn
    ).flatten()
    assert diags == []
    # sw & se both on 'ground' -> M inherits it -> both south sub-faces associate
    assert sum(1 for a in topo.associations if a.entity is circle) == 2


def test_split_with_missing_parent_is_flagged():
    conn = {
        "nodes": {"a": {"xy": [0, 0]}, "m": {"split": ["a", "ghost"], "t": 0.5}},
        "edges": [{"a": "a", "b": "m"}],
    }
    topo, diags = ExplicitTopology(connectivity=conn).flatten()
    assert topo is None
    assert any(d.kind == "stale_ref" for d in diags)


def test_per_edge_resolution_propagates_around_the_loop():
    """A per-edge ``res`` drives every edge that must stay consistent with it —
    the whole structured loop, across blocks. Setting one vertical edge sets the
    axis-1 resolution of both side-by-side blocks (they share a vertical edge);
    a horizontal edge stays local to its own block."""
    conn = deepcopy(TWO_QUADS)
    for e in conn["edges"]:
        if {e["a"], e["b"]} == {"n0", "n3"}:  # one vertical edge of the left block
            e["res"] = 9
    topo, diags = ExplicitTopology(connectivity=conn).flatten()
    assert not [d for d in diags if not d.kind.startswith("warn")]
    # axis-1 (vertical) is 9 in BOTH blocks; axis-0 keeps the blocking default 4
    assert sorted(s.resolutions for s in topo.block_specs.values()) == [(4, 9), (4, 9)]

    conn = deepcopy(TWO_QUADS)
    for e in conn["edges"]:
        if {e["a"], e["b"]} == {"n0", "n1"}:  # a horizontal edge of the left block
            e["res"] = 7
    topo, diags = ExplicitTopology(connectivity=conn).flatten()
    assert not [d for d in diags if not d.kind.startswith("warn")]
    assert sorted(s.resolutions for s in topo.block_specs.values()) == [(4, 4), (7, 4)]


def test_per_edge_resolution_restores_density_after_a_reblock():
    """Cut sub-edges inherit the full base resolution (doubling the count); an
    explicit per-edge ``res`` on each half lets the user restore the original
    density, and it propagates to the opposite sub-edge of each sub-quad."""
    base = TopologyBuilder(d=2)
    for nm, xy in [("s0", (0, 0)), ("s1", (1, 0)), ("s2", (0, 1)), ("s3", (1, 1))]:
        base.add_corner(nm, xy)
    base.add_block("b0", sw="s0", se="s1", nw="s2", ne="s3", res=(6, 20))
    conn = {
        "nodes": {"M": {"xy": [0.5, 0]}, "N": {"xy": [0.5, 1]}},
        "edges": [
            # the south (6) is cut in two; set each half to 3 so the sub-quads
            # sum back to the original 6 instead of inheriting 6 each (=12).
            {"a": "s0", "b": "M", "base": ["s0", "s1"], "res": 3},
            {"a": "M", "b": "s1", "base": ["s0", "s1"], "res": 3},
            {"a": "s2", "b": "N", "base": ["s2", "s3"]},
            {"a": "N", "b": "s3", "base": ["s2", "s3"]},
            {"a": "M", "b": "N"},
        ],
    }
    topo, diags = ExplicitTopology(base=base, connectivity=conn).flatten()
    assert not [d for d in diags if not d.kind.startswith("warn")]
    assert len(topo.block_specs) == 2
    # each sub-quad: axis-0 = 3 (user, propagated to the north half), axis-1 = 20
    assert sorted(s.resolutions for s in topo.block_specs.values()) == [
        (3, 20),
        (3, 20),
    ]
