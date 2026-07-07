# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Named entities: auto-derived markers, the entities map, and carried clustering.

Naming a geometry entity (``.named(...)`` / ``Edge(..., name=...)``) is the
single source of truth: ``build()`` auto-tags every associated face with the
entity name, an explicit ``tag_boundary`` overrides it, and the built topology
exposes ``{name: entity}`` as ``.entities``. An entity may also carry a
boundary-layer request via ``.clustered(...)``, collected the same way into
``boundary_layer_specs`` (so a drawn, base-less topology clusters by name).
"""

from egg.geometry import Edge, Line, Vector3
from egg.topology import ExplicitTopology
from egg.topology.builder import TopologyBuilder


def _square():
    sw, se = Vector3(0, 0, fixed=True), Vector3(4, 0, fixed=True)
    ne, nw = Vector3(4, 4, fixed=True), Vector3(0, 4, fixed=True)
    return sw, se, ne, nw


def _faces(tags):
    """boundary_tags -> {marker: {(block, axis, side), ...}}."""
    return {
        name: {(f.block_name, f.axis, f.side) for f in faces}
        for name, faces in tags.items()
    }


def test_named_forwards_through_edge_to_entity():
    line = Line(p0=Vector3(0, 0), p1=Vector3(1, 0))
    assert line.name is None
    assert line.named("wall") is line
    assert line.name == "wall"

    e = Edge(Line(p0=Vector3(0, 0), p1=Vector3(1, 0)), name="inflow")
    assert e.name == "inflow"
    assert e.entity.name == "inflow"  # lives on the unwrapped entity
    assert e.named("outflow") is e
    assert e.entity.name == "outflow"


def test_association_of_named_entity_auto_tags_face():
    sw, se, ne, nw = _square()
    left = Edge(Line(p0=sw, p1=nw), name="inflow")
    b = TopologyBuilder(d=2)
    b.add_block("c", sw=sw, nw=nw, se=se, ne=ne, res=(4, 4))
    b.associate("c", 0, 0, left)  # west face -> inflow
    topo = b.build()
    assert _faces(topo.boundary_tags) == {"inflow": {("c", 0, 0)}}


def test_tag_defaults_to_name_and_overrides_it():
    line = Line(p0=Vector3(0, 0), p1=Vector3(1, 0)).named("wall_top")
    assert line.tag == "wall_top"  # inherits name
    line.tag = "wall"
    assert line.tag == "wall" and line.name == "wall_top"  # independent

    e = Edge(Line(p0=Vector3(0, 0), p1=Vector3(1, 0)), name="wall_bottom", tag="wall")
    assert e.name == "wall_bottom" and e.tag == "wall"
    assert e.entity.tag == "wall"


def test_entity_tag_collapses_two_walls_under_one_marker():
    # the egg case: distinct names (both draw) but a shared 'wall' export marker,
    # with no per-face tag_boundary calls
    sw, se, ne, nw = _square()
    bottom = Edge(Line(p0=sw, p1=se), name="wall_bottom", tag="wall")
    top = Edge(Line(p0=ne, p1=nw), name="wall_top", tag="wall")
    b = TopologyBuilder(d=2)
    b.add_block("c", sw=sw, nw=nw, se=se, ne=ne, res=(4, 4))
    b.associate("c", 1, 0, bottom)
    b.associate("c", 1, 1, top)
    topo = b.build()
    assert _faces(topo.boundary_tags) == {"wall": {("c", 1, 0), ("c", 1, 1)}}
    # the entities map still carries both distinctly named walls for rendering
    assert topo.entities == {"wall_bottom": bottom.entity, "wall_top": top.entity}


def test_explicit_tag_boundary_overrides_on_a_single_face():
    sw, se, ne, nw = _square()
    left = Edge(Line(p0=sw, p1=nw), name="inflow")
    b = TopologyBuilder(d=2)
    b.add_block("c", sw=sw, nw=nw, se=se, ne=ne, res=(4, 4))
    b.associate("c", 0, 0, left)  # would auto-tag 'inflow'
    b.tag_boundary("special", "c", 0, 0)  # explicit per-face override wins
    topo = b.build()
    assert _faces(topo.boundary_tags) == {"special": {("c", 0, 0)}}
    assert topo.entities == {"inflow": left.entity}  # name unaffected


def test_tag_none_suppresses_marker_but_keeps_entity():
    sw, se, ne, nw = _square()
    left = Edge(Line(p0=sw, p1=nw), name="interface", tag=None)
    b = TopologyBuilder(d=2)
    b.add_block("c", sw=sw, nw=nw, se=se, ne=ne, res=(4, 4))
    b.associate("c", 0, 0, left)
    topo = b.build()
    assert topo.boundary_tags == {}  # no marker emitted
    assert topo.entities == {"interface": left.entity}  # still drawn


def test_unnamed_entity_derives_no_tag():
    sw, se, ne, nw = _square()
    left = Edge(Line(p0=sw, p1=nw))  # no name
    b = TopologyBuilder(d=2)
    b.add_block("c", sw=sw, nw=nw, se=se, ne=ne, res=(4, 4))
    b.associate("c", 0, 0, left)
    topo = b.build()
    assert topo.boundary_tags == {}
    assert topo.entities == {}


def test_entities_dedupes_by_name_across_faces():
    # one entity associated to several faces appears once in .entities
    sw, se, ne, nw = _square()
    mid_s, mid_n = Vector3(2, 0, fixed=True), Vector3(2, 4, fixed=True)
    wall = Line(p0=sw, p1=se).named("wall")
    b = TopologyBuilder(d=2)
    b.add_block("l", sw=sw, nw=nw, se=mid_s, ne=mid_n, res=(4, 4))
    b.add_block("r", sw=mid_s, nw=mid_n, se=se, ne=ne, res=(4, 4))
    b.associate("l", 1, 0, wall)
    b.associate("r", 1, 0, wall)
    topo = b.build()
    assert topo.entities == {"wall": wall}
    assert _faces(topo.boundary_tags) == {"wall": {("l", 1, 0), ("r", 1, 0)}}


# --- entity-carried boundary-layer clustering (.clustered) -------------------


def test_clustered_entity_is_collected_into_boundary_layer_specs():
    sw, se, ne, nw = _square()
    wall = (
        Line(p0=sw, p1=se)
        .named("wall")
        .clustered(first_height=5e-3, growth=1.5, n_fixed=2)
    )
    b = TopologyBuilder(d=2)
    b.add_block("c", sw=sw, nw=nw, se=se, ne=ne, res=(4, 4))
    b.associate("c", 1, 0, wall)
    topo = b.build()
    spec = topo.boundary_layer_specs[id(wall)]
    assert spec["first_height"] == 5e-3
    assert spec["growth"] == 1.5
    assert spec["n_fixed"] == 2


def test_clustered_only_collected_when_associated():
    # a clustered entity that is never associated contributes no spec
    sw, se, ne, nw = _square()
    Line(p0=sw, p1=se).named("wall").clustered(first_height=5e-3, growth=1.5)
    b = TopologyBuilder(d=2)
    b.add_block("c", sw=sw, nw=nw, se=se, ne=ne, res=(4, 4))  # no associate
    assert b.build().boundary_layer_specs == {}


def test_explicit_set_boundary_layer_overrides_clustered():
    sw, se, ne, nw = _square()
    wall = Line(p0=sw, p1=se).named("wall").clustered(first_height=9.9, growth=2.0)
    b = TopologyBuilder(d=2)
    b.add_block("c", sw=sw, nw=nw, se=se, ne=ne, res=(4, 4))
    b.associate("c", 1, 0, wall)
    b.set_boundary_layer(wall, first_height=5e-3, growth=1.5)  # builder method wins
    assert b.build().boundary_layer_specs[id(wall)]["first_height"] == 5e-3


def test_relax_orthogonality_by_name_resolves_to_entity():
    sw, se, ne, nw = _square()
    egg = (
        Line(p0=nw, p1=ne)
        .named("egg")
        .clustered(first_height=1e-2, growth=1.3, relax_orthogonality=("wall",))
    )
    wall = Line(p0=sw, p1=se).named("wall")
    b = TopologyBuilder(d=2)
    b.add_block("c", sw=sw, nw=nw, se=se, ne=ne, res=(4, 4))
    b.associate("c", 1, 1, egg)
    b.associate("c", 1, 0, wall)
    spec = b.build().boundary_layer_specs[id(egg)]
    assert spec["relax_orthogonality"] == (wall,)  # name -> the named entity


def test_clustered_flows_through_a_drawn_topology():
    # the payoff: a by-name clustered entity clusters a drawn (base=None) topology
    egg = (
        Line(p0=Vector3(0, 0), p1=Vector3(4, 0))
        .named("egg")
        .clustered(first_height=2e-3, growth=1.4, n_fixed=1)
    )
    conn = {
        "nodes": {
            "sw": {"xy": [0, 0]},
            "se": {"xy": [4, 0]},
            "ne": {"xy": [4, 4]},
            "nw": {"xy": [0, 4]},
        },
        "edges": [
            {"a": "sw", "b": "se", "bind": "egg"},
            {"a": "se", "b": "ne"},
            {"a": "ne", "b": "nw"},
            {"a": "nw", "b": "sw"},
        ],
        "res": 4,
    }
    topo, diags = ExplicitTopology(geometry={"egg": egg}, connectivity=conn).flatten()
    assert diags == []
    assert id(egg) in topo.boundary_layer_specs
