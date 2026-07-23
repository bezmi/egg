# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""relax_orthogonality as a topology-level declaration: the builder method,
the ExplicitTopology argument, deprecation of the clustering-spec kwargs, and
consumption by build_topology_target."""

import numpy as np
import pytest

from egg.geometry import Edge, Line, Plane, Vector3
from egg.topology import ExplicitTopology, TopologyBuilder


def _square():
    return Vector3(0, 0), Vector3(1, 0), Vector3(1, 1), Vector3(0, 1)


def _one_block(wall, side_ent):
    sw, se, ne, nw = _square()
    b = TopologyBuilder(d=2)
    b.add_block("c", sw=sw, nw=nw, se=se, ne=ne, res=(4, 4))
    b.associate("c", 1, 0, wall)
    b.associate("c", 0, 1, side_ent)
    return b


def test_builder_declaration_resolves_names_edges_entities():
    sw, se, ne, nw = _square()
    wall = Line(p0=sw, p1=se).named("wall")
    out = Line(p0=se, p1=ne).named("out")
    b = _one_block(wall, out)
    b.relax_orthogonality("out", Edge(wall))
    topo = b.build()
    assert topo.relax_orthogonality == (out, wall)


def test_unknown_name_raises_at_build():
    sw, se, ne, nw = _square()
    wall = Line(p0=sw, p1=se).named("wall")
    out = Line(p0=se, p1=ne).named("out")
    b = _one_block(wall, out)
    b.relax_orthogonality("no_such_boundary")
    with pytest.raises(ValueError, match="no_such_boundary"):
        b.build()


def test_set_boundary_layer_kwarg_deprecated_and_forwarded():
    sw, se, ne, nw = _square()
    wall = Line(p0=sw, p1=se).named("wall")
    out = Line(p0=se, p1=ne).named("out")
    b = _one_block(wall, out)
    with pytest.deprecated_call():
        b.set_boundary_layer(
            wall, first_height=1e-2, growth=1.2, relax_orthogonality=(out,)
        )
    topo = b.build()
    assert topo.relax_orthogonality == (out,)
    assert topo.boundary_layer_specs[id(wall)]["relax_orthogonality"] == ()


def test_target_builder_reads_topology_declaration():
    from egg.smoothing.targets import MultiBlockTarget, build_topology_target

    sw, se, ne, nw = _square()
    # oblique east boundary so the shear taper has real work to do
    ne = Vector3(1.4, 1)
    wall = Line(p0=sw, p1=se).named("wall")
    out = Line(p0=se, p1=ne).named("out")
    b = TopologyBuilder(d=2)
    b.add_block("c", sw=sw, nw=nw, se=se, ne=ne, res=(6, 6))
    b.associate("c", 1, 0, wall)
    b.associate("c", 0, 1, out)
    b.set_boundary_layer(wall, first_height=1e-2, growth=1.2)
    b.relax_orthogonality(out)
    topo = b.build()
    grid = topo.initialize_grid()
    tgt = build_topology_target(topo, grid)
    assert isinstance(tgt, MultiBlockTarget)
    blt = next(iter(tgt.per_block.values()))
    assert blt.boundary_shear  # the declared boundary reached the wall target


def test_explicit_topology_argument_carries_through_flatten():
    wall = Line(Vector3(0, 0), Vector3(2, 0)).named("wall")
    lid = Line(Vector3(0, 1), Vector3(2, 1)).named("lid")
    conn = {
        "nodes": {
            "a": {"xy": [0, 0]},
            "b": {"xy": [2, 0]},
            "c": {"xy": [2, 1]},
            "d": {"xy": [0, 1]},
        },
        "edges": [
            {"a": "a", "b": "b", "bind": "wall"},
            {"a": "b", "b": "c"},
            {"a": "c", "b": "d", "bind": "lid"},
            {"a": "d", "b": "a"},
        ],
        "res": 4,
    }
    et = ExplicitTopology(
        base=None,
        geometry={"wall": wall, "lid": lid},
        connectivity=conn,
        relax_orthogonality=("lid",),
    )
    topo = et.build()
    assert topo.relax_orthogonality == (lid,)
    # and the printed replication block re-declares it
    import io

    text = et.print_topology(file=io.StringIO())
    assert 'relax_orthogonality=("lid",)' in text


def test_3d_declaration_carries():
    floor = Plane((0, 0, 0), (1, 0, 0), (0, 1, 0))
    side = Plane((0, 0, 0), (0, 1, 0), (0, 0, 1))
    b = TopologyBuilder(d=3)
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                b.add_corner(f"p{x}{y}{z}", (x, y, z), fixed=False)
    corners = [f"p{i}{j}{k}" for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    b.add_block("H", corners=corners, resolutions=(3, 3, 3))
    b.associate("H", 2, 0, floor)
    b.associate("H", 0, 0, side)
    b.relax_orthogonality(side)
    topo = b.build()
    assert topo.relax_orthogonality == (side,)
    assert np.allclose(topo.corners["p000"].position, [0, 0, 0])
