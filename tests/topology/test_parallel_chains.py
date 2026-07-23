# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Parallel-chain declarations: the builder API, resolution (fine-node lists
and the logical column walk to the boundary) on the built topology, the
ExplicitTopology kwarg and connectivity section, and the export."""

import math

import pytest

from egg.geometry import Circle, LineSegment, Vector3
from egg.geometry.analytic3d import Plane
from egg.topology import ExplicitTopology, TopologyBuilder
from egg.topology.block_topology import ParallelWalkWarning


def _strip2d(res: int = 3, wall_blocks: tuple = ("L0", "L1")):
    """Two columns x two rows of blocks over a wall segment along y=0.

    Corner grid: w* (y=0), m* (y=1), t* (y=2) at x = 0, 1, 2. The wall
    entity is associated to the bottom faces of the blocks named in
    ``wall_blocks``.
    """
    wall = LineSegment((0.0, 0.0), (2.0, 0.0)).named("wall")
    b = TopologyBuilder(d=2)
    for j, row in enumerate(("w", "m", "t")):
        for i in range(3):
            b.add_corner(f"{row}{i}", (float(i), float(j)))
    for i, name in enumerate(("L0", "L1")):
        b.add_block(
            name,
            corners=(f"w{i}", f"m{i}", f"w{i + 1}", f"m{i + 1}"),
            resolutions=(res, 2),
        )
    for i, name in enumerate(("U0", "U1")):
        b.add_block(
            name,
            corners=(f"m{i}", f"t{i}", f"m{i + 1}", f"t{i + 1}"),
            resolutions=(res, 2),
        )
    for name in wall_blocks:
        b.associate(name, 1, 0, wall)
    return b, wall


def _fan2d(res: int = 3):
    """Five quads around a shared interior corner C (valence 5), rim faces
    associated to a circle."""
    rim = Circle((0.0, 0.0), 1.5).named("rim")
    C = Vector3(0.0, 0.0)
    R = [
        Vector3(math.cos(2 * math.pi * k / 5), math.sin(2 * math.pi * k / 5))
        for k in range(5)
    ]
    M = [
        Vector3(
            1.5 * math.cos(2 * math.pi * (k + 0.5) / 5),
            1.5 * math.sin(2 * math.pi * (k + 0.5) / 5),
        )
        for k in range(5)
    ]
    b = TopologyBuilder(d=2)
    for k in range(5):
        b.add_block(
            f"b{k}",
            corners=(C, R[(k + 1) % 5], R[k], M[k]),
            resolutions=(res, res),
        )
        b.associate(f"b{k}", 0, 1, rim)
        b.associate(f"b{k}", 1, 1, rim)
    return b, rim, C, R, M


def _strip3d(res: int = 2):
    """Two blocks stacked in y over a wall plane at y=0, extruded in z."""
    wall = Plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)).named("wall")
    b = TopologyBuilder(d=3)
    for x in (0, 1):
        for y in (0, 1, 2):
            for z in (0, 1):
                b.add_corner(f"n{x}{y}{z}", (float(x), float(y), float(z)))
    for name, y0 in (("lo", 0), ("hi", 1)):
        b.add_block(
            name,
            corners=tuple(
                f"n{x}{y0 + y}{z}" for x in (0, 1) for y in (0, 1) for z in (0, 1)
            ),
            resolutions=(res, res, res),
        )
    b.associate("lo", 1, 0, wall)
    return b, wall


def _block_dof_map(topo, name):
    return topo.grid.block_dof_maps[list(topo.block_specs.keys()).index(name)]


# ---- builder API + resolution ----------------------------------------------


def test_one_row_off_wall_resolves():
    b, wall = _strip2d(res=3)
    b.parallel_to("wall", chain=("m0", "m1", "m2"))
    topo = b.build()
    assert len(topo.parallel_chains) == 1
    p = topo.parallel_chains[0]
    assert p.to is wall
    assert p.chain == ("m0", "m1", "m2")
    assert len(p.dofs) == 7 and len(set(p.dofs)) == 7
    # correspondence: straight down the column of the wall-adjacent block
    dm_L0, dm_L1 = _block_dof_map(topo, "L0"), _block_dof_map(topo, "L1")
    expected = [int(dm_L0[i, 0]) for i in range(4)] + [
        int(dm_L1[i, 0]) for i in range(1, 4)
    ]
    assert list(p.wall_dofs) == expected


def test_two_rows_off_wall_walks_across_the_interface():
    b, wall = _strip2d(res=3)
    b.parallel_to(wall, chain=("t0", "t1", "t2"), weight=2.5, taper=0.1)
    topo = b.build()
    p = topo.parallel_chains[0]
    assert (p.weight, p.taper) == (2.5, 0.1)
    dm_L0, dm_L1 = _block_dof_map(topo, "L0"), _block_dof_map(topo, "L1")
    expected = [int(dm_L0[i, 0]) for i in range(4)] + [
        int(dm_L1[i, 0]) for i in range(1, 4)
    ]
    assert list(p.wall_dofs) == expected


def test_unreachable_nodes_fall_back_with_a_warning():
    b, wall = _strip2d(res=3, wall_blocks=("L0",))
    b.parallel_to("wall", chain=("t0", "t1", "t2"))
    with pytest.warns(ParallelWalkWarning, match="3 of 7"):
        topo = b.build()
    p = topo.parallel_chains[0]
    dm_L0 = _block_dof_map(topo, "L0")
    assert list(p.wall_dofs[:4]) == [int(dm_L0[i, 0]) for i in range(4)]
    assert list(p.wall_dofs[4:]) == [-1, -1, -1]


def test_chain_through_a_singular_fan_corner():
    b, rim, C, R, M = _fan2d(res=3)
    b.parallel_to(rim, chain=(R[0], C, R[2]))
    topo = b.build()
    p = topo.parallel_chains[0]
    assert len(p.dofs) == 7 and len(set(p.dofs)) == 7
    assert all(w >= 0 for w in p.wall_dofs)


def test_resolves_in_3d():
    b, wall = _strip3d(res=2)
    b.parallel_to("wall", chain=("n020", "n120"))
    topo = b.build()
    p = topo.parallel_chains[0]
    assert len(p.dofs) == 3
    dm_lo = _block_dof_map(topo, "lo")
    dm_hi = _block_dof_map(topo, "hi")
    assert list(p.dofs) == [int(dm_hi[i, 2, 0]) for i in range(3)]
    assert list(p.wall_dofs) == [int(dm_lo[i, 0, 0]) for i in range(3)]


def test_short_chain_raises():
    b, wall = _strip2d()
    with pytest.raises(ValueError, match="at least two"):
        b.parallel_to("wall", chain=("m0",))


def test_bad_weight_raises():
    b, wall = _strip2d()
    with pytest.raises(ValueError, match="weight"):
        b.parallel_to("wall", chain=("m0", "m1"), weight=0.0)


def test_unknown_corner_raises():
    b, wall = _strip2d()
    with pytest.raises(ValueError, match="unknown corner"):
        b.parallel_to("wall", chain=("m0", "nope"))


def test_non_edge_pair_raises_at_build():
    b, wall = _strip2d()
    b.parallel_to("wall", chain=("w0", "m1"))
    with pytest.raises(ValueError, match="no block edge"):
        b.build()


def test_unknown_boundary_name_raises_at_build():
    b, wall = _strip2d()
    b.parallel_to("nope", chain=("m0", "m1"))
    with pytest.raises(ValueError, match="no associated entity"):
        b.build()


def test_unassociated_boundary_raises_at_build():
    b, wall = _strip2d()
    other = LineSegment((0.0, 2.0), (2.0, 2.0)).named("top")
    b.parallel_to(other, chain=("m0", "m1"))
    with pytest.raises(ValueError, match="not an associated boundary"):
        b.build()


# ---- ExplicitTopology ------------------------------------------------------


def _strip_connectivity(section: bool = True, bind_all: bool = True):
    nodes = {}
    for j, row in enumerate(("w", "m", "t")):
        for i in range(3):
            nodes[f"{row}{i}"] = {"xy": [float(i), float(j)]}
    edges = []
    for j, row in enumerate(("w", "m", "t")):
        for i in range(2):
            e = {"a": f"{row}{i}", "b": f"{row}{i + 1}"}
            if row == "w" and (bind_all or i == 0):
                e["bind"] = "wall"
            edges.append(e)
    for ra, rb in (("w", "m"), ("m", "t")):
        for i in range(3):
            edges.append({"a": f"{ra}{i}", "b": f"{rb}{i}"})
    conn = {"nodes": nodes, "edges": edges, "res": 3}
    if section:
        conn["parallel"] = [{"to": "wall", "chain": ["t0", "t1", "t2"]}]
    return conn


def _wall_geom():
    return {"wall": LineSegment((0.0, 0.0), (2.0, 0.0)).named("wall")}


def test_connectivity_section_resolves():
    et = ExplicitTopology(geometry=_wall_geom(), connectivity=_strip_connectivity())
    topo, diags = et.flatten()
    assert topo is not None and not diags
    p = topo.parallel_chains[0]
    assert p.chain == ("t0", "t1", "t2")
    assert all(w >= 0 for w in p.wall_dofs)


def test_kwarg_form_resolves():
    et = ExplicitTopology(
        geometry=_wall_geom(),
        connectivity=_strip_connectivity(section=False),
        parallel=[{"to": "wall", "chain": ["t0", "t1", "t2"]}],
    )
    topo, diags = et.flatten()
    assert topo is not None and not diags
    assert topo.parallel_chains[0].chain == ("t0", "t1", "t2")


def test_malformed_entry_diagnostic():
    conn = _strip_connectivity(section=False)
    conn["parallel"] = [{"to": "wall"}]
    et = ExplicitTopology(geometry=_wall_geom(), connectivity=conn)
    topo, diags = et.flatten()
    assert topo is None
    assert any(d.kind == "bad_parallel" for d in diags)


def test_projection_fallback_is_a_warn_diagnostic():
    et = ExplicitTopology(
        geometry=_wall_geom(),
        connectivity=_strip_connectivity(bind_all=False),
    )
    topo, diags = et.flatten()
    assert topo is not None  # advisory only, does not block
    assert any(d.kind == "warn_parallel_projection" for d in diags)


def test_export_round_trips():
    et = ExplicitTopology(geometry=_wall_geom(), connectivity=_strip_connectivity())
    conn = et.to_connectivity()
    assert conn["parallel"] == [{"to": "wall", "chain": ["t0", "t1", "t2"]}]
    text = et.print_topology(file=open(__import__("os").devnull, "w"))
    assert '"parallel"' in text
    et2 = ExplicitTopology(geometry=_wall_geom(), connectivity=conn)
    topo2, diags2 = et2.flatten()
    assert topo2 is not None and not diags2
    assert topo2.parallel_chains[0].chain == ("t0", "t1", "t2")


def test_webui_blocking_format_keeps_section():
    from egg.webui.scene import _format_blocking

    conn = _strip_connectivity()
    text = _format_blocking(conn, 0)
    assert '"parallel"' in text
    assert eval(text)["parallel"] == conn["parallel"]
