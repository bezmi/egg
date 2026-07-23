# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Named faces, block handles, batch association, and the BlockArray handle.

Every fluent form must lower to exactly the same topology state as the integer
``(axis, side)`` form it replaces, in both 2D and 3D; the integer form and the
tuple-unpacking of ``add_block_array`` stay working (backward compatible).
"""

from __future__ import annotations

import itertools

import pytest

from egg.geometry import Edge, Line, Vector3
from egg.geometry.frontend3d import Vector3 as Vector3D
from egg.topology import Block, BlockArray, Face, TopologyBuilder


def _assoc(builder):
    return [(a.face.block_name, a.face.axis, a.face.side) for a in builder._associations]


def _tags(builder, name):
    return [(f.block_name, f.axis, f.side) for f in builder._boundary_tags[name]]


def _unit_2d(builder, name, x0=0.0):
    builder.add_block(
        name,
        sw=Vector3(x0, 0),
        se=Vector3(x0 + 1, 0),
        nw=Vector3(x0, 1),
        ne=Vector3(x0 + 1, 1),
        res=(4, 4),
    )


def _unit_3d(builder, name):
    corner_names = []
    for i, (a, b, c) in enumerate(itertools.product((0, 1), repeat=3)):
        nm = f"{name}_k{i}"
        builder.add_corner(nm, Vector3D(a, b, c))
        corner_names.append(nm)
    builder.add_block(name, corners=tuple(corner_names), resolutions=(4, 4, 4))


# --- #1 named faces + #2 block handle: single-face equivalence ----------------


@pytest.mark.parametrize(
    "form",
    [
        lambda b, e: b.associate("blk0", 1, 1, e),  # integer
        lambda b, e: b.associate("blk0", "north", e),  # named (str)
        lambda b, e: b.associate("blk0", Face.NORTH, e),  # named (enum)
        lambda b, e: b["blk0"].north.on(e),  # block handle
    ],
    ids=["int", "name", "enum", "handle"],
)
def test_associate_single_face_forms_are_equivalent(form):
    b = TopologyBuilder(d=2)
    _unit_2d(b, "blk0")
    edge = Line(Vector3(0, 1), Vector3(1, 1))
    form(b, edge)
    assert _assoc(b) == [("blk0", 1, 1)]


def test_block_handle_from_add_block_factory():
    b = TopologyBuilder(d=2)
    handle = b.block("blk0", sw=Vector3(0, 0), se=Vector3(1, 0), nw=Vector3(0, 1), ne=Vector3(1, 1), res=(4, 4))
    assert isinstance(handle, Block)
    edge = Line(Vector3(0, 1), Vector3(1, 1))
    handle.north.on(edge)
    assert _assoc(b) == [("blk0", 1, 1)]


# --- #3 batch association -----------------------------------------------------


def test_batch_association_matches_a_per_block_loop():
    edge = Line(Vector3(0, 1), Vector3(2, 1))
    loop = TopologyBuilder(d=2)
    batch = TopologyBuilder(d=2)
    for i, name in enumerate(("b0", "b1")):
        _unit_2d(loop, name, x0=i)
        _unit_2d(batch, name, x0=i)
    for name in ("b0", "b1"):
        loop.associate(name, 1, 1, edge)  # the old loop
    batch.associate(edge, north=["b0", "b1"])  # the new batch form
    assert _assoc(batch) == _assoc(loop) == [("b0", 1, 1), ("b1", 1, 1)]


# --- tag_boundary + connect (named + handle) ----------------------------------


def test_tag_and_connect_named_and_handle():
    b = TopologyBuilder(d=2)
    _unit_2d(b, "b0")
    _unit_2d(b, "b1", x0=1)
    b["b0"].east.tag("wall")
    b["b0"].east.join(b["b1"].west)
    assert _tags(b, "wall") == [("b0", 0, 1)]
    c = b._connections[0]
    assert (c.face_a.block_name, c.face_a.axis, c.face_a.side) == ("b0", 0, 1)
    assert (c.face_b.block_name, c.face_b.axis, c.face_b.side) == ("b1", 0, 0)

    # connect's named positional form lowers identically
    b2 = TopologyBuilder(d=2)
    _unit_2d(b2, "b0")
    _unit_2d(b2, "b1", x0=1)
    b2.connect("b0", "east", "b1", "west")
    c2 = b2._connections[0]
    assert (c2.face_a.axis, c2.face_a.side) == (0, 1)
    assert (c2.face_b.axis, c2.face_b.side) == (0, 0)


# --- #4 BlockArray handle + backward-compatible unpacking ---------------------


def test_block_array_handle_and_legacy_unpacking():
    b = TopologyBuilder(d=2)
    south = Edge(Line(Vector3(0, 0), Vector3(2, 0)))
    north = Edge(Line(Vector3(0, 2), Vector3(2, 2)))
    west = Edge(Line(Vector3(0, 0), Vector3(0, 2)))
    east = Edge(Line(Vector3(2, 0), Vector3(2, 2)))
    arr = b.add_block_array(
        south=south, north=north, west=west, east=east, nib=2, njb=2, res=(8, 8)
    )
    assert isinstance(arr, BlockArray)
    # named lookups
    assert arr.name(0, 0) == "b0_0"
    assert isinstance(arr.block(1, 1), Block)
    assert arr.corner(0, 0) is arr.corners[0, 0]
    # outer-edge helper: the west column, as handles
    west_blocks = [blk.name for blk in arr.edge("west")]
    assert west_blocks == ["b0_0", "b0_1"]
    east_blocks = [blk.name for blk in arr.edge(Face.EAST)]
    assert east_blocks == ["b1_0", "b1_1"]
    # backward compatible: still unpacks as (corner, names)
    corner, names = arr
    assert corner is arr.corners and names is arr.names


# --- dimensionality guards ----------------------------------------------------


def test_k_face_rejected_in_2d():
    b = TopologyBuilder(d=2)
    _unit_2d(b, "blk0")
    with pytest.raises(ValueError, match="axis must be in 0..1"):
        b["blk0"].top  # noqa: B018  (property access raises)
    with pytest.raises(ValueError, match="axis must be in 0..1"):
        b.associate("blk0", "top", Line(Vector3(0, 0), Vector3(1, 0)))


def test_named_faces_3d():
    b = TopologyBuilder(d=3)
    _unit_3d(b, "cube")
    _unit_3d(b, "cube2")
    surf = object()
    b.associate("cube", "top", surf)  # (2, 1)
    b.associate("cube", Face.BOTTOM, surf)  # (2, 0)
    b["cube"].east.on(surf)  # (0, 1)
    assert _assoc(b) == [("cube", 2, 1), ("cube", 2, 0), ("cube", 0, 1)]
    b["cube"].top.tag("outer")
    assert _tags(b, "outer") == [("cube", 2, 1)]
    b["cube"].top.join(b["cube2"].bottom)
    c = b._connections[0]
    assert (c.face_a.axis, c.face_a.side) == (2, 1)
    assert (c.face_b.axis, c.face_b.side) == (2, 0)
