# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Fan-frame declarations: the builder API, resolution/validation on the
built topology, the ExplicitTopology kwarg and connectivity section, and the
print_topology export."""

import math

import pytest

from egg.geometry import Vector3
from egg.geometry.frontend3d import Vector3 as Vector3d
from egg.topology import ExplicitTopology, TopologyBuilder


def _fan2d(res: int = 3):
    """Five quads around a shared interior corner C (valence 5)."""
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
    return b, C, R, M


def _fan3d(res: int = 2):
    """The 2D five-fan extruded in z: five hexes sharing the C0-C1 edge."""
    C = [Vector3d(0.0, 0.0, float(z)) for z in (0, 1)]
    R = [
        [
            Vector3d(
                math.cos(2 * math.pi * k / 5),
                math.sin(2 * math.pi * k / 5),
                float(z),
            )
            for z in (0, 1)
        ]
        for k in range(5)
    ]
    M = [
        [
            Vector3d(
                1.5 * math.cos(2 * math.pi * (k + 0.5) / 5),
                1.5 * math.sin(2 * math.pi * (k + 0.5) / 5),
                float(z),
            )
            for z in (0, 1)
        ]
        for k in range(5)
    ]
    b = TopologyBuilder(d=3)
    for k in range(5):
        kk = (k + 1) % 5
        b.add_block(
            f"b{k}",
            corners=(
                C[0],
                C[1],
                R[kk][0],
                R[kk][1],
                R[k][0],
                R[k][1],
                M[k][0],
                M[k][1],
            ),
            resolutions=(res, res, res),
        )
    return b, C, R, M


# ---- builder API -----------------------------------------------------------


def test_builder_resolves_frame_2d():
    b, C, R, M = _fan2d(res=3)
    b.fan_frame(C, through=(R[0], R[2]), normal=R[1])
    topo = b.build()
    assert len(topo.fan_frames) == 1
    f = topo.fan_frames[0]
    assert f.dof >= 0
    # every rail starts at the fan corner's DOF and spans res+1 fine nodes
    for rail in (*f.through_rails, f.normal_rail):
        assert rail[0] == f.dof
        assert len(rail) == 4
        assert len(set(rail)) == 4
    # the three legs leave through three distinct fine edges
    first = {f.through_rails[0][1], f.through_rails[1][1], f.normal_rail[1]}
    assert len(first) == 3


def test_builder_resolves_frame_3d():
    b, C, R, M = _fan3d(res=2)
    # top center: five radial legs plus the axial leg down to C[0]
    b.fan_frame(C[1], through=(R[0][1], R[2][1]), normal=C[0])
    topo = b.build()
    f = topo.fan_frames[0]
    for rail in (*f.through_rails, f.normal_rail):
        assert rail[0] == f.dof
        assert len(rail) == 3
        assert len(set(rail)) == 3


def test_unknown_corner_name_raises():
    b, C, R, M = _fan2d()
    with pytest.raises(ValueError, match="unknown corner"):
        b.fan_frame("nope", through=("also", "nope2"), normal="none")


def test_regular_corner_raises_at_build():
    b, C, R, M = _fan2d()
    # R[0] is a regular boundary node (legs: C, M[0], M[4])
    b.fan_frame(R[0], through=(C, M[0]), normal=M[4])
    with pytest.raises(ValueError, match="regular node"):
        b.build()


def test_non_incident_leg_raises_at_build():
    b, C, R, M = _fan2d()
    b.fan_frame(C, through=(R[0], M[0]), normal=R[1])
    with pytest.raises(ValueError, match="no block edge"):
        b.build()


def test_repeated_leg_raises_at_build():
    b, C, R, M = _fan2d()
    b.fan_frame(C, through=(R[0], R[1]), normal=R[1])
    with pytest.raises(ValueError, match="distinct"):
        b.build()


def test_double_frame_raises_at_build():
    b, C, R, M = _fan2d()
    b.fan_frame(C, through=(R[0], R[2]), normal=R[1])
    b.fan_frame(C, through=(R[1], R[3]), normal=R[2])
    with pytest.raises(ValueError, match="framed twice"):
        b.build()


# ---- ExplicitTopology ------------------------------------------------------


def _fan_connectivity(section: bool = True):
    nodes = {"C": {"xy": [0.0, 0.0]}}
    edges = []
    for k in range(5):
        a = 2 * math.pi * k / 5
        am = 2 * math.pi * (k + 0.5) / 5
        nodes[f"r{k}"] = {"xy": [math.cos(a), math.sin(a)]}
        nodes[f"m{k}"] = {"xy": [1.5 * math.cos(am), 1.5 * math.sin(am)]}
    for k in range(5):
        edges.append({"a": "C", "b": f"r{k}"})
        edges.append({"a": f"r{k}", "b": f"m{k}"})
        edges.append({"a": f"m{k}", "b": f"r{(k + 1) % 5}"})
    conn = {"nodes": nodes, "edges": edges, "res": 3}
    if section:
        conn["fan_frames"] = {"C": {"through": ["r0", "r2"], "normal": "r1"}}
    return conn


def test_connectivity_section_resolves():
    et = ExplicitTopology(geometry={}, connectivity=_fan_connectivity())
    topo, diags = et.flatten()
    assert topo is not None and not diags
    f = topo.fan_frames[0]
    assert (f.corner, f.through, f.normal) == ("C", ("r0", "r2"), "r1")


def test_kwarg_form_resolves():
    et = ExplicitTopology(
        geometry={},
        connectivity=_fan_connectivity(section=False),
        fan_frames={"C": {"through": ["r0", "r2"], "normal": "r1"}},
    )
    topo, diags = et.flatten()
    assert topo is not None and not diags
    assert topo.fan_frames[0].corner == "C"


def test_kwarg_and_section_conflict():
    et = ExplicitTopology(
        geometry={},
        connectivity=_fan_connectivity(),
        fan_frames={"C": {"through": ["r1", "r3"], "normal": "r2"}},
    )
    topo, diags = et.flatten()
    assert topo is None
    assert any(d.kind == "fan_frame_conflict" for d in diags)


def test_malformed_section_diagnostic():
    conn = _fan_connectivity(section=False)
    conn["fan_frames"] = {"C": {"through": ["r0"], "normal": "r1"}}
    et = ExplicitTopology(geometry={}, connectivity=conn)
    topo, diags = et.flatten()
    assert topo is None
    assert any(d.kind == "bad_fan_frame" for d in diags)


def test_invalid_frame_is_a_flatten_diagnostic():
    conn = _fan_connectivity(section=False)
    conn["fan_frames"] = {"C": {"through": ["r0", "m0"], "normal": "r1"}}
    et = ExplicitTopology(geometry={}, connectivity=conn)
    topo, diags = et.flatten()
    assert topo is None
    assert any("m0" in d.msg for d in diags)


def test_export_round_trips():
    et = ExplicitTopology(geometry={}, connectivity=_fan_connectivity())
    conn = et.to_connectivity()
    assert conn["fan_frames"] == {"C": {"through": ["r0", "r2"], "normal": "r1"}}
    text = et.print_topology(file=open(__import__("os").devnull, "w"))
    assert '"fan_frames"' in text
    et2 = ExplicitTopology(geometry={}, connectivity=conn)
    topo2, diags2 = et2.flatten()
    assert topo2 is not None and not diags2
    f = topo2.fan_frames[0]
    assert (f.corner, f.through, f.normal) == ("C", ("r0", "r2"), "r1")


def test_webui_blocking_format_keeps_section():
    from egg.webui.scene import _format_blocking

    conn = _fan_connectivity()
    text = _format_blocking(conn, 0)
    assert '"fan_frames"' in text
    assert eval(text)["fan_frames"] == conn["fan_frames"]
