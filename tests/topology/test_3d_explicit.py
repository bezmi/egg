# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""ExplicitTopology(d=3): the explicit-hex 3D authoring model.

Nodes carry ``xyz`` + optional ``on`` surface bindings; blocks list 8 corner ids
in product order. A boundary face whose four corners share a bound surface rides
it. Covers a plain slab round-trip, surface-association inference, diagnostics,
and a cubed-sphere reproduced entirely through the schema.
"""

import numpy as np
import pytest

from egg.geometry.analytic3d import Plane, Sphere
from egg.topology.explicit import ExplicitTopology, editable

SLAB = {
    "a00": (0, 0, 0),
    "a01": (0, 0, 1),
    "a10": (0, 1, 0),
    "a11": (0, 1, 1),
    "s00": (1, 0, 0),
    "s01": (1, 0, 1),
    "s10": (1, 1, 0),
    "s11": (1, 1, 1),
    "b00": (2, 0, 0),
    "b01": (2, 0, 1),
    "b10": (2, 1, 0),
    "b11": (2, 1, 1),
}


def test_slab_round_trip():
    nodes = {k: {"xyz": list(v)} for k, v in SLAB.items()}
    blocks = [
        {
            "name": "A",
            "corners": ["a00", "a01", "a10", "a11", "s00", "s01", "s10", "s11"],
        },
        {
            "name": "B",
            "corners": ["s00", "s01", "s10", "s11", "b00", "b01", "b10", "b11"],
        },
    ]
    et = ExplicitTopology(
        d=3, connectivity=editable({"nodes": nodes, "blocks": blocks, "res": 3})
    )
    topo, diags = et.flatten()
    assert diags == []
    assert (
        topo.grid.global_node_count == 4 * 4 * 4 * 2 - 16
    )  # two 4^3 blocks, shared 4x4
    assert topo.singularities == []


def test_on_inference_projects_face_onto_surface():
    sph = Sphere((0, 0, 0), 1.0, (1, 0, 0), (0, 1, 0))
    pts = {
        "c000": (-0.4, -0.4, 0.6),
        "c001": (-0.4, -0.4, 1.4),
        "c010": (-0.4, 0.4, 0.6),
        "c011": (-0.4, 0.4, 1.4),
        "c100": (0.4, -0.4, 0.6),
        "c101": (0.4, -0.4, 1.4),
        "c110": (0.4, 0.4, 0.6),
        "c111": (0.4, 0.4, 1.4),
    }
    zmin = {"c000", "c010", "c100", "c110"}
    nodes = {
        k: {"xyz": list(v), **({"on": ["sph"]} if k in zmin else {})}
        for k, v in pts.items()
    }
    blocks = [{"name": "B", "corners": list(pts), "res": [3, 3, 3]}]
    et = ExplicitTopology(
        d=3, geometry={"sph": sph}, connectivity={"nodes": nodes, "blocks": blocks}
    )
    grid = et.initialize_grid()
    face = grid.blocks[0].nodes[:, :, 0].reshape(-1, 3)
    np.testing.assert_allclose(np.linalg.norm(face, axis=1), 1.0, atol=1e-9)


def test_bad_block_reports_diagnostic():
    nodes = {k: {"xyz": list(v)} for k, v in SLAB.items()}
    bad = [{"name": "A", "corners": ["a00", "a01", "a10", "a11", "s00", "s01", "s10"]}]
    topo, diags = ExplicitTopology(
        d=3, connectivity={"nodes": nodes, "blocks": bad}
    ).flatten()
    assert topo is None
    assert any(d.kind == "bad_block" for d in diags)


def _cubed_sphere_schema(r0=0.5, cw=1.0, n_rad=3, n_tan=3):
    """The cubed-sphere O-shell as a (geometry, connectivity) pair."""
    signs = (-1, 1)
    sign_ij = {0: 1, 1: -1, 2: 1}
    geom, nodes = {"sphere": Sphere((0, 0, 0), r0, (1, 0, 0), (0, 1, 0))}, {}
    for sx in signs:
        for sy in signs:
            for sz in signs:
                d = np.array([sx, sy, sz], float)
                sname = "S_%+d%+d%+d" % (sx, sy, sz)
                cname = "C_%+d%+d%+d" % (sx, sy, sz)
                nodes[sname] = {
                    "xyz": list(r0 * d / np.linalg.norm(d)),
                    "on": ["sphere"],
                }
                planes = ["plane_%d_%d" % (a, 1 if d[a] > 0 else 0) for a in (0, 1, 2)]
                nodes[cname] = {"xyz": list(cw * d), "on": planes}
    blocks = []
    for k in (0, 1, 2):
        i, j = (a for a in (0, 1, 2) if a != k)
        for s in signs:
            geom["plane_%d_%d" % (k, 1 if s > 0 else 0)] = Plane(
                [s * cw if m == k else 0.0 for m in (0, 1, 2)],
                [1.0 if m == i else 0.0 for m in (0, 1, 2)],
                [1.0 if m == j else 0.0 for m in (0, 1, 2)],
            )
            a1, a2 = (i, j) if s == sign_ij[k] else (j, i)

            def corner(rad, t1, t2, _k=k, _s=s, _a1=a1, _a2=a2):
                dd = [0, 0, 0]
                dd[_k], dd[_a1], dd[_a2] = _s, t1, t2
                pre = "S" if rad == 0 else "C"
                return "%s_%+d%+d%+d" % (pre, dd[0], dd[1], dd[2])

            corners = [
                corner(i0, signs[i1], signs[i2])
                for i0 in (0, 1)
                for i1 in (0, 1)
                for i2 in (0, 1)
            ]
            blocks.append(
                {
                    "name": "blk_%d%+d" % (k, s),
                    "corners": corners,
                    "res": [n_rad, n_tan, n_tan],
                }
            )
    return geom, {"nodes": nodes, "blocks": blocks}


def test_cubed_sphere_via_schema_flattens():
    geom, conn = _cubed_sphere_schema()
    topo, diags = ExplicitTopology(d=3, geometry=geom, connectivity=conn).flatten()
    assert topo is not None, [d.msg for d in diags]
    assert len(topo.block_specs) == 6
    assert len(topo.singularities) > 0  # octant fans
    grid = topo.initialize_grid()
    # inner (radial side-0) faces project onto the sphere
    for block in grid.blocks:
        inner = block.nodes[0].reshape(-1, 3)
        np.testing.assert_allclose(np.linalg.norm(inner, axis=1), 0.5, atol=1e-9)


def _has_cpp():
    try:
        from egg._cpp import cpp_core  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_cpp(), reason="egg._cpp.cpp_core not built")
def test_cubed_sphere_via_schema_solves():
    from egg.pipeline import PipelineConfig, generate_steps

    geom, conn = _cubed_sphere_schema()
    grid = ExplicitTopology(d=3, geometry=geom, connectivity=conn).initialize_grid()
    cfg = PipelineConfig(device="cpu", tmop_sweeps=60, tmop_chunk=30)
    final = None
    for _phase, info in generate_steps(grid, config=cfg, untangle_direct=True):
        final = info
    assert final["min_det"] > 0.0
    assert np.isfinite(final["energy"])


def _slab_edges():
    from egg.topology.trace3d import block_edges

    a = ["a00", "a01", "a10", "a11", "s00", "s01", "s10", "s11"]
    b = ["s00", "s01", "s10", "s11", "b00", "b01", "b10", "b11"]
    eset = set(frozenset(e) for e in block_edges(a) + block_edges(b))
    return [list(e) for e in eset]


def test_infer_slab_from_edges():
    nodes = {k: {"xyz": list(v)} for k, v in SLAB.items()}
    et = ExplicitTopology(
        d=3, connectivity={"nodes": nodes, "edges": _slab_edges(), "res": 3}
    )
    topo, diags = et.flatten()
    assert [d.kind for d in diags] == []
    assert len(topo.block_specs) == 2
    assert topo.grid.global_node_count == 4 * 4 * 4 * 2 - 16


def test_infer_cubed_sphere_prunes_voids():
    from egg.topology.trace3d import block_edges

    geom, conn = _cubed_sphere_schema()
    eset = set()
    for b in conn["blocks"]:
        for e in block_edges(b["corners"]):
            eset.add(frozenset(e))
    conn2 = {"nodes": conn["nodes"], "edges": [list(e) for e in eset]}
    topo, diags = ExplicitTopology(d=3, geometry=geom, connectivity=conn2).flatten()
    assert any(
        d.kind == "warn_pruned_void" for d in diags
    )  # cavity + outer cube dropped
    assert len(topo.block_specs) == 6
    grid = topo.initialize_grid()
    # inferred blocks have valid but arbitrary axis orientation, so check the
    # actual sphere-associated face of each block (not a fixed array slice).
    names = list(topo.block_specs)
    checked = 0
    for assoc in topo.associations:
        if assoc.entity is geom["sphere"]:
            spec = topo.block_specs[assoc.face.block_name]
            shape = spec.logical_shape
            sl = [slice(None)] * 3
            sl[assoc.face.axis] = (
                0 if assoc.face.side == 0 else shape[assoc.face.axis] - 1
            )
            face = grid.blocks[names.index(assoc.face.block_name)].nodes[tuple(sl)]
            np.testing.assert_allclose(
                np.linalg.norm(face.reshape(-1, 3), axis=1), 0.5, atol=1e-9
            )
            checked += 1
    assert checked == 6  # one sphere-bound face per O-shell block


def _unit_base():
    from egg.topology.builder import TopologyBuilder

    tb = TopologyBuilder(d=3)
    bc = {
        "c000": (0, 0, 0),
        "c001": (0, 0, 1),
        "c010": (0, 1, 0),
        "c011": (0, 1, 1),
        "c100": (1, 0, 0),
        "c101": (1, 0, 1),
        "c110": (1, 1, 0),
        "c111": (1, 1, 1),
    }
    for nm, p in bc.items():
        tb.add_corner(nm, p)
    tb.add_block(
        "B",
        corners=["c000", "c001", "c010", "c011", "c100", "c101", "c110", "c111"],
        resolutions=(2, 2, 2),
    )
    return tb


def test_base_merge_subdivides_a_block():
    """A drawn mid-plane cuts the base block into two conforming sub-blocks."""
    base = _unit_base()
    mids = {
        "m00": (0, 0, 0.5),
        "m01": (0, 1, 0.5),
        "m10": (1, 0, 0.5),
        "m11": (1, 1, 0.5),
    }
    nodes = {k: {"xyz": list(v)} for k, v in mids.items()}
    edges = [
        ["c000", "m00"],
        ["c010", "m01"],
        ["c100", "m10"],
        ["c110", "m11"],
        ["m00", "c001"],
        ["m01", "c011"],
        ["m10", "c101"],
        ["m11", "c111"],
        ["m00", "m01"],
        ["m00", "m10"],
        ["m01", "m11"],
        ["m10", "m11"],
    ]
    et = ExplicitTopology(
        d=3, base=base, connectivity={"nodes": nodes, "edges": edges, "res": 3}
    )
    topo, diags = et.flatten()
    assert [d.kind for d in diags] == []
    assert len(topo.block_specs) == 2  # the parent was dropped by containment
    grid = topo.initialize_grid()
    assert not np.any(np.isnan(grid.global_nodes))
    assert (
        grid.global_node_count == 4 * 4 * 4 * 2 - 16
    )  # two 3^3-cell blocks, shared face


def test_base_merge_no_overlay_reproduces_base():
    """A base with an empty overlay re-infers exactly the base blocking."""
    base = _unit_base()
    topo, diags = ExplicitTopology(
        d=3, base=base, connectivity={"nodes": {}, "res": 2}
    ).flatten()
    assert [d.kind for d in diags] == []
    assert len(topo.block_specs) == 1
