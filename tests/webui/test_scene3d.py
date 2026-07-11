# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""The 3D webui scene payload (webui/scene3d.py)."""

import numpy as np

from egg.webui import scene3d

from egg.geometry.frontend3d import Bezier3, Edge  # noqa: E402
from egg.topology.explicit import ExplicitTopology  # noqa: E402


def _slab_grid(res=3):
    C = {
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
    nodes = {k: {"xyz": list(v)} for k, v in C.items()}
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
    return ExplicitTopology(
        d=3, connectivity={"nodes": nodes, "blocks": blocks, "res": res}
    ).initialize_grid()


def test_topology_payload_has_block_meshes():
    grid = _slab_grid(res=3)
    payload = scene3d.topology_scene3d(grid)
    assert len(payload["blocks"]) == 2
    blk = payload["blocks"][0]
    assert blk["name"] in ("A", "B")
    # a 3^3-cell block: 4^3 = 64 verts, 6 faces x 3x3 quads = 54
    assert len(blk["verts"]) == 64
    assert len(blk["quads"]) == 54
    # every quad index is in range
    for q in blk["quads"]:
        assert all(0 <= i < 64 for i in q) and len(q) == 4
    assert payload["bounds"]["min"] == [0.0, 0.0, 0.0]
    assert payload["bounds"]["max"] == [2.0, 1.0, 1.0]


def test_geometry_payload_samples_curves():
    curve = Edge(Bezier3([[0, 0, 0], [0.5, 1.0, 0.5], [1, 0, 0]]))
    out = scene3d.geometry_scene3d({"c": curve})
    assert len(out) == 1
    assert out[0]["name"] == "c" and out[0]["kind"] == "curve"
    line = np.array(out[0]["line"])
    assert line.shape[1] == 3
    np.testing.assert_allclose(line[0], [0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(line[-1], [1, 0, 0], atol=1e-9)


def test_full_payload_shape():
    grid = _slab_grid()
    payload = scene3d.scene3d_payload(grid, {})
    assert set(payload) == {"blocks", "bounds", "curves"}
    assert payload["curves"] == []
