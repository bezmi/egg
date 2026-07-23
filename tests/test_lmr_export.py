# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""lmr grid export tests: per-block files, i-fastest ordering, grid.lua stub.

The parsers here mirror gdtk's own readers (``read_from_gzip_file`` /
``read_from_raw_binary_file`` / ``read_from_text_file``) so a round trip pins
the vertex ordering, which is the one thing that can silently corrupt geometry.
"""

import gzip
import struct

import numpy as np
import pytest

from egg.io.lmr import export_lmr
from egg.topology.builder import TopologyBuilder


# ---------------------------------------------------------------------------
# Readers matching gdtk's structured-grid parsers.
# ---------------------------------------------------------------------------


def read_gziptext(path):
    with gzip.open(path, "rt", encoding="ascii") as f:
        lines = f.read().splitlines()
    assert lines[0] == "structured_grid 1.1"
    label = lines[1].split("label:", 1)[1].strip()
    dims = int(lines[2].split(":")[1])
    niv = int(lines[3].split(":")[1])
    njv = int(lines[4].split(":")[1])
    nkv = int(lines[5].split(":")[1])
    n = niv * njv * nkv
    pts = np.array([[float(c) for c in lines[6 + k].split()] for k in range(n)])
    assert lines[6 + n].startswith("ntags:")
    return label, dims, (niv, njv, nkv), pts


def read_rawbinary(path):
    with open(path, "rb") as f:
        header = f.read(len("structured_grid 1.1"))
        assert header == b"structured_grid 1.1"
        (label_len,) = struct.unpack("<i", f.read(4))
        label = f.read(label_len).decode("ascii")
        dims, niv, njv, nkv = struct.unpack("<4i", f.read(16))
        n = niv * njv * nkv
        pts = np.frombuffer(f.read(n * 3 * 8), dtype="<f8").reshape(n, 3)
        (ntags,) = struct.unpack("<i", f.read(4))
        assert ntags == 0
    return label, dims, (niv, njv, nkv), pts.copy()


def read_vtk(path):
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    assert lines[0].startswith("# vtk DataFile")
    label = lines[1]
    assert lines[4] == "DATASET STRUCTURED_GRID"
    niv, njv, nkv = (int(c) for c in lines[5].split()[1:4])
    n = niv * njv * nkv
    assert lines[6].startswith("POINTS")
    pts = np.array([[float(c) for c in lines[7 + k].split()] for k in range(n)])
    return label, None, (niv, njv, nkv), pts


READERS = {"gziptext": read_gziptext, "rawbinary": read_rawbinary, "vtk": read_vtk}


# ---------------------------------------------------------------------------
# Grid builders (shared with the SU2 suite's shapes).
# ---------------------------------------------------------------------------


def two_block_2d():
    """Two unit-square blocks side by side, sharing one edge."""
    b = TopologyBuilder(d=2)
    for n, p in [
        ("sw", (0, 0)),
        ("s", (1, 0)),
        ("se", (2, 0)),
        ("nw", (0, 1)),
        ("n", (1, 1)),
        ("ne", (2, 1)),
    ]:
        b.add_corner(n, p)
    b.add_block("L", ("sw", "nw", "s", "n"), (4, 3))
    b.add_block("R", ("s", "n", "se", "ne"), (5, 3))
    b.connect("L", 0, 1, "R", 0, 0)
    b.tag_boundary("inlet", "L", 0, 0)
    b.tag_boundary("outlet", "R", 0, 1)
    for blk in ("L", "R"):
        b.tag_boundary("wall", blk, 1, 0)
        b.tag_boundary("wall", blk, 1, 1)
    topo = b.build()
    grid = topo.initialize_grid()
    return topo, grid


def one_block_3d():
    """A single unit-cube block."""
    from itertools import product

    b = TopologyBuilder(d=3)
    corners = {
        (0, 0, 0): "c000",
        (0, 0, 1): "c001",
        (0, 1, 0): "c010",
        (0, 1, 1): "c011",
        (1, 0, 0): "c100",
        (1, 0, 1): "c101",
        (1, 1, 0): "c110",
        (1, 1, 1): "c111",
    }
    for idx, name in corners.items():
        b.add_corner(name, np.array(idx, dtype=float))
    names = tuple(corners[idx] for idx in product((0, 1), repeat=3))
    b.add_block("cube", names, (3, 4, 5))
    topo = b.build()
    grid = topo.grid
    dof_map = grid.block_dof_maps[0]
    axes = [np.linspace(0.0, 1.0, n) for n in dof_map.shape]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    grid.global_nodes = np.empty((grid.global_node_count, 3))
    grid.global_nodes[dof_map.reshape(-1)] = coords.reshape(-1, 3)
    return topo, grid


# ---------------------------------------------------------------------------
# Per-format round-trip and ordering.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["gziptext", "rawbinary", "vtk"])
def test_2d_roundtrip_dims_and_labels(tmp_path, fmt):
    topo, grid = two_block_2d()
    written = export_lmr(grid, tmp_path, fmt=fmt)
    ext = {"gziptext": "gz", "rawbinary": "bin", "vtk": "vtk"}[fmt]

    block0 = tmp_path / f"block-0000.{ext}"
    block1 = tmp_path / f"block-0001.{ext}"
    assert block0 in [__import__("pathlib").Path(p) for p in written]

    label, dims, (niv, njv, nkv), pts = READERS[fmt](block0)
    assert (niv, njv, nkv) == (5, 4, 1)  # L: (4,3) cells -> (5,4) nodes
    assert len(pts) == 5 * 4
    if dims is not None:
        assert dims == 2
    if fmt != "vtk":
        assert label == "L"

    _, _, (niv1, njv1, _), pts1 = READERS[fmt](block1)
    assert (niv1, njv1) == (6, 4)  # R: (5,3) cells -> (6,4) nodes


@pytest.mark.parametrize("fmt", ["gziptext", "rawbinary", "vtk"])
def test_i_fastest_ordering(tmp_path, fmt):
    """The exported point list must match gdtk's single_index = i + niv*(j+njv*k)."""
    topo, grid = two_block_2d()
    export_lmr(grid, tmp_path, fmt=fmt)
    ext = {"gziptext": "gz", "rawbinary": "bin", "vtk": "vtk"}[fmt]
    _, _, (niv, njv, nkv), pts = READERS[fmt](tmp_path / f"block-0000.{ext}")

    dof_map = grid.block_dof_maps[0]  # shape (niv, njv)
    coords = grid.global_nodes[dof_map]  # [i, j, comp]
    for k in range(nkv):
        for j in range(njv):
            for i in range(niv):
                flat = i + niv * (j + njv * k)
                assert pts[flat][0] == pytest.approx(coords[i, j, 0])
                assert pts[flat][1] == pytest.approx(coords[i, j, 1])
                assert pts[flat][2] == pytest.approx(0.0)  # z padded in 2D


def test_3d_ordering_and_z(tmp_path):
    topo, grid = one_block_3d()
    export_lmr(grid, tmp_path, fmt="gziptext")
    _, dims, (niv, njv, nkv), pts = read_gziptext(tmp_path / "block-0000.gz")
    assert dims == 3
    assert (niv, njv, nkv) == (4, 5, 6)  # (3,4,5) cells -> (4,5,6) nodes
    dof_map = grid.block_dof_maps[0]
    coords = grid.global_nodes[dof_map]  # [i, j, k, comp]
    for k in range(nkv):
        for j in range(njv):
            for i in range(niv):
                flat = i + niv * (j + njv * k)
                assert np.allclose(pts[flat], coords[i, j, k])


# ---------------------------------------------------------------------------
# grid.lua stub.
# ---------------------------------------------------------------------------


def test_grid_lua_registration(tmp_path):
    topo, grid = two_block_2d()
    export_lmr(grid, tmp_path, fmt="gziptext")
    lua = (tmp_path / "grid.lua").read_text()

    assert "config.dimensions = 2" in lua
    assert 'StructuredGrid:new{filename="block-0000.gz", fmt="gziptext"}' in lua
    assert 'StructuredGrid:new{filename="block-0001.gz", fmt="gziptext"}' in lua
    assert "identifyGridConnections()" in lua

    # External faces carry the topology's boundary tags; the shared interface
    # (L east / R west) must NOT appear as a bcTag.
    assert 'west="inlet"' in lua  # L axis0 side0
    assert 'east="outlet"' in lua  # R axis0 side1
    assert 'south="wall"' in lua
    assert 'north="wall"' in lua
    # L's east face is the interface -> not tagged.
    l_block = lua.split("grid1 =")[0]
    assert '"east"' not in l_block
    # every external face here has a real tag, none auto-assigned as a bcTag
    # value (the instruction header mentions egg-untagged, so match the ="form)
    assert '="egg-untagged' not in lua


def test_untagged_no_geometry_is_per_block_edge(tmp_path):
    """External faces with no geometry each get their own egg-untagged-N."""
    from egg.io.lmr import untagged_external_faces

    b = TopologyBuilder(d=2)
    for n, p in [("sw", (0, 0)), ("nw", (0, 1)), ("se", (1, 0)), ("ne", (1, 1))]:
        b.add_corner(n, p)
    b.add_block("B", ("sw", "nw", "se", "ne"), (2, 2))
    topo = b.build()
    grid = topo.initialize_grid()
    export_lmr(grid, tmp_path, fmt="gziptext")
    lua = (tmp_path / "grid.lua").read_text()
    assert "TODO" not in lua
    # all four faces external and geometry-free -> four separate markers.
    for i in range(4):
        assert f'"egg-untagged-{i}"' in lua

    groups = untagged_external_faces(grid)
    assert len(groups) == 4
    assert {g["tag"] for g in groups} == {f"egg-untagged-{i}" for i in range(4)}
    for g in groups:
        assert g["geometry"] is None  # no geometry
        assert len(g["faces"]) == 1  # one block edge each
    faces = {(f["block"], f["face"]) for g in groups for f in g["faces"]}
    assert faces == {("B", "west"), ("B", "east"), ("B", "south"), ("B", "north")}


def test_untagged_shared_geometry_gets_one_marker(tmp_path):
    """Faces of different blocks lying on one tag-less entity share a marker."""
    from egg.geometry import Line, Vector3
    from egg.io.lmr import untagged_external_faces

    b = TopologyBuilder(d=2)
    # Two disjoint unit squares stacked with a gap; both west faces lie on one
    # (unnamed, so tag-less) vertical line spanning both.
    for n, p in [
        ("a_sw", (0, 0)),
        ("a_nw", (0, 1)),
        ("a_se", (1, 0)),
        ("a_ne", (1, 1)),
        ("b_sw", (0, 2)),
        ("b_nw", (0, 3)),
        ("b_se", (1, 2)),
        ("b_ne", (1, 3)),
    ]:
        b.add_corner(n, p)
    b.add_block("A", ("a_sw", "a_nw", "a_se", "a_ne"), (2, 2))
    b.add_block("B", ("b_sw", "b_nw", "b_se", "b_ne"), (2, 2))
    line = Line(Vector3(0.0, 0.0), Vector3(0.0, 3.0))  # unnamed -> tag None
    b.associate("A", 0, 0, line)
    b.associate("B", 0, 0, line)
    topo = b.build()
    # The grouping reads the topology only; set finite node coords directly so
    # the export's NaN guard passes without initialize_grid's cpp projection.
    grid = topo.grid
    grid.global_nodes = np.zeros((grid.global_node_count, 2))

    groups = untagged_external_faces(grid)
    # exactly one group covers both west faces (same entity)
    geo_groups = [g for g in groups if g["geometry"] is not None]
    assert len(geo_groups) == 1
    shared = geo_groups[0]
    assert {(f["block"], f["face"]) for f in shared["faces"]} == {
        ("A", "west"),
        ("B", "west"),
    }
    export_lmr(grid, tmp_path, fmt="gziptext")
    lua = (tmp_path / "grid.lua").read_text()
    # both A and B register their west face under the same marker
    assert lua.count(f'west="{shared["tag"]}"') == 2


def test_grid_lua_instructions_toggle(tmp_path):
    """The 'how to run in lmr' header is on by default and can be suppressed.

    two_block_2d tags every external face, so the untagged step is dropped and
    the gas step is renumbered to 1.
    """
    topo, grid = two_block_2d()

    on = tmp_path / "on"
    export_lmr(grid, on, fmt="gziptext", grid_lua_instructions=True)
    lua_on = (on / "grid.lua").read_text()
    assert "To run this grid in lmr:" in lua_on
    assert "lmr prep-gas" in lua_on and "makeFluidBlocks" in lua_on
    # nothing untagged -> no untagged step, gas is step 1
    assert "untagged boundaries" not in lua_on
    assert "1. Gas:" in lua_on
    # instructions are comments only; the registration still follows
    assert "config.dimensions = 2" in lua_on
    assert "identifyGridConnections()" in lua_on

    off = tmp_path / "off"
    export_lmr(grid, off, fmt="gziptext", grid_lua_instructions=False)
    lua_off = (off / "grid.lua").read_text()
    assert "To run this grid in lmr:" not in lua_off
    assert "config.dimensions = 2" in lua_off  # the stub itself is unchanged
    assert "identifyGridConnections()" in lua_off


def test_grid_lua_instructions_list_untagged(tmp_path):
    """With untagged faces, step 1 names every egg-untagged-* placeholder and
    the following steps renumber after it."""
    b = TopologyBuilder(d=2)
    for n, p in [("sw", (0, 0)), ("nw", (0, 1)), ("se", (1, 0)), ("ne", (1, 1))]:
        b.add_corner(n, p)
    b.add_block("B", ("sw", "nw", "se", "ne"), (2, 2))
    grid = b.build().initialize_grid()
    export_lmr(grid, tmp_path, fmt="gziptext", grid_lua_instructions=True)
    header = (tmp_path / "grid.lua").read_text().split("config.dimensions")[0]

    assert "untagged boundaries" in header
    for i in range(4):  # all four placeholders named in the header
        assert f"egg-untagged-{i}" in header
    assert "2. Gas:" in header  # gas renumbered after the untagged step


def test_no_grid_lua(tmp_path):
    topo, grid = two_block_2d()
    written = export_lmr(grid, tmp_path, fmt="gziptext", write_grid_lua=False)
    assert not (tmp_path / "grid.lua").exists()
    assert all("grid.lua" not in p for p in written)


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


def test_uninitialized_grid_rejected(tmp_path):
    b = TopologyBuilder(d=2)
    for n, p in [("sw", (0, 0)), ("nw", (0, 1)), ("se", (1, 0)), ("ne", (1, 1))]:
        b.add_corner(n, p)
    b.add_block("B", ("sw", "nw", "se", "ne"), (2, 2))
    topo = b.build()
    with pytest.raises(ValueError, match="NaN"):
        export_lmr(topo.grid, tmp_path)


def test_bad_format_rejected(tmp_path):
    topo, grid = two_block_2d()
    with pytest.raises(ValueError, match="Unknown lmr grid format"):
        export_lmr(grid, tmp_path, fmt="nonsense")


def test_overwrite_guard(tmp_path):
    """Re-exporting into a dir that already holds an export raises unless forced."""
    topo, grid = two_block_2d()
    export_lmr(grid, tmp_path, fmt="gziptext")
    with pytest.raises(FileExistsError, match="already holds an lmr export"):
        export_lmr(grid, tmp_path, fmt="gziptext")
    # explicit opt-in replaces it
    export_lmr(grid, tmp_path, fmt="gziptext", overwrite=True)
    assert (tmp_path / "grid.lua").exists()
    # a fresh directory is never blocked
    export_lmr(grid, tmp_path / "fresh", fmt="gziptext")
