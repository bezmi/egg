"""SU2 export tests: 2D multiblock, 3D single block, markers, orientation."""

import numpy as np
import pytest

from egg.io.su2 import export_su2
from egg.topology.builder import TopologyBuilder


# ---------------------------------------------------------------------------
# Minimal SU2 ASCII parser for round-trip checks.
# ---------------------------------------------------------------------------


def parse_su2(path):
    """Parse an SU2 mesh into (ndime, points, elements, markers)."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0

    def expect(key):
        nonlocal i
        assert lines[i].startswith(key), f"expected {key}, got {lines[i]!r}"
        val = lines[i].split("=", 1)[1].strip()
        i += 1
        return val

    ndime = int(expect("NDIME="))

    npoin = int(expect("NPOIN="))
    points = np.array([[float(c) for c in lines[i + k].split()] for k in range(npoin)])
    i += npoin
    assert points.shape == (npoin, ndime)

    nelem = int(expect("NELEM="))
    elements = []
    for k in range(nelem):
        parts = [int(c) for c in lines[i + k].split()]
        elements.append((parts[0], parts[1:]))
    i += nelem

    nmark = int(expect("NMARK="))
    markers = {}
    for _ in range(nmark):
        tag = expect("MARKER_TAG=")
        cnt = int(expect("MARKER_ELEMS="))
        elems = []
        for k in range(cnt):
            parts = [int(c) for c in lines[i + k].split()]
            elems.append((parts[0], parts[1:]))
        i += cnt
        markers[tag] = elems
    assert i == len(lines)
    return ndime, points, elements, markers


# ---------------------------------------------------------------------------
# Grid builders.
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
    # corner order: product((0,1), repeat=2) = (lo,lo),(lo,hi),(hi,lo),(hi,hi)
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
    from itertools import product

    names = tuple(corners[idx] for idx in product((0, 1), repeat=3))
    b.add_block("cube", names, (3, 4, 5))
    b.tag_boundary("xlo", "cube", 0, 0)
    b.tag_boundary("xhi", "cube", 0, 1)
    for axis, side, tag in [
        (1, 0, "walls"),
        (1, 1, "walls"),
        (2, 0, "walls"),
        (2, 1, "walls"),
    ]:
        b.tag_boundary(tag, "cube", axis, side)
    topo = b.build()
    # initialize_grid's edge/face fill is 2D-only for now; place the
    # single-cube nodes directly through the DOF map.
    grid = topo.grid
    dof_map = grid.block_dof_maps[0]
    shape = dof_map.shape
    axes = [np.linspace(0.0, 1.0, n) for n in shape]
    coords = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    grid.global_nodes = np.empty((grid.global_node_count, 3))
    grid.global_nodes[dof_map.reshape(-1)] = coords.reshape(-1, 3)
    return topo, grid


# ---------------------------------------------------------------------------
# 2D tests.
# ---------------------------------------------------------------------------


def test_2d_counts_and_dedup(tmp_path):
    topo, grid = two_block_2d()
    out = tmp_path / "mesh.su2"
    export_su2(grid, out)
    ndime, points, elements, markers = parse_su2(out)

    assert ndime == 2
    # 5x4 + 6x4 nodes minus the shared 4-node interface column.
    assert len(points) == 5 * 4 + 6 * 4 - 4
    assert len(elements) == 4 * 3 + 5 * 3
    # No duplicate coordinates (interface written once).
    assert len(np.unique(np.round(points, 12), axis=0)) == len(points)


def test_2d_element_types_and_orientation(tmp_path):
    topo, grid = two_block_2d()
    out = tmp_path / "mesh.su2"
    export_su2(grid, out)
    _, points, elements, _ = parse_su2(out)

    for etype, conn in elements:
        assert etype == 9  # VTK quadrilateral
        assert len(conn) == 4
        p = points[conn]
        area = 0.0
        for k in range(4):
            x0, y0 = p[k]
            x1, y1 = p[(k + 1) % 4]
            area += x0 * y1 - x1 * y0
        assert area > 0, "quad must be counter-clockwise"


def test_2d_markers(tmp_path):
    topo, grid = two_block_2d()
    out = tmp_path / "mesh.su2"
    export_su2(grid, out)
    _, points, _, markers = parse_su2(out)

    assert set(markers) == {"inlet", "outlet", "wall"}
    assert len(markers["inlet"]) == 3  # L has 3 cells along axis 1
    assert len(markers["outlet"]) == 3
    assert len(markers["wall"]) == 4 + 5 + 4 + 5

    for tag, elems in markers.items():
        for etype, conn in elems:
            assert etype == 3  # VTK line
            assert len(conn) == 2

    # Geometric placement of marker nodes.
    inlet_nodes = {i for _, conn in markers["inlet"] for i in conn}
    assert np.allclose(points[sorted(inlet_nodes)][:, 0], 0.0)
    outlet_nodes = {i for _, conn in markers["outlet"] for i in conn}
    assert np.allclose(points[sorted(outlet_nodes)][:, 0], 2.0)
    wall_nodes = {i for _, conn in markers["wall"] for i in conn}
    y = points[sorted(wall_nodes)][:, 1]
    assert np.all((np.abs(y) < 1e-12) | (np.abs(y - 1.0) < 1e-12))


def test_2d_explicit_marker_mapping(tmp_path):
    topo, grid = two_block_2d()
    out = tmp_path / "mesh.su2"
    export_su2(grid, out, markers={"left_edge": [("L", 0, 0)]})
    _, _, _, markers = parse_su2(out)
    assert set(markers) == {"left_edge"}
    assert len(markers["left_edge"]) == 3


def test_flipped_block_exports_ccw(tmp_path):
    """A block whose logical axes are left-handed still exports CCW quads."""
    b = TopologyBuilder(d=2)
    # Swapped corner roles produce a negative-Jacobian logical orientation.
    for n, p in [("sw", (0, 0)), ("se", (1, 0)), ("nw", (0, 1)), ("ne", (1, 1))]:
        b.add_corner(n, p)
    b.add_block("flipped", ("sw", "se", "nw", "ne"), (3, 3))
    topo = b.build()
    grid = topo.initialize_grid()
    out = tmp_path / "mesh.su2"
    export_su2(grid, out)
    _, points, elements, _ = parse_su2(out)
    for _, conn in elements:
        p = points[conn]
        area = 0.0
        for k in range(4):
            x0, y0 = p[k]
            x1, y1 = p[(k + 1) % 4]
            area += x0 * y1 - x1 * y0
        assert area > 0


# ---------------------------------------------------------------------------
# 3D tests.
# ---------------------------------------------------------------------------


def test_3d_counts_types_volume(tmp_path):
    topo, grid = one_block_3d()
    out = tmp_path / "mesh.su2"
    export_su2(grid, out)
    ndime, points, elements, markers = parse_su2(out)

    assert ndime == 3
    assert len(points) == 4 * 5 * 6
    assert len(elements) == 3 * 4 * 5

    total_vol = 0.0
    for etype, conn in elements:
        assert etype == 12  # VTK hexahedron
        assert len(conn) == 8
        p = points[conn]
        e1, e2, e3 = p[1] - p[0], p[3] - p[0], p[4] - p[0]
        vol = float(np.dot(np.cross(e1, e2), e3))
        assert vol > 0, "hex must be positively oriented"
        total_vol += vol
    # For an axis-aligned uniform cube the corner triple product equals the
    # cell volume, so they sum to the unit-cube volume.
    assert total_vol == pytest.approx(1.0)


def test_3d_markers(tmp_path):
    topo, grid = one_block_3d()
    out = tmp_path / "mesh.su2"
    export_su2(grid, out)
    _, points, _, markers = parse_su2(out)

    assert set(markers) == {"xlo", "xhi", "walls"}
    assert len(markers["xlo"]) == 4 * 5
    assert len(markers["xhi"]) == 4 * 5
    assert len(markers["walls"]) == 2 * (3 * 5) + 2 * (3 * 4)
    for tag, elems in markers.items():
        for etype, conn in elems:
            assert etype == 9  # boundary quads
            assert len(conn) == 4
    xlo_nodes = {i for _, conn in markers["xlo"] for i in conn}
    assert np.allclose(points[sorted(xlo_nodes)][:, 0], 0.0)


# ---------------------------------------------------------------------------
# Validation / error paths.
# ---------------------------------------------------------------------------


def test_uninitialized_grid_rejected(tmp_path):
    b = TopologyBuilder(d=2)
    for n, p in [("sw", (0, 0)), ("nw", (0, 1)), ("se", (1, 0)), ("ne", (1, 1))]:
        b.add_corner(n, p)
    b.add_block("B", ("sw", "nw", "se", "ne"), (2, 2))
    topo = b.build()
    with pytest.raises(ValueError, match="NaN"):
        export_su2(topo.grid, tmp_path / "mesh.su2")


def test_tag_on_interface_face_rejected():
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
    b.add_block("L", ("sw", "nw", "s", "n"), (2, 2))
    b.add_block("R", ("s", "n", "se", "ne"), (2, 2))
    b.connect("L", 0, 1, "R", 0, 0)
    b.tag_boundary("oops", "L", 0, 1)
    with pytest.raises(ValueError, match="shared interface"):
        b.build()


def test_tag_unknown_block_rejected():
    b = TopologyBuilder(d=2)
    with pytest.raises(ValueError, match="unknown block"):
        b.tag_boundary("inlet", "nope", 0, 0)


def test_export_unknown_marker_block_rejected(tmp_path):
    topo, grid = two_block_2d()
    with pytest.raises(ValueError, match="unknown block"):
        export_su2(grid, tmp_path / "mesh.su2", markers={"m": [("nope", 0, 0)]})
