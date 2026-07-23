# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""print_topology / to_connectivity: export a flattened topology (base blocks
included) as a standalone blocking that replicates it."""

import io

from egg.geometry import Line, Plane, Vector3
from egg.topology import ExplicitTopology, TopologyBuilder, editable


def _corner_classes(topo):
    """corner name -> 'frozen' | 'slide:<entity name>' | 'free'."""
    grid = topo.initialize_grid()
    names = list(topo.block_specs.keys())
    cls = {}
    for bi, bname in enumerate(names):
        spec = topo.block_specs[bname]
        shape = spec.logical_shape
        dof_map = grid.block_dof_maps[bi]
        for ci, cl in enumerate(spec._corner_logical_indices(topo.d)):
            actual = tuple(0 if c == 0 else shape[k] - 1 for k, c in enumerate(cl))
            g = int(dof_map[actual])
            cname = spec.corner_names[ci]
            if not grid.free_mask[g]:
                cls[cname] = "frozen"
            elif g in grid.dof_constraints:
                cls[cname] = f"slide:{getattr(grid.dof_constraints[g], 'name', '?')}"
            else:
                cls[cname] = "free"
    return cls


def _block_map(topo):
    return {
        frozenset(s.corner_names): tuple(sorted(s.resolutions))
        for s in topo.block_specs.values()
    }


def _tag_map(topo):
    return {
        tag: {
            frozenset(
                topo.block_specs[f.block_name].face_corner_names(f.axis, f.side, topo.d)
            )
            for f in faces
        }
        for tag, faces in topo.boundary_tags.items()
    }


def _base_2d():
    """Two conforming blocks: wall along y=0, tagged lid along y=1, one pinned
    corner, distinct per-axis resolutions."""
    wall = Line(Vector3(0, 0), Vector3(2, 0)).named("wall")
    lid = Line(Vector3(0, 1), Vector3(2, 1)).named("lid", tag="top")
    A, B, C = Vector3(0, 0), Vector3(1, 0, fixed=True), Vector3(2, 0)
    D, E, F = Vector3(0, 1), Vector3(1, 1), Vector3(2, 1)
    bld = TopologyBuilder(d=2)
    bld.add_block("L", sw=A, se=B, nw=D, ne=E, res=(4, 6))
    bld.add_block("R", sw=B, se=C, nw=E, ne=F, res=(5, 6))
    for name in ("L", "R"):
        bld.associate(name, 1, 0, wall)
        bld.associate(name, 1, 1, lid)
    return bld, {"wall": wall, "lid": lid}


class Test2DRoundtrip:
    def test_standalone_rebuild_matches(self):
        bld, geometry = _base_2d()
        et = ExplicitTopology(base=bld, geometry=geometry, connectivity={})
        orig = et.build()
        conn = et.to_connectivity()
        new = ExplicitTopology(base=None, geometry=geometry, connectivity=conn).build()

        assert _block_map(new) == _block_map(orig)
        assert _corner_classes(new) == _corner_classes(orig)
        assert _tag_map(new) == _tag_map(orig)

    def test_printed_block_is_valid_python_and_rebuilds(self):
        bld, geometry = _base_2d()
        et = ExplicitTopology(base=bld, geometry=geometry, connectivity={})
        text = et.print_topology(geometry_var="geometry", file=io.StringIO())

        # Booleans must be Python literals, never JSON's lowercase.
        assert '"fixed": True' in text
        assert "true" not in text

        expr = "\n".join(
            ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
        )
        rebuilt = eval(  # noqa: S307 - round-trip of our own printed literal
            expr,
            {
                "ExplicitTopology": ExplicitTopology,
                "editable": editable,
                "geometry": geometry,
            },
        )
        assert _block_map(rebuilt.build()) == _block_map(et.build())

    def test_exact_on_curve_free_node_stays_free(self):
        # A free node placed exactly on a named curve it never declared must
        # export (and re-flatten) as free — membership is declaration-only.
        bld, geometry = _base_2d()
        et = ExplicitTopology(base=bld, geometry=geometry, connectivity={})
        conn = et.to_connectivity()
        mid = Line(Vector3(0, 0.5), Vector3(2, 0.5)).named("mid")
        geometry2 = dict(geometry, mid=mid)
        new = ExplicitTopology(base=None, geometry=geometry2, connectivity=conn).build()
        # B–E and E–F interfaces have endpoints off y=0.5; the corners D/E/F
        # of the lid row are off it too. No corner may bind to `mid`.
        assert all("mid" not in v for v in _corner_classes(new).values())


class Test3DRoundtrip:
    def test_two_hexes_with_surface(self):
        floor = Plane((0, 0, 0), (1, 0, 0), (0, 1, 0))
        geometry = {"floor": floor}

        # Two conforming hexes sharing the x=1 face, bottom corners on the
        # floor plane; product((0,1), repeat=3) corner order, z fastest.
        nodes = {
            f"p{x}{y}{z}": (
                {"xyz": [x, y, z], "on": ["floor"]} if z == 0 else {"xyz": [x, y, z]}
            )
            for x in (0, 1, 2)
            for y in (0, 1)
            for z in (0, 1)
        }
        c0 = [f"p{i}{j}{k}" for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        c1 = [f"p{i + 1}{j}{k}" for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        et = ExplicitTopology(
            base=None,
            geometry=geometry,
            connectivity={
                "nodes": nodes,
                "blocks": [
                    {"name": "H0", "corners": c0, "res": [3, 4, 5]},
                    {"name": "H1", "corners": c1, "res": [6, 4, 5]},
                ],
            },
            d=3,
        )
        orig = et.build()
        conn = et.to_connectivity()

        # 3D blocks keep their names and per-axis resolutions verbatim.
        by_name = {b["name"]: b for b in conn["blocks"]}
        assert set(by_name) == {"H0", "H1"}
        assert by_name["H0"]["res"] == [3, 4, 5]
        assert by_name["H1"]["res"] == [6, 4, 5]
        # z=0 corners of both hexes carry the surface binding.
        on_floor = {n for n, s in conn["nodes"].items() if "floor" in s.get("on", ())}
        assert len(on_floor) == 6

        new = ExplicitTopology(
            base=None, geometry=geometry, connectivity=conn, d=3
        ).build()
        assert {
            n: (set(s.corner_names), tuple(s.resolutions))
            for n, s in new.block_specs.items()
        } == {
            n: (set(s.corner_names), tuple(s.resolutions))
            for n, s in orig.block_specs.items()
        }
        assert len(new.interface_connections) == len(orig.interface_connections) == 1
        floor_faces = lambda t: {  # noqa: E731
            (a.face.block_name, a.face.axis, a.face.side)
            for a in t.associations
            if a.entity is floor
        }
        assert floor_faces(new) == floor_faces(orig)
