# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Compass-form add_block, object corners, and connect/associate inference."""

import numpy as np
import pytest

from egg.geometry import Edge, Line, Vector3
from egg.topology.builder import TopologyBuilder


def _wall_edges(w=4.0, h=4.0):
    sw, se = Vector3(0, 0, fixed=True), Vector3(w, 0, fixed=True)
    ne, nw = Vector3(w, h, fixed=True), Vector3(0, h, fixed=True)
    return {
        "sw": sw,
        "se": se,
        "ne": ne,
        "nw": nw,
        "bottom": Edge(Line(p0=sw, p1=se)),
        "right": Edge(Line(p0=se, p1=ne)),
        "top": Edge(Line(p0=ne, p1=nw)),
        "left": Edge(Line(p0=sw, p1=nw)),
    }


class TestCompassAddBlock:
    def test_object_corners_auto_registered(self):
        g = _wall_edges()
        b = TopologyBuilder(d=2)
        b.add_block(sw=g["sw"], se=g["se"], nw=g["nw"], ne=g["ne"], res=(4, 4))
        assert len(b._corners) == 4
        spec = b._block_specs["blk0"]  # auto-named
        # product order: (0,0)=sw, (0,1)=nw, (1,0)=se, (1,1)=ne
        pos = {n: b._corners[n].position for n in spec.corner_names}
        np.testing.assert_allclose(pos[spec.corner_names[0]], [0, 0])
        np.testing.assert_allclose(pos[spec.corner_names[1]], [0, 4])
        np.testing.assert_allclose(pos[spec.corner_names[2]], [4, 0])
        np.testing.assert_allclose(pos[spec.corner_names[3]], [4, 4])

    def test_fixed_taken_from_object(self):
        b = TopologyBuilder(d=2)
        e = Edge(Line(p0=Vector3(0, 0), p1=Vector3(4, 0)))
        b.add_block(
            sw=Vector3(0, 0, fixed=True),
            se=e.place_node(1.0),
            nw=Vector3(0, 4),
            ne=e.place_node(0.5, fixed=True),
            res=(2, 2),
        )
        fixed = {n: c.fixed for n, c in b._corners.items()}
        assert sorted(fixed.values()) == [False, False, True, True]

    def test_shared_object_is_same_corner(self):
        g = _wall_edges()
        mid_b = g["bottom"].place_node(0.5)
        mid_t = g["top"].place_node(0.5)
        b = TopologyBuilder(d=2)
        b.add_block(sw=g["sw"], se=mid_b, nw=g["nw"], ne=mid_t, res=(2, 2))
        b.add_block(sw=mid_b, se=g["se"], nw=mid_t, ne=g["ne"], res=(2, 2))
        assert len(b._corners) == 6  # mid corners deduplicated by identity

    def test_mixing_forms_rejected(self):
        b = TopologyBuilder(d=2)
        b.add_corner("a", (0, 0))
        with pytest.raises(ValueError):
            b.add_block("x", ("a", "a", "a", "a"), (1, 1), sw=Vector3(0, 0))
        with pytest.raises(ValueError):
            b.add_block(sw=Vector3(0, 0), se=Vector3(1, 0), res=(1, 1))
        with pytest.raises(ValueError):
            b.add_block(
                sw=Vector3(0, 0), se=Vector3(1, 0), nw=Vector3(0, 1), ne=Vector3(1, 1)
            )
        with pytest.raises(TypeError):
            b.add_block(sw=(0, 0), se=(1, 0), nw=(0, 1), ne=(1, 1), res=(1, 1))

    def test_positional_form_unchanged(self):
        b = TopologyBuilder(d=2)
        for n, p in [("sw", (0, 0)), ("se", (1, 0)), ("nw", (0, 1)), ("ne", (1, 1))]:
            b.add_corner(n, p)
        b.add_block("blk", ("sw", "nw", "se", "ne"), (2, 2))
        assert b._block_specs["blk"].corner_names == ("sw", "nw", "se", "ne")


class TestConnectInference:
    def _two_blocks(self, share=True):
        g = _wall_edges()
        mid_b = g["bottom"].place_node(0.5)
        mid_t = g["top"].place_node(0.5)
        mid_b2 = mid_b if share else g["bottom"].place_node(0.5)
        mid_t2 = mid_t if share else g["top"].place_node(0.5)
        b = TopologyBuilder(d=2)
        b.add_block("L", res=(2, 2), sw=g["sw"], se=mid_b, nw=g["nw"], ne=mid_t)
        b.add_block("R", res=(2, 2), sw=mid_b2, se=g["se"], nw=mid_t2, ne=g["ne"])
        return b, g

    def test_shared_corners_imply_connection(self):
        b, _ = self._two_blocks(share=True)
        topo = b.build()
        assert len(topo.interface_connections) == 1
        conn = topo.interface_connections[0]
        assert {conn.face_a.block_name, conn.face_b.block_name} == {"L", "R"}

    def test_distinct_objects_stay_disconnected(self):
        # Coincident but distinct corner objects: a slit, not a joint.
        b, _ = self._two_blocks(share=False)
        topo = b.build()
        assert len(topo.interface_connections) == 0

    def test_explicit_connection_not_duplicated(self):
        b, _ = self._two_blocks(share=True)
        b.connect("L", 0, 1, "R", 0, 0)
        topo = b.build()
        assert len(topo.interface_connections) == 1


class TestAssociateInference:
    def test_nodes_on_common_edge_imply_association(self):
        g = _wall_edges()
        b = TopologyBuilder(d=2)
        b.add_block(
            res=(2, 2),
            sw=g["bottom"].place_node(0.0),
            se=g["bottom"].place_node(1.0),
            nw=g["top"].place_node(1.0),
            ne=g["top"].place_node(0.0),
        )
        topo = b.build()
        entities = {id(a.entity) for a in topo.associations}
        assert entities == {id(g["bottom"].entity), id(g["top"].entity)}

    def test_explicit_association_not_duplicated(self):
        g = _wall_edges()
        b = TopologyBuilder(d=2)
        b.add_block(
            "blk",
            res=(2, 2),
            sw=g["bottom"].place_node(0.0),
            se=g["bottom"].place_node(1.0),
            nw=Vector3(0, 4),
            ne=Vector3(4, 4),
        )
        b.associate("blk", 1, 0, g["bottom"])
        topo = b.build()
        assert len(topo.associations) == 1

    def test_no_inference_across_edges_or_plain_points(self):
        g = _wall_edges()
        b = TopologyBuilder(d=2)
        b.add_block(
            res=(2, 2),
            sw=g["bottom"].place_node(0.0),
            se=g["right"].place_node(0.0),  # different edge
            nw=Vector3(0, 4),
            ne=Vector3(4, 4),
        )
        topo = b.build()
        assert len(topo.associations) == 0

    def test_connected_faces_not_associated(self):
        g = _wall_edges()
        m0 = g["bottom"].place_node(0.4)
        m1 = g["bottom"].place_node(0.6)
        b = TopologyBuilder(d=2)
        # Two stacked blocks whose shared face corners both sit on `bottom`:
        # the shared face must be connected, not associated.
        b.add_block(
            "lo", res=(2, 2), sw=Vector3(0, -1), se=Vector3(4, -1), nw=m0, ne=m1
        )
        b.add_block("hi", res=(2, 2), sw=m0, se=m1, nw=Vector3(0, 4), ne=Vector3(4, 4))
        topo = b.build()
        assert len(topo.interface_connections) == 1
        faces = {
            (a.face.block_name, a.face.axis, a.face.side) for a in topo.associations
        }
        assert ("lo", 1, 1) not in faces and ("hi", 1, 0) not in faces


class TestInferredGridInitializes:
    def test_full_inference_single_block(self):
        g = _wall_edges()
        b = TopologyBuilder(d=2)
        b.add_block(
            res=(4, 4),
            sw=g["bottom"].place_node(0.0, fixed=True),
            se=g["bottom"].place_node(1.0, fixed=True),
            nw=g["top"].place_node(1.0, fixed=True),
            ne=g["top"].place_node(0.0, fixed=True),
        )
        grid = b.build().initialize_grid()
        assert not np.any(np.isnan(grid.global_nodes))


class TestEdgeParamFaceInit:
    """initialize_grid spaces wall nodes in the edge parameter when both
    face corners are Nodes on one Edge — a chord projected onto a strongly
    curved wall leaves spans of it empty (no chord point projects onto a
    narrow tip's apex), and the smoother cannot recover the gap."""

    @staticmethod
    def _egg_edge():
        from egg.geometry import Spline

        theta = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
        ring = [
            Vector3(2.0 + (0.66 - 0.15 * np.sin(t)) * np.cos(t), 2.0 + 0.85 * np.sin(t))
            for t in theta
        ]
        return Edge(Spline(ring, closed=True))

    @staticmethod
    def _wall_row(egg, t0, t1, sw, se):
        b = TopologyBuilder(d=2)
        b.add_block(
            "blk",
            sw=Vector3(*sw, fixed=True),
            nw=egg.place_node(t0),
            se=Vector3(*se, fixed=True),
            ne=egg.place_node(t1),
            res=(16, 4),
        )
        grid = b.build().initialize_grid()
        return grid.blocks[0].nodes[:, -1]

    def test_wall_row_covers_high_curvature_tip(self):
        egg = self._egg_edge()
        # The face spans the egg's narrow apex (fraction 0.25, ~(2, 2.85)).
        row = self._wall_row(egg, 0.125, 0.375, sw=(3, 3), se=(1, 3))
        for p in row:
            assert np.linalg.norm(np.asarray(egg.project(p)) - p) < 1e-9
        assert (row[:, 1] > 2.8).any()  # apex covered
        seg = np.linalg.norm(np.diff(row, axis=0), axis=1)
        assert seg.max() / seg.min() < 1.5  # near-uniform, no chord gap

    def test_closed_composite_wraps_the_short_way(self):
        egg = self._egg_edge()
        # 0.875 -> 0.125 must run through the wrap point (the right side),
        # not 3/4 of the way around; CompositePath.closed is a class attr
        # (False), so closure is detected from coincident endpoints.
        row = self._wall_row(egg, 0.875, 0.125, sw=(3, 1), se=(3, 3))
        assert row[:, 0].min() > 2.2
        seg = np.linalg.norm(np.diff(row, axis=0), axis=1)
        assert seg.sum() < 1.7  # short arc, not the 3.5-long way round


class TestBlockArraySkip:
    """add_block_array(skip=...) leaves cells for hand-built replacements."""

    def _edges(self, w=4.0, h=4.0):
        sw, se = Vector3(0, 0, fixed=True), Vector3(w, 0, fixed=True)
        ne, nw = Vector3(w, h, fixed=True), Vector3(0, h, fixed=True)
        return (
            Edge(Line(p0=sw, p1=se)),  # south, west -> east
            Edge(Line(p0=nw, p1=ne)),  # north, west -> east
            Edge(Line(p0=sw, p1=nw)),  # west, south -> north
            Edge(Line(p0=se, p1=ne)),  # east, south -> north
        )

    def test_skipped_cell_absent_but_corners_and_associations_survive(self):
        south, north, west, east = self._edges()
        b = TopologyBuilder(d=2)
        corner, names = b.add_block_array(
            south=south,
            north=north,
            west=west,
            east=east,
            nib=2,
            njb=2,
            res=(8, 8),
            skip={(1, 1)},
        )
        # every array corner still exists for replacement blocks to share
        assert set(corner) == {(i, j) for i in range(3) for j in range(3)}
        topo = b.build()
        assert set(topo.block_specs) == {"b0_0", "b0_1", "b1_0"}
        grid = topo.initialize_grid()  # L-shaped domain initializes fine
        assert len(grid.blocks) == 3

    def test_dipole_insertion_reports_the_three_five_pair(self):
        # Replace the skipped ne cell with three blocks meeting at a free
        # interior corner: lines run to the north face, the east face, and
        # the cell's sw corner — which gains a block and goes 5-valent.
        south, north, west, east = self._edges()
        b = TopologyBuilder(d=2)
        corner, _names = b.add_block_array(
            south=south,
            north=north,
            west=west,
            east=east,
            nib=2,
            njb=2,
            res=(8, 8),
            skip={(1, 1)},
        )
        b.add_corner("s3", (3.2, 3.4), fixed=False)
        b.add_corner("an", north.place_node(0.8), fixed=False)
        b.add_corner("ae", east.place_node(0.8), fixed=False)
        b.add_block(
            "pa", sw=corner[1, 1], se="s3", ne="an", nw=corner[1, 2], res=(2, 4)
        )
        b.add_block(
            "pb", sw=corner[1, 1], se=corner[2, 1], ne="ae", nw="s3", res=(4, 2)
        )
        b.add_block("pc", sw="s3", se="ae", ne=corner[2, 2], nw="an", res=(4, 4))
        topo = b.build()
        valences = sorted(s.valence for s in topo.singularities)
        assert valences == [3, 5]
        grid = topo.initialize_grid()
        assert len(grid.blocks) == 6
        for blk in grid.blocks:  # TFI init untangled around the singularity
            p = blk.nodes[..., :2]
            e0 = p[1:, :-1] - p[:-1, :-1]
            e1 = p[:-1, 1:] - p[:-1, :-1]
            assert float(np.cross(e0, e1).min()) > 0.0
