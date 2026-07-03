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
