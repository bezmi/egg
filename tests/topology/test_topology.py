"""Tests for TopologyBuilder and BlockTopology."""

import pytest

from egg.topology.builder import TopologyBuilder


class TestTopologyBuilder:
    def test_rejects_wrong_corner_count(self):
        builder = TopologyBuilder(d=2)
        builder.add_corner("A", (0., 0.))
        builder.add_corner("B", (1., 0.))
        builder.add_corner("C", (1., 1.))
        with pytest.raises(ValueError, match="4"):
            builder.add_block("bad", ("A", "B", "C"), (4, 4))

    def test_rejects_unknown_corner(self):
        builder = TopologyBuilder(d=2)
        builder.add_corner("A", (0., 0.))
        with pytest.raises(ValueError, match="unknown"):
            builder.add_block("bad", ("A", "X", "Y", "Z"), (4, 4))

    def test_rejects_wrong_resolution_count(self):
        builder = TopologyBuilder(d=2)
        for n in "ABCD":
            builder.add_corner(n, (0., 0.))
        with pytest.raises(ValueError):
            builder.add_block("bad", ("A", "B", "C", "D"), (4,))

    def test_connect_rejects_nonexistent_block(self):
        builder = TopologyBuilder(d=2)
        with pytest.raises(ValueError):
            builder.connect("ghost", 0, 0, "phantom", 0, 0)

    def test_connect_accepts_matching_faces(self):
        builder = TopologyBuilder(d=2)
        for name, pos in [("A", (0., 0.)), ("B", (2., 0.)),
                          ("C", (2., 2.)), ("D", (0., 2.)),
                          ("E", (4., 0.)), ("F", (4., 2.))]:
            builder.add_corner(name, pos)
        builder.add_block("L", ("A", "D", "B", "C"), (4, 4))
        builder.add_block("R", ("B", "C", "E", "F"), (4, 4))
        builder.connect("L", 0, 1, "R", 0, 0)
        topo = builder.build()
        assert topo is not None

    def test_connect_rejects_mismatched_corners(self):
        """Faces with different corner sets should raise."""
        builder = TopologyBuilder(d=2)
        for name, pos in [("A", (0., 0.)), ("B", (2., 0.)),
                          ("C", (2., 2.)), ("D", (0., 2.)),
                          ("E", (4., 0.)), ("F", (4., 2.))]:
            builder.add_corner(name, pos)
        builder.add_block("L", ("A", "D", "B", "C"), (4, 4))
        builder.add_block("R", ("A", "B", "E", "F"), (4, 4))
        builder.connect("L", 0, 1, "R", 0, 0)
        with pytest.raises(ValueError, match="do not match"):
            builder.build()

    def test_associate_rejects_nonexistent_block(self):
        builder = TopologyBuilder(d=2)
        with pytest.raises(ValueError, match="unknown"):
            builder.associate("ghost", 0, 0, None)


class TestSingularityDetection:
    def test_cartesian_one_block_no_singularities(self):
        builder = TopologyBuilder(d=2)
        for name, pos in [("sw", (0., 0.)), ("se", (4., 0.)),
                          ("ne", (4., 4.)), ("nw", (0., 4.))]:
            builder.add_corner(name, pos, fixed=True)
        builder.add_block("main", ("sw", "nw", "se", "ne"), (4, 4))
        topo = builder.build()
        assert len(topo.singularities) == 0

    def test_two_block_shared_edge_no_singularities(self):
        builder = TopologyBuilder(d=2)
        (builder
         .add_corner("A", (0., 0.), fixed=True)
         .add_corner("B", (2., 0.), fixed=True)
         .add_corner("C", (2., 2.), fixed=True)
         .add_corner("D", (0., 2.), fixed=True)
         .add_corner("E", (4., 0.), fixed=True)
         .add_corner("F", (4., 2.), fixed=True)
         .add_block("L", ("A", "D", "B", "C"), (4, 4))
         .add_block("R", ("B", "C", "E", "F"), (4, 4))
         .connect("L", 0, 1, "R", 0, 0))
        topo = builder.build()
        assert len(topo.singularities) == 0

    def test_ogrid_inner_corners_are_not_graph_singularities(self):
        """O-grid inner corners have valence == 2*d (4 neighbours), not singularities.

        The inner O-ring corners are shared between two blocks and have 4 graph
        neighbours (2 within-block + 2 cross-interface via the two shared faces).
        They are geometric (grid-line-focus) singularities, not graph-theoretic
        (edge-valence) ones.
        """
        builder = TopologyBuilder(d=2)
        for name, pos in [("sw", (0., 0.)), ("se", (4., 0.)),
                          ("ne", (4., 4.)), ("nw", (0., 4.))]:
            builder.add_corner(name, pos, fixed=True)
        for name, pos in [("isw", (1.5, 1.5)), ("ise", (2.5, 1.5)),
                          ("ine", (2.5, 2.5)), ("inw", (1.5, 2.5))]:
            builder.add_corner(name, pos, fixed=False)

        builder.add_block("south", ("sw", "isw", "se", "ise"), (8, 4))
        builder.add_block("east", ("se", "ise", "ne", "ine"), (8, 4))
        builder.add_block("north", ("ne", "ine", "nw", "inw"), (8, 4))
        builder.add_block("west", ("nw", "inw", "sw", "isw"), (8, 4))

        builder.connect("south", 0, 1, "east", 0, 0)
        builder.connect("east", 0, 1, "north", 0, 0)
        builder.connect("north", 0, 1, "west", 0, 0)
        builder.connect("west", 0, 1, "south", 0, 0)

        topo = builder.build()
        assert len(topo.singularities) == 0

    def test_three_block_junction_regular_valence(self):
        """Three blocks meeting at a corner with 2 shared faces → valence 4.

        Junction of A, B, C at the shared corner: the within-block neighbours
        and cross-interface neighbours fill all 4 directions (2× within-block
        + 2× cross-interface), so the node has valence == 2*d — not a singularity.
        """
        builder = TopologyBuilder(d=2)
        builder.add_corner("sw", (0., 0.), fixed=False)
        builder.add_corner("nw", (0., 2.), fixed=False)
        builder.add_corner("se", (2., 0.), fixed=False)
        builder.add_corner("ne", (2., 2.), fixed=False)
        builder.add_corner("e_se", (4., 0.), fixed=False)
        builder.add_corner("e_ne", (4., 2.), fixed=False)
        builder.add_corner("n_nw", (0., 4.), fixed=False)
        builder.add_corner("n_ne", (2., 4.), fixed=False)

        builder.add_block("A", ("sw", "nw", "se", "ne"), (4, 4))
        builder.add_block("B", ("se", "ne", "e_se", "e_ne"), (4, 4))
        builder.add_block("C", ("nw", "n_nw", "ne", "n_ne"), (4, 4))

        builder.connect("A", 0, 1, "B", 0, 0)
        builder.connect("A", 1, 1, "C", 1, 0)

        topo = builder.build()
        assert len(topo.singularities) == 0
