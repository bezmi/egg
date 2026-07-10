# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""3D grid initialization: curve-aware edge spacing and surface projection."""

import numpy as np

from egg.geometry.analytic3d import Sphere
from egg.geometry.frontend3d import Bezier3, Edge
from egg.topology.builder import TopologyBuilder

CTRL = [[0, 0, 0], [0.5, 1.0, 0.5], [1, 0, 0]]  # bulges in +y and +z


def test_3d_curve_aware_edge_spacing():
    """A hex edge whose endpoints are Nodes on a curve is sampled on the curve."""
    curve = Edge(Bezier3(CTRL))
    n0, n1 = curve.place_node(0.0), curve.place_node(1.0)

    tb = TopologyBuilder(d=3)
    tb.add_corner("c000", n0)
    tb.add_corner("c100", n1)
    for nm, pos in [
        ("c001", (0, 0, 1)),
        ("c010", (0, 1, 0)),
        ("c011", (0, 1, 1)),
        ("c101", (1, 0, 1)),
        ("c110", (1, 1, 0)),
        ("c111", (1, 1, 1)),
    ]:
        tb.add_corner(nm, pos)
    tb.add_block(
        "B",
        corners=("c000", "c001", "c010", "c011", "c100", "c101", "c110", "c111"),
        resolutions=(6, 3, 3),
    )
    grid = tb.build().initialize_grid()
    assert not np.any(np.isnan(grid.global_nodes))

    # The edge runs along axis 0 at (y-index 0, z-index 0), sampled by curve
    # parameter: interior node k sits at eval(k / n_cells).
    edge_line = grid.blocks[0].nodes[:, 0, 0]
    n_cells = edge_line.shape[0] - 1
    for k in range(1, n_cells):
        np.testing.assert_allclose(
            edge_line[k], Bezier3(CTRL).eval(k / n_cells), atol=1e-9
        )

    # It is genuinely curved, not the chord (which stays at y=z=0).
    assert np.max(np.abs(edge_line[1:-1, 1:])) > 0.05


def test_3d_face_projects_onto_surface():
    """A block face associated to a Sphere lands on the sphere after init.

    In 3D the face interior is NaN until the volume TFI, so initialize_grid
    fills the face from its edges before projecting it onto the entity.
    """
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
    tb = TopologyBuilder(d=3)
    for nm, pos in pts.items():
        tb.add_corner(nm, pos)
    tb.add_block(
        "B",
        corners=("c000", "c001", "c010", "c011", "c100", "c101", "c110", "c111"),
        resolutions=(4, 4, 4),
    )
    tb.associate("B", 2, 0, sph)  # z-min face onto the sphere
    grid = tb.build().initialize_grid()

    assert not np.any(np.isnan(grid.global_nodes))
    face = grid.blocks[0].nodes[:, :, 0].reshape(-1, 3)
    np.testing.assert_allclose(np.linalg.norm(face, axis=1), 1.0, atol=1e-9)
