# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""CAD import adapter (egg.io.cad): build123d faces/edges to egg entities.

Skips without the ``cad`` group (build123d). Covers face -> BSplineSurface with
UV trim loops (planar, periodic, and a face with a hole), edge -> Line3, and
attaching an extracted surface to a TopologyBuilder(d=3) block so init projects
onto it.
"""

import numpy as np
import pytest


def _has_cad() -> bool:
    try:
        import build123d  # noqa: F401
        import OCP  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cad(), reason="build123d not installed (uv sync --group cad)"
)


def test_box_faces_to_named_surfaces():
    from build123d import Box

    from egg.io import cad
    from egg.geometry.surfaces3d import BSplineSurface

    sm = cad.surfaces(Box(2, 2, 2).faces(), tag="wall")
    assert len(sm) == 6
    s = sm["face_0"]
    assert isinstance(s, BSplineSurface)
    assert s.name == "face_0" and s.tag == "wall"
    assert s.trim is not None and len(s.trim) == 1  # a single outer loop


def test_planar_face_extraction_evaluates_on_plane():
    from build123d import Box

    from egg.io import cad

    face = Box(2, 2, 2).faces().sort_by_distance((10, 0, 0))[0]  # +x plane, x=1
    s = cad.face_to_surface(face)
    # every extracted node evaluates on the x=1 plane
    for u in np.linspace(s._u0, s._u1, 4):
        for v in np.linspace(s._v0, s._v1, 4):
            assert abs(s.eval(u, v)[0] - 1.0) < 1e-9


def test_periodic_surface_projects_radially():
    from build123d import Sphere

    from egg.io import cad

    s = cad.face_to_surface(Sphere(1.0).faces()[0], name="sph")
    p = s.project(np.array([3.0, 0.5, -0.2]))
    assert abs(np.linalg.norm(p) - 1.0) < 1e-6


def test_face_with_hole_has_two_trim_loops():
    from build123d import Box, Cylinder

    from egg.io import cad

    plate = Box(4, 4, 1) - Cylinder(radius=0.8, height=2)
    top = plate.faces().sort_by_distance((0, 0, 10))[0]
    s = cad.face_to_surface(top)
    assert len(s.trim) == 2  # outer rectangle + the hole


def test_straight_edge_to_line3():
    from build123d import Box

    from egg.io import cad
    from egg.geometry.analytic3d import Line3

    ln = cad.edge_to_curve(Box(2, 2, 2).edges()[0], name="e")
    assert isinstance(ln, Line3) and ln.name == "e"


def test_curved_edge_raises():
    from build123d import Cylinder

    from egg.io import cad

    circle = [
        e for e in Cylinder(radius=1, height=2).edges() if e.geom_type.name != "LINE"
    ][0]
    with pytest.raises(NotImplementedError):
        cad.edge_to_curve(circle)


def test_attach_extracted_surface_to_block():
    from build123d import Box

    from egg.io import cad
    from egg.topology.builder import TopologyBuilder

    patch = cad.face_to_surface(Box(2, 2, 2).faces().sort_by_distance((10, 0, 0))[0])
    tb = TopologyBuilder(d=3)
    pts = {
        "c000": (1.2, -0.5, -0.5),
        "c001": (1.2, -0.5, 0.5),
        "c010": (1.2, 0.5, -0.5),
        "c011": (1.2, 0.5, 0.5),
        "c100": (2.0, -0.5, -0.5),
        "c101": (2.0, -0.5, 0.5),
        "c110": (2.0, 0.5, -0.5),
        "c111": (2.0, 0.5, 0.5),
    }
    for nm, pp in pts.items():
        tb.add_corner(nm, pp)
    tb.add_block("B", corners=tuple(pts), resolutions=(3, 3, 3))
    tb.associate("B", 0, 0, patch)
    grid = tb.build().initialize_grid()

    assert not np.any(np.isnan(grid.global_nodes))
    face = grid.blocks[0].nodes[0, :, :].reshape(-1, 3)  # x-min face
    np.testing.assert_allclose(face[:, 0], 1.0, atol=1e-9)
