# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Sphere-in-cube O-shell with build123d-imported geometry builds and solves.

Skips without build123d (the ``cad`` group) or the C++ core. The topology is the
6-block cubed-sphere O-shell authored through ``TopologyBuilder(d=3)`` and
smoothed by the ``generate_steps`` pipeline; its surfaces are extracted from a
build123d Sphere and Box, so this exercises the whole CAD -> egg -> solve path.
"""

import importlib.util
import pathlib

import numpy as np
import pytest


def _has(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not (_has("build123d") and _has("egg._cpp.cpp_core")),
    reason="needs the cad group (build123d) and the built C++ core",
)

_EX = (
    pathlib.Path(__file__).resolve().parent.parent
    / "examples/3D/sphere_in_cube_cad/sphere_in_cube_cad.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("sphere_in_cube_cad_ex", _EX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cad_sphere_in_cube_builds_and_solves():
    from egg.pipeline import PipelineConfig, generate_steps

    mod = _load()
    r0 = 0.5
    # node counts (n odd -> even tangential cells so the fans untangle)
    topo = mod.sphere_in_cube_cad(5, 4, 3, r0=r0).build()
    assert len(topo.block_specs) == 32  # 6 O-shell + 26 H-grid
    assert len(topo.singularities) > 0  # octant fans

    grid = topo.initialize_grid()
    assert not np.any(np.isnan(grid.global_nodes))
    # every DOF bound to the imported sphere projects onto it at radius r0
    on_sphere = mod._sphere_mask(grid)
    assert on_sphere.sum() > 0
    r = np.linalg.norm(grid.global_nodes[on_sphere], axis=1)
    np.testing.assert_allclose(r, r0, atol=1e-6)

    cfg = PipelineConfig(device="cpu", tmop_sweeps=120, tmop_chunk=40)
    final = None
    for _phase, info in generate_steps(grid, config=cfg, untangle_direct=True):
        final = info
    assert final["min_det"] > 0.0
    assert np.isfinite(final["energy"])


def test_cad_sphere_in_cube_su2_export(tmp_path):
    from egg.io.su2 import export_su2

    mod = _load()
    grid = mod.sphere_in_cube_cad(5, 4, 3, r0=0.5).build().initialize_grid()
    out = tmp_path / "cad.su2"
    export_su2(grid, out)
    text = out.read_text()
    assert "NDIME= 3" in text
    # the imported entity tags become SU2 markers (inner sphere, outer walls)
    assert "MARKER_TAG= sphere" in text
    assert "MARKER_TAG= wall" in text
