# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Sphere-in-cube O-shell built with TopologyBuilder(d=3) solves.

Integration test for the whole 3D path: a 6-block cubed-sphere with singular
octant fans, Sphere + Plane face associations, surface-projected init, and a
pipeline solve monitored through the C++ device reduction.
"""

import importlib.util
import pathlib

import numpy as np
import pytest


def _has_cpp() -> bool:
    try:
        from egg._cpp import cpp_core  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cpp(), reason="egg._cpp.cpp_core not built (requires cmake build)"
)

_EX = (
    pathlib.Path(__file__).resolve().parent.parent
    / "examples/3D/cubed_sphere/cubed_sphere.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("cubed_sphere_ex", _EX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cubed_sphere_builds_and_solves():
    from egg.pipeline import PipelineConfig, generate_steps

    mod = _load()
    r0 = 0.5
    topo = mod.cubed_sphere(3, 4, r0=r0).build()
    assert len(topo.block_specs) == 6
    assert len(topo.singularities) > 0  # the octant fans

    grid = topo.initialize_grid()
    assert not np.any(np.isnan(grid.global_nodes))

    cfg = PipelineConfig(device="cpu", tmop_sweeps=80, tmop_chunk=40)
    phases = {}
    for phase, info in generate_steps(grid, config=cfg, untangle_direct=True):
        phases[phase] = info
        assert np.isfinite(info["min_det"])
    assert phases["final"]["min_det"] > 0.0
    assert np.isfinite(phases["final"]["energy"])

    # Every block's inner (radial side-0) face stays on the sphere.
    for block in grid.blocks:
        inner = block.nodes[0].reshape(-1, 3)
        np.testing.assert_allclose(np.linalg.norm(inner, axis=1), r0, atol=1e-6)
