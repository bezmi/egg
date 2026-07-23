# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Regrid workflow in 3D: save a net, load it onto a new resolution.

The Resample/re-tabulate path is dimension-general. On the 6-block cubed sphere
(Sphere + Plane walls, octant fans) a cold control solve saves the net, and a
finer grid loads it, staying valid and watertight.
"""

import importlib.util
import pathlib

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


def _cubed_sphere():
    spec = importlib.util.spec_from_file_location("cubed_sphere_ex", _EX)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cubed_sphere


def _cold(grid, net_path):
    from egg.pipeline import (
        ControlPointSmoother,
        JacobiSmoother,
        Presmooth,
        Refit,
        Save,
        Untangle,
        generate_steps,
    )

    stages = [
        Untangle(),
        Presmooth(JacobiSmoother(sweeps=100)),
        ControlPointSmoother(ratio=2, max_outer=15),
        Refit(),
        Save(net_path),
    ]
    seen, final = [], {}
    for phase, info in generate_steps(grid, stages=stages):
        seen.append(phase)
        final = info
    return seen, final


def _warm(grid, net_path):
    from egg.pipeline import (
        ControlPointSmoother,
        Resample,
        generate_steps,
    )

    stages = [Resample(net_path), ControlPointSmoother(ratio=2, max_outer=15)]
    seen, final = [], {}
    for phase, info in generate_steps(grid, stages=stages):
        seen.append(phase)
        final = info
    return seen, final


def test_3d_regrid_stays_valid_and_watertight(tmp_path):
    from egg.smoothing.control_topology import watertight_mismatch

    build = _cubed_sphere()
    net = str(tmp_path / "net.npz")

    # Cold solve at a coarse resolution saves the net.
    coarse = build(3, 4, r0=0.5).build().initialize_grid()
    seen1, final1 = _cold(coarse, net)
    assert "resample" not in seen1
    assert final1["min_det"] > 0.0

    # A finer grid (same topology) loads the saved net (re-tabulated).
    fine = build(5, 6, r0=0.5).build().initialize_grid()
    seen2, final2 = _warm(fine, net)
    assert "resample" in seen2
    assert final2["min_det"] > 0.0
    assert fine.control_net is not None
    assert watertight_mismatch(fine.control_net) < 1e-9
