# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""End-to-end: the block-interface C2 curvature term runs through the C++
structured sweep — de-kinking a clustered/sheared seam without inverting, and
staying inert (stable) on an already-straight grid."""

from __future__ import annotations

import numpy as np
import pytest

from egg.pipeline import PipelineConfig, run_pipeline
from egg.smoothing import highorder as H
from egg.smoothing.solver import build_sweep_context
from egg.smoothing.targets import IdentityTarget
from egg.smoothing.untangle import grid_min_det
from egg.topology.builder import TopologyBuilder


def _has_cpp():
    try:
        from egg._cpp import cpp_core  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cpp(), reason="egg._cpp.cpp_core not built (requires cmake build)"
)


def _pgram(res=(12, 12), shear=1.0):
    """Two sheared blocks sharing an oblique seam (curved grid lines near it)."""
    b = TopologyBuilder(d=2)
    for n, p in [
        ("A", (0, 0)),
        ("D", (shear, 2)),
        ("B", (2, 0)),
        ("C", (2 + shear, 2)),
        ("E", (4, 0)),
        ("F", (4 + shear, 2)),
    ]:
        b.add_corner(n, p, fixed=True)
    b.add_block("L", ("A", "D", "B", "C"), res)
    b.add_block("R", ("B", "C", "E", "F"), res)
    b.connect("L", 0, 1, "R", 0, 0)
    return b.build().initialize_grid()


def _lr(res=(10, 10)):
    b = TopologyBuilder(d=2)
    for n, p in [
        ("A", (0, 0)),
        ("D", (0, 2)),
        ("B", (2, 0)),
        ("C", (2, 2)),
        ("E", (4, 0)),
        ("F", (4, 2)),
    ]:
        b.add_corner(n, p, fixed=True)
    b.add_block("L", ("A", "D", "B", "C"), res)
    b.add_block("R", ("B", "C", "E", "F"), res)
    b.connect("L", 0, 1, "R", 0, 0)
    return b.build().initialize_grid()


def _mindet(grid):
    es = build_sweep_context(grid, IdentityTarget(2)).energy_stencil
    return grid_min_det(grid.global_nodes, es)


def _cfg(interface_c2):
    return PipelineConfig(
        tmop_sweeps=300,
        tmop_chunk=50,
        tmop_metric="shape",
        interface_c2=interface_c2,
        device="cpu",
    )


def test_curvature_reduces_seam_waviness_without_inverting():
    base = _pgram()
    run_pipeline(base, config=_cfg(None), verbose=False)
    d0 = H.line_diagnostics(base)

    c2 = _pgram()
    run_pipeline(c2, config=_cfg({"weight": 0.5, "iface_boost": 10.0}), verbose=False)
    d1 = H.line_diagnostics(c2)

    assert _mindet(c2) > 0.0  # barrier held
    assert d1["turn_max"] < 0.6 * d0["turn_max"]  # markedly less kinked


def test_curvature_is_stable_and_inert_on_a_straight_grid():
    # An axis-aligned grid smooths perfectly straight; the C2 term has nothing to
    # do and must not amplify roundoff (the biharmonic-Jacobi failure mode).
    ref = _lr()
    run_pipeline(ref, config=_cfg(None), verbose=False)
    X_ref = np.asarray(ref.global_nodes).copy()

    c2 = _lr()
    run_pipeline(c2, config=_cfg({"weight": 0.5, "iface_boost": 10.0}), verbose=False)
    assert np.abs(np.asarray(c2.global_nodes) - X_ref).max() < 1e-6
