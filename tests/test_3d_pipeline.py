# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""A TopologyBuilder(d=3) grid solves through the pipeline.

Exercises the 3D solve path: the pipeline's min-det/energy monitoring routes
through the d-general C++ device reduction (the NumPy metric is 2D-only), and
the structured session smooths at dim=3.
"""

import numpy as np
import pytest

from egg.topology.builder import TopologyBuilder


def _has_cpp() -> bool:
    try:
        from egg._cpp import cpp_core  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cpp(), reason="egg._cpp.cpp_core not built (requires cmake build)"
)

CORNERS = {
    "a00": (0, 0, 0),
    "a01": (0, 0, 1),
    "a10": (0, 1, 0),
    "a11": (0, 1, 1),
    "s00": (1, 0, 0),
    "s01": (1, 0, 1),
    "s10": (1, 1, 0),
    "s11": (1, 1, 1),
    "b00": (2, 0, 0),
    "b01": (2, 0, 1),
    "b10": (2, 1, 0),
    "b11": (2, 1, 1),
}


def test_3d_two_block_pipeline_solves():
    """Two aligned hex blocks: the pipeline runs and yields a valid grid."""
    from egg.pipeline import PipelineConfig, generate_steps

    tb = TopologyBuilder(d=3)
    for name, pos in CORNERS.items():
        tb.add_corner(name, pos)
    tb.add_block(
        "A",
        corners=("a00", "a01", "a10", "a11", "s00", "s01", "s10", "s11"),
        resolutions=(3, 3, 3),
    )
    tb.add_block(
        "B",
        corners=("s00", "s01", "s10", "s11", "b00", "b01", "b10", "b11"),
        resolutions=(3, 3, 3),
    )
    grid = tb.build().initialize_grid()

    cfg = PipelineConfig(
        device="cpu", tmop_sweeps=40, tmop_chunk=20, sweeps_per_delta=20
    )
    phases = {}
    for phase, info in generate_steps(grid, config=cfg, untangle_direct=True):
        phases[phase] = info
        assert np.isfinite(info["min_det"])

    assert "final" in phases
    assert phases["final"]["min_det"] > 0.0
    assert np.isfinite(phases["final"]["energy"])
    # A straight slab is TMOP-optimal, so the shape energy relaxes to ~0
    # (floored at the fp32 parity level in the float build).
    from tests.real_tol import real_tol

    assert phases["final"]["energy"] < real_tol(1e-6)
