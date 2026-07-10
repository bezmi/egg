# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Trimmed BSplineSurface: the trim reaches the device through the SoA wire.

A block face is bound to a flat patch trimmed to the inner square [0.3,0.7]^2.
Without the trim the face would project onto the full [0,1]^2 patch (out to the
block's [0.1,0.9] footprint); with it, both the host init and the device solve
clamp every face node into the trim square.
"""

import numpy as np
import pytest

from egg.geometry.surfaces3d import BSplineSurface
from egg.topology.builder import TopologyBuilder


def _has_cpp() -> bool:
    try:
        from egg._cpp import cpp_core  # noqa: F401

        return True
    except ImportError:
        return False


CTRL = np.array([[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]], float)
KN = [0, 0, 1, 1]
TRIM = [[[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]]


def _grid():
    patch = BSplineSurface(1, 1, KN, KN, CTRL, trim=TRIM)
    tb = TopologyBuilder(d=3)
    # bottom (z=0) slides on the patch; top (z=0.3) is a fixed [0.1,0.9] cap.
    pts = {
        "c000": (0.1, 0.1, 0.0, 0),
        "c001": (0.1, 0.1, 0.3, 1),
        "c010": (0.1, 0.9, 0.0, 0),
        "c011": (0.1, 0.9, 0.3, 1),
        "c100": (0.9, 0.1, 0.0, 0),
        "c101": (0.9, 0.1, 0.3, 1),
        "c110": (0.9, 0.9, 0.0, 0),
        "c111": (0.9, 0.9, 0.3, 1),
    }
    for nm, (x, y, z, fx) in pts.items():
        tb.add_corner(nm, (x, y, z), fixed=bool(fx))
    tb.add_block(
        "B",
        corners=("c000", "c001", "c010", "c011", "c100", "c101", "c110", "c111"),
        resolutions=(4, 4, 4),
    )
    tb.associate("B", 2, 0, patch)
    return tb.build().initialize_grid()


def test_trim_clamps_face_on_host_init():
    grid = _grid()
    bottom = grid.blocks[0].nodes[:, :, 0].reshape(-1, 3)
    assert bottom[:, :2].min() >= 0.3 - 1e-9
    assert bottom[:, :2].max() <= 0.7 + 1e-9
    np.testing.assert_allclose(bottom[:, 2], 0.0, atol=1e-9)


@pytest.mark.skipif(not _has_cpp(), reason="egg._cpp.cpp_core not built")
def test_trim_reaches_device_solve():
    from egg.pipeline import PipelineConfig, generate_steps

    grid = _grid()
    cfg = PipelineConfig(device="cpu", tmop_sweeps=150, tmop_chunk=75)
    for _phase, _info in generate_steps(grid, config=cfg, untangle_direct=True):
        pass

    # The face hugs the trim boundary (a near-degenerate cap by construction), so
    # min-det is not the point; the trim invariant is: the device keeps every
    # face node inside the trim square and on the surface.
    bottom = grid.blocks[0].nodes[:, :, 0].reshape(-1, 3)
    # Still on the trimmed surface: within the square (untrimmed would reach the
    # block's 0.9 footprint) and on z=0.
    assert bottom[:, :2].max() <= 0.7 + 1e-4
    assert bottom[:, :2].min() >= 0.3 - 1e-4
    np.testing.assert_allclose(bottom[:, 2], 0.0, atol=1e-5)
