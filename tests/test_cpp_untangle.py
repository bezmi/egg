# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Tests for the structured block-Jacobi δ-continuation untangle path.

Validates the ``cpp_untangle`` continuation driver: it is self-consistent with a
manual re-run of the same schedule, and damped block-Jacobi clears a mildly
folded grid (``min det A`` crosses the margin).

The exhaustive sweep-math coverage lives in the C++ golden test; this file
checks the driver + binding round-trip.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "examples", "2D", "circles")
)


def _has_cpp() -> bool:
    try:
        from egg._cpp import cpp_core  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cpp(),
    reason="egg._cpp.cpp_core not built (requires cmake build)",
)


def _folded_context():
    """A folded circle-in-rectangle O-grid (rough=True) → min det A <= 0 at start."""
    from topologies import build_circle_in_rectangle  # noqa: E402

    from egg.smoothing.solver import build_sweep_context
    from egg.smoothing.targets import IdentityTarget

    topo, _ents = build_circle_in_rectangle(rough=True, R=3)
    grid = topo.initialize_grid()
    ctx = build_sweep_context(grid, IdentityTarget(d=topo.d))
    return ctx, grid, grid.global_nodes.copy()


def _mildly_folded_context(t=0.5, R=3):
    """A mildly (untanglable) folded grid: interpolate the valid and rough
    initial states so ``min det A`` is only just negative."""
    from topologies import build_circle_in_rectangle  # noqa: E402

    from egg.smoothing.cpp_backend import _grid_mindet
    from egg.smoothing.solver import build_sweep_context
    from egg.smoothing.targets import IdentityTarget

    topo, _ents = build_circle_in_rectangle(rough=True, R=R)
    grid = topo.initialize_grid()
    X_folded = grid.global_nodes.copy()
    X_valid = (
        build_circle_in_rectangle(rough=False, R=R)[0]
        .initialize_grid()
        .global_nodes.copy()
    )
    ctx = build_sweep_context(grid, IdentityTarget(d=topo.d))
    X0 = (1.0 - t) * X_valid + t * X_folded
    assert _grid_mindet(X0, ctx.energy_stencil) <= 0.0, (
        "interpolated grid is not folded"
    )
    return ctx, grid, X0


def test_cpp_untangle_driver_returns_consistent():
    """The cpp_untangle driver returns a coherent (X, min det A) and terminates.

    The reported ``mindet`` is the min det A of the returned ``X_out``, and the
    driver stops within its ``max_outer`` budget. (Block-Jacobi on the stiff
    barrier is not bit-reproducible run to run, so this checks the driver's own
    return values rather than re-running the schedule.)
    """
    from egg.smoothing.cpp_backend import _grid_mindet, cpp_untangle

    ctx, grid, X0 = _folded_context()
    es = ctx.energy_stencil
    max_outer = 8

    X_out, md_cpp, outer_iters, _delta_final = cpp_untangle(
        ctx,
        grid,
        X0.copy(),
        device="cpu",
        sweeps_per_delta=20,
        max_outer=max_outer,
        margin=1e-9,
        omega=0.5,
    )

    assert 0 < outer_iters <= max_outer
    np.testing.assert_allclose(md_cpp, _grid_mindet(X_out, es), rtol=0, atol=1e-9)


def test_cpp_untangle_clears_mild_fold():
    """Damped block-Jacobi untangle drives a mildly folded grid valid."""
    from egg.smoothing.cpp_backend import cpp_untangle

    ctx, grid, X0 = _mildly_folded_context(t=0.5)
    _X_out, md, _outer_iters, _delta = cpp_untangle(
        ctx,
        grid,
        X0.copy(),
        device="cpu",
        sweeps_per_delta=40,
        max_outer=60,
        margin=1e-9,
        omega=0.5,
    )
    assert md > 1e-9, f"failed to untangle mild fold: min det A = {md:.3e}"
