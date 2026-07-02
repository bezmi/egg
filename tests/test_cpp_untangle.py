"""Parity tests for the C++ δ-continuation untangle path.

Validates ``cpp_sweep(..., phase="untangle", delta=δ)`` self-consistency:
the one-shot kernel matches the session path, and the driver ``cpp_untangle``
recovers ``min det A > margin`` on a folded grid.

The exhaustive sweep-math coverage lives in the C++ golden test; this file
checks the binding round-trip is lossless.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "circles"))


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

_RTOL_X = 1e-8
_ATOL = 1e-8


def _folded_context():
    """A folded circle-in-rectangle O-grid (rough=True) → min det A <= 0 at start."""
    from topologies import build_circle_in_rectangle  # noqa: E402

    from egg.smoothing.solver import build_sweep_context
    from egg.smoothing.targets import IdentityTarget

    topo, _ents = build_circle_in_rectangle(rough=True, R=3)
    grid = topo.initialize_grid()
    ctx = build_sweep_context(grid, IdentityTarget(d=topo.d))
    X0 = grid.global_nodes.copy()
    return ctx, X0


@pytest.mark.parametrize("delta", [1e-1, 1e-2, 1e-3])
@pytest.mark.parametrize("n_sweeps", [1, 3])
def test_cpp_untangle_self_consistent(n_sweeps, delta):
    """One-shot cpp_sweep(phase="untangle") is identical to session-mode.

    Pins ``report_every=1`` so the per-sweep min-det arrays are directly
    comparable (the binding's default of 0 would collapse one side to a
    single value).
    """
    from egg.smoothing.cpp_backend import CppSweepSession, cpp_sweep

    ctx, X0 = _folded_context()
    X_os, _, m_os = cpp_sweep(ctx, X0.copy(), n_sweeps, device="cpu",
                               phase="untangle", delta=delta, report_every=1)

    sess = CppSweepSession(ctx, X0.copy(), device="cpu")
    m_parts = []
    for _ in range(n_sweeps):
        _, m = sess.run(1, phase="untangle", delta=delta, report_every=1)
        m_parts.append(m)
    m_sess = np.concatenate(m_parts)

    np.testing.assert_array_equal(m_os, m_sess)
    np.testing.assert_array_equal(X_os, sess.get_X())


def test_cpp_untangle_driver_self_consistent():
    """The cpp_untangle continuation driver matches a manual reimplementation."""
    from egg.smoothing.cpp_backend import _grid_mindet, cpp_untangle, CppSweepSession

    ctx, X0 = _folded_context()
    es = ctx.energy_stencil
    sweeps_per_delta, max_outer, margin = 20, 8, 1e-9

    X_out, md_cpp, outer_iters, delta_final = cpp_untangle(
        ctx, X0.copy(), device="cpu",
        sweeps_per_delta=sweeps_per_delta,
        max_outer=max_outer, margin=margin)

    md = _grid_mindet(X0, es)
    delta = 2.0 * max(abs(md), 1e-12)
    md_man = md
    iters_man = 0
    sess = CppSweepSession(ctx, X0.copy(), device="cpu")
    for _ in range(max_outer):
        _, mds = sess.run(sweeps_per_delta, phase="untangle", delta=delta)
        iters_man += 1
        md_man = float(mds[-1])
        if md_man > margin:
            break
        delta *= 0.5

    np.testing.assert_allclose(md_cpp, md_man, rtol=1e-8, atol=1e-10,
                               err_msg="driver final min det A mismatch vs manual")
    assert outer_iters == iters_man, "driver outer-iter count mismatch vs manual"
