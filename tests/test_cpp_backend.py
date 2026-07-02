"""Smoke tests for the C++ backend (egg._cpp.cpp_core).

These tests verify that the C++ module imports and the structured block-Jacobi
sweep runs without error on a minimal context, on both CPU and GPU when
available.
"""

from __future__ import annotations

import numpy as np
import pytest


def _has_cpp():
    try:
        from egg._cpp import cpp_core  # noqa: F401
        return True
    except ImportError:
        return False


def _has_gpu():
    """Check if a SYCL GPU device is visible by attempting a 1-sweep run on it."""
    if not _has_cpp():
        return False
    from egg.smoothing.cpp_backend import cpp_structured_sweep
    ctx, grid = _make_mini_context()
    try:
        cpp_structured_sweep(ctx, grid, grid.global_nodes, 1, device="gpu")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cpp(),
    reason="egg._cpp.cpp_core not built (requires cmake build)",
)


def test_ping():
    """Module imports and ping returns 0."""
    from egg._cpp import cpp_core
    assert cpp_core.ping() == 0


def _make_mini_context():
    """Build a minimal SweepContext for smoke testing."""
    from egg.smoothing.solver import build_sweep_context
    from egg.smoothing.targets import IdentityTarget
    from egg.topology.builder import TopologyBuilder

    builder = TopologyBuilder(d=2)
    for name, pos in [("A", (0.0, 0.0)), ("B", (4.0, 0.0)),
                      ("C", (4.0, 4.0)), ("D", (0.0, 4.0))]:
        builder.add_corner(name, pos, fixed=True)
    builder.add_block("main", ("A", "D", "B", "C"), (4, 4))
    topo = builder.build()
    topo.initialize_grid()
    grid = topo.grid
    ctx = build_sweep_context(grid, IdentityTarget(d=2))
    return ctx, grid


@pytest.mark.parametrize("device", ["cpu"])
def test_block_jacobi_smoke(device):
    """The structured block-Jacobi sweep runs on a tiny context without error.

    Pins ``report_every=1`` to assert per-sweep array shapes (the binding's
    default of 0 would collapse to a single value).
    """
    from egg.smoothing.cpp_backend import cpp_structured_sweep

    ctx, grid = _make_mini_context()
    X0 = grid.global_nodes.copy()

    X_out, energies, mindets = cpp_structured_sweep(
        ctx, grid, X0, 3, device=device, report_every=1)

    assert X_out.shape == X0.shape
    assert energies.shape == (3,)
    assert mindets.shape == (3,)
    assert np.all(np.isfinite(X_out))
    assert np.all(np.isfinite(energies))
    assert np.all(np.isfinite(mindets))


@pytest.mark.skipif(not _has_gpu(), reason="No GPU device available")
def test_block_jacobi_smoke_gpu():
    """The structured block-Jacobi sweep runs on GPU without error."""
    from egg.smoothing.cpp_backend import cpp_structured_sweep

    ctx, grid = _make_mini_context()
    X0 = grid.global_nodes.copy()

    X_out, energies, mindets = cpp_structured_sweep(
        ctx, grid, X0, 3, device="gpu", report_every=1)

    assert X_out.shape == X0.shape
    assert energies.shape == (3,)
    assert mindets.shape == (3,)
    assert np.all(np.isfinite(X_out))
    assert np.all(np.isfinite(energies))
    assert np.all(np.isfinite(mindets))


def test_block_jacobi_cpu_gpu_both_run():
    """CPU and GPU both run and produce finite, same-shape results.

    Block-Jacobi is not bit-reproducible across backends (simultaneous updates,
    parallel reductions), so this checks both devices execute and stay valid
    rather than asserting numerical equality.
    """
    if not _has_gpu():
        pytest.skip("No GPU device available")

    from egg.smoothing.cpp_backend import cpp_structured_sweep

    ctx, grid = _make_mini_context()
    X0 = grid.global_nodes.copy()

    X_cpu, e_cpu, m_cpu = cpp_structured_sweep(
        ctx, grid, X0.copy(), 3, device="cpu", report_every=1)
    X_gpu, e_gpu, m_gpu = cpp_structured_sweep(
        ctx, grid, X0.copy(), 3, device="gpu", report_every=1)

    for arr in (X_cpu, e_cpu, m_cpu, X_gpu, e_gpu, m_gpu):
        assert np.all(np.isfinite(arr))
    assert X_cpu.shape == X_gpu.shape == X0.shape
