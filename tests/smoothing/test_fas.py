# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Integration tests for the FAS (nonlinear geometric multigrid) session API.

Drives ``CppStructuredSweepSession.run_fas`` end-to-end through the Python
context builders on smoothly-perturbed structured grids: hierarchy shape,
per-cycle energy monotonicity and positive min-det (the safeguarded
correction's guarantees), acceleration over plain block-Jacobi at an equal
fine-sweep budget, agreement with the Jacobi minimiser, and the untangle
rejection. Tolerances go through ``real_tol`` (identity in the fp64
correctness gate, floored in the fp32 build); energies are never compared
exactly (console values fluctuate in the 4th-5th digit between runs).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("egg._cpp.cpp_core", reason="egg._cpp.cpp_core not built")

from egg.smoothing.cpp_backend import (
    CppStructuredSweepSession,
    build_block_structured_context,
)
from egg.smoothing.solver import build_sweep_context
from egg.smoothing.targets import IdentityTarget
from egg.topology.builder import TopologyBuilder
from tests.real_tol import real_tol


def _single_block_grid(res=(16, 16), size=4.0):
    b = TopologyBuilder(d=2)
    for name, pos in [
        ("A", (0.0, 0.0)),
        ("B", (size, 0.0)),
        ("C", (size, size)),
        ("D", (0.0, size)),
    ]:
        b.add_corner(name, pos, fixed=True)
    b.add_block("main", ("A", "D", "B", "C"), res)
    topo = b.build()
    topo.initialize_grid()
    return topo.grid


def _perturbed(grid, amp):
    """Smooth low-frequency displacement vanishing on the boundary — the error
    component plain Jacobi damps slowest."""
    X = grid.global_nodes.copy()
    lo = X.min(axis=0)
    span = X.max(axis=0) - lo
    s = np.sin(np.pi * (X - lo) / span)
    bump = s[:, 0] * s[:, 1]
    X[:, 0] += amp * bump
    X[:, 1] += 0.7 * amp * bump
    return X


def _session(grid, X):
    ctx = build_sweep_context(grid, IdentityTarget(d=2))
    bsc = build_block_structured_context(grid)
    return CppStructuredSweepSession(ctx, bsc, X, device="cpu")


def test_mg_levels_shape():
    """17x17-node block ladders 9x9 -> 5x5 -> 3x3 (factor 2 both axes each
    level; 3 has no coarsenable axis, so the hierarchy stops there)."""
    grid = _single_block_grid(res=(16, 16))
    sess = _session(grid, grid.global_nodes.copy())
    levels = sess.mg_levels()
    assert levels == [[(9, 9)], [(5, 5)], [(3, 3)]]


def test_fas_monotone_and_barrier():
    """Per-cycle energies non-increasing, min det positive every cycle."""
    grid = _single_block_grid()
    sess = _session(grid, _perturbed(grid, amp=0.08))
    energies, mindets = sess.run_fas(6, omega=0.8)

    assert energies.shape == (6,)
    assert np.all(np.isfinite(energies))
    assert np.all(mindets > 0.0)
    slack = real_tol(1e-10) * np.maximum(1.0, np.abs(energies[:-1]))
    assert np.all(energies[1:] <= energies[:-1] + slack)


def test_fas_beats_jacobi_at_equal_fine_budget():
    """FAS with nu_pre + nu_post fine sweeps per cycle must land well below
    plain Jacobi given the same number of fine sweeps — the coarse grid's
    entire purpose on smooth error."""
    grid = _single_block_grid()
    X0 = _perturbed(grid, amp=0.08)
    n_cycles, nu = 5, 2

    fas = _session(grid, X0.copy())
    e_fas, _ = fas.run_fas(n_cycles, nu_pre=nu, nu_post=nu, omega=0.8)

    jac = _session(grid, X0.copy())
    e_jac, _ = jac.run(n_cycles * 2 * nu, omega=0.8)

    floor = real_tol(1e-9)
    assert e_fas[-1] <= 0.5 * e_jac[-1] + floor, (e_fas[-1], e_jac[-1])


def test_fas_agrees_with_jacobi_minimiser():
    """Long FAS and long Jacobi converge to the same energy (same minimiser,
    different path)."""
    grid = _single_block_grid()
    X0 = _perturbed(grid, amp=0.08)

    fas = _session(grid, X0.copy())
    e_fas, _ = fas.run_fas(12, omega=0.8)

    jac = _session(grid, X0.copy())
    e_jac, _ = jac.run(600, omega=0.8)

    tol = real_tol(1e-8) * max(1.0, abs(e_jac[-1]))
    assert e_fas[-1] <= e_jac[-1] + tol, (e_fas[-1], e_jac[-1])


def _two_block_grid(res=16):
    b = TopologyBuilder(d=2)
    for name, pos in [
        ("A", (0.0, 0.0)),
        ("D", (0.0, 4.0)),
        ("B", (4.0, 0.0)),
        ("C", (4.0, 4.0)),
        ("E", (8.0, 0.0)),
        ("F", (8.0, 4.0)),
    ]:
        b.add_corner(name, pos, fixed=True)
    b.add_block("L", ("A", "D", "B", "C"), (res, res))
    b.add_block("R", ("B", "C", "E", "F"), (res, res))
    b.connect("L", 0, 1, "R", 0, 0)
    topo = b.build()
    topo.initialize_grid()
    return topo.grid


def test_fas_two_blocks():
    """Two connected blocks coarsen and converge (conforming interface nodes
    relax on the coarse levels too)."""
    grid = _two_block_grid()
    sess = _session(grid, _perturbed(grid, amp=0.15))
    assert sess.mg_levels() == [
        [(9, 9), (9, 9)],
        [(5, 5), (5, 5)],
        [(3, 3), (3, 3)],
    ]

    energies, mindets = sess.run_fas(6, omega=0.8)
    assert np.all(np.isfinite(energies))
    assert np.all(mindets > 0.0)
    slack = real_tol(1e-10) * np.maximum(1.0, np.abs(energies[:-1]))
    assert np.all(energies[1:] <= energies[:-1] + slack)
    assert np.all(np.isfinite(sess.get_X()))


def test_fas_two_blocks_beats_jacobi():
    """The coarse-interface acceptance check: with interface DOFs relaxing on
    the coarse levels, two-block FAS must reach (and pass) a long Jacobi run's
    energy — the smooth cross-block error component frozen interfaces stall on."""
    grid = _two_block_grid(res=32)
    X0 = _perturbed(grid, amp=0.08)

    fas = _session(grid, X0.copy())
    e_fas, m_fas = fas.run_fas(12, omega=0.8)

    jac = _session(grid, X0.copy())
    e_jac, _ = jac.run(1200, omega=0.8)

    assert np.all(m_fas > 0.0)
    tol = real_tol(1e-8) * max(1.0, abs(e_jac[-1]))
    assert e_fas[-1] <= e_jac[-1] + tol, (e_fas[-1], e_jac[-1])


def test_fas_max_levels():
    """max_levels clips the cycle depth per call: 2 forces the two-level
    cycle, 1 disables the hierarchy entirely (plain-Jacobi fallback at the
    per-cycle report cadence). Both stay monotone with a positive min det."""
    grid = _single_block_grid()
    X0 = _perturbed(grid, amp=0.08)

    two = _session(grid, X0.copy())
    e2, m2 = two.run_fas(4, omega=0.8, max_levels=2)
    assert e2.shape == (4,)
    assert np.all(np.isfinite(e2))
    assert np.all(m2 > 0.0)
    slack = real_tol(1e-10) * np.maximum(1.0, np.abs(e2[:-1]))
    assert np.all(e2[1:] <= e2[:-1] + slack)

    one = _session(grid, X0.copy())
    e1, m1 = one.run_fas(4, omega=0.8, max_levels=1)
    assert e1.shape == (4,)
    assert np.all(np.isfinite(e1))
    assert np.all(m1 > 0.0)


def test_fas_rejects_untangle():
    grid = _single_block_grid(res=(8, 8))
    sess = _session(grid, grid.global_nodes.copy())
    with pytest.raises(ValueError, match="untangle"):
        sess.run_fas(1, phase="untangle")


def test_fas_carries_uncoarsenable_block():
    """A block that cannot coarsen (even node counts) rides along unchanged
    while the other block ladders down — it no longer vetoes the hierarchy."""
    b = TopologyBuilder(d=2)
    for name, pos in [
        ("A", (0.0, 0.0)),
        ("B", (4.0, 0.0)),
        ("C", (4.0, 4.0)),
        ("D", (0.0, 4.0)),
        ("E", (6.0, 0.0)),
        ("F", (6.0, 2.0)),
        ("G", (8.0, 0.0)),
        ("H", (8.0, 2.0)),
    ]:
        b.add_corner(name, pos, fixed=True)
    b.add_block("big", ("A", "D", "B", "C"), (16, 16))  # 17x17 nodes — ladders
    b.add_block("small", ("E", "F", "G", "H"), (5, 5))  # 6x6 nodes — carried
    topo = b.build()
    topo.initialize_grid()
    grid = topo.grid

    sess = _session(grid, _perturbed(grid, amp=0.08))
    assert sess.mg_levels() == [
        [(9, 9), (6, 6)],
        [(5, 5), (6, 6)],
        [(3, 3), (6, 6)],
    ]
    energies, mindets = sess.run_fas(6, omega=0.8)
    assert np.all(np.isfinite(energies))
    assert np.all(mindets > 0.0)
    slack = real_tol(1e-10) * np.maximum(1.0, np.abs(energies[:-1]))
    assert np.all(energies[1:] <= energies[:-1] + slack)


def test_fas_falls_back_without_coarse_level():
    """Even node counts (res (5,5) -> 6x6 nodes) admit no coarse level; run_fas
    falls back to plain Jacobi with the per-cycle report cadence."""
    grid = _single_block_grid(res=(5, 5))
    sess = _session(grid, _perturbed(grid, amp=0.05))
    assert sess.mg_levels() == []
    energies, mindets = sess.run_fas(3, omega=0.8)
    assert energies.shape == (3,)
    assert np.all(np.isfinite(energies))
    assert np.all(mindets > 0.0)
