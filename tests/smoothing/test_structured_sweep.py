"""Parity of cpp_structured_sweep against cpp_sweep (Phase 1.3, steps 1-2).

Step 1 (single block): no shared interfaces, so the per-sweep halo exchange is a
no-op and the structured colored-GS sweep over the halo-padded store reproduces
the unstructured ``cpp_sweep`` bit-for-bit (same X0, same sweep count). This
locks the Python->C++ structured-context re-homing (global indices -> structured
padded node indices, X -> packed buffer, gather back).

Step 2 (conforming multiblock): a shared interface node is relaxed only by its
owner block; cross-block patch neighbours read the owner's frozen ghost copy and
the non-owner copies are refreshed each sweep (frozen-halo additive Schwarz,
cadence 1.4b). This is NOT bit-for-bit per sweep, but converges to the *same*
minimiser as the global ``cpp_sweep``, gated on the conforming L/R pair.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from egg.smoothing.solver import build_sweep_context
from egg.smoothing.targets import IdentityTarget
from egg.topology.builder import TopologyBuilder
from tests.real_tol import real_tol

# The circle-in-rectangle O-grid builder lives with the examples.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "circles"))


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


def _single_block_grid(size=(6, 6)):
    """One 4x4 block with all four corners fixed (no shared interfaces)."""
    b = TopologyBuilder(d=2)
    for name, pos in [
        ("A", (0.0, 0.0)), ("B", (4.0, 0.0)),
        ("C", (4.0, 4.0)), ("D", (0.0, 4.0)),
    ]:
        b.add_corner(name, pos, fixed=True)
    b.add_block("main", ("A", "D", "B", "C"), size)
    topo = b.build()
    return topo.initialize_grid()


def _perturb(grid, seed=42, scale=0.1):
    """Perturb free, unconstrained interior DOFs to create a non-optimal state."""
    rng = np.random.default_rng(seed)
    free = np.array(grid.free_mask)
    constrained = set(grid.dof_constraints.keys())
    gn = np.array(grid.global_nodes)
    for i in range(grid.global_node_count):
        if free[i] and i not in constrained:
            gn[i] += rng.normal(0, scale, 2)
    grid.global_nodes = gn
    for bi, blk in enumerate(grid.blocks):
        blk.nodes[...] = gn[grid.block_dof_maps[bi]]
    return gn.copy()


def test_single_block_parity_with_cpp_sweep():
    from egg.smoothing.cpp_backend import cpp_structured_sweep, cpp_sweep

    grid = _single_block_grid()
    X0 = _perturb(grid)
    ctx = build_sweep_context(grid, IdentityTarget(2))

    n_sweeps = 20
    # report_every=1 preserves per-sweep arrays for the full-array comparison
    # below (the binding default of 0 would collapse to a single value).
    X_u, e_u, m_u = cpp_sweep(ctx, X0, n_sweeps, device="cpu", report_every=1)
    X_s, e_s, m_s = cpp_structured_sweep(
        ctx, grid, X0, n_sweeps, device="cpu", report_every=1
    )

    # Single block => empty halo => bit-for-bit identical to the unstructured path.
    np.testing.assert_allclose(e_s, e_u, rtol=0, atol=1e-12)
    np.testing.assert_allclose(m_s, m_u, rtol=0, atol=1e-12)
    np.testing.assert_allclose(X_s, X_u, rtol=0, atol=1e-12)


def _lr_grid(res=(4, 4)):
    """Two unit blocks L (x:0..2) and R (x:2..4) sharing the x=2 edge.

    The shared face's endpoints (corners B, C) are fixed; its interior nodes are
    free and shared between the blocks — the conforming-multiblock case step 2
    must own/ghost-remap correctly.
    """
    b = TopologyBuilder(d=2)
    for name, pos in [
        ("A", (0.0, 0.0)), ("D", (0.0, 2.0)),
        ("B", (2.0, 0.0)), ("C", (2.0, 2.0)),
        ("E", (4.0, 0.0)), ("F", (4.0, 2.0)),
    ]:
        b.add_corner(name, pos, fixed=True)
    b.add_block("L", ("A", "D", "B", "C"), res)
    b.add_block("R", ("B", "C", "E", "F"), res)
    b.connect("L", 0, 1, "R", 0, 0)
    return b.build().initialize_grid()


def test_conforming_lr_pair_converges_to_same_minimiser():
    """Frozen-halo additive Schwarz over the L/R pair reaches the same minimiser
    (final energy / min-det / positions) as the global colored-GS cpp_sweep."""
    from egg.smoothing.cpp_backend import cpp_structured_sweep, cpp_sweep

    grid = _lr_grid()
    X0 = _perturb(grid)
    ctx = build_sweep_context(grid, IdentityTarget(2))

    # Run both to convergence. The structured path is additive Schwarz across the
    # interface, so it needs more sweeps than the global GS, but both land on the
    # same unique minimiser (fixed boundary, shape objective, identity target).
    n_sweeps = 400
    # report_every=1 preserves the per-sweep array (for the np.diff monotonicity
    # check below); the binding default of 0 would collapse to one value.
    _X_u, e_u, m_u = cpp_sweep(ctx, X0, n_sweeps, device="cpu", report_every=1)
    X_s, e_s, m_s = cpp_structured_sweep(
        ctx, grid, X0, n_sweeps, device="cpu", report_every=1
    )

    # Structured energy decreases monotonically (frozen halos can't increase it
    # beyond FP noise) and converges to the global minimiser's energy / min-det.
    assert np.all(np.diff(e_s) <= real_tol(1e-9))
    np.testing.assert_allclose(e_s[-1], e_u[-1], rtol=real_tol(0.0), atol=real_tol(1e-9))
    np.testing.assert_allclose(m_s[-1], m_u[-1], rtol=real_tol(0.0), atol=real_tol(1e-9))

    # And to the same node positions as a fully-converged global sweep.
    _X_ref, _e_ref, _m_ref = cpp_sweep(ctx, X0, 2 * n_sweeps, device="cpu")
    np.testing.assert_allclose(X_s, _X_ref, rtol=real_tol(0.0), atol=real_tol(1e-7))


def test_ogrid_singular_fan_converges_to_same_minimiser():
    """The 12-block circle-in-rectangle O-grid has 4 valence-5 singular nodes whose
    fans cross non-axis-aligned interfaces. The structured path mirrors those fan
    neighbours into spare ghost slots and must still reach the global minimiser."""
    from topologies import build_circle_in_rectangle  # noqa: E402

    from egg.smoothing.cpp_backend import cpp_structured_sweep, cpp_sweep

    topo, _ents = build_circle_in_rectangle(rough=False, R=1)
    grid = topo.initialize_grid()
    ctx = build_sweep_context(grid, IdentityTarget(d=topo.d))
    X0 = np.array(grid.global_nodes)

    n_sweeps = 300
    # report_every=1 preserves the per-sweep array (for the np.diff monotonicity
    # check below); the binding default of 0 would collapse to one value.
    _X_u, e_u, _m_u = cpp_sweep(ctx, X0, n_sweeps, device="cpu", report_every=1)
    X_s, e_s, _m_s = cpp_structured_sweep(
        ctx, grid, X0, n_sweeps, device="cpu", report_every=1
    )

    assert np.all(np.diff(e_s) <= real_tol(1e-9))  # monotone (frozen-halo additive Schwarz)
    np.testing.assert_allclose(e_s[-1], e_u[-1], rtol=real_tol(0.0), atol=real_tol(1e-9))

    _X_ref, _e_ref, _m_ref = cpp_sweep(ctx, X0, 2 * n_sweeps, device="cpu")
    np.testing.assert_allclose(X_s, _X_ref, rtol=real_tol(0.0), atol=real_tol(1e-6))


def test_block_jacobi_single_block_converges_to_same_minimiser():
    """Block-Jacobi (double-buffered, one merged launch) on a single block reaches
    the same minimiser as the global colored-GS cpp_sweep. It is NOT bit-for-bit
    per sweep (simultaneous vs sequential updates), but lands on the same unique
    minimiser and decreases energy monotonically at omega=1 on this smooth case."""
    from egg.smoothing.cpp_backend import cpp_structured_sweep, cpp_sweep

    grid = _single_block_grid()
    X0 = _perturb(grid)
    ctx = build_sweep_context(grid, IdentityTarget(2))

    # Jacobi smooths less per sweep than GS, so it needs more sweeps to reach the
    # same residual (the whole reason net wall-time, not sweep count, is the metric).
    n_sweeps = 2000
    _X_u, e_u, m_u = cpp_sweep(ctx, X0, n_sweeps, device="cpu")
    X_j, e_j, m_j = cpp_structured_sweep(
        ctx, grid, X0, n_sweeps, device="cpu", smoother="block-jacobi")

    np.testing.assert_allclose(e_j[-1], e_u[-1], rtol=real_tol(0.0), atol=real_tol(1e-9))
    np.testing.assert_allclose(m_j[-1], m_u[-1], rtol=real_tol(0.0), atol=real_tol(1e-9))

    _X_ref, _e_ref, _m_ref = cpp_sweep(ctx, X0, 2 * n_sweeps, device="cpu")
    np.testing.assert_allclose(X_j, _X_ref, rtol=real_tol(0.0), atol=real_tol(1e-7))


def test_block_jacobi_lr_pair_converges_to_same_minimiser():
    """Block-Jacobi across the conforming L/R pair (cross-block frozen halos +
    intra-sweep Jacobi) reaches the same minimiser as the global cpp_sweep."""
    from egg.smoothing.cpp_backend import cpp_structured_sweep, cpp_sweep

    grid = _lr_grid()
    X0 = _perturb(grid)
    ctx = build_sweep_context(grid, IdentityTarget(2))

    n_sweeps = 1000
    _X_u, e_u, m_u = cpp_sweep(ctx, X0, n_sweeps, device="cpu")
    X_j, e_j, m_j = cpp_structured_sweep(
        ctx, grid, X0, n_sweeps, device="cpu", smoother="block-jacobi")

    np.testing.assert_allclose(e_j[-1], e_u[-1], rtol=real_tol(0.0), atol=real_tol(1e-9))
    np.testing.assert_allclose(m_j[-1], m_u[-1], rtol=real_tol(0.0), atol=real_tol(1e-9))

    _X_ref, _e_ref, _m_ref = cpp_sweep(ctx, X0, 2 * n_sweeps, device="cpu")
    np.testing.assert_allclose(X_j, _X_ref, rtol=real_tol(0.0), atol=real_tol(1e-7))


def test_block_jacobi_ogrid_converges_to_same_minimiser():
    """Block-Jacobi on the 12-block O-grid (singular fans mirrored into ghosts)
    reaches the same minimiser as the global cpp_sweep."""
    from topologies import build_circle_in_rectangle  # noqa: E402

    from egg.smoothing.cpp_backend import cpp_structured_sweep, cpp_sweep

    topo, _ents = build_circle_in_rectangle(rough=False, R=1)
    grid = topo.initialize_grid()
    ctx = build_sweep_context(grid, IdentityTarget(d=topo.d))
    X0 = np.array(grid.global_nodes)

    n_sweeps = 1200
    _X_u, e_u, _m_u = cpp_sweep(ctx, X0, n_sweeps, device="cpu")
    X_j, e_j, _m_j = cpp_structured_sweep(
        ctx, grid, X0, n_sweeps, device="cpu", smoother="block-jacobi")

    np.testing.assert_allclose(e_j[-1], e_u[-1], rtol=real_tol(0.0), atol=real_tol(1e-9))

    _X_ref, _e_ref, _m_ref = cpp_sweep(ctx, X0, 2 * n_sweeps, device="cpu")
    np.testing.assert_allclose(X_j, _X_ref, rtol=real_tol(0.0), atol=real_tol(1e-6))


def test_single_and_disjoint_blocks_still_bit_identical():
    """A single block has an empty share table, so step 2's machinery leaves the
    bit-for-bit single-block parity (step 1) intact."""
    from egg.smoothing.cpp_backend import cpp_structured_sweep, cpp_sweep

    grid = _single_block_grid()
    X0 = _perturb(grid)
    ctx = build_sweep_context(grid, IdentityTarget(2))

    n_sweeps = 20
    # report_every=1 preserves per-sweep arrays for the full-array comparison
    # below (the binding default of 0 would collapse to a single value).
    X_u, e_u, m_u = cpp_sweep(ctx, X0, n_sweeps, device="cpu", report_every=1)
    X_s, e_s, m_s = cpp_structured_sweep(
        ctx, grid, X0, n_sweeps, device="cpu", report_every=1
    )
    np.testing.assert_allclose(e_s, e_u, rtol=0, atol=1e-12)
    np.testing.assert_allclose(m_s, m_u, rtol=0, atol=1e-12)
    np.testing.assert_allclose(X_s, X_u, rtol=0, atol=1e-12)
