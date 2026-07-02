"""Block-Jacobi convergence over the halo-padded structured store.

The structured sweep re-homes the global context onto per-block halo-padded
arrays (coalesced stencil reads) with a per-sweep halo exchange + shared-node
broadcast. Under the frozen-halo cadence (additive Schwarz) every free DOF reads
the previous sweep's snapshot, so block-Jacobi converges to a stable minimiser:
these gates check it reaches an energy plateau, decreases energy from a perturbed
start, and keeps the mesh valid — on a single block, the conforming L/R pair, and
the 12-block O-grid (singular fans mirrored into ghosts).
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "2D", "circles"))


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


def _energy(X, ctx) -> float:
    from egg.smoothing.objective import assemble_energy_vec
    es = ctx.energy_stencil
    return assemble_energy_vec(
        X, es["gc"], es["gn0"], es["gn1"], es["s0"], es["s1"], es["W_inv"])


def _assert_converges(ctx, grid, X0, n_sweeps):
    """Run block-Jacobi and assert it descends to a valid, converged minimiser."""
    from egg.smoothing.cpp_backend import cpp_structured_sweep

    e0 = _energy(X0, ctx)
    X_j, e_j, m_j = cpp_structured_sweep(
        ctx, grid, X0, n_sweeps, device="cpu", report_every=1)

    assert m_j[-1] > 0, f"block-Jacobi left the mesh invalid: min det A = {m_j[-1]:.2e}"
    total_drop = e0 - e_j[-1]
    assert total_drop > 0, "block-Jacobi did not decrease energy"
    # Converged: the energy moved a negligible fraction of its total descent over
    # the tail of the run (a plateau, robust to the O(1e2-1e3) energy scale).
    tail_move = abs(e_j[-1] - e_j[-10])
    assert tail_move <= 1e-3 * total_drop + real_tol(1e-9), (
        f"block-Jacobi has not converged: tail move {tail_move:.2e} "
        f"vs total drop {total_drop:.2e}")
    # The device's reported energy / min det A match the NumPy reference on the
    # returned state — the objective math is correct, not merely self-consistent.
    from egg.smoothing.untangle import grid_min_det
    np.testing.assert_allclose(e_j[-1], _energy(X_j, ctx),
                               rtol=real_tol(0.0), atol=real_tol(1e-9))
    np.testing.assert_allclose(m_j[-1], grid_min_det(X_j, ctx.energy_stencil),
                               rtol=real_tol(0.0), atol=real_tol(1e-9))
    return X_j


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


def test_block_jacobi_single_block_converges():
    """Block-Jacobi on a single block (empty halo) descends to a valid minimiser."""
    grid = _single_block_grid()
    X0 = _perturb(grid)
    ctx = build_sweep_context(grid, IdentityTarget(2))
    _assert_converges(ctx, grid, X0, 2000)


def test_block_jacobi_reaches_analytic_minimiser():
    """Ground-truth gate: a Cartesian block under an identity target minimises to
    the uniform grid (closed form). Block-Jacobi must converge to it from a
    perturbed start — validating the whole device pipeline (metric + Newton +
    backtracking) against the exact answer, not a captured snapshot."""
    from egg.smoothing.cpp_backend import cpp_structured_sweep

    grid = _single_block_grid()
    uniform = np.array(grid.global_nodes)   # TFI of a square == the uniform grid
    ctx = build_sweep_context(grid, IdentityTarget(2))
    X0 = _perturb(grid)                     # perturb the interior; uniform is the min

    X_j, _e, m_j = cpp_structured_sweep(
        ctx, grid, X0, 3000, device="cpu")

    assert m_j[-1] > 0
    np.testing.assert_allclose(X_j, uniform, rtol=real_tol(0.0), atol=real_tol(1e-9))


def test_block_jacobi_lr_pair_converges():
    """Block-Jacobi across the conforming L/R pair (cross-block frozen halos +
    intra-sweep Jacobi) descends to a valid minimiser."""
    grid = _lr_grid()
    X0 = _perturb(grid)
    ctx = build_sweep_context(grid, IdentityTarget(2))
    _assert_converges(ctx, grid, X0, 1000)


def test_block_jacobi_ogrid_converges():
    """Block-Jacobi on the 12-block O-grid (singular fans mirrored into ghosts)
    descends to a valid minimiser."""
    from topologies import build_circle_in_rectangle  # noqa: E402

    topo, _ents = build_circle_in_rectangle(rough=False, R=1)
    grid = topo.initialize_grid()
    ctx = build_sweep_context(grid, IdentityTarget(d=topo.d))
    X0 = np.array(grid.global_nodes)
    _assert_converges(ctx, grid, X0, 1200)
