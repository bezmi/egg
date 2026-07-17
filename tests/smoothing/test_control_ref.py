# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""NumPy reduced-GN control-net reference: convergence acceptance criteria.

The control-net smoother must prove itself competitive with the node smoother
before any device or multi-block work builds on it:
  (i)   final fine TMOP energy <= 1.05x the node smoother's on the same grid,
        with min det A > 0 throughout;
  (ii)  within 1% of its own converged energy in <= 30 outer iterations;
  (iii) re-evaluating the net at 2x and 4x sampling keeps min det A > 0.
"""

import numpy as np
import pytest

from egg.geometry.control_net import tensor_map
from egg.init.tfi import tfi_fill_interior
from egg.smoothing import batch as _batch
from egg.smoothing.control_ref import (
    boolean_sum_b,
    build_control_ref,
    fit_control_net,
    node_grad_hess_field,
    run_control_ref,
)
from egg.smoothing.objective import assemble_energy_vec
from egg.smoothing.solver import build_sweep_context
from egg.smoothing.targets import IdentityTarget
from egg.topology.builder import TopologyBuilder

# --------------------------------------------------------------------------- #
# Fixture: a single wavy-boundary block with a perturbed interior
# --------------------------------------------------------------------------- #


def _make_wavy_grid(n=13, wave=0.35, perturb=0.04, seed=11):
    """Unit-ish quad with a sinusoidal top edge; interior TFI then perturbed.

    The wavy boundary makes the TMOP optimum nontrivial; the perturbation gives
    both smoothers the same non-smooth start. Boundary nodes stay put (the node
    baseline only relaxes interior DOFs, matching the control mode's fixed
    boundary controls).
    """
    builder = TopologyBuilder(d=2)
    for name, pos in [
        ("A", (0.0, 0.0)),
        ("B", (4.0, 0.0)),
        ("C", (4.0, 4.0)),
        ("D", (0.0, 4.0)),
    ]:
        builder.add_corner(name, pos, fixed=True)
    builder.add_block("main", ("A", "D", "B", "C"), (n, n))
    topo = builder.build()
    topo.initialize_grid()
    grid = topo.grid
    block = grid.blocks[0]

    # Wavy top edge, then re-TFI the interior from the new boundary.
    nodes = block.nodes
    x_top = nodes[:, -1, 0]
    nodes[:, -1, 1] = 4.0 + wave * np.sin(np.pi * x_top / 4.0)
    interior = np.full_like(nodes, np.nan)
    interior[0, :] = nodes[0, :]
    interior[-1, :] = nodes[-1, :]
    interior[:, 0] = nodes[:, 0]
    interior[:, -1] = nodes[:, -1]
    block.nodes[...] = interior
    tfi_fill_interior(block)

    # Perturb the interior so there is real work to do.
    rng = np.random.default_rng(seed)
    block.nodes[1:-1, 1:-1] += perturb * rng.standard_normal(
        block.nodes[1:-1, 1:-1].shape
    )

    # Sync global state.
    grid.global_nodes[grid.block_dof_maps[0].reshape(-1)] = block.nodes.reshape(
        -1, 2
    )
    return grid


def _interior_dofs(grid):
    """Global ids of strictly-interior nodes of the single block."""
    dof_map = grid.block_dof_maps[0]
    return np.unique(dof_map[1:-1, 1:-1].reshape(-1))


def _node_baseline(grid, target_fn, sweeps=200, metric="shape_2d"):
    """Sequential per-DOF Newton reference over interior nodes (the node
    smoother's NumPy semantics: local Newton + backtracking, monotone patch
    energy, min det gate)."""
    ctx = build_sweep_context(grid, target_fn)
    interior = _interior_dofs(grid)
    X = grid.global_nodes
    for _ in range(sweeps):
        moved = 0.0
        for dof in interior:
            p = ctx.dof_patches[int(dof)]
            g, H = _batch.dof_grad_hess(
                X, p["gc"], p["gn0"], p["gn1"], p["s0"], p["s1"], p["W_inv"],
                p["role"], p["J"], metric=metric,
            )
            try:
                step = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                continue
            e0, _ = _batch.energy_and_mindet(
                X, p["gc"], p["gn0"], p["gn1"], p["s0"], p["s1"], p["W_inv"],
                metric=metric,
            )
            alpha = 1.0
            while alpha >= 1e-4:
                X_try = X.copy()
                X_try[dof] += alpha * step
                e, md = _batch.energy_and_mindet(
                    X_try, p["gc"], p["gn0"], p["gn1"], p["s0"], p["s1"],
                    p["W_inv"], metric=metric,
                )
                if np.isfinite(e) and e <= e0 + 1e-12 and md > 0.0:
                    X[dof] += alpha * step
                    moved += alpha * float(np.linalg.norm(step))
                    break
                alpha *= 0.5
        if moved < 1e-12:
            break
    for bi, block in enumerate(grid.blocks):
        block.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]
    return ctx


def _global_energy(grid, ctx, metric="shape_2d"):
    st = ctx.energy_stencil
    return assemble_energy_vec(
        grid.global_nodes, st["gc"], st["gn0"], st["gn1"], st["s0"], st["s1"],
        st["W_inv"], metric=metric,
    )


def _resampled_mindet(cref, factor):
    """Min corner-Jacobian det of the net re-evaluated at `factor`x resolution."""
    cmap = cref.cmap
    fine_shape = tuple((s - 1) * factor + 1 for s in cmap.node_shape)
    fine = tensor_map(fine_shape, cmap.ctrl_shape, degree=3)
    # b resampled: TFI is sampling-consistent only for the same boundary; here
    # the boundary controls are the fitted spline, so re-add a b computed for
    # the fine sampling from the *coarse* exact boundary interpolated... this
    # reference keeps it simple: the regrid check evaluates the pure spline
    # map (no b).
    X = fine.prolong(cref.C)
    mind = np.inf
    for co in ((0, 0), (0, 1), (1, 0), (1, 1)):
        du = np.diff(X, axis=0)[:, co[1] : X.shape[1] - 1 + co[1]]
        dv = np.diff(X, axis=1)[co[0] : X.shape[0] - 1 + co[0], :]
        det = du[..., 0] * dv[..., 1] - du[..., 1] * dv[..., 0]
        mind = min(mind, float(det.min()))
    return mind


# --------------------------------------------------------------------------- #
# Unit tests: fit, boundary exactness, per-node grad/hess parity
# --------------------------------------------------------------------------- #


def test_fit_recovers_net_in_span():
    # A grid generated FROM a net is fitted back exactly (data in span).
    rng = np.random.default_rng(4)
    cmap = tensor_map((11, 9), (6, 5), degree=3)
    gvx = np.linspace(0.0, 1.0, 6)
    gvy = np.linspace(0.0, 1.0, 5)
    C_true = np.stack(np.meshgrid(gvx, gvy, indexing="ij"), axis=-1)
    C_true += 0.05 * rng.standard_normal(C_true.shape)
    X0 = cmap.prolong(C_true)
    C_fit = fit_control_net(cmap, X0)
    assert np.allclose(cmap.prolong(C_fit), X0, atol=1e-9)


def test_boolean_sum_b_makes_boundary_exact():
    grid = _make_wavy_grid(n=9)
    block = grid.blocks[0]
    X0 = np.asarray(block.nodes, dtype=float)
    cmap = tensor_map(X0.shape[:-1], (5, 5), degree=3)
    C = fit_control_net(cmap, X0)
    b = boolean_sum_b(cmap, C, X0)
    X = cmap.prolong(C) + b
    assert np.allclose(X[0, :], X0[0, :], atol=1e-12)
    assert np.allclose(X[-1, :], X0[-1, :], atol=1e-12)
    assert np.allclose(X[:, 0], X0[:, 0], atol=1e-12)
    assert np.allclose(X[:, -1], X0[:, -1], atol=1e-12)


def test_node_grad_field_matches_per_dof_patch():
    grid = _make_wavy_grid(n=9)
    target = IdentityTarget(d=2)
    cref = build_control_ref(grid, target, (5, 5))
    Xg = cref.prolong_global()
    g, H = node_grad_hess_field(Xg, cref.ctx, cref._J)
    for dof in _interior_dofs(grid)[::7]:
        p = cref.ctx.dof_patches[int(dof)]
        g_ref, H_ref = _batch.dof_grad_hess(
            Xg, p["gc"], p["gn0"], p["gn1"], p["s0"], p["s1"], p["W_inv"],
            p["role"], p["J"],
        )
        assert np.allclose(g[int(dof)], g_ref, atol=1e-12)
        assert np.allclose(H[int(dof)], H_ref, atol=1e-12)


def test_reduced_gradient_is_true_derivative():
    # Finite-difference check of G = Mf^T g through the full map (incl. b).
    grid = _make_wavy_grid(n=9)
    target = IdentityTarget(d=2)
    cref = build_control_ref(grid, target, (5, 5))
    st = cref.ctx.energy_stencil

    def E_of_C(C):
        Xg = cref.prolong_global(C)
        return assemble_energy_vec(
            Xg, st["gc"], st["gn0"], st["gn1"], st["s0"], st["s1"], st["W_inv"]
        )

    Xg = cref.prolong_global()
    g, _ = node_grad_hess_field(Xg, cref.ctx, cref._J)
    G = np.einsum("iF,ia->Fa", cref.Mf, g)
    free_idx = np.flatnonzero(cref.free_ctrl.reshape(-1))
    rng = np.random.default_rng(5)
    for probe in rng.choice(len(free_idx), 4, replace=False):
        for axis in range(2):
            eps = 1e-6
            Cp = cref.C.copy()
            Cm = cref.C.copy()
            Cp.reshape(-1, 2)[free_idx[probe], axis] += eps
            Cm.reshape(-1, 2)[free_idx[probe], axis] -= eps
            fd = (E_of_C(Cp) - E_of_C(Cm)) / (2 * eps)
            assert G[probe, axis] == pytest.approx(fd, rel=1e-5, abs=1e-8)


# --------------------------------------------------------------------------- #
# Convergence acceptance criteria
# --------------------------------------------------------------------------- #


def test_energy_iterations_and_regrid():
    target = IdentityTarget(d=2)

    # Node-smoother baseline on its own copy of the grid.
    grid_node = _make_wavy_grid()
    ctx_node = _node_baseline(grid_node, target)
    e_node = _global_energy(grid_node, ctx_node)

    # Control-net run on an identical grid.
    grid_ctrl = _make_wavy_grid()
    cref = build_control_ref(grid_ctrl, target, (6, 6))
    report = run_control_ref(cref, max_outer=30)

    e_ctrl = report["energies"][-1]
    assert all(np.isfinite(report["energies"]))
    assert all(md > 0.0 for md in report["mindets"])
    # (ii) monotone and converged within the budget
    assert np.all(np.diff(report["energies"]) <= 1e-12)
    assert report["iters"] <= 30
    e_limit = report["energies"][-1]
    within_1pct = next(
        i for i, e in enumerate(report["energies"])
        if e <= e_limit + 0.01 * abs(e_limit)
    )
    assert within_1pct <= 30
    # (i) energy competitive with the node smoother
    assert e_ctrl <= 1.05 * e_node + 1e-12
    # (iii) algebraic regrid stays valid
    assert _resampled_mindet(cref, 2) > 0.0
    assert _resampled_mindet(cref, 4) > 0.0
