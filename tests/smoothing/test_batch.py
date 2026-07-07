# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Tests for vectorized batch patch evaluation against per-corner reference."""

from itertools import product

import time

import numpy as np

from egg.smoothing import batch as _batch
from egg.smoothing.solver import (
    _patch_energy_and_mindet,
    _patch_grad_hess,
    build_sweep_context,
)
from egg.smoothing.targets import IdentityTarget
from egg.topology.builder import TopologyBuilder


def _make_test_grid_2d():
    builder = TopologyBuilder(d=2)
    for name, pos in [
        ("A", (0.0, 0.0)),
        ("B", (4.0, 0.0)),
        ("C", (4.0, 4.0)),
        ("D", (0.0, 4.0)),
    ]:
        builder.add_corner(name, pos, fixed=True)
    builder.add_block("main", ("A", "D", "B", "C"), (6, 6))
    topo = builder.build()
    topo.initialize_grid()
    return topo.grid


def _perturb_grid(grid):
    """Perturb free nodes so gradient/Hessian is non-trivial."""
    x = np.zeros(grid.total_free_dofs * 2)
    x += 0.1 * np.random.default_rng(7).standard_normal(len(x))
    # unpack
    d = grid.topology.d
    n_free = grid.total_free_dofs
    grid.global_nodes[grid.free_mask] = x.reshape(n_free, d)
    for bi, block in enumerate(grid.blocks):
        block.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]


def _extract_patch_stencils(grid, ctx, dof_idx):
    """Manually extract batch stencil arrays for a DOF from ctx's cells/w_inv.

    Does *not* use ctx.dof_patches — this is the independent reference builder
    used to verify that the new precomputation matches the old per-corner path.
    """
    d = grid.topology.d
    gc_list, gn0_list, gn1_list = [], [], []
    s0_list, s1_list = [], []
    w_inv_list = []
    role_list = []
    seen: set[tuple[int, tuple]] = set()

    for bi, cell_base in ctx.dof_to_cells[dof_idx]:
        key = (bi, cell_base)
        if key in seen:
            continue
        seen.add(key)

        dof_map = grid.block_dof_maps[bi]
        base_arr = np.asarray(cell_base, dtype=int)

        for co in product((0, 1), repeat=d):
            o_arr = np.asarray(co, dtype=int)
            corner_idx = tuple(base_arr + o_arr)
            gidx_c = int(dof_map[corner_idx])

            s0 = 1 if o_arr[0] == 0 else -1
            nbr0 = (base_arr + o_arr).copy()
            nbr0[0] += s0
            gidx_n0 = int(dof_map[tuple(nbr0)])

            s1 = 1 if o_arr[1] == 0 else -1
            nbr1 = (base_arr + o_arr).copy()
            nbr1[1] += s1
            gidx_n1 = int(dof_map[tuple(nbr1)])

            gc_list.append(gidx_c)
            gn0_list.append(gidx_n0)
            gn1_list.append(gidx_n1)
            s0_list.append(s0)
            s1_list.append(s1)
            w_inv_list.append(ctx.w_inv[(bi, cell_base, co)])

            if gidx_c == dof_idx:
                role_list.append(0)
            elif gidx_n0 == dof_idx:
                role_list.append(1)
            elif gidx_n1 == dof_idx:
                role_list.append(2)
            else:
                role_list.append(-1)

    return {
        "gc": np.array(gc_list, dtype=np.intp),
        "gn0": np.array(gn0_list, dtype=np.intp),
        "gn1": np.array(gn1_list, dtype=np.intp),
        "s0": np.array(s0_list, dtype=np.intp),
        "s1": np.array(s1_list, dtype=np.intp),
        "W_inv": np.stack(w_inv_list) if w_inv_list else np.zeros((0, 2, 2)),
        "role": np.array(role_list, dtype=np.intp),
    }


class TestEnergyAndMinDetEquivalence:
    """batch.energy_and_mindet must match _patch_energy_and_mindet."""

    def test_single_block(self):
        grid = _make_test_grid_2d()
        _perturb_grid(grid)
        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        # Pick an interior DOF (not on boundary) so its patch has full stencil
        free_dofs = np.where(grid.free_mask)[0]
        dof_idx = free_dofs[len(free_dofs) // 2]

        stencil = _extract_patch_stencils(grid, ctx, dof_idx)

        e_batch, det_batch = _batch.energy_and_mindet(
            grid.global_nodes,
            stencil["gc"],
            stencil["gn0"],
            stencil["gn1"],
            stencil["s0"],
            stencil["s1"],
            stencil["W_inv"],
        )
        e_ref, det_ref = _patch_energy_and_mindet(grid, dof_idx, ctx)

        assert abs(e_batch - e_ref) < 1e-10, (
            f"Energy mismatch: batch={e_batch}, ref={e_ref}"
        )
        assert abs(det_batch - det_ref) < 1e-10, (
            f"Min-det mismatch: batch={det_batch}, ref={det_ref}"
        )


class TestGradHessEquivalence:
    """batch.dof_grad_hess must match _patch_grad_hess."""

    def test_single_block(self):
        grid = _make_test_grid_2d()
        _perturb_grid(grid)
        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        free_dofs = np.where(grid.free_mask)[0]
        dof_idx = free_dofs[len(free_dofs) // 2]

        stencil = _extract_patch_stencils(grid, ctx, dof_idx)

        g_batch, H_batch = _batch.dof_grad_hess(
            grid.global_nodes,
            stencil["gc"],
            stencil["gn0"],
            stencil["gn1"],
            stencil["s0"],
            stencil["s1"],
            stencil["W_inv"],
            stencil["role"],
        )
        g_ref, H_ref = _patch_grad_hess(grid, dof_idx, ctx)

        np.testing.assert_allclose(g_batch, g_ref, atol=1e-10)
        np.testing.assert_allclose(H_batch, H_ref, atol=1e-10)


class TestPatchEvalEquivalence:
    """patch_eval must match dof_grad_hess + energy_and_mindet separately."""

    def test_single_block(self):
        grid = _make_test_grid_2d()
        _perturb_grid(grid)
        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        free_dofs = np.where(grid.free_mask)[0]
        dof_idx = free_dofs[len(free_dofs) // 2]

        stencil = _extract_patch_stencils(grid, ctx, dof_idx)
        X = grid.global_nodes
        args = (
            X,
            stencil["gc"],
            stencil["gn0"],
            stencil["gn1"],
            stencil["s0"],
            stencil["s1"],
            stencil["W_inv"],
            stencil["role"],
        )

        g_fused, H_fused, e_fused, d_fused = _batch.patch_eval(*args)
        g_sep, H_sep = _batch.dof_grad_hess(*args)
        e_sep, d_sep = _batch.energy_and_mindet(*args[:7])

        np.testing.assert_allclose(g_fused, g_sep, atol=1e-10)
        np.testing.assert_allclose(H_fused, H_sep, atol=1e-10)
        assert abs(e_fused - e_sep) < 1e-10
        assert abs(d_fused - d_sep) < 1e-10

    def test_multi_block(self):
        builder = TopologyBuilder(d=2)
        for name, pos in [
            ("A", (0.0, 0.0)),
            ("B", (2.0, 0.0)),
            ("C", (2.0, 2.0)),
            ("D", (0.0, 2.0)),
            ("E", (4.0, 0.0)),
            ("F", (4.0, 2.0)),
        ]:
            builder.add_corner(name, pos, fixed=True)
        builder.add_block("L", ("A", "D", "B", "C"), (4, 4))
        builder.add_block("R", ("B", "C", "E", "F"), (4, 4))
        builder.connect("L", 0, 1, "R", 0, 0)
        topo = builder.build()
        topo.initialize_grid()
        grid = topo.grid
        _perturb_grid(grid)

        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        free_dofs = np.where(grid.free_mask)[0]
        dof_idx = free_dofs[len(free_dofs) // 2]

        stencil = _extract_patch_stencils(grid, ctx, dof_idx)
        X = grid.global_nodes
        args = (
            X,
            stencil["gc"],
            stencil["gn0"],
            stencil["gn1"],
            stencil["s0"],
            stencil["s1"],
            stencil["W_inv"],
            stencil["role"],
        )

        g_fused, H_fused, e_fused, d_fused = _batch.patch_eval(*args)
        g_sep, H_sep = _batch.dof_grad_hess(*args)
        e_sep, d_sep = _batch.energy_and_mindet(*args[:7])

        np.testing.assert_allclose(g_fused, g_sep, atol=1e-10)
        np.testing.assert_allclose(H_fused, H_sep, atol=1e-10)
        assert abs(e_fused - e_sep) < 1e-10
        assert abs(d_fused - d_sep) < 1e-10

    def test_fused_is_faster(self):
        """patch_eval should be faster than calling both dof_grad_hess + energy_and_mindet."""
        grid = _make_test_grid_2d()
        _perturb_grid(grid)
        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        free_dofs = np.where(grid.free_mask)[0]
        dof_idx = free_dofs[len(free_dofs) // 2]
        stencil = _extract_patch_stencils(grid, ctx, dof_idx)
        X = grid.global_nodes
        args = (
            X,
            stencil["gc"],
            stencil["gn0"],
            stencil["gn1"],
            stencil["s0"],
            stencil["s1"],
            stencil["W_inv"],
            stencil["role"],
        )
        args_energy = args[:7]

        # warm-up
        for _ in range(50):
            _batch.patch_eval(*args)
            _batch.dof_grad_hess(*args)
            _batch.energy_and_mindet(*args_energy)

        n = 500
        t0 = time.perf_counter()
        for _ in range(n):
            _batch.patch_eval(*args)
        t_fused = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n):
            _batch.dof_grad_hess(*args)
            _batch.energy_and_mindet(*args_energy)
        t_sep = time.perf_counter() - t0

        assert t_fused < t_sep, (
            f"Fused {t_fused:.4f}s not faster than separate {t_sep:.4f}s"
        )


class TestPrecomputedJEquivalence:
    """make_chain_J precomputation must match on-the-fly and be faster."""

    def test_correctness_patch_eval(self):
        grid = _make_test_grid_2d()
        _perturb_grid(grid)
        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        free_dofs = np.where(grid.free_mask)[0]
        dof_idx = free_dofs[len(free_dofs) // 2]
        stencil = _extract_patch_stencils(grid, ctx, dof_idx)

        J = _batch.make_chain_J(stencil["s0"], stencil["s1"], stencil["W_inv"])
        X = grid.global_nodes
        args = (
            X,
            stencil["gc"],
            stencil["gn0"],
            stencil["gn1"],
            stencil["s0"],
            stencil["s1"],
            stencil["W_inv"],
            stencil["role"],
        )

        # with precomputed J
        g1, H1, e1, d1 = _batch.patch_eval(*args, J=J)
        # without (fallback to auto-compute)
        g2, H2, e2, d2 = _batch.patch_eval(*args, J=None)

        np.testing.assert_allclose(g1, g2, atol=1e-10)
        np.testing.assert_allclose(H1, H2, atol=1e-10)
        assert abs(e1 - e2) < 1e-10
        assert abs(d1 - d2) < 1e-10

    def test_correctness_dof_grad_hess(self):
        grid = _make_test_grid_2d()
        _perturb_grid(grid)
        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        free_dofs = np.where(grid.free_mask)[0]
        dof_idx = free_dofs[len(free_dofs) // 2]
        stencil = _extract_patch_stencils(grid, ctx, dof_idx)

        J = _batch.make_chain_J(stencil["s0"], stencil["s1"], stencil["W_inv"])
        X = grid.global_nodes
        args = (
            X,
            stencil["gc"],
            stencil["gn0"],
            stencil["gn1"],
            stencil["s0"],
            stencil["s1"],
            stencil["W_inv"],
            stencil["role"],
        )

        g1, H1 = _batch.dof_grad_hess(*args, J=J)
        g2, H2 = _batch.dof_grad_hess(*args, J=None)

        np.testing.assert_allclose(g1, g2, atol=1e-10)
        np.testing.assert_allclose(H1, H2, atol=1e-10)

    def test_precomputed_is_faster(self):
        """patch_eval with precomputed J should be faster than without."""
        grid = _make_test_grid_2d()
        _perturb_grid(grid)
        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        free_dofs = np.where(grid.free_mask)[0]
        dof_idx = free_dofs[len(free_dofs) // 2]
        stencil = _extract_patch_stencils(grid, ctx, dof_idx)

        J = _batch.make_chain_J(stencil["s0"], stencil["s1"], stencil["W_inv"])
        X = grid.global_nodes
        args = (
            X,
            stencil["gc"],
            stencil["gn0"],
            stencil["gn1"],
            stencil["s0"],
            stencil["s1"],
            stencil["W_inv"],
            stencil["role"],
        )

        # warm-up
        for _ in range(50):
            _batch.patch_eval(*args, J=J)
            _batch.patch_eval(*args, J=None)

        n = 500
        t0 = time.perf_counter()
        for _ in range(n):
            _batch.patch_eval(*args, J=J)
        t_j = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n):
            _batch.patch_eval(*args, J=None)
        t_none = time.perf_counter() - t0

        assert t_j < t_none, (
            f"Precomputed J {t_j:.4f}s was not faster than auto-compute {t_none:.4f}s"
        )


class TestAssembleAEquivalence:
    """batch.assemble_A must produce the same A matrices as corner_sample."""

    def test_single_block(self):
        grid = _make_test_grid_2d()
        _perturb_grid(grid)
        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        free_dofs = np.where(grid.free_mask)[0]
        dof_idx = free_dofs[len(free_dofs) // 2]
        stencil = _extract_patch_stencils(grid, ctx, dof_idx)

        A_batch = _batch.assemble_A(
            grid.global_nodes,
            stencil["gc"],
            stencil["gn0"],
            stencil["gn1"],
            stencil["s0"],
            stencil["s1"],
        )  # shape (P, 2, 2)

        # Build reference A arrays via per-corner corner_sample
        seen = set()
        d = grid.topology.d
        k = 0
        for bi, cell_base in ctx.dof_to_cells[dof_idx]:
            key = (bi, cell_base)
            if key in seen:
                continue
            seen.add(key)
            nodes = grid.blocks[bi].nodes
            for co in product((0, 1), repeat=d):
                from egg.smoothing.jacobian import corner_sample

                A_ref, _, _ = corner_sample(nodes, cell_base, co)
                np.testing.assert_allclose(
                    A_batch[k], A_ref, atol=1e-12, err_msg=f"Mismatch at sample {k}"
                )
                k += 1

        assert k == stencil["gc"].shape[0], "Sample count mismatch"

    def test_preallocated_is_correct_and_faster(self):
        """Pre-allocated assemble_A must match the old np.stack approach and be faster."""
        grid = _make_test_grid_2d()
        _perturb_grid(grid)
        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        free_dofs = np.where(grid.free_mask)[0]
        dof_idx = free_dofs[len(free_dofs) // 2]
        stencil = _extract_patch_stencils(grid, ctx, dof_idx)
        X = grid.global_nodes
        gc, gn0, gn1 = stencil["gc"], stencil["gn0"], stencil["gn1"]
        s0, s1 = stencil["s0"], stencil["s1"]

        def old_assemble_A(X, gc, gn0, gn1, s0, s1):
            Xc = X[gc]
            col0 = s0[:, None] * (X[gn0] - Xc)
            col1 = s1[:, None] * (X[gn1] - Xc)
            return np.stack([col0, col1], axis=2)

        A_new = _batch.assemble_A(X, gc, gn0, gn1, s0, s1)
        A_old = old_assemble_A(X, gc, gn0, gn1, s0, s1)
        np.testing.assert_allclose(A_new, A_old, atol=1e-12)

        # warm-up
        for _ in range(200):
            _batch.assemble_A(X, gc, gn0, gn1, s0, s1)
            old_assemble_A(X, gc, gn0, gn1, s0, s1)

        n = 2000
        t0 = time.perf_counter()
        for _ in range(n):
            _batch.assemble_A(X, gc, gn0, gn1, s0, s1)
        t_new = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n):
            old_assemble_A(X, gc, gn0, gn1, s0, s1)
        t_old = time.perf_counter() - t0

        assert t_new < t_old, f"New {t_new:.4f}s not faster than old {t_old:.4f}s"


class TestNonParticipatingSamples:
    """Samples where the DOF does not participate must contribute zero."""

    def test_role_minus_one_gradient_zero(self):
        grid = _make_test_grid_2d()
        _perturb_grid(grid)
        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        free_dofs = np.where(grid.free_mask)[0]
        dof_idx = free_dofs[len(free_dofs) // 2]
        stencil = _extract_patch_stencils(grid, ctx, dof_idx)

        # Verify at least one sample has role=-1 in a multi-cell patch
        assert np.any(stencil["role"] == -1), "Expected some non-participating samples"

        g_batch, H_batch = _batch.dof_grad_hess(
            grid.global_nodes,
            stencil["gc"],
            stencil["gn0"],
            stencil["gn1"],
            stencil["s0"],
            stencil["s1"],
            stencil["W_inv"],
            stencil["role"],
        )

        # Build gradient/Hessian from *only* participating samples
        mask = stencil["role"] >= 0
        p_stencil = {
            k: v[mask] for k, v in stencil.items() if isinstance(v, np.ndarray)
        }
        g_part, H_part = _batch.dof_grad_hess(
            grid.global_nodes,
            p_stencil["gc"],
            p_stencil["gn0"],
            p_stencil["gn1"],
            p_stencil["s0"],
            p_stencil["s1"],
            p_stencil["W_inv"],
            p_stencil["role"],
        )

        np.testing.assert_allclose(
            g_batch,
            g_part,
            atol=1e-12,
            err_msg="Non-participating samples leaked into gradient",
        )
        np.testing.assert_allclose(
            H_batch,
            H_part,
            atol=1e-12,
            err_msg="Non-participating samples leaked into Hessian",
        )


class TestMultiBlock:
    """Batch functions must work on multi-block grids."""

    def test_two_block_grid(self):
        builder = TopologyBuilder(d=2)
        for name, pos in [
            ("A", (0.0, 0.0)),
            ("B", (2.0, 0.0)),
            ("C", (2.0, 2.0)),
            ("D", (0.0, 2.0)),
            ("E", (4.0, 0.0)),
            ("F", (4.0, 2.0)),
        ]:
            builder.add_corner(name, pos, fixed=True)
        builder.add_block("L", ("A", "D", "B", "C"), (4, 4))
        builder.add_block("R", ("B", "C", "E", "F"), (4, 4))
        builder.connect("L", 0, 1, "R", 0, 0)
        topo = builder.build()
        topo.initialize_grid()
        grid = topo.grid
        _perturb_grid(grid)

        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        free_dofs = np.where(grid.free_mask)[0]
        dof_idx = free_dofs[len(free_dofs) // 2]

        stencil = _extract_patch_stencils(grid, ctx, dof_idx)

        e_batch, det_batch = _batch.energy_and_mindet(
            grid.global_nodes,
            stencil["gc"],
            stencil["gn0"],
            stencil["gn1"],
            stencil["s0"],
            stencil["s1"],
            stencil["W_inv"],
        )
        e_ref, det_ref = _patch_energy_and_mindet(grid, dof_idx, ctx)

        assert abs(e_batch - e_ref) < 1e-10
        assert abs(det_batch - det_ref) < 1e-10

        g_batch, H_batch = _batch.dof_grad_hess(
            grid.global_nodes,
            stencil["gc"],
            stencil["gn0"],
            stencil["gn1"],
            stencil["s0"],
            stencil["s1"],
            stencil["W_inv"],
            stencil["role"],
        )
        g_ref, H_ref = _patch_grad_hess(grid, dof_idx, ctx)

        np.testing.assert_allclose(g_batch, g_ref, atol=1e-10)
        np.testing.assert_allclose(H_batch, H_ref, atol=1e-10)


class TestInteriorSynthesisGate:
    """The C++ interior-patch synthesis assumes an identity W_inv, so the
    wire may only classify interior DOFs when the target is (a scalar
    multiple of) the identity everywhere."""

    def test_identity_target_synthesizes_interior(self):
        grid = _make_test_grid_2d()
        ctx = build_sweep_context(grid, IdentityTarget(2))
        g = ctx.wire["groups"][0]
        assert np.any(np.asarray(g["interior_block"]) >= 0)

    def test_varying_target_ships_stored_stencils(self):
        grid = _make_test_grid_2d()

        def varying(bi, block, cell_base, corner_offset):
            s = 0.1 + 0.05 * cell_base[0]
            return np.diag([s, 3.0 * s])

        ctx = build_sweep_context(grid, varying)
        g = ctx.wire["groups"][0]
        assert np.all(np.asarray(g["interior_block"]) == -1)
        # Interior DOFs must own stored samples again (P_of > 0 for all
        # moving DOFs on a single fully-free block interior).
        assert np.all(np.asarray(g["P_of"]) > 0)
