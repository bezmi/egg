"""Tests for local Gauss-Seidel relaxation solver."""

import time

import numpy as np

from egg.smoothing.objective import assemble_energy, pack_x, unpack_x
from egg.smoothing.solver import local_relaxation, local_relaxation_sweep
from egg.smoothing.targets import IdentityTarget
from egg.topology.builder import TopologyBuilder


def _make_grid_2d():
    builder = TopologyBuilder(d=2)
    for name, pos in [("A", (0.0, 0.0)), ("B", (4.0, 0.0)),
                      ("C", (4.0, 4.0)), ("D", (0.0, 4.0))]:
        builder.add_corner(name, pos, fixed=True)
    builder.add_block("main", ("A", "D", "B", "C"), (6, 6))
    topo = builder.build()
    topo.initialize_grid()
    return topo.grid


class TestSolver:
    def test_monotone_descent(self):
        """Energy is non-increasing across sweeps."""
        grid = _make_grid_2d()
        target = IdentityTarget(d=2)

        # Perturb to create non-optimal state
        x0 = pack_x(grid)
        x0 = x0 + 0.1 * np.random.default_rng(42).normal(size=len(x0))
        unpack_x(x0, grid)

        history = local_relaxation(grid, target, "shape_2d", max_sweeps=15)
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + 1e-12, (
                f"Energy increased at sweep {i}: {history[i-1]} -> {history[i]}"
            )

    def test_energy_decreases_significantly(self):
        """Energy drops by at least an order of magnitude."""
        grid = _make_grid_2d()
        target = IdentityTarget(d=2)
        x0 = pack_x(grid)
        x0 = x0 + 0.1 * np.random.default_rng(42).normal(size=len(x0))
        unpack_x(x0, grid)
        F0 = assemble_energy(grid, target, "shape_2d")

        local_relaxation(grid, target, "shape_2d", max_sweeps=30)
        F1 = assemble_energy(grid, target, "shape_2d")
        assert F1 < F0 * 0.01, f"Energy only dropped from {F0} to {F1}"

    def test_validity_preserved(self):
        """min(det A) > 0 maintained at every sweep."""
        grid = _make_grid_2d()
        target = IdentityTarget(d=2)
        x0 = pack_x(grid)
        x0 = x0 + 0.05 * np.random.default_rng(99).normal(size=len(x0))
        unpack_x(x0, grid)

        # Quick check after each of a few sweeps
        for _ in range(5):
            history = local_relaxation(grid, target, "shape_2d", max_sweeps=3)
            for block in grid.blocks:
                nodes = block.nodes
                for i in range(nodes.shape[0] - 1):
                    for j in range(nodes.shape[1] - 1):
                        A = np.zeros((2, 2))
                        A[:, 0] = nodes[i + 1, j] - nodes[i, j]
                        A[:, 1] = nodes[i, j + 1] - nodes[i, j]
                        assert np.linalg.det(A) > 0, (
                            f"Negative det at ({i},{j}): {np.linalg.det(A):.2e}"
                        )

    def test_fixed_point(self):
        """Uniform grid with W=I → nodes barely move."""
        grid = _make_grid_2d()
        target = IdentityTarget(d=2)
        # Grid is already uniform (TFI gives uniform), don't perturb

        x_before = pack_x(grid).copy()
        local_relaxation(grid, target, "shape_2d", max_sweeps=5, tol=1e-10)
        x_after = pack_x(grid)
        max_move = np.max(np.abs(x_before - x_after))
        assert max_move < 1e-8, f"Nodes moved by {max_move} from fixed point"

    def test_multiblock_relaxation(self):
        """Relaxation works on a 2-block connected grid."""
        builder = TopologyBuilder(d=2)
        for name, pos in [("A", (0., 0.)), ("B", (2., 0.)),
                          ("C", (2., 2.)), ("D", (0., 2.)),
                          ("E", (4., 0.)), ("F", (4., 2.))]:
            builder.add_corner(name, pos, fixed=True)
        builder.add_block("L", ("A", "D", "B", "C"), (4, 4))
        builder.add_block("R", ("B", "C", "E", "F"), (4, 4))
        builder.connect("L", 0, 1, "R", 0, 0)
        topo = builder.build()
        topo.initialize_grid()
        grid = topo.grid

        target = IdentityTarget(d=2)
        x0 = pack_x(grid)
        x0 = x0 + 0.05 * np.random.default_rng(7).normal(size=len(x0))
        unpack_x(x0, grid)

        F0 = assemble_energy(grid, target, "shape_2d")
        local_relaxation(grid, target, "shape_2d", max_sweeps=15)
        F1 = assemble_energy(grid, target, "shape_2d")

        assert F1 < F0, "Energy did not decrease on multiblock grid"
        # Shared interface nodes should still match after relaxation
        dof_L = grid.block_dof_maps[0]
        dof_R = grid.block_dof_maps[1]
        nodes_L = grid.blocks[0].nodes
        nodes_R = grid.blocks[1].nodes
        for j in range(5):
            assert np.allclose(nodes_L[4, j], nodes_R[0, j]), (
                f"Shared interface mismatch at j={j}"
            )


class TestBatchSweepContext:
    """Verify the SweepContext dof_patches precomputation is correct."""

    @staticmethod
    def _extract_patch_stencils(grid, ctx, dof_idx):
        """Independent stencil extraction from dof_to_cells + w_inv (not from dof_patches)."""
        from itertools import product as _product
        d = grid.topology.d
        gc_list, gn0_list, gn1_list = [], [], []
        s0_list, s1_list = [], []
        w_inv_list = []
        role_list = []
        seen = set()

        for bi, cell_base in ctx.dof_to_cells[dof_idx]:
            key = (bi, cell_base)
            if key in seen:
                continue
            seen.add(key)

            dof_map = grid.block_dof_maps[bi]
            base_arr = np.asarray(cell_base, dtype=int)

            for co in _product((0, 1), repeat=d):
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

    def test_dof_patches_match_manual_extraction(self):
        from egg.smoothing.solver import build_sweep_context
        grid = _make_grid_2d()
        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        free_dofs = np.where(grid.free_mask)[0]
        for dof_idx in free_dofs[:5]:  # spot-check first 5 free DOFs
            expected = self._extract_patch_stencils(grid, ctx, dof_idx)
            actual = ctx.dof_patches[dof_idx]
            np.testing.assert_array_equal(expected["gc"], actual["gc"])
            np.testing.assert_array_equal(expected["gn0"], actual["gn0"])
            np.testing.assert_array_equal(expected["gn1"], actual["gn1"])
            np.testing.assert_array_equal(expected["s0"], actual["s0"])
            np.testing.assert_array_equal(expected["s1"], actual["s1"])
            np.testing.assert_allclose(expected["W_inv"], actual["W_inv"])
            np.testing.assert_array_equal(expected["role"], actual["role"])

    def test_fixed_dof_patches_match_manual_extraction(self):
        from egg.smoothing.solver import build_sweep_context
        grid = _make_grid_2d()
        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        fixed_dofs = np.where(~grid.free_mask)[0]
        for dof_idx in fixed_dofs[:5]:  # spot-check
            # Fixed corner DOFs are still incident to cells; patches may be non-empty
            patch = ctx.dof_patches[dof_idx]
            expected = self._extract_patch_stencils(grid, ctx, dof_idx)
            assert patch["gc"].shape[0] == expected["gc"].shape[0]
            np.testing.assert_array_equal(expected["gc"], patch["gc"])
            np.testing.assert_array_equal(expected["role"], patch["role"])

    def test_multiblock_dof_patches(self):
        from egg.smoothing.solver import build_sweep_context
        builder = TopologyBuilder(d=2)
        for name, pos in [("A", (0., 0.)), ("B", (2., 0.)),
                          ("C", (2., 2.)), ("D", (0., 2.)),
                          ("E", (4., 0.)), ("F", (4., 2.))]:
            builder.add_corner(name, pos, fixed=True)
        builder.add_block("L", ("A", "D", "B", "C"), (4, 4))
        builder.add_block("R", ("B", "C", "E", "F"), (4, 4))
        builder.connect("L", 0, 1, "R", 0, 0)
        topo = builder.build()
        topo.initialize_grid()
        grid = topo.grid

        target = IdentityTarget(d=2)
        ctx = build_sweep_context(grid, target)

        free_dofs = np.where(grid.free_mask)[0]
        for dof_idx in free_dofs[:5]:
            expected = self._extract_patch_stencils(grid, ctx, dof_idx)
            actual = ctx.dof_patches[dof_idx]
            np.testing.assert_array_equal(expected["gc"], actual["gc"])
            np.testing.assert_array_equal(expected["role"], actual["role"])


class TestQuadraticFilter:
    """Quadratic pre-filter must produce identical results to unfiltered path."""

    @staticmethod
    def _perturbed_grid(size=6):
        builder = TopologyBuilder(d=2)
        L = float(size - 1)
        for name, pos in [("A", (0., 0.)), ("B", (L, 0.)),
                          ("C", (L, L)), ("D", (0., L))]:
            builder.add_corner(name, pos, fixed=True)
        builder.add_block("main", ("A", "D", "B", "C"), (size, size))
        topo = builder.build()
        topo.initialize_grid()
        grid = topo.grid
        x0 = pack_x(grid)
        x0 = x0 + 0.1 * np.random.default_rng(7).normal(size=len(x0))
        unpack_x(x0, grid)
        return grid

    def test_identical_nodes_one_sweep(self):
        """One sweep with filter ON vs OFF must produce identical global_nodes."""
        target = IdentityTarget(d=2)

        grid_a = self._perturbed_grid()
        local_relaxation_sweep(grid_a, target, "shape_2d", quadratic_filter=False)
        nodes_off = grid_a.global_nodes.copy()

        grid_b = self._perturbed_grid()
        local_relaxation_sweep(grid_b, target, "shape_2d", quadratic_filter=True)
        nodes_on = grid_b.global_nodes.copy()

        np.testing.assert_allclose(nodes_off, nodes_on, atol=1e-10)

    def test_identical_nodes_multiblock(self):
        """Filter parity must hold on multi-block grids."""
        target = IdentityTarget(d=2)

        def _initial_state():
            builder = TopologyBuilder(d=2)
            for name, pos in [("A", (0., 0.)), ("B", (2., 0.)),
                              ("C", (2., 2.)), ("D", (0., 2.)),
                              ("E", (4., 0.)), ("F", (4., 2.))]:
                builder.add_corner(name, pos, fixed=True)
            builder.add_block("L", ("A", "D", "B", "C"), (4, 4))
            builder.add_block("R", ("B", "C", "E", "F"), (4, 4))
            builder.connect("L", 0, 1, "R", 0, 0)
            topo = builder.build()
            topo.initialize_grid()
            grid = topo.grid
            x0 = pack_x(grid)
            x0 = x0 + 0.1 * np.random.default_rng(42).normal(size=len(x0))
            unpack_x(x0, grid)
            return grid

        grid_off = _initial_state()
        local_relaxation_sweep(grid_off, target, "shape_2d", quadratic_filter=False)
        nodes_off = grid_off.global_nodes.copy()

        grid_on = _initial_state()
        local_relaxation_sweep(grid_on, target, "shape_2d", quadratic_filter=True)
        nodes_on = grid_on.global_nodes.copy()

        np.testing.assert_allclose(nodes_off, nodes_on, atol=1e-10)

    def test_filter_not_slower(self):
        """Filtered sweep should not be slower than unfiltered."""
        target = IdentityTarget(d=2)
        N = 10

        # warm
        grid_warm = self._perturbed_grid()
        local_relaxation_sweep(grid_warm, target, "shape_2d", quadratic_filter=False)
        local_relaxation_sweep(grid_warm, target, "shape_2d", quadratic_filter=True)

        t0 = time.perf_counter()
        for _ in range(N):
            g = self._perturbed_grid()
            local_relaxation_sweep(g, target, "shape_2d", quadratic_filter=False)
        t_off = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(N):
            g = self._perturbed_grid()
            local_relaxation_sweep(g, target, "shape_2d", quadratic_filter=True)
        t_on = time.perf_counter() - t0

        assert t_on <= t_off * 1.05, (
            f"Filtered {t_on:.4f}s > unfiltered {t_off:.4f}s + 5% margin"
        )

    def test_monotone_descent_with_filter(self):
        """Energy is non-increasing across sweeps with filter enabled."""
        grid = self._perturbed_grid()
        target = IdentityTarget(d=2)
        ctx = None
        prev = assemble_energy(grid, target, "shape_2d")
        for _ in range(10):
            local_relaxation_sweep(grid, target, "shape_2d", ctx, quadratic_filter=True)
            curr = assemble_energy(grid, target, "shape_2d")
            assert curr <= prev + 1e-12, f"Energy increased: {prev} -> {curr}"
            prev = curr

    def test_multiblock_with_filter(self):
        """Filtered relaxation descends on multiblock grid."""
        builder = TopologyBuilder(d=2)
        for name, pos in [("A", (0., 0.)), ("B", (2., 0.)),
                          ("C", (2., 2.)), ("D", (0., 2.)),
                          ("E", (4., 0.)), ("F", (4., 2.))]:
            builder.add_corner(name, pos, fixed=True)
        builder.add_block("L", ("A", "D", "B", "C"), (4, 4))
        builder.add_block("R", ("B", "C", "E", "F"), (4, 4))
        builder.connect("L", 0, 1, "R", 0, 0)
        topo = builder.build()
        topo.initialize_grid()
        grid = topo.grid

        target = IdentityTarget(d=2)
        x0 = pack_x(grid)
        x0 = x0 + 0.05 * np.random.default_rng(7).normal(size=len(x0))
        unpack_x(x0, grid)

        F0 = assemble_energy(grid, target, "shape_2d")
        local_relaxation_sweep(grid, target, "shape_2d", quadratic_filter=True)
        F1 = assemble_energy(grid, target, "shape_2d")
        assert F1 < F0, "Energy did not decrease with filter"
