# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""The size-aware ``shape_size`` metric end to end.

Parity: the batched NumPy evaluators and the C++ closed forms against the
scalar oracle (:mod:`egg.smoothing.metrics`). Behavior: on a two-block grid
with deliberately mismatched cell sizes, the shape metric (scale-invariant)
leaves the size disparity in place while ``shape_size`` under a mean-size
target equalises cell areas — the whole point of the metric. Cell-area
statistics are ratios, so the gates hold in fp32 too.
"""

from __future__ import annotations

import numpy as np
import pytest

from egg.smoothing.batch import _grad_T, _hess_T, _mu_batch
from egg.smoothing.metrics import (
    _shape_size_hess_T,
    metric_value_and_grad,
)
from egg.topology.builder import TopologyBuilder


def _has_cpp() -> bool:
    try:
        from egg._cpp import cpp_core  # noqa: F401

        return True
    except ImportError:
        return False


needs_cpp = pytest.mark.skipif(
    not _has_cpp(),
    reason="egg._cpp.cpp_core not built (requires cmake build)",
)


def _random_valid_T(rng, n):
    """Random 2x2 T matrices bounded away from det = 0."""
    out = []
    while len(out) < n:
        T = np.eye(2) + 0.6 * rng.standard_normal((2, 2))
        if np.linalg.det(T) > 0.05:
            out.append(T)
    return np.stack(out)


class TestBatchOracleParity:
    """Vectorized batch evaluators == scalar oracle, per sample."""

    def test_value_grad_hess_match_scalar(self):
        rng = np.random.default_rng(3)
        T = _random_valid_T(rng, 50)
        mu_b = _mu_batch(T, "shape_size")
        g_b = _grad_T(T, "shape_size")
        h_b = _hess_T(T, "shape_size")
        for p in range(T.shape[0]):
            mu_s, g_s = metric_value_and_grad(T[p], metric="shape_size")
            np.testing.assert_allclose(mu_b[p], mu_s, rtol=1e-12)
            np.testing.assert_allclose(g_b[p], g_s, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(
                h_b[p], _shape_size_hess_T(T[p]), rtol=1e-10, atol=1e-12
            )

    def test_unknown_metric_raises(self):
        T = np.eye(2)[None]
        with pytest.raises(ValueError, match="Unknown metric"):
            _mu_batch(T, "nope")


@needs_cpp
class TestCppOracleParity:
    """C++ closed forms (cpp_core.metric_eval) == NumPy oracle."""

    def test_metric_eval_shape_size(self):
        from egg._cpp import cpp_core
        from tests.real_tol import real_tol

        tol = real_tol(1e-9)
        rng = np.random.default_rng(11)
        for T in _random_valid_T(rng, 50):
            mu_c, g_c, h_c = cpp_core.metric_eval(T.reshape(-1), metric="shape_size")
            mu_p, g_p = metric_value_and_grad(T, metric="shape_size")
            h_p = _shape_size_hess_T(T)
            assert abs(mu_c - mu_p) <= tol * (1.0 + abs(mu_p))
            np.testing.assert_allclose(
                np.asarray(g_c), g_p.reshape(-1), rtol=tol, atol=tol
            )
            np.testing.assert_allclose(
                np.asarray(h_c).reshape(4, 4), h_p, rtol=tol, atol=tol
            )

    def test_metric_eval_default_is_shape_2d(self):
        from egg._cpp import cpp_core

        t = np.array([1.0, 0.2, -0.1, 1.3])
        mu_default, _, _ = cpp_core.metric_eval(t)
        mu_named, _, _ = cpp_core.metric_eval(t, metric="shape_2d")
        assert mu_default == mu_named

    def test_metric_eval_unknown_metric_raises(self):
        from egg._cpp import cpp_core

        with pytest.raises(ValueError, match="unknown metric"):
            cpp_core.metric_eval(np.eye(2).reshape(-1), metric="nope")


def _mismatched_grid(res=(6, 6)):
    """Two blocks of equal resolution but 3x different width: L (x 0..1) and
    R (x 1..4), sharing the x=1 edge. Only the four outline corners are
    fixed — the interface (corners B/C included) is free, so its position is
    a flat direction of the scale-invariant shape energy (any split gives
    per-block-uniform optima) while the size-aware metric pulls it toward
    the equal-area split at x=2."""
    b = TopologyBuilder(d=2)
    for name, pos, fixed in [
        ("A", (0.0, 0.0), True),
        ("D", (0.0, 2.0), True),
        ("B", (1.0, 0.0), False),
        ("C", (1.0, 2.0), False),
        ("E", (4.0, 0.0), True),
        ("F", (4.0, 2.0), True),
    ]:
        b.add_corner(name, pos, fixed=fixed)
    b.add_block("L", ("A", "D", "B", "C"), res)
    b.add_block("R", ("B", "C", "E", "F"), res)
    b.connect("L", 0, 1, "R", 0, 0)
    return b.build().initialize_grid()


def _cell_area_cv(grid) -> float:
    areas = []
    for blk in grid.blocks:
        n = np.asarray(blk.nodes)[..., :2]
        p00, p10 = n[:-1, :-1], n[1:, :-1]
        p11, p01 = n[1:, 1:], n[:-1, 1:]
        a = 0.5 * (
            (p00[..., 0] * p10[..., 1] - p10[..., 0] * p00[..., 1])
            + (p10[..., 0] * p11[..., 1] - p11[..., 0] * p10[..., 1])
            + (p11[..., 0] * p01[..., 1] - p01[..., 0] * p11[..., 1])
            + (p01[..., 0] * p00[..., 1] - p00[..., 0] * p01[..., 1])
        )
        areas.append(a.ravel())
    areas = np.concatenate(areas)
    return float(areas.std() / areas.mean())


@needs_cpp
class TestShapeSizeSweeps:
    """The structured backend under phase="shape_size" equalises cell sizes
    where the shape phase (scale-invariant by construction) cannot."""

    def test_shape_size_equalises_where_shape_cannot(self):
        from egg.smoothing.cpp_backend import cpp_structured_sweep
        from egg.smoothing.solver import build_sweep_context
        from egg.smoothing.targets import IdentityTarget, mean_size_target

        results = {}
        for phase in ("barrier", "shape_size"):
            grid = _mismatched_grid()
            target = (
                mean_size_target(grid) if phase == "shape_size" else IdentityTarget(2)
            )
            ctx = build_sweep_context(grid, target)
            X_out, energies, mindets = cpp_structured_sweep(
                ctx,
                grid,
                np.asarray(grid.global_nodes),
                400,
                device="cpu",
                phase=phase,
                omega=0.8,
                report_every=1,
            )
            grid.global_nodes = X_out
            for bi, blk in enumerate(grid.blocks):
                blk.nodes[...] = X_out[grid.block_dof_maps[bi]]
            assert float(np.min(mindets)) > 0.0
            e = np.asarray(energies)
            # Per-sweep energies are monotone up to reduction-order noise.
            assert float(e[-1]) <= float(e[0])
            results[phase] = _cell_area_cv(grid)

        # The initial TFI grid has CV ~= 0.5 (3x width mismatch). The shape
        # phase keeps it (scale-invariant); shape_size must collapse it.
        assert results["barrier"] > 0.3
        assert results["shape_size"] < 0.5 * results["barrier"]

    def test_pipeline_knob_and_default_target(self):
        """tmop_metric="shape_size" through run_pipeline: the default
        mean-size target is built internally and the grid ends more
        size-uniform than the shape run of the same budget."""
        from egg.pipeline import PipelineConfig, run_pipeline

        cvs = {}
        for metric in ("shape", "shape_size"):
            grid = _mismatched_grid()
            cfg = PipelineConfig(
                tmop_sweeps=200,
                tmop_chunk=100,
                tmop_metric=metric,
                device="cpu",
            )
            run_pipeline(grid, None, cfg)
            cvs[metric] = _cell_area_cv(grid)
        assert cvs["shape_size"] < 0.5 * cvs["shape"]

    def test_fas_accepts_shape_size(self):
        """FAS V-cycles run under the shape_size objective (falling back to
        Jacobi when nothing coarsens) and keep the mesh valid."""
        from egg.pipeline import PipelineConfig, run_pipeline
        from egg.smoothing.untangle import grid_min_det
        from egg.smoothing.solver import build_sweep_context
        from egg.smoothing.targets import IdentityTarget

        grid = _mismatched_grid(res=(12, 12))
        cfg = PipelineConfig(
            tmop_sweeps=20,
            tmop_chunk=10,
            tmop_smoother="fas",
            tmop_metric="shape_size",
            device="cpu",
        )
        run_pipeline(grid, None, cfg)
        es = build_sweep_context(grid, IdentityTarget(2)).energy_stencil
        assert grid_min_det(np.asarray(grid.global_nodes), es) > 0.0

    def test_bad_metric_rejected(self):
        from egg.pipeline import PipelineConfig

        with pytest.raises(ValueError, match="tmop_metric"):
            PipelineConfig(tmop_metric="sizeshape").validate()
