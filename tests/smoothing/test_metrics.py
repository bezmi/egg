# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Tests for TMOP quality metrics."""

import numpy as np

from egg.smoothing.metrics import (
    shape_metric,
    shape_metric_2d,
    shape_size_metric,
    untangle_surrogate,
    metric_value_and_grad,
)


class TestShapeMetric:
    def test_mu_I_is_zero_2d(self):
        assert shape_metric(np.eye(2)) < 1e-14

    def test_mu_I_is_zero_3d(self):
        assert shape_metric(np.eye(3)) < 1e-14

    def test_mu_nonnegative_2d(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            T = rng.normal(size=(2, 2))
            T = T + 2 * np.eye(2)
            assert shape_metric(T) >= -1e-14

    def test_mu_nonnegative_3d(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            T = rng.normal(size=(3, 3))
            T = T + 3 * np.eye(3)
            assert shape_metric(T) >= -1e-14

    def test_rotation_invariance(self):
        T = np.array([[2.0, 0.0], [0.0, 0.5]])
        theta = 0.7
        Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        T_rot = Q @ T @ Q.T
        np.testing.assert_allclose(shape_metric(T), shape_metric(T_rot), atol=1e-12)

    def test_scale_invariance(self):
        T = np.array([[2.0, 0.0], [0.0, 0.5]])
        np.testing.assert_allclose(shape_metric(T), shape_metric(3.0 * T), atol=1e-12)

    def test_barrier_blowup(self):
        """μ → ∞ as det(T) → 0⁺."""
        T = np.array([[1.0, 0.0], [0.0, 1e-12]])
        mu = shape_metric(T)
        assert mu > 1e10

    def test_dimension_parity(self):
        """Same metric function works for d=2 and d=3."""
        assert shape_metric(np.eye(2)) < 1e-14
        assert shape_metric(np.eye(3)) < 1e-14
        T2 = np.diag([2.0, 0.5])
        T3 = np.diag([2.0, 0.5, 1.0])
        assert np.isfinite(shape_metric(T2))
        assert np.isfinite(shape_metric(T3))


class TestShapeMetric2D:
    def test_mu_I_is_zero(self):
        assert shape_metric_2d(np.eye(2)) < 1e-14

    def test_mu_nonnegative(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            T = rng.normal(size=(2, 2))
            T = T + 2 * np.eye(2)
            assert shape_metric_2d(T) >= -1e-14

    def test_rotation_invariance(self):
        T = np.array([[2.0, 0.0], [0.0, 0.5]])
        theta = 0.7
        Q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        T_rot = Q @ T @ Q.T
        np.testing.assert_allclose(
            shape_metric_2d(T), shape_metric_2d(T_rot), atol=1e-12
        )

    def test_scale_invariance(self):
        T = np.array([[2.0, 0.0], [0.0, 0.5]])
        np.testing.assert_allclose(
            shape_metric_2d(T), shape_metric_2d(3.0 * T), atol=1e-12
        )


class TestShapeSizeMetric:
    def test_mu_I_is_zero(self):
        assert shape_size_metric(np.eye(2)) < 1e-14

    def test_not_scale_invariant(self):
        """Shape+size IS sensitive to scale via (det-1)² term."""
        T = np.array([[2.0, 0.0], [0.0, 0.5]])
        mu_T = shape_size_metric(T)
        mu_2T = shape_size_metric(2.0 * T)
        assert abs(mu_T - mu_2T) > 1e-6

    def test_det_one_gives_zero_size_term(self):
        """When det(T) = 1, the size term is zero."""
        T = np.array([[1.5, 0.0], [0.0, 1.0 / 1.5]])
        np.testing.assert_allclose(shape_size_metric(T), shape_metric(T), atol=1e-14)


class TestUntangleSurrogate:
    def test_h_positive(self):
        assert untangle_surrogate(-10.0, 0.5) > 0
        assert untangle_surrogate(0.0, 0.5) > 0
        assert untangle_surrogate(10.0, 0.5) > 0

    def test_h_converges_as_delta_zero(self):
        """h(τ, δ) → τ as δ → 0 for τ > 0."""
        np.testing.assert_allclose(untangle_surrogate(2.0, 0.0), 2.0, atol=1e-14)
        np.testing.assert_allclose(untangle_surrogate(2.0, 1e-10), 2.0, atol=1e-8)

    def test_h_at_negative_tau(self):
        """h is finite and positive for τ < 0."""
        h = untangle_surrogate(-5.0, 1.0)
        assert h > 0
        assert np.isfinite(h)


class TestGradient:
    def test_grad_at_identity_zero(self):
        _, grad = metric_value_and_grad(np.eye(2), "shape")
        np.testing.assert_allclose(grad, np.zeros((2, 2)), atol=1e-14)

    def test_grad_2d_at_identity_zero(self):
        _, grad = metric_value_and_grad(np.eye(2), "shape_2d")
        np.testing.assert_allclose(grad, np.zeros((2, 2)), atol=1e-14)

    def test_grad_finite_difference_consistency(self):
        """Autodiff gradient ≈ finite-difference gradient."""
        T = np.array([[1.5, 0.3], [-0.1, 0.8]])
        val, grad_analytic = metric_value_and_grad(T, "shape")
        eps = 1e-6
        grad_fd = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                T_pert = T.copy()
                T_pert[i, j] += eps
                val_pert = shape_metric(T_pert)
                grad_fd[i, j] = (val_pert - val) / eps
        np.testing.assert_allclose(grad_analytic, grad_fd, rtol=1e-4)
