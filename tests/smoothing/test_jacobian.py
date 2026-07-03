"""Tests for Jacobian computation."""

from itertools import product

import numpy as np

from egg.smoothing.jacobian import (
    compute_all_jacobians,
    compute_corner_jacobian,
    compute_jacobian,
    corner_sample,
    iter_sample_points,
)


class TestJacobian:
    def test_affine_exact_2d(self):
        """FD Jacobian on affine grid matches analytic A to machine precision."""
        ni, nj = 5, 7
        nodes = np.zeros((ni, nj, 2))
        # Affine: x = 2*i + 0.5*j, y = 0.3*i + 1.5*j
        for i in range(ni):
            for j in range(nj):
                nodes[i, j, 0] = 2.0 * i + 0.5 * j
                nodes[i, j, 1] = 0.3 * i + 1.5 * j

        # Analytic Jacobian: A = [[dx/di, dx/dj], [dy/di, dy/dj]] = [[2, 0.5], [0.3, 1.5]]
        A_analytic = np.array([[2.0, 0.5], [0.3, 1.5]])

        for i in range(ni - 1):
            for j in range(nj - 1):
                A_fd = compute_jacobian(nodes, (i, j))
                np.testing.assert_allclose(A_fd, A_analytic, atol=1e-14)

    def test_affine_exact_3d(self):
        """FD Jacobian on affine 3D grid matches analytic."""
        ni, nj, nk = 4, 5, 6
        nodes = np.zeros((ni, nj, nk, 3))
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    nodes[i, j, k, 0] = 2.0 * i + 0.5 * j - 0.2 * k
                    nodes[i, j, k, 1] = 0.3 * i + 1.5 * j + 0.1 * k
                    nodes[i, j, k, 2] = -0.1 * i + 0.2 * j + 2.0 * k

        A_analytic = np.array([[2.0, 0.5, -0.2], [0.3, 1.5, 0.1], [-0.1, 0.2, 2.0]])

        for i in range(ni - 1):
            for j in range(nj - 1):
                for k in range(nk - 1):
                    A_fd = compute_jacobian(nodes, (i, j, k))
                    np.testing.assert_allclose(A_fd, A_analytic, atol=1e-14)

    def test_det_equals_area_2d(self):
        """det(A) equals cell area for a known rectangular grid."""
        ni, nj = 5, 7
        dx, dy = 0.5, 0.3
        nodes = np.zeros((ni, nj, 2))
        for i in range(ni):
            for j in range(nj):
                nodes[i, j, 0] = dx * i
                nodes[i, j, 1] = dy * j

        expected_area = dx * dy
        for i in range(ni - 1):
            for j in range(nj - 1):
                A = compute_jacobian(nodes, (i, j))
                det_A = np.linalg.det(A)
                np.testing.assert_allclose(det_A, expected_area, atol=1e-14)

    def test_det_equals_volume_3d(self):
        """det(A) equals cell volume for a known 3D grid."""
        ni, nj, nk = 4, 5, 6
        dx, dy, dz = 0.5, 0.3, 0.4
        nodes = np.zeros((ni, nj, nk, 3))
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    nodes[i, j, k, 0] = dx * i
                    nodes[i, j, k, 1] = dy * j
                    nodes[i, j, k, 2] = dz * k

        expected_vol = dx * dy * dz
        for i in range(ni - 1):
            for j in range(nj - 1):
                for k in range(nk - 1):
                    A = compute_jacobian(nodes, (i, j, k))
                    det_A = np.linalg.det(A)
                    np.testing.assert_allclose(det_A, expected_vol, atol=1e-14)

    def test_compute_all_jacobians_shape(self):
        """compute_all_jacobians returns correct shape."""
        ni, nj = 5, 7
        nodes = np.zeros((ni, nj, 2))
        all_A = compute_all_jacobians(nodes)
        assert all_A.shape == (ni - 1, nj - 1, 2, 2)

    def test_compute_all_jacobians_3d_shape(self):
        ni, nj, nk = 4, 5, 6
        nodes = np.zeros((ni, nj, nk, 3))
        all_A = compute_all_jacobians(nodes)
        assert all_A.shape == (ni - 1, nj - 1, nk - 1, 3, 3)


class TestCornerSampling:
    def test_iter_sample_points_all_corners(self):
        """Each cell yields 2**d corner samples."""
        nodes = np.zeros((4, 3, 2))  # 3x2 = 6 cells, d=2 -> 4 corners each
        pts = list(iter_sample_points(nodes))
        assert len(pts) == 6 * 4
        offsets = {o for _, o in pts if _ == (0, 0)}
        assert offsets == set(product((0, 1), repeat=2))

    def test_affine_same_at_every_corner(self):
        """On an affine cell, the corner Jacobian is identical at all corners."""
        ni, nj = 4, 5
        nodes = np.zeros((ni, nj, 2))
        for i in range(ni):
            for j in range(nj):
                nodes[i, j] = [2.0 * i + 0.5 * j, 0.3 * i + 1.5 * j]
        A_analytic = np.array([[2.0, 0.5], [0.3, 1.5]])
        for o in product((0, 1), repeat=2):
            A = compute_corner_jacobian(nodes, (1, 1), o)
            np.testing.assert_allclose(A, A_analytic, atol=1e-13)

    def test_lower_left_corner_matches_forward_diff(self):
        """Corner sample at offset (0,0) equals the forward-difference Jacobian."""
        rng = np.random.default_rng(1)
        nodes = rng.standard_normal((3, 3, 2))
        np.testing.assert_allclose(
            compute_corner_jacobian(nodes, (1, 1), (0, 0)),
            compute_jacobian(nodes, (1, 1)),
            atol=1e-14,
        )

    def test_fold_detected_at_nonbase_corner(self):
        """A cell folded at its upper-right corner is invisible to the lower-left
        sample but is caught by all-corner sampling."""
        nodes = np.zeros((2, 2, 2))
        nodes[0, 0] = [0, 0]
        nodes[1, 0] = [1, 0]
        nodes[0, 1] = [0, 1]
        nodes[1, 1] = [-0.3, -0.3]  # fold UR node across
        ll_det = np.linalg.det(compute_jacobian(nodes, (0, 0)))
        corner_dets = [
            np.linalg.det(compute_corner_jacobian(nodes, (0, 0), o))
            for o in product((0, 1), repeat=2)
        ]
        assert ll_det > 0  # lower-left corner is blind to the fold
        assert min(corner_dets) < 0  # but some corner sees it

    def test_corner_sample_stencil(self):
        """corner_sample returns the corner node plus one in-cell neighbour per axis."""
        nodes = np.zeros((3, 3, 2))
        for i in range(3):
            for j in range(3):
                nodes[i, j] = [i, j]
        # Upper-right corner of cell (1,1): offset (1,1) -> neighbours step inward
        A, corner_idx, nbrs = corner_sample(nodes, (1, 1), (1, 1))
        assert corner_idx == (2, 2)
        assert {idx for idx, _ in nbrs} == {(1, 2), (2, 1)}
        assert all(s == -1 for _, s in nbrs)  # both neighbours are -e_k
        np.testing.assert_allclose(A, np.eye(2), atol=1e-14)
