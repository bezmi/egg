# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Vectorized closed-form patch evaluation (``shape_2d`` / ``shape_size``).

The patch relaxation re-evaluates a node's *patch* — every corner sample
of every cell incident to that node — many times per sweep (once for the
gradient/Hessian, then once per backtracking trial). Doing that with a Python
loop over corners, each building tiny 2x2 arrays via tuple indexing, is the hot
spot. These functions evaluate **all P corner samples of a patch at once** with
batched NumPy, reading positions straight from the flat ``global_nodes`` array
through precomputed integer index stencils.

Every evaluator takes ``metric`` ("shape_2d" default, or the size-aware
"shape_size"), matching :mod:`egg.smoothing.metrics` per sample.

A corner sample is ``A[:, k] = s_k * (nbr_k - corner)`` with ``T = A @ W_inv``
(see :func:`egg.smoothing.jacobian.corner_sample`). ``vec(T) = [T00, T01,
T10, T11]`` (row-major) throughout, matching :mod:`egg.smoothing.metrics`.
"""

from __future__ import annotations

import numpy as np

# d vec(C)/d vec(T) with cofactor vec(C) = [d, -c, -b, a].
# MIRROR: keep in sync with egg.smoothing.metrics._HESS_P (Python) and
# egg::kHessP in src/metric.hpp (C++). Three copies across two languages; any
# change must be applied to all three.
_HESS_P = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
)
_I4 = np.eye(4)


def assemble_A(X, gc, gn0, gn1, s0, s1):
    """Batched Jacobians A (P, 2, 2) for P corner samples.

    ``A[:, :, 0]`` is column 0 = ``s0 * (nbr0 - corner)``; ``A[:, :, 1]`` column 1.
    """
    Xc = X[gc]
    A = np.empty((gc.shape[0], 2, 2))
    A[:, :, 0] = s0[:, None] * (X[gn0] - Xc)
    A[:, :, 1] = s1[:, None] * (X[gn1] - Xc)
    return A


def _detA(A):
    return A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]


def _mu_batch(T, metric):
    """Batched metric values, shape (P,)."""
    a, b = T[:, 0, 0], T[:, 0, 1]
    c, d = T[:, 1, 0], T[:, 1, 1]
    D = a * d - b * c
    s = a * a + b * b + c * c + d * d
    cond = s / (2.0 * D)
    if metric == "shape_2d":
        return cond - 1.0
    if metric == "shape_size":
        return cond * cond - 1.0 + (D - 1.0) ** 2
    raise ValueError(f"Unknown metric: {metric}")


def energy_and_mindet(X, gc, gn0, gn1, s0, s1, W_inv, metric="shape_2d"):
    """Sum of mu and min det(A) over a patch's P corner samples."""
    A = assemble_A(X, gc, gn0, gn1, s0, s1)
    det_A = _detA(A)
    T = np.einsum("pij,pjk->pik", A, W_inv)
    return float(_mu_batch(T, metric).sum()), float(det_A.min())


def _grad_T(T, metric="shape_2d"):
    """dmu/dT for a batch, shape (P, 2, 2)."""
    a, b = T[:, 0, 0], T[:, 0, 1]
    c, d = T[:, 1, 0], T[:, 1, 1]
    D = a * d - b * c
    s = a * a + b * b + c * c + d * d
    g = np.empty_like(T)
    if metric == "shape_2d":
        coef = s / (2.0 * D * D)
        g[:, 0, 0] = a / D - coef * d
        g[:, 0, 1] = b / D + coef * c
        g[:, 1, 0] = c / D + coef * b
        g[:, 1, 1] = d / D - coef * a
    elif metric == "shape_size":
        coef_s = s / (D * D)
        coef_c = s * s / (2.0 * D**3) - 2.0 * (D - 1.0)
        g[:, 0, 0] = coef_s * a - coef_c * d
        g[:, 0, 1] = coef_s * b + coef_c * c
        g[:, 1, 0] = coef_s * c + coef_c * b
        g[:, 1, 1] = coef_s * d - coef_c * a
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return g


def _hess_T(T, metric="shape_2d"):
    """d^2 mu / d vec(T)^2 for a batch, shape (P, 4, 4)."""
    a, b = T[:, 0, 0], T[:, 0, 1]
    c, d = T[:, 1, 0], T[:, 1, 1]
    D = a * d - b * c
    s = a * a + b * b + c * c + d * d
    t = np.stack([a, b, c, d], axis=1)  # (P, 4)
    cof = np.stack([d, -c, -b, a], axis=1)  # (P, 4)
    tc = t[:, :, None] * cof[:, None, :]  # outer(t, cof)
    cc = cof[:, :, None] * cof[:, None, :]  # outer(cof, cof)
    inv_D = (1.0 / D)[:, None, None]
    inv_D2 = (1.0 / (D * D))[:, None, None]
    s_D3 = (s / (D**3))[:, None, None]
    s_2D2 = (s / (2.0 * D * D))[:, None, None]
    H2 = (
        inv_D * _I4[None]
        - inv_D2 * (tc + tc.transpose(0, 2, 1))
        + s_D3 * cc
        - s_2D2 * _HESS_P[None]
    )
    if metric == "shape_2d":
        return H2
    if metric == "shape_size":
        # H = 2 g_c g_c^T + 2 (s/2D) H2 + 2 cof cof^T + 2(D-1) P, with g_c/H2
        # the shape_2d closed forms (mirrors metrics._shape_size_hess_T).
        g_c = _grad_T(T, "shape_2d").reshape(-1, 4)
        gg = g_c[:, :, None] * g_c[:, None, :]
        cond = (s / (2.0 * D))[:, None, None]
        dm1 = (2.0 * (D - 1.0))[:, None, None]
        return 2.0 * gg + 2.0 * cond * H2 + 2.0 * cc + dm1 * _HESS_P[None]
    raise ValueError(f"Unknown metric: {metric}")


def make_chain_J_nd(S, W_inv):
    """Constant J = d vec(T)/d coords per sample, any d; shape (P, d², d(d+1)).

    Generalizes :func:`make_chain_J` to arbitrary spatial dimension.
    ``S`` is the (P, d) per-axis sign array; ``W_inv`` is (P, d, d).
    coords = [corner(d), nbr_0(d), ..., nbr_{d-1}(d)];
    A[i, k] = s_k (nbr_k[i] - corner[i]); T = A @ W_inv (vec(T) row-major), so
    J[i*d+c, j] = sum_k W_inv[k, c] * dA[i, k]/d coord[j].
    """
    S = np.asarray(S, dtype=np.float64)
    W_inv = np.asarray(W_inv, dtype=np.float64)
    P, d = S.shape
    n_coords = d * (d + 1)
    dA = np.zeros((P, d, d, n_coords))
    for i in range(d):
        for k in range(d):
            dA[:, i, k, i] = -S[:, k]  # d/d corner[i]
            dA[:, i, k, (k + 1) * d + i] = S[:, k]  # d/d nbr_k[i]
    # J[p, i*d+c, j] = sum_k W_inv[p, k, c] dA[p, i, k, j]
    J = np.einsum("pkc,pikj->picj", W_inv, dA).reshape(P, d * d, n_coords)
    return J


def make_chain_J(s0, s1, W_inv):
    """Precompute constant J = d vec(T)/d coords per sample, shape (P, 4, 6).

    coords = [corner_x, corner_y, nbr0_x, nbr0_y, nbr1_x, nbr1_y].
    Depends only on the sign vectors and target matrices, not on positions,
    so it can be cached in the sweep context.
    """
    P = s0.shape[0]
    dA = np.zeros((P, 4, 6))
    dA[:, 0, 0], dA[:, 0, 2] = -s0, s0  # A00 = s0*(nbr0_x - corner_x)
    dA[:, 1, 0], dA[:, 1, 4] = -s1, s1  # A01 = s1*(nbr1_x - corner_x)
    dA[:, 2, 1], dA[:, 2, 3] = -s0, s0  # A10 = s0*(nbr0_y - corner_y)
    dA[:, 3, 1], dA[:, 3, 5] = -s1, s1  # A11 = s1*(nbr1_y - corner_y)
    w = W_inv
    M = np.zeros((P, 4, 4))
    M[:, 0, 0], M[:, 0, 1] = w[:, 0, 0], w[:, 1, 0]
    M[:, 1, 0], M[:, 1, 1] = w[:, 0, 1], w[:, 1, 1]
    M[:, 2, 2], M[:, 2, 3] = w[:, 0, 0], w[:, 1, 0]
    M[:, 3, 2], M[:, 3, 3] = w[:, 0, 1], w[:, 1, 1]
    return np.einsum("pij,pjk->pik", M, dA)


def dof_grad_hess(X, gc, gn0, gn1, s0, s1, W_inv, role, J=None, metric="shape_2d"):
    """Gradient (2,) and Hessian (2, 2) of patch energy w.r.t. one moving DOF.

    Each sample contributes only through the column(s) the DOF participates in,
    selected by ``role`` (0 = corner, 1 = nbr0, 2 = nbr1).

    If ``J`` is provided (precomputed via :func:`make_chain_J`), the chain-rule
    Jacobian is not recomputed.
    """
    A = assemble_A(X, gc, gn0, gn1, s0, s1)
    T = np.einsum("pij,pjk->pik", A, W_inv)

    dmu_dT = _grad_T(T, metric)
    # dmu/dA = dmu/dT @ W_inv^T  -> columns col_k = dmu_dA[:, :, k]
    dmu_dA = np.einsum("pij,pkj->pik", dmu_dT, W_inv)
    col0, col1 = dmu_dA[:, :, 0], dmu_dA[:, :, 1]

    contrib = np.zeros((s0.shape[0], 2))
    is_c, is_0, is_1 = role == 0, role == 1, role == 2
    contrib[is_c] = -(s0[is_c, None] * col0[is_c] + s1[is_c, None] * col1[is_c])
    contrib[is_0] = s0[is_0, None] * col0[is_0]
    contrib[is_1] = s1[is_1, None] * col1[is_1]
    grad = contrib.sum(0)

    H_T = _hess_T(T, metric)
    if J is None:
        J = make_chain_J(s0, s1, W_inv)
    valid = role >= 0
    role_safe = np.maximum(role, 0)
    cols = (2 * role_safe)[:, None] + np.array([0, 1])[None]  # (P, 2)
    Jb = np.take_along_axis(J, cols[:, None, :], axis=2)  # (P, 4, 2)
    Jb[~valid] = 0.0
    blk = np.einsum("pai,pab,pbj->pij", Jb, H_T, Jb)
    H = blk.sum(0)
    return grad, H


def patch_eval(X, gc, gn0, gn1, s0, s1, W_inv, role, J=None, metric="shape_2d"):
    """Combined patch evaluation: (grad, hess, energy, mindet) in one pass.

    Computes ``A`` and ``T = A @ W_inv`` once, then derives the gradient,
    Hessian (2, 2), scalar energy, and min-det from them.  Numerically
    identical to calling :func:`dof_grad_hess` and :func:`energy_and_mindet`
    separately on the same position array ``X``.

    If ``J`` is provided (precomputed via :func:`make_chain_J`), the chain-rule
    Jacobian is not recomputed.
    """
    A = assemble_A(X, gc, gn0, gn1, s0, s1)
    det_A = _detA(A)
    T = np.einsum("pij,pjk->pik", A, W_inv)

    # --- energy & mindet ---
    energy = float(_mu_batch(T, metric).sum())
    mindet = float(det_A.min())

    # --- gradient ---
    dmu_dT = _grad_T(T, metric)
    dmu_dA = np.einsum("pij,pkj->pik", dmu_dT, W_inv)
    col0, col1 = dmu_dA[:, :, 0], dmu_dA[:, :, 1]

    contrib = np.zeros((s0.shape[0], 2))
    is_c, is_0, is_1 = role == 0, role == 1, role == 2
    contrib[is_c] = -(s0[is_c, None] * col0[is_c] + s1[is_c, None] * col1[is_c])
    contrib[is_0] = s0[is_0, None] * col0[is_0]
    contrib[is_1] = s1[is_1, None] * col1[is_1]
    grad = contrib.sum(0)

    # --- hessian ---
    H_T = _hess_T(T, metric)
    if J is None:
        J = make_chain_J(s0, s1, W_inv)
    valid = role >= 0
    role_safe = np.maximum(role, 0)
    cols = (2 * role_safe)[:, None] + np.array([0, 1])[None]
    Jb = np.take_along_axis(J, cols[:, None, :], axis=2)
    Jb[~valid] = 0.0
    blk = np.einsum("pai,pab,pbj->pij", Jb, H_T, Jb)
    hess = blk.sum(0)

    return grad, hess, energy, mindet
