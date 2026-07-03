"""mu(T): shape, shape+size, barrier + untangling variants.

All metrics are pure scalar functions of T = A W^{-1}.
Derivatives use closed-form analytic expressions (pure NumPy).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "shape_metric",
    "shape_metric_2d",
    "shape_size_metric",
    "untangle_surrogate",
    "metric_value_and_grad",
    "metric_value",
]

# MIRROR: keep in sync with egg.smoothing.batch._HESS_P (Python) and
# egg::kHessP in src/metric.hpp (C++). Three copies across two languages.
_HESS_P = np.array(
    [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
)


def _shape2d_value(T: np.ndarray) -> float:
    """Classic 2D shape metric: |T|^2 / (2*det(T)) - 1."""
    a, b = T[0, 0], T[0, 1]
    c, d = T[1, 0], T[1, 1]
    D = a * d - b * c
    s = a * a + b * b + c * c + d * d
    return float(s / (2.0 * D) - 1.0)


def _shape2d_value_grad(T: np.ndarray) -> tuple[float, np.ndarray]:
    """Return (mu, dmu/dT) in 2x2 layout matching T."""
    a, b = T[0, 0], T[0, 1]
    c, d = T[1, 0], T[1, 1]
    D = a * d - b * c
    s = a * a + b * b + c * c + d * d
    val = s / (2.0 * D) - 1.0
    coef = s / (2.0 * D * D)
    grad = np.array(
        [[a / D - coef * d, b / D + coef * c], [c / D + coef * b, d / D - coef * a]]
    )
    return float(val), grad


def _shape2d_hess_T(T: np.ndarray) -> np.ndarray:
    """d^2 mu / d vec(T)^2 as a 4x4 matrix, vec(T) = [T00, T01, T10, T11]."""
    a, b = T[0, 0], T[0, 1]
    c, d = T[1, 0], T[1, 1]
    D = a * d - b * c
    s = a * a + b * b + c * c + d * d
    t = np.array([a, b, c, d])
    cof = np.array([d, -c, -b, a])
    H = (
        (1.0 / D) * np.eye(4)
        - (1.0 / (D * D)) * (np.outer(t, cof) + np.outer(cof, t))
        + (s / (D**3)) * np.outer(cof, cof)
        - (s / (2.0 * D * D)) * _HESS_P
    )
    return H


# --- shape metric (general-d condition number, |T|^2 * |T^{-1}|^2 / d^2 - 1) ---
#
# For d=2: mu = s^2 / (4 D^2) - 1  with s = |T|^2, D = det(T).
# Gradient:  dmu/dT = s*T/D^2 - (s^2/(2 D^3)) * cof(T)
# Hessian:   H = 2 * g_c * g_c^T + 2*(mu_c+1) * H_c
#             where g_c, H_c are the classic 2D shape metric gradient/Hessian.


def _shape_value(T: np.ndarray) -> float:
    d = T.shape[-1]
    if d == 2:
        a, b = T[0, 0], T[0, 1]
        c, d_ = T[1, 0], T[1, 1]
        D = a * d_ - b * c
        s = a * a + b * b + c * c + d_ * d_
        return float(s * s / (4.0 * D * D) - 1.0)
    T_inv = np.linalg.inv(T)
    s = np.sum(T * T)
    u = np.sum(T_inv * T_inv)
    return float(s * u / (d * d) - 1.0)


def _shape_value_grad(T: np.ndarray) -> tuple[float, np.ndarray]:
    """Shape metric value and gradient (d=2 closed-form)."""
    if T.shape[-1] != 2:
        raise NotImplementedError("Shape gradient only implemented for d=2")
    a, b = T[0, 0], T[0, 1]
    c, d_ = T[1, 0], T[1, 1]
    D = a * d_ - b * c
    s = a * a + b * b + c * c + d_ * d_
    val = s * s / (4.0 * D * D) - 1.0
    coef_s = s / (D * D)
    coef_cof = s * s / (2.0 * D * D * D)
    grad = np.array(
        [
            [coef_s * a - coef_cof * d_, coef_s * b + coef_cof * c],
            [coef_s * c + coef_cof * b, coef_s * d_ - coef_cof * a],
        ]
    )
    return float(val), grad


def _shape_hess_T(T: np.ndarray) -> np.ndarray:
    """Hessian of shape metric w.r.t. vec(T) (d=2 closed-form)."""
    if T.shape[-1] != 2:
        raise NotImplementedError("Shape Hessian only implemented for d=2")
    a, b = T[0, 0], T[0, 1]
    c, d_ = T[1, 0], T[1, 1]
    D = a * d_ - b * c
    s = a * a + b * b + c * c + d_ * d_
    t = np.array([a, b, c, d_])
    cof = np.array([d_, -c, -b, a])
    g_c = t / D - s * cof / (2.0 * D * D)
    H_c = _shape2d_hess_T(T)
    mu_c_plus_1 = s / (2.0 * D)
    H = 2.0 * np.outer(g_c, g_c) + 2.0 * mu_c_plus_1 * H_c
    return H


# --- shape+size metric: shape + (det(T)-1)^2 ---


def _shape_size_value(T: np.ndarray) -> float:
    if T.shape[-1] != 2:
        raise NotImplementedError("Shape+size value not implemented for d!=2")
    a, b = T[0, 0], T[0, 1]
    c, d_ = T[1, 0], T[1, 1]
    D = a * d_ - b * c
    s = a * a + b * b + c * c + d_ * d_
    return float(s * s / (4.0 * D * D) - 1.0 + (D - 1.0) ** 2)


def _shape_size_value_grad(T: np.ndarray) -> tuple[float, np.ndarray]:
    """Shape+size metric value and gradient (d=2 closed-form)."""
    if T.shape[-1] != 2:
        raise NotImplementedError("Shape+size gradient only implemented for d=2")
    a, b = T[0, 0], T[0, 1]
    c, d_ = T[1, 0], T[1, 1]
    D = a * d_ - b * c
    s = a * a + b * b + c * c + d_ * d_
    val = s * s / (4.0 * D * D) - 1.0 + (D - 1.0) ** 2
    coef_s = s / (D * D)
    coef_cof = s * s / (2.0 * D * D * D)
    cof = np.array([[d_, -c], [-b, a]])
    grad_g = coef_s * T - coef_cof * cof
    grad_size = 2.0 * (D - 1.0) * cof
    return float(val), grad_g + grad_size


def _shape_size_hess_T(T: np.ndarray) -> np.ndarray:
    """Hessian of shape+size w.r.t. vec(T) (d=2 closed-form)."""
    if T.shape[-1] != 2:
        raise NotImplementedError("Shape+size Hessian only implemented for d=2")
    a, b = T[0, 0], T[0, 1]
    c, d_ = T[1, 0], T[1, 1]
    D = a * d_ - b * c
    cof = np.array([d_, -c, -b, a])
    H_g = _shape_hess_T(T)
    H = H_g + 2.0 * np.outer(cof, cof) + 2.0 * (D - 1.0) * _HESS_P
    return H


# --- internal helpers ---


def _untangle_surrogate(det_T: np.ndarray | float, delta: float) -> np.ndarray | float:
    """Smooth positive surrogate: h(tau) = 0.5*(tau + sqrt(tau^2 + 4*delta^2))."""
    return 0.5 * (det_T + np.sqrt(det_T * det_T + 4.0 * delta * delta))


def _get_hess_T_fn(metric: str):
    if metric == "shape_2d":
        return _shape2d_hess_T
    if metric == "shape":
        return _shape_hess_T
    if metric == "shape_size":
        return _shape_size_hess_T
    raise ValueError(f"Unknown metric: {metric}")


# --- public API ---


def shape_metric(T: np.ndarray) -> float:
    """Condition-number shape metric (any d)."""
    return _shape_value(np.asarray(T))


def shape_metric_2d(T: np.ndarray) -> float:
    """Classic 2D shape metric."""
    return _shape2d_value(np.asarray(T))


def shape_size_metric(T: np.ndarray) -> float:
    """Shape + size metric."""
    return _shape_size_value(np.asarray(T))


def untangle_surrogate(det_T: float, delta: float) -> float:
    """Smooth positive surrogate."""
    return float(_untangle_surrogate(float(det_T), delta))


def metric_value(T: np.ndarray, metric: str = "shape") -> float:
    """Evaluate a named metric on T."""
    T = np.asarray(T)
    if metric == "shape_2d":
        return _shape2d_value(T)
    if metric == "shape":
        return _shape_value(T)
    if metric == "shape_size":
        return _shape_size_value(T)
    raise ValueError(f"Unknown metric: {metric}")


def metric_value_and_grad(
    T: np.ndarray, metric: str = "shape"
) -> tuple[float, np.ndarray]:
    """Evaluate a named metric and its gradient dmu/dT (shape (d, d))."""
    T = np.asarray(T)
    if metric == "shape_2d":
        return _shape2d_value_grad(T)
    if metric == "shape":
        return _shape_value_grad(T)
    if metric == "shape_size":
        return _shape_size_value_grad(T)
    raise ValueError(f"Unknown metric: {metric}")


# --- Jacobian helpers ---


def _corner_jacobian_chain(s0: int, s1: int, W_inv: np.ndarray) -> np.ndarray:
    """Constant J = d vec(T) / d coords, shape (4, 6).

    coords = [corner_x, corner_y, nbr0_x, nbr0_y, nbr1_x, nbr1_y];
    A[:, k] = s_k * (nbr_k - corner); T = A @ W_inv;
    vec(T) = [T00, T01, T10, T11] (row-major).
    """
    s0, s1 = int(s0), int(s1)
    wi = np.asarray(W_inv)
    dA = np.array(
        [
            [-s0, 0.0, s0, 0.0, 0.0, 0.0],
            [-s1, 0.0, 0.0, 0.0, s1, 0.0],
            [0.0, -s0, 0.0, s0, 0.0, 0.0],
            [0.0, -s1, 0.0, 0.0, 0.0, s1],
        ]
    )
    M = np.array(
        [
            [wi[0, 0], wi[1, 0], 0.0, 0.0],
            [wi[0, 1], wi[1, 1], 0.0, 0.0],
            [0.0, 0.0, wi[0, 0], wi[1, 0]],
            [0.0, 0.0, wi[0, 1], wi[1, 1]],
        ]
    )
    return M @ dA


# --- Cell-level Hessians (2D, analytic NumPy) ---


def cell_hessian_2d(
    base: np.ndarray,
    nb0: np.ndarray,
    nb1: np.ndarray,
    W_inv: np.ndarray,
    metric: str = "shape_2d",
) -> np.ndarray:
    """Compute 6x6 Hessian of mu w.r.t. cell node coordinates.

    Nodes are: [base, neighbor_axis0, neighbor_axis1].
    Returns (6, 6) hessian matrix.
    """
    base = np.asarray(base)
    nb0 = np.asarray(nb0)
    nb1 = np.asarray(nb1)
    wi = np.asarray(W_inv)

    A = np.array(
        [[nb0[0] - base[0], nb1[0] - base[0]], [nb0[1] - base[1], nb1[1] - base[1]]]
    )
    T = A @ wi
    H_T = _get_hess_T_fn(metric)(T)
    dA = np.array(
        [
            [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    M = np.array(
        [
            [wi[0, 0], wi[1, 0], 0.0, 0.0],
            [wi[0, 1], wi[1, 1], 0.0, 0.0],
            [0.0, 0.0, wi[0, 0], wi[1, 0]],
            [0.0, 0.0, wi[0, 1], wi[1, 1]],
        ]
    )
    J = M @ dA
    return J.T @ H_T @ J


def corner_cell_hessian_2d(
    corner: np.ndarray,
    nbr0: np.ndarray,
    nbr1: np.ndarray,
    s0: int,
    s1: int,
    W_inv: np.ndarray,
    metric: str = "shape_2d",
) -> np.ndarray:
    """6x6 Hessian of mu for a single corner sample (analytic, NumPy).

    Node order in the Hessian is [corner, nbr0, nbr1].
    Uses J^T @ H_T @ J with analytic H_T and constant J.
    """
    corner = np.asarray(corner)
    nbr0 = np.asarray(nbr0)
    nbr1 = np.asarray(nbr1)
    s0, s1 = int(s0), int(s1)
    wi = np.asarray(W_inv)

    A = np.array(
        [
            [s0 * (nbr0[0] - corner[0]), s1 * (nbr1[0] - corner[0])],
            [s0 * (nbr0[1] - corner[1]), s1 * (nbr1[1] - corner[1])],
        ]
    )
    T = A @ wi
    H_T = _get_hess_T_fn(metric)(T)
    J = _corner_jacobian_chain(s0, s1, wi)
    return J.T @ H_T @ J


# Alias preserved for backward compatibility (formerly JAX-based, now identical).
_corner_cell_hessian_2d_jax = corner_cell_hessian_2d
