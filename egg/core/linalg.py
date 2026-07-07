# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""d×d Jacobian, determinant, inverse, Frobenius norm helpers.

All functions accept (..., d, d) arrays for batch operation over the leading axes.
"""

import numpy as np

__all__ = ["det", "inv", "frobenius_norm_sq", "condition_number"]


def det(A: np.ndarray) -> np.ndarray:
    """Determinant of the last two axes (d×d matrix/matrices)."""
    return np.linalg.det(A)


def inv(A: np.ndarray) -> np.ndarray:
    """Inverse of the last two axes."""
    return np.linalg.inv(A)


def frobenius_norm_sq(A: np.ndarray) -> np.ndarray:
    """|A|_F^2 = sum of squares of all elements over the last two axes."""
    return np.sum(A * A, axis=(-2, -1))


def condition_number(A: np.ndarray) -> np.ndarray:
    """|A|_F · |A^{-1}|_F / d. Safe for singular matrices (returns inf)."""
    d = A.shape[-1]
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return np.asarray(np.inf)
    norm_A = np.sqrt(frobenius_norm_sq(A))
    norm_Ainv = np.sqrt(frobenius_norm_sq(A_inv))
    with np.errstate(invalid="ignore"):
        result = np.where(
            np.isfinite(norm_A) & np.isfinite(norm_Ainv),
            norm_A * norm_Ainv / d,
            np.inf,
        )
    return np.asarray(result)
