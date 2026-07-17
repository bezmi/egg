# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Transfinite interpolation (Boolean sum, d-general)."""

from __future__ import annotations

from itertools import combinations, product
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from egg.core.types import Block

__all__ = ["tfi_fill_interior"]


def tfi_fill_interior(block: Block) -> None:
    """Fill a block's non-corner nodes by transfinite interpolation.

    Works in any dimension. Nodes are filled by increasing facet dimension
    (edges, then faces, then the volume), so each facet interpolates from an
    already-filled boundary. Pre-set boundary nodes (corners, curved edges,
    surface-projected faces) are kept; only NaN facet nodes are filled, while
    the top-dimensional interior is always (re)computed. Modifies
    ``block.nodes`` in place.

    Each facet fills as one vectorized Boolean sum
    (``sum_ax lerp_ax - (k-1) * cornerblend``, identical to the per-node
    ``_tfi_point`` it replaces in the hot path — grid initialization time is
    dominated by this fill at scale).
    """
    nodes = block.nodes
    d = block.d
    shape = block.logical_shape
    if d == 0:
        return

    for k in range(1, d + 1):
        for interior_axes in combinations(range(d), k):
            fixed_axes = [a for a in range(d) if a not in interior_axes]
            for ends in product(*[(0, shape[a] - 1) for a in fixed_axes]):
                sl: list = [slice(None)] * d
                for a, e in zip(fixed_axes, ends):
                    sl[a] = e
                sub = nodes[tuple(sl)]  # view: k facet axes + coords
                core = tuple(slice(1, -1) for _ in range(k))
                if k < d:
                    nan_nodes = np.isnan(sub[core]).any(axis=-1)
                    if not nan_nodes.any():
                        continue
                filled = _boolean_sum(sub)
                if k < d:
                    m = np.zeros(sub.shape[:-1], dtype=bool)
                    m[core] = nan_nodes
                    sub[m] = filled[m]
                else:
                    sub[core] = filled[core]


def _boolean_sum(sub: np.ndarray) -> np.ndarray:
    """Vectorized Boolean-sum fill of a facet from its own boundary."""
    dims = sub.ndim - 1
    xi = [np.linspace(0.0, 1.0, n) for n in sub.shape[:-1]]

    def lerp_axis(F, ax):
        sh = [1] * (dims + 1)
        sh[ax] = sub.shape[ax]
        t = xi[ax].reshape(sh)
        f0 = np.take(F, [0], axis=ax)
        f1 = np.take(F, [-1], axis=ax)
        return (1.0 - t) * f0 + t * f1

    p_sum = lerp_axis(sub, 0)
    for ax in range(1, dims):
        p_sum = p_sum + lerp_axis(sub, ax)
    corner = sub
    for ax in range(dims):
        corner = lerp_axis(corner, ax)
    return p_sum - (dims - 1) * corner


def _tfi_point(nodes, idx, shape, interior):
    """Boolean-sum TFI at ``idx`` over the facet spanned by the interior axes."""
    xi = {ax: idx[ax] / (shape[ax] - 1) for ax in interior}
    result = _corner_interp(nodes, idx, shape, interior, xi)
    for ax in interior:
        for side in (0, 1):
            fixed = 0 if side == 0 else shape[ax] - 1
            face_idx = list(idx)
            face_idx[ax] = fixed
            face_val = nodes[tuple(face_idx)]

            xi_face = dict(xi)
            xi_face[ax] = float(side)
            corner_val = _corner_interp(nodes, idx, shape, interior, xi_face)

            w = (1.0 - xi[ax]) if side == 0 else xi[ax]
            result = result + w * (face_val - corner_val)
    return result


def _corner_interp(nodes, idx, shape, interior, xi):
    """Multilinear blend of the ``2**len(interior)`` facet corners.

    Axes not in ``interior`` stay fixed at their value in ``idx``, so this
    interpolates within the facet rather than the whole block.
    """
    result = np.zeros(nodes.shape[-1])
    for offset in product((0, 1), repeat=len(interior)):
        weight = 1.0
        corner_idx = list(idx)
        for bit, ax in zip(offset, interior):
            if bit == 0:
                weight *= 1.0 - xi[ax]
                corner_idx[ax] = 0
            else:
                weight *= xi[ax]
                corner_idx[ax] = shape[ax] - 1
        result = result + weight * nodes[tuple(corner_idx)]
    return result
