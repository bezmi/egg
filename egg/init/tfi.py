# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Transfinite interpolation (Boolean sum, d-general)."""

from __future__ import annotations

from itertools import product
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
    """
    nodes = block.nodes
    d = block.d
    shape = block.logical_shape
    if d == 0:
        return

    for k in range(1, d + 1):
        for idx in product(*(range(s) for s in shape)):
            interior = tuple(ax for ax in range(d) if 0 < idx[ax] < shape[ax] - 1)
            if len(interior) != k:
                continue
            if k < d and not np.any(np.isnan(nodes[idx])):
                continue
            nodes[idx] = _tfi_point(nodes, idx, shape, interior)


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
