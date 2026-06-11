"""Geometry entity encoding for constrained DOFs.

Encodes a geometry entity into ``(type_tag, params)`` for use by the C++ core
and the NumPy solver. ``params`` is a length-``PARAM_PAD_SIZE`` numpy float64
array, zero-padded.

Each constrained DOF stores ``(type_tag, params)``; the spatial dimension ``d``
is implicit from the DOF's position array.

Layouts (2D; ``d`` is the spatial dimension):

| Entity      | tag | params layout                    |
|-------------|-----|----------------------------------|
| free        | 0   | (unused)                         |
| LineSegment | 1   | ``[start(d), end(d), ...pad]``   |
| Circle      | 2   | ``[center(d), radius, ...pad]``  |
| Ellipse     | 3   | ``[center(d), radii(d), ...pad]``|
| Sphere      | 4   | ``[center(d), radius, ...pad]``  |
| Plane       | 5   | ``[q(d), n(d), ...pad]``         |
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "encode_entity",
    "PARAM_PAD_SIZE",
    "TAG_FREE",
    "TAG_LINESEG",
    "TAG_CIRCLE",
    "TAG_ELLIPSE",
    "TAG_SPHERE",
    "TAG_PLANE",
]

PARAM_PAD_SIZE = 12  # >= 2*d for all 2D/3D entities (3D plane needs 2*3 = 6)

TAG_FREE = 0
TAG_LINESEG = 1
TAG_CIRCLE = 2
TAG_ELLIPSE = 3
TAG_SPHERE = 4
TAG_PLANE = 5


def encode_entity(entity, d: int = 2):
    """Encode a geometry entity into ``(type_tag, params)``.

    ``params`` is a length-``PARAM_PAD_SIZE`` numpy float64 array, zero-padded.
    Returns ``(TAG_FREE, zeros)`` for ``entity is None``.
    """
    from egg.geometry.analytic2d import Circle, Ellipse, LineSegment

    params = np.zeros(PARAM_PAD_SIZE, dtype=np.float64)
    if entity is None:
        return TAG_FREE, params
    if isinstance(entity, LineSegment):
        params[:d] = entity.start[:d]
        params[d:2 * d] = entity.end[:d]
        return TAG_LINESEG, params
    if isinstance(entity, Circle):
        params[:d] = entity.center[:d]
        params[d] = entity.radius
        return TAG_CIRCLE, params
    if isinstance(entity, Ellipse):
        params[:d] = entity.center[:d]
        params[d:2 * d] = np.array([entity.rx, entity.ry])
        return TAG_ELLIPSE, params
    raise NotImplementedError(f"Entity type {type(entity)} not encodable yet")
