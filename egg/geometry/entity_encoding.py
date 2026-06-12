"""Geometry entity encoding for constrained DOFs.

Encodes a geometry entity into ``(type_tag, params)`` for use by the C++ core
and the NumPy solver. ``params`` is a length-``PARAM_PAD_SIZE`` numpy float64
array, zero-padded.

Each constrained DOF stores ``(type_tag, params)``; the spatial dimension ``d``
is implicit from the DOF's position array.

Layouts (2D; ``d`` is the spatial dimension; angles/parameters in radians):

| Entity       | tag | params layout                                         |
|--------------|-----|-------------------------------------------------------|
| free         | 0   | (unused)                                              |
| LineSegment  | 1   | ``[start(2), end(2), ...pad]``                        |
| Circle       | 2   | ``[center(2), radius, ...pad]``                       |
| Ellipse      | 3   | ``[center(2), radii(2), ...pad]``                     |
| Sphere       | 4   | ``[center(d), radius, ...pad]`` (3D surface)          |
| Plane        | 5   | ``[q(d), n(d), ...pad]`` (3D surface)                 |
| CircleArc    | 6   | ``[center(2), radius, t0, t1, closed, ...pad]``       |
| EllipseArc   | 7   | ``[center(2), a, b, phi, t0, t1, closed, ...pad]``    |
| QuadBezier   | 8   | ``[P0(2), P1(2), P2(2), t0, t1, ...pad]``             |
| CubicBezier  | 9   | ``[P0(2), P1(2), P2(2), P3(2), t0, t1]``              |
| BSpline      | 10  | ``[degree, n_ctrl, knot_off, ctrl_off, t0, t1]``      |

The arc/Bézier/B-spline entities have no Python geometry class yet (they enter
through the vector/CAD importer front-end); the tags are defined here so the C++
and Python sides share one numbering. ``encode_entity`` covers the analytic 2D
set only.

The B-spline is variable-length: its knot vector and control net live in a
per-group ``arena`` (a flat float64 array uploaded alongside ``params``), and the
blob stores only the offsets (``knot_off``, ``ctrl_off``, in float64 units) and
counts (knots = ``n_ctrl + degree + 1``; control net = ``2 * n_ctrl``, x/y
interleaved). Emitting the arena from the sweep-context builder is front-end work.
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
    "TAG_CIRCLEARC",
    "TAG_ELLIPSEARC",
    "TAG_QUADBEZIER",
    "TAG_CUBICBEZIER",
    "TAG_BSPLINE",
]

PARAM_PAD_SIZE = 12  # >= 2*d for all 2D/3D entities (3D plane needs 2*3 = 6)

TAG_FREE = 0
TAG_LINESEG = 1
TAG_CIRCLE = 2
TAG_ELLIPSE = 3
TAG_SPHERE = 4
TAG_PLANE = 5
TAG_CIRCLEARC = 6
TAG_ELLIPSEARC = 7
TAG_QUADBEZIER = 8
TAG_CUBICBEZIER = 9
TAG_BSPLINE = 10


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
