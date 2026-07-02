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
| BSpline      | 10  | ``[degree, n_ctrl, knot_off, ctrl_off, t0, t1, w_off, has_w]`` |
| Composite    | 11  | ``[n_segs, rec_off]``                                 |

A B-spline with ``has_w != 0`` is rational (NURBS): the ``n_ctrl`` weights live
in the arena at ``w_off``.

The B-spline and composite path are variable-length: their data lives in a
shared ``arena`` (a flat float64 array uploaded alongside ``params``), and the
blob stores only offsets (in float64 units) and counts. The B-spline arena data
is the knot vector (length ``n_ctrl + degree + 1``) and control net
(``2 * n_ctrl``, x/y interleaved). The composite arena data is ``n_segs``
records of ``1 + PARAM_PAD_SIZE`` doubles, each ``[tag, params...]`` — one
encoded (non-composite) segment entity per record. Callers that may encode
variable-length entities pass an ``arena`` list to :func:`encode_entity`, which
appends to it and is later flattened into the per-group upload array.
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
    "TAG_COMPOSITE",
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
TAG_COMPOSITE = 11


def encode_entity(entity, d: int = 2, arena: list | None = None):
    """Encode a geometry entity into ``(type_tag, params)``.

    ``params`` is a length-``PARAM_PAD_SIZE`` numpy float64 array, zero-padded.
    Returns ``(TAG_FREE, zeros)`` for ``entity is None``.

    Variable-length entities (B-spline, composite path) append their data to
    ``arena`` (a growable list of floats shared by all DOFs of the upload) and
    store only offsets/counts in ``params``; encoding one without an arena
    raises ``ValueError``.
    """
    from egg.geometry.analytic2d import Circle, Ellipse, LineSegment
    from egg.geometry.curves2d import (
        BSplineCurve,
        CircleArc,
        CompositePath,
        CubicBezier,
        EllipseArc,
        QuadBezier,
    )

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
    if isinstance(entity, CircleArc):
        params[:2] = entity.center
        params[2] = entity.radius
        # Normalise a reversed (t1 < t0) traversal for the C++ clamp.
        params[3:5] = sorted((entity.t0, entity.t1))
        params[5] = float(entity.closed)
        return TAG_CIRCLEARC, params
    if isinstance(entity, EllipseArc):
        params[:2] = entity.center
        params[2:5] = (entity.a, entity.b, entity.phi)
        params[5:7] = sorted((entity.t0, entity.t1))
        params[7] = float(entity.closed)
        return TAG_ELLIPSEARC, params
    if isinstance(entity, QuadBezier):
        params[:6] = entity.p.ravel()
        params[6:8] = (entity.t0, entity.t1)
        return TAG_QUADBEZIER, params
    if isinstance(entity, CubicBezier):
        params[:8] = entity.p.ravel()
        params[8:10] = (entity.t0, entity.t1)
        return TAG_CUBICBEZIER, params
    if isinstance(entity, BSplineCurve):
        if arena is None:
            raise ValueError("encoding a BSplineCurve requires an arena")
        knot_off = len(arena)
        arena.extend(entity.knots.tolist())
        ctrl_off = len(arena)
        arena.extend(entity.ctrl.ravel().tolist())
        w_off = 0.0
        if entity.weights is not None:
            w_off = len(arena)
            arena.extend(entity.weights.tolist())
        params[:8] = (entity.degree, entity.ctrl.shape[0],
                      knot_off, ctrl_off, entity.t0, entity.t1,
                      w_off, float(entity.weights is not None))
        return TAG_BSPLINE, params
    if isinstance(entity, CompositePath):
        if arena is None:
            raise ValueError("encoding a CompositePath requires an arena")
        # Encode segment data (e.g. B-spline knots/nets) first so the records
        # land contiguously at rec_off.
        recs = []
        for seg in entity.segments:
            seg_tag, seg_params = encode_entity(seg, d=d, arena=arena)
            recs.append((seg_tag, seg_params))
        rec_off = len(arena)
        for seg_tag, seg_params in recs:
            arena.append(float(seg_tag))
            arena.extend(seg_params.tolist())
        params[:2] = (len(recs), rec_off)
        return TAG_COMPOSITE, params
    raise NotImplementedError(f"Entity type {type(entity)} not encodable yet")
