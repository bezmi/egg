"""Typed entity field schema (Python-side mirror of C++ ``EntitySoA<E>``).

This is the single Python source for the packed-record SoA layout of the
2D entity types. It replaces the positional ``(tag, params)`` blob of
:mod:`egg.geometry.entity_encoding` for every 2D type, including the composite
path (Phase 3) — a composite encodes to ``(n_segs, rec_off)`` plus one CSR slot
holding a self-contained per-composite arena (any B-spline sub-segment data,
then the segment records). A Polyline/Spline of any curve type is supported
(lines, arcs, Béziers, B-splines — the NURBS-relevant case). Only nested
composites are unsupported, and ``CompositePath`` rejects those at construction.

Each per-type encoder returns ``(tag, kFields, records_row, seg)`` where
``records_row`` is a length-``kFields`` ``float64`` array laid out with the
same named field offsets as the C++ ``EntitySoA<E>`` specialization, and
``seg`` is a list of 1D ``float64`` arrays (the variable-length CSR
payloads) for segmented types, or ``None`` for fixed-size types. The
grouping utility :func:`group_entities_by_type` partitions a set of
constrained DOFs by entity type and emits, per present type::

    {tag_name: {"tag": int, "kFields": int,
                 "dof_local": int32[k],
                 "records":   f64[k * kFields],
                 "seg":       [{"data": f64, "off": i32}, ...]}}

The ``records`` array is packed contiguous records (stride ``kFields`` per
entity), matching ``EntitySoA<E>::Host::records`` on the C++ side — one
coalesced read of entity ``i``'s ``kFields`` doubles per work item. The
``seg`` list carries the segmented (CSR) fields (absent for fixed-size
types) — flat concatenated data + per-entity offset tables, matching
``EntitySoA<E>::Host::seg`` / ``SoAHostRecord::seg``.

The field layouts (offsets within each record) are frozen to match the C++
``EntitySoA<E>`` specializations in ``src/geometry.hpp`` exactly:

| Entity       | tag | kFields | kSeg | record layout (offsets)                         |
|--------------|-----|---------|------|-------------------------------------------------|
| Free         | 0   | 0       | 0    | (none)                                          |
| LineSegment  | 1   | 4       | 0    | [sx, sy, ex, ey]                                |
| Circle       | 2   | 3       | 0    | [cx, cy, r]                                     |
| Ellipse      | 3   | 4       | 0    | [cx, cy, rx, ry]                                |
| CircleArc    | 6   | 6       | 0    | [cx, cy, r, t0, t1, closed]                     |
| EllipseArc   | 7   | 8       | 0    | [cx, cy, a, b, phi, t0, t1, closed]             |
| QuadBezier   | 8   | 8       | 0    | [P0x, P0y, P1x, P1y, P2x, P2y, t0, t1]          |
| CubicBezier  | 9   | 10      | 0    | [P0x..P3y(8), t0, t1]                           |
| BSpline      | 10  | 6       | 3    | [degree, n_ctrl, t0, t1, closed, has_w] + seg: knots, ctrl, weights |
| Composite    | 11  | 2       | 1    | [n_segs, rec_off] + seg: self-contained arena slice |
| Sphere       | 4   | 10      | 0    | [c(3), r, ax(3), ay(3)]                         |
| Plane        | 5   | 9       | 0    | [o(3), ax(3), ay(3)]                           |
| Cylinder     | 12  | 10      | 0    | [o(3), ax(3), ay(3), r]                        |
| Line3        | 13  | 8       | 0    | [p0(3), p1(3), t0, t1]                         |
| BSplineSurf  | 14  | 9       | 4    | [pu,pv,nu,nv,ku_off,kv_off,ctrl_off,w_off,has_w] + seg: knots_u, knots_v, ctrl, weights |
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "encode_entity_soa",
    "group_entities_by_type",
    "TAG_NAMES",
    "TAG_FREE",
    "TAG_LINESEG",
    "TAG_CIRCLE",
    "TAG_ELLIPSE",
    "TAG_CIRCLEARC",
    "TAG_ELLIPSEARC",
    "TAG_QUADBEZIER",
    "TAG_CUBICBEZIER",
    "TAG_BSPLINE",
    "TAG_COMPOSITE",
    "TAG_SPHERE",
    "TAG_PLANE",
    "TAG_CYLINDER",
    "TAG_LINE3",
    "TAG_BSPLINESURF",
]

# Frozen tag values (shared with entity_encoding.py / C++ EntityTag).
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
TAG_CYLINDER = 12
TAG_LINE3 = 13
TAG_BSPLINESURF = 14

# Per-type field counts (mirror C++ EntitySoA<E>::kFields).
_KFIELDS = {
    TAG_FREE: 0,
    TAG_LINESEG: 4,
    TAG_CIRCLE: 3,
    TAG_ELLIPSE: 4,
    TAG_CIRCLEARC: 6,
    TAG_ELLIPSEARC: 8,
    TAG_QUADBEZIER: 8,
    TAG_CUBICBEZIER: 10,
    TAG_BSPLINE: 6,  # degree, n_ctrl, t0, t1, closed, has_w
    TAG_COMPOSITE: 2,  # n_segs, rec_off
    TAG_PLANE: 9,  # o(3), ax(3), ay(3)
    TAG_SPHERE: 10,  # c(3), r, ax(3), ay(3)
    TAG_CYLINDER: 10,  # o(3), ax(3), ay(3), r
    TAG_LINE3: 8,  # p0(3), p1(3), t0, t1
    TAG_BSPLINESURF: 9,  # pu, pv, nu, nv, ku_off, kv_off, ctrl_off, w_off, has_w
}

# Per-type segmented field counts (mirror C++ EntitySoA<E>::kSeg).
_KSEG = {
    TAG_BSPLINE: 3,  # knots, ctrl, weights
    TAG_COMPOSITE: 1,  # segment records
    TAG_BSPLINESURF: 4,  # knots_u, knots_v, ctrl, weights
}

# Tag -> wire-format name (used as dict key in the per-group `entities` sub-dict).
TAG_NAMES = {
    TAG_FREE: "free",
    TAG_LINESEG: "lineseg",
    TAG_CIRCLE: "circle",
    TAG_ELLIPSE: "ellipse",
    TAG_CIRCLEARC: "circlearc",
    TAG_ELLIPSEARC: "ellipsearc",
    TAG_QUADBEZIER: "quadbezier",
    TAG_CUBICBEZIER: "cubicbezier",
    TAG_BSPLINE: "bspline",
    TAG_COMPOSITE: "composite",
    TAG_SPHERE: "sphere",
    TAG_PLANE: "plane",
    TAG_CYLINDER: "cylinder",
    TAG_LINE3: "line3",
    TAG_BSPLINESURF: "bsplinesurf",
}


def _encode_lineseg(entity) -> np.ndarray:
    return np.array(
        [
            entity.start[0],
            entity.start[1],
            entity.end[0],
            entity.end[1],
        ],
        dtype=np.float64,
    )


def _encode_circle(entity) -> np.ndarray:
    return np.array(
        [
            entity.center[0],
            entity.center[1],
            entity.radius,
        ],
        dtype=np.float64,
    )


def _encode_ellipse(entity) -> np.ndarray:
    return np.array(
        [
            entity.center[0],
            entity.center[1],
            entity.rx,
            entity.ry,
        ],
        dtype=np.float64,
    )


def _encode_circlearc(entity) -> np.ndarray:
    # t1 < t0 encodes a reversed traversal in Python; the C++ projection
    # clamps to the interval, so normalise it (same point set either way).
    return np.array(
        [
            entity.center[0],
            entity.center[1],
            entity.radius,
            min(entity.t0, entity.t1),
            max(entity.t0, entity.t1),
            float(entity.closed),
        ],
        dtype=np.float64,
    )


def _encode_ellipsearc(entity) -> np.ndarray:
    return np.array(
        [
            entity.center[0],
            entity.center[1],
            entity.a,
            entity.b,
            entity.phi,
            min(entity.t0, entity.t1),
            max(entity.t0, entity.t1),
            float(entity.closed),
        ],
        dtype=np.float64,
    )


def _encode_quadbezier(entity) -> np.ndarray:
    p = entity.p
    return np.array(
        [
            p[0, 0],
            p[0, 1],
            p[1, 0],
            p[1, 1],
            p[2, 0],
            p[2, 1],
            entity.t0,
            entity.t1,
        ],
        dtype=np.float64,
    )


def _encode_cubicbezier(entity) -> np.ndarray:
    p = entity.p
    return np.array(
        [
            p[0, 0],
            p[0, 1],
            p[1, 0],
            p[1, 1],
            p[2, 0],
            p[2, 1],
            p[3, 0],
            p[3, 1],
            entity.t0,
            entity.t1,
        ],
        dtype=np.float64,
    )


def _bspline_curve_degree_guard(degree: int) -> None:
    """Reject a B-spline *curve* whose degree exceeds the compiled de Boor cap.

    The device de Boor work arrays are fixed-size (``kBSplineCurveCap`` in
    ``geometry.hpp``); a higher-degree curve would index them out of bounds and
    segfault. The standalone-curve C++ loader guards this, but a curve nested in
    a composite is encoded positionally and bypasses that loader, so the guard is
    applied here too. The cap is read from the compiled extension when present
    (so it tracks any ``-DEGG_MAX_BSPLINE_CURVE_DEGREE`` override); without the
    extension there is no device de Boor to overflow, so the check is skipped.
    """
    try:
        from egg._cpp import cpp_core
    except ImportError:
        return
    cap = getattr(cpp_core, "MAX_BSPLINE_CURVE_DEGREE", None)
    if cap is not None and degree > cap:
        raise ValueError(
            f"B-spline curve degree {degree} exceeds the compiled cap {cap}; "
            f"rebuild with a larger -DEGG_MAX_BSPLINE_CURVE_DEGREE."
        )


def _encode_bspline(entity):
    """Encode a BSplineCurve into (scalars_row, segmented_data).

    Returns (row, seg) where row is the 6-double scalar record
    [degree, n_ctrl, t0, t1, closed, has_w] and seg is a list of three
    1D arrays [knots, ctrl_flat, weights] (ctrl flattened x/y interleaved,
    matching the C++ BSplineCurveParam::ctrl span layout; weights is empty
    for the polynomial path). has_w == 0.0 selects the polynomial path.
    """
    _bspline_curve_degree_guard(int(entity.degree))
    has_w = entity.weights is not None
    row = np.array(
        [
            float(entity.degree),
            float(entity.ctrl.shape[0]),
            entity.t0,
            entity.t1,
            float(entity.closed),
            float(has_w),
        ],
        dtype=np.float64,
    )
    seg = [
        np.asarray(entity.knots, dtype=np.float64).ravel(),
        np.asarray(entity.ctrl, dtype=np.float64).ravel(),
        (
            np.asarray(entity.weights, dtype=np.float64).ravel()
            if has_w
            else np.empty(0, dtype=np.float64)
        ),
    ]
    return row, seg


def _encode_composite(entity):
    """Encode a CompositePath into (scalars_row, segmented_data).

    Returns ``(row, seg)`` where ``row`` is the 2-double scalar record
    ``[n_segs, rec_off]`` and ``seg`` is a single-element list
    ``[arena_slice]`` — a self-contained per-composite arena: any
    variable-length sub-segment data (B-spline knots/control nets) laid down
    first, then the ``n_segs`` fixed-stride segment records (each
    ``[seg_tag, params(PARAM_PAD_SIZE)]``) at offset ``rec_off``. This is the
    same positional layout :func:`egg.geometry.entity_encoding.encode_entity`
    builds in the *global* upload arena, but scoped to one composite, so the
    record offsets (incl. a B-spline segment's ``knot_off``/``ctrl_off``) are
    relative to this slice's base — matching the C++
    ``EntitySoA<CompositePath>::load`` reconstruction.

    Reusing ``encode_entity`` with a fresh local arena means a Polyline/Spline
    of *any* curve type (lines, arcs, Béziers, B-splines) is supported — the
    NURBS-relevant case. Nested composites are rejected by ``CompositePath``
    itself at construction, so they never reach here.
    """
    from egg.geometry.curves2d import BSplineCurve
    from egg.geometry.entity_encoding import encode_entity

    for seg in entity.segments:
        if isinstance(seg, BSplineCurve):
            _bspline_curve_degree_guard(int(seg.degree))

    local_arena: list[float] = []
    _tag, params = encode_entity(entity, d=2, arena=local_arena)
    # params[:2] = (n_segs, rec_off) — offsets into local_arena (see encoder).
    row = np.array([params[0], params[1]], dtype=np.float64)
    arena_slice = np.asarray(local_arena, dtype=np.float64)
    return row, [arena_slice]


# --- 3D entity encoders ----------------------------------------------------


def _encode_plane(entity) -> np.ndarray:
    """Encode a Plane into a 9-double record [o(3), ax(3), ay(3)]."""
    return np.concatenate(
        [
            np.asarray(entity.o, dtype=np.float64).ravel(),
            np.asarray(entity.ax, dtype=np.float64).ravel(),
            np.asarray(entity.ay, dtype=np.float64).ravel(),
        ]
    )


def _encode_sphere(entity) -> np.ndarray:
    """Encode a Sphere into a 10-double record [c(3), r, ax(3), ay(3)]."""
    return np.concatenate(
        [
            np.asarray(entity.c, dtype=np.float64).ravel(),
            np.array([entity.r], dtype=np.float64),
            np.asarray(entity.ax, dtype=np.float64).ravel(),
            np.asarray(entity.ay, dtype=np.float64).ravel(),
        ]
    )


def _encode_cylinder(entity) -> np.ndarray:
    """Encode a Cylinder into a 10-double record [o(3), ax(3), ay(3), r]."""
    return np.concatenate(
        [
            np.asarray(entity.o, dtype=np.float64).ravel(),
            np.asarray(entity.ax, dtype=np.float64).ravel(),
            np.asarray(entity.ay, dtype=np.float64).ravel(),
            np.array([entity.r], dtype=np.float64),
        ]
    )


def _encode_line3(entity) -> np.ndarray:
    """Encode a Line3 into an 8-double record [p0(3), p1(3), t0, t1]."""
    return np.concatenate(
        [
            np.asarray(entity.p0, dtype=np.float64).ravel(),
            np.asarray(entity.p1, dtype=np.float64).ravel(),
            np.array([entity.t0, entity.t1], dtype=np.float64),
        ]
    )


def _encode_bsplinesurface(entity):
    """Encode a BSplineSurface into (scalars_row, segmented_data).

    Returns (row, seg) where row is the 9-double scalar record
    [pu, pv, nu, nv, ku_off, kv_off, ctrl_off, w_off, has_w] and seg is a
    list of four 1D arrays [knots_u, knots_v, ctrl_flat, weights] (ctrl
    flattened xyz-interleaved, matching the C++ BSplineSurfaceParam::ctrl
    span layout; weights is empty for the polynomial path). The offset
    fields in the record are stored as 0 (unused — the CSR layout makes
    per-entity offsets implicit). has_w == 0.0 selects the polynomial path.
    """
    has_w = entity.weights is not None
    row = np.array(
        [
            float(entity.pu),
            float(entity.pv),
            float(entity.nu),
            float(entity.nv),
            0.0,
            0.0,
            0.0,
            0.0,
            float(has_w),
        ],
        dtype=np.float64,
    )
    seg = [
        np.asarray(entity.knots_u, dtype=np.float64).ravel(),
        np.asarray(entity.knots_v, dtype=np.float64).ravel(),
        np.asarray(entity.ctrl, dtype=np.float64).ravel(),
        (
            np.asarray(entity.weights, dtype=np.float64).ravel()
            if has_w
            else np.empty(0, dtype=np.float64)
        ),
    ]
    return row, seg


# Tag -> (encoder function, kFields).  Tags not listed here (Composite has a
# bespoke per-segment encoder below) use the simple fixed-size encoders.
_ENCODERS = {
    TAG_LINESEG: _encode_lineseg,
    TAG_CIRCLE: _encode_circle,
    TAG_ELLIPSE: _encode_ellipse,
    TAG_CIRCLEARC: _encode_circlearc,
    TAG_ELLIPSEARC: _encode_ellipsearc,
    TAG_QUADBEZIER: _encode_quadbezier,
    TAG_CUBICBEZIER: _encode_cubicbezier,
}


def encode_entity_soa(entity, d: int = 2):
    """Encode a 2D entity into a packed SoA record row (+ segmented data).

    Returns ``(tag, kFields, records, seg)`` where ``records`` is a length-
    ``kFields`` ``float64`` array with the same named-offset layout as the
    C++ ``EntitySoA<E>`` specialization, and ``seg`` is a list of 1D
    ``float64`` arrays (the variable-length CSR payloads) for segmented
    types, or ``None`` for fixed-size types. Returns
    ``(TAG_FREE, 0, empty, None)`` for ``entity is None``.
    """
    if entity is None:
        return TAG_FREE, 0, np.empty(0, dtype=np.float64), None

    from egg.geometry.analytic2d import Circle, Ellipse, LineSegment
    from egg.geometry.analytic3d import Cylinder, Line3, Plane, Sphere
    from egg.geometry.curves2d import (
        BSplineCurve,
        CircleArc,
        CompositePath,
        CubicBezier,
        EllipseArc,
        QuadBezier,
    )
    from egg.geometry.surfaces3d import BSplineSurface

    if isinstance(entity, LineSegment):
        return TAG_LINESEG, _KFIELDS[TAG_LINESEG], _encode_lineseg(entity), None
    if isinstance(entity, Circle):
        return TAG_CIRCLE, _KFIELDS[TAG_CIRCLE], _encode_circle(entity), None
    if isinstance(entity, Ellipse):
        return TAG_ELLIPSE, _KFIELDS[TAG_ELLIPSE], _encode_ellipse(entity), None
    if isinstance(entity, CircleArc):
        return TAG_CIRCLEARC, _KFIELDS[TAG_CIRCLEARC], _encode_circlearc(entity), None
    if isinstance(entity, EllipseArc):
        return (
            TAG_ELLIPSEARC,
            _KFIELDS[TAG_ELLIPSEARC],
            _encode_ellipsearc(entity),
            None,
        )
    if isinstance(entity, QuadBezier):
        return (
            TAG_QUADBEZIER,
            _KFIELDS[TAG_QUADBEZIER],
            _encode_quadbezier(entity),
            None,
        )
    if isinstance(entity, CubicBezier):
        return (
            TAG_CUBICBEZIER,
            _KFIELDS[TAG_CUBICBEZIER],
            _encode_cubicbezier(entity),
            None,
        )
    if isinstance(entity, BSplineCurve):
        row, seg = _encode_bspline(entity)
        return TAG_BSPLINE, _KFIELDS[TAG_BSPLINE], row, seg
    if isinstance(entity, CompositePath):
        row, seg = _encode_composite(entity)
        return TAG_COMPOSITE, _KFIELDS[TAG_COMPOSITE], row, seg
    if isinstance(entity, Plane):
        return TAG_PLANE, _KFIELDS[TAG_PLANE], _encode_plane(entity), None
    if isinstance(entity, Sphere):
        return TAG_SPHERE, _KFIELDS[TAG_SPHERE], _encode_sphere(entity), None
    if isinstance(entity, Cylinder):
        return TAG_CYLINDER, _KFIELDS[TAG_CYLINDER], _encode_cylinder(entity), None
    if isinstance(entity, Line3):
        return TAG_LINE3, _KFIELDS[TAG_LINE3], _encode_line3(entity), None
    if isinstance(entity, BSplineSurface):
        row, seg = _encode_bsplinesurface(entity)
        return TAG_BSPLINESURF, _KFIELDS[TAG_BSPLINESURF], row, seg
    raise NotImplementedError(f"Entity type {type(entity)} not encodable yet")


def group_entities_by_type(
    dof_indices: list[int],
    entities: dict[int, object],
    d: int = 2,
) -> dict[str, dict]:
    """Partition a set of DOFs by entity type into packed-record SoA groups.

    Parameters
    ----------
    dof_indices : list[int]
        Global DOF indices in the order they appear in the group.
    entities : dict[int, object]
        Mapping from global DOF index -> geometry entity (or ``None`` for free).
    d : int
        Spatial dimension (passed through to encoders; currently always 2).

    Returns
    -------
    dict[str, dict]
        Per present entity type (keyed by wire-format name), a dict with::

            {"tag": int, "kFields": int,
             "dof_local": int32[k],   # indices into dof_indices (group-local)
             "records":   f64[k*kFields],  # packed (k, kFields) row-major
             "seg": [{"data": f64, "off": i32}, ...]}  # CSR fields (absent if kSeg==0)

    DOFs whose entities have no SoA encoder (only a composite with a
    variable-length/nested segment) are returned under the special key
    ``"__blob__"`` as a list of group-local indices, so the caller can keep
    them on the legacy blob path. Free DOFs (entity is None or TAG_FREE) are
    included as a ``"free"`` group with empty records.
    """
    groups: dict[str, dict] = {}
    blob_local: list[int] = []

    for local_idx, dof in enumerate(dof_indices):
        entity = entities.get(dof)
        try:
            tag, kfields, row, seg = encode_entity_soa(entity, d=d)
        except ValueError:
            # Composite — stays on the legacy blob path.
            blob_local.append(local_idx)
            continue

        name = TAG_NAMES[tag]
        if name not in groups:
            groups[name] = {
                "tag": tag,
                "kFields": kfields,
                "dof_local": [],
                "records": [],
                "_seg_lists": [],  # per-slot list of arrays (finalized to CSR)
            }
            kseg = _KSEG.get(tag, 0)
            if kseg > 0:
                groups[name]["_seg_lists"] = [[] for _ in range(kseg)]
        g = groups[name]
        g["dof_local"].append(local_idx)
        if kfields > 0:
            g["records"].append(row)
        if seg is not None:
            for j, arr in enumerate(seg):
                g["_seg_lists"][j].append(arr)

    # Finalize: convert lists to numpy arrays + build CSR for segmented fields.
    for name, g in groups.items():
        g["dof_local"] = np.asarray(g["dof_local"], dtype=np.int32)
        kf = g["kFields"]
        if kf > 0 and g["records"]:
            g["records"] = np.concatenate(g["records"]).astype(np.float64)
        else:
            g["records"] = np.empty(0, dtype=np.float64)

        seg_lists = g.pop("_seg_lists")
        if seg_lists:
            g["seg"] = []
            for arrays in seg_lists:
                data = (
                    np.concatenate(arrays).astype(np.float64)
                    if arrays
                    else np.empty(0, dtype=np.float64)
                )
                off = np.zeros(len(arrays) + 1, dtype=np.int32)
                for i, arr in enumerate(arrays):
                    off[i + 1] = off[i] + arr.shape[0]
                g["seg"].append({"data": data, "off": off})

    if blob_local:
        groups["__blob__"] = {"dof_local": np.asarray(blob_local, dtype=np.int32)}

    return groups
