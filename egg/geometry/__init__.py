"""Geometry entity classes (2D and 3D) mirroring the C++ parametrizations."""

from .analytic2d import Circle, Ellipse, LineSegment
from .analytic3d import Cylinder, Line3, Plane, Sphere
from .base import GeometryEntity
from .curves2d import (
    BSplineCurve,
    CircleArc,
    CompositePath,
    CubicBezier,
    EllipseArc,
    QuadBezier,
)
from .frontend2d import (
    Arc,
    Bezier,
    Edge,
    Line,
    Node,
    Polyline,
    Spline,
    Vector3,
    split_cells,
    tfi_point,
)
from .surfaces3d import BSplineSurface

__all__ = [
    # base
    "GeometryEntity",
    # 2D construction front-end (gdtk/Eilmer-style)
    "Vector3",
    "Line",
    "Arc",
    "Bezier",
    "Polyline",
    "Spline",
    "Edge",
    "Node",
    "split_cells",
    "tfi_point",
    # 2D analytic
    "Circle",
    "Ellipse",
    "LineSegment",
    # 2D curves
    "BSplineCurve",
    "CircleArc",
    "CompositePath",
    "CubicBezier",
    "EllipseArc",
    "QuadBezier",
    # 3D analytic
    "Cylinder",
    "Line3",
    "Plane",
    "Sphere",
    # 3D surfaces
    "BSplineSurface",
]
