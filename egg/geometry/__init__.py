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
from .surfaces3d import BSplineSurface

__all__ = [
    # base
    "GeometryEntity",
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
