# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

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
from .svg2d import SvgDomain, SvgItem, svg_import, svg_topology

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
    # SVG import
    "svg_import",
    "svg_topology",
    "SvgDomain",
    "SvgItem",
]
