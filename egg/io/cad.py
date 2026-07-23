# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""CAD import adapter: build123d / OCP shapes to egg geometry entities.

Optional; needs the ``cad`` group (``uv sync --group cad``: build123d). Import
STEP files and build/boolean domain volumes with build123d directly, then hand
the resulting solid here: each face becomes an egg
:class:`~egg.geometry.surfaces3d.BSplineSurface` carrying its UV trim loops, and
straight edges become :class:`~egg.geometry.analytic3d.Line3`. The result is the
``{name: entity}`` map that
:class:`~egg.topology.explicit.ExplicitTopology` and
:meth:`~egg.topology.builder.TopologyBuilder.associate` consume.

The heavy conversion (OCCT surface to NURBS, trim wires to UV polylines) runs
once at import; the solver never sees OCCT.
"""

from __future__ import annotations

import numpy as np

from egg.geometry.analytic3d import Line3
from egg.geometry.base import _INHERIT
from egg.geometry.surfaces3d import (
    BSplineSurface,
    _orthonormalize2,
    _trim_clamp,
    _trim_contains,
)

__all__ = [
    "import_step",
    "face_to_surface",
    "edge_to_curve",
    "surfaces",
    "CadBSplineSurface",
]


class CadBSplineSurface(BSplineSurface):
    """A BSplineSurface backed by its OCCT carrier for native projection.

    Serializes to the device wire exactly like a plain ``BSplineSurface``
    (``isinstance`` holds, so ``encode_entity_soa`` dispatches identically);
    the Python-side projection family (:meth:`invert` and everything composed
    from it) is overridden to route through OCCT's
    ``GeomAPI_ProjectPointOnSurf``, which is far faster than the egg
    parametrization Newton and handles the surface-of-revolution pole/seam
    natively. This touches only host projection (``initialize_grid`` /
    ``project_nodes`` / target frames); the C++ device path still runs the SoA
    reconstruction. Falls back to the egg ``invert`` when the OCCT handle is
    absent (e.g. after a deep copy that drops it).
    """

    def __init__(self, *args, occ_surface=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._occ_surface = occ_surface
        self._occ_proj = None

    def _projector(self):
        """A reusable ``GeomAPI_ProjectPointOnSurf`` over the carrier surface."""
        if self._occ_proj is None and self._occ_surface is not None:
            from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
            from OCP.gp import gp_Pnt

            self._occ_proj = GeomAPI_ProjectPointOnSurf(
                gp_Pnt(0.0, 0.0, 0.0),
                self._occ_surface,
                self._u0,
                self._u1,
                self._v0,
                self._v1,
            )
        return self._occ_proj

    def invert(self, p):
        proj = self._projector()
        if proj is None:
            return super().invert(p)
        from OCP.gp import gp_Pnt

        p = np.asarray(p, dtype=float)
        proj.Perform(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
        if proj.NbPoints() < 1:
            return super().invert(p)
        u, v = proj.LowerDistanceParameters()
        u = float(np.clip(u, self._u0, self._u1))
        v = float(np.clip(v, self._v0, self._v1))
        if self.trim and not _trim_contains(self.trim, u, v):
            u, v = _trim_clamp(self.trim, u, v)
        return u, v

    # Host projection stays with OCCT (the GeometryEntity base methods route
    # through the C++ SoA reconstruction, which would bypass the carrier):
    # everything composes invert -> eval / frame.

    def project(self, p):
        u, v = self.invert(p)
        return self.eval(u, v)

    def project_many(self, pts):
        return np.stack([self.project(p) for p in np.asarray(pts, dtype=float)])

    def tangent_space(self, q):
        u, v = self.invert(q)
        return _orthonormalize2(*self.frame(u, v))

    def tangent_space_many(self, Q):
        return np.stack([self.tangent_space(q) for q in np.asarray(Q, dtype=float)])

    def normal(self, q):
        u, v = self.invert(q)
        su, sv = self.frame(u, v)
        n = np.cross(su, sv)
        return n / np.linalg.norm(n)


def _require_cad() -> None:
    try:
        import build123d  # noqa: F401
        import OCP  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "egg.io.cad needs the 'cad' dependency group: uv sync --group cad"
        ) from exc


def import_step(path):
    """Load a STEP file into a build123d shape (for face/edge extraction)."""
    _require_cad()
    from build123d import import_step as _import_step

    return _import_step(str(path))


def _topods(shape):
    return shape.wrapped if hasattr(shape, "wrapped") else shape


def _nurbs_face(tface):
    """NURBS-convert a TopoDS_Face and return the converted face."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    conv = BRepBuilderAPI_NurbsConvert(tface, True).Shape()
    exp = TopExp_Explorer(conv, TopAbs_FACE)
    return TopoDS.Face_s(exp.Current())


def _extract_surface(bs) -> CadBSplineSurface:
    """A Geom_BSplineSurface to an egg CadBSplineSurface (poles, weights, carrier).

    The returned surface keeps a handle to ``bs`` (post-clamp) so its host-side
    projection runs through OCCT; the control net / knots still feed the device.
    """
    if bs.IsUPeriodic():
        bs.SetUNotPeriodic()
    if bs.IsVPeriodic():
        bs.SetVNotPeriodic()
    pu, pv = bs.UDegree(), bs.VDegree()
    nu, nv = bs.NbUPoles(), bs.NbVPoles()
    ctrl = np.empty((nu, nv, 3))
    rational = bs.IsURational() or bs.IsVRational()
    weights = np.ones((nu, nv)) if rational else None
    for i in range(1, nu + 1):
        for j in range(1, nv + 1):
            p = bs.Pole(i, j)
            ctrl[i - 1, j - 1] = (p.X(), p.Y(), p.Z())
            if rational:
                weights[i - 1, j - 1] = bs.Weight(i, j)

    def kseq(arr):
        return np.array([arr.Value(k) for k in range(arr.Lower(), arr.Upper() + 1)])

    return CadBSplineSurface(
        pu,
        pv,
        kseq(bs.UKnotSequence()),
        kseq(bs.VKnotSequence()),
        ctrl,
        weights,
        occ_surface=bs,
    )


def _face_trim(tface, n_samp=24):
    """Face wires (outer first, then holes) as UV-polygon loops."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve2d
    from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
    from OCP.TopAbs import TopAbs_REVERSED, TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    outer = BRepTools.OuterWire_s(tface)
    wires = []
    we = TopExp_Explorer(tface, TopAbs_WIRE)
    while we.More():
        wires.append(TopoDS.Wire_s(we.Current()))
        we.Next()
    wires.sort(key=lambda w: not w.IsSame(outer))  # outer loop first, then holes

    loops = []
    for w in wires:
        pts = []
        wx = BRepTools_WireExplorer(w, tface)
        while wx.More():
            e = wx.Current()
            ad = BRepAdaptor_Curve2d(e, tface)
            ts = np.linspace(ad.FirstParameter(), ad.LastParameter(), n_samp)
            if e.Orientation() == TopAbs_REVERSED:
                ts = ts[::-1]
            for t in ts[:-1]:  # drop the shared end vertex
                p = ad.Value(float(t))
                pts.append((p.X(), p.Y()))
            wx.Next()
        loops.append(np.array(pts))
    return loops


def face_to_surface(face, *, name=None, tag=_INHERIT, trim=True) -> BSplineSurface:
    """A face's carrier NURBS as an egg BSplineSurface with its UV trim loops."""
    _require_cad()
    from OCP.BRep import BRep_Tool

    tface = _nurbs_face(_topods(face))
    surf = _extract_surface(BRep_Tool.Surface_s(tface))
    if trim:
        surf.trim = _face_trim(tface)
    if name is not None:
        surf.named(name, tag=tag)
    return surf


def edge_to_curve(edge, *, name=None, tag=_INHERIT) -> Line3:
    """A straight edge as a Line3.

    Curved 3D edges are not yet device-encodable (the core has no 3D B-spline
    curve entity), so they raise ``NotImplementedError``.
    """
    _require_cad()
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomAbs import GeomAbs_Line

    ad = BRepAdaptor_Curve(_topods(edge))
    if ad.GetType() != GeomAbs_Line:
        raise NotImplementedError(
            "only straight (Line3) edges are device-encodable in 3D; a curved "
            "edge needs a 3D B-spline curve entity (not yet in the core)"
        )
    p0, p1 = ad.Value(ad.FirstParameter()), ad.Value(ad.LastParameter())
    line = Line3((p0.X(), p0.Y(), p0.Z()), (p1.X(), p1.Y(), p1.Z()))
    if name is not None:
        line.named(name, tag=tag)
    return line


def surfaces(faces, *, prefix="face", tag=None) -> dict[str, BSplineSurface]:
    """Extract an iterable of faces to a named ``{f'{prefix}_{i}': surface}`` map."""
    out: dict[str, BSplineSurface] = {}
    for i, face in enumerate(faces):
        nm = f"{prefix}_{i}"
        out[nm] = face_to_surface(face, name=nm, tag=tag)
    return out
