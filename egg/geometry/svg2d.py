# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Inkscape SVG import: labeled drawing objects become geometry entities.

:func:`svg_import` reads an SVG (typically drawn in Inkscape), converts
every path/shape into the corresponding egg curve entities, and returns an
:class:`SvgDomain` that looks objects up by their Inkscape *label*
(``inkscape:label``, the name shown in Inkscape's Objects/Layers panel;
the ``id`` attribute is the fallback). Geometry setup scripts can then
attach topology to named parts of a drawing::

    dom = svg_import("domain.svg")
    wall = dom.edge("wall")            # Edge over the imported curve
    n = wall.place_node(0.25)          # slides on the drawn geometry

Everything is standalone Python (``xml.etree`` + numpy); the produced
entities are the ordinary ones from :mod:`egg.geometry` and feed the
topology builder and the C++ encoding path unchanged.

Coordinate mapping
------------------

SVG is y-down; model space is y-up. By default the drawing is flipped
about its viewBox (``y_model = y_min + height - y_svg``), so what you see
in Inkscape is what you get, in the document's *user units* (for a
typical Inkscape document, ``viewBox`` units — check the document
properties). ``scale`` multiplies the result; ``y_up=False`` imports raw
SVG coordinates.

Supported content
-----------------

``path`` (the full ``d`` grammar: M/L/H/V/C/S/Q/T/A/Z and the relative
forms), ``rect`` (sharp-cornered), ``circle``, ``ellipse``, ``line``,
``polyline``, ``polygon``; ``transform`` stacks (matrix / translate /
scale / rotate / skewX / skewY) are composed and applied exactly — line
and Bézier control points map through the affine directly, elliptical
arcs are remapped exactly via an SVD decomposition of the transformed
axes. Arcs are split at the angle-branch cuts of the arc entities (and
into <= 90-degree pieces), so the imported trims are always representable
by the C++ projection kernels. Elements with ``display:none`` (hidden
Inkscape layers) are skipped; unsupported elements (text, images, ...)
are skipped with a note in :attr:`SvgDomain.warnings`.

A multi-segment drawing object becomes one
:class:`~egg.geometry.curves2d.CompositePath` (a closed subpath gets its
closing segment, like ``Polyline(closed=True)``); a full ``circle`` /
``ellipse`` element becomes a periodic
:class:`~egg.geometry.analytic2d.Circle` or a closed
:class:`~egg.geometry.curves2d.EllipseArc`. Paths with several subpaths
are rejected — split them into separate objects in Inkscape (Path >
Break Apart) so each can carry its own label.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np

from .analytic2d import Circle, LineSegment
from .curves2d import CircleArc, CompositePath, CubicBezier, EllipseArc, QuadBezier
from .frontend2d import Edge

__all__ = ["svg_import", "svg_topology", "SvgDomain", "SvgItem"]

_SVG_NS = "http://www.w3.org/2000/svg"
_INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"

# Elements that never contain drawable geometry.
_SKIP_TAGS = {
    "defs",
    "metadata",
    "namedview",
    "title",
    "desc",
    "style",
    "script",
    "clipPath",
    "mask",
    "marker",
    "pattern",
    "symbol",
    "filter",
    "linearGradient",
    "radialGradient",
}
_GROUP_TAGS = {"svg", "g", "a", "switch"}

_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------
# Affine transforms: y = A x + b, composed down the element tree.


def _rot(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


_TRANSFORM_RE = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")


def _parse_transform(text: str) -> tuple[np.ndarray, np.ndarray]:
    """SVG ``transform`` attribute -> (A, b). Functions compose left to right."""
    A = np.eye(2)
    b = np.zeros(2)
    for m in _TRANSFORM_RE.finditer(text):
        name = m.group(1)
        args = [float(v) for v in re.split(r"[\s,]+", m.group(2).strip()) if v]
        if name == "matrix":
            if len(args) != 6:
                raise ValueError(f"matrix() needs 6 numbers, got {args}")
            Ai = np.array([[args[0], args[2]], [args[1], args[3]]])
            bi = np.array(args[4:6])
        elif name == "translate":
            Ai = np.eye(2)
            bi = np.array([args[0], args[1] if len(args) > 1 else 0.0])
        elif name == "scale":
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            Ai = np.diag([sx, sy])
            bi = np.zeros(2)
        elif name == "rotate":
            Ai = _rot(math.radians(args[0]))
            bi = np.zeros(2)
            if len(args) == 3:
                c = np.array(args[1:3])
                bi = c - Ai @ c
        elif name == "skewX":
            Ai = np.array([[1.0, math.tan(math.radians(args[0]))], [0.0, 1.0]])
            bi = np.zeros(2)
        else:  # skewY
            Ai = np.array([[1.0, 0.0], [math.tan(math.radians(args[0])), 1.0]])
            bi = np.zeros(2)
        A, b = A @ Ai, A @ bi + b
    return A, b


# --------------------------------------------------------------------------
# Path `d` scanning.

_WSC = r"[\s,]*"
_NUM_RE = re.compile(_WSC + r"([+-]?(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?)")
_FLAG_RE = re.compile(_WSC + r"([01])")
_CMD_RE = re.compile(_WSC + r"([MmLlHhVvCcSsQqTtAaZz])")
_END_RE = re.compile(_WSC + r"$")


class _DScanner:
    """Tokenizer over a path ``d`` string.

    Numbers and arc flags need separate rules: Inkscape (legally) writes
    the two ``A`` flags without separators (``... 0 01 3,4``), so flags
    are single ``[01]`` characters, never part of a number.
    """

    def __init__(self, d: str):
        self.d = d
        self.pos = 0

    def _take(self, rx: re.Pattern, what: str) -> str:
        m = rx.match(self.d, self.pos)
        if not m:
            raise ValueError(
                f"bad path data: expected {what} at position {self.pos}: "
                f"{self.d[self.pos : self.pos + 24]!r}"
            )
        self.pos = m.end()
        return m.group(1)

    def num(self) -> float:
        return float(self._take(_NUM_RE, "a number"))

    def flag(self) -> bool:
        return self._take(_FLAG_RE, "an arc flag") == "1"

    def cmd(self) -> str | None:
        m = _CMD_RE.match(self.d, self.pos)
        if m:
            self.pos = m.end()
            return m.group(1)
        return None

    def has_number(self) -> bool:
        return _NUM_RE.match(self.d, self.pos) is not None

    def at_end(self) -> bool:
        return _END_RE.match(self.d, self.pos) is not None


# Local-coordinate primitives collected while scanning; transformed into
# entities afterwards, so the S/T reflection logic runs untransformed.
# ("line", p0, p1) | ("quad", p0, p1, p2) | ("cubic", p0..p3)
# | ("arc", center, rx, ry, phi, t0, t1)
_Prim = tuple


def _parse_path_d(d: str) -> tuple[list[_Prim], bool]:
    """Parse one-subpath ``d`` into local primitives; returns (prims, closed)."""
    sc = _DScanner(d)
    prims: list[_Prim] = []
    cur = np.zeros(2)
    start = np.zeros(2)
    started = False
    closed = False
    last_cubic_ctrl: np.ndarray | None = None
    last_quad_ctrl: np.ndarray | None = None
    cmd: str | None = None

    def _line_to(p: np.ndarray) -> None:
        nonlocal cur
        if np.linalg.norm(p - cur) > 0.0:
            prims.append(("line", cur, p))
        cur = p

    while True:
        nxt = sc.cmd()
        if nxt is not None:
            cmd = nxt
        elif sc.at_end():
            break
        elif cmd in (None, "Z", "z") or not sc.has_number():
            raise ValueError(f"bad path data at position {sc.pos}: {d[sc.pos :]!r}")
        # else: implicit repeat of the previous command.

        if closed and cmd not in ("Z", "z"):
            raise ValueError(
                "path has multiple subpaths — break it apart in Inkscape so "
                "each piece can carry its own label"
            )
        rel = cmd.islower()
        C = cmd.upper()
        new_cubic_ctrl: np.ndarray | None = None
        new_quad_ctrl: np.ndarray | None = None

        if C == "M":
            p = np.array([sc.num(), sc.num()])
            if rel and started:
                p = cur + p
            if started:
                raise ValueError(
                    "path has multiple subpaths — break it apart in Inkscape "
                    "so each piece can carry its own label"
                )
            cur = start = p
            started = True
            cmd = "l" if rel else "L"  # extra M pairs are lineto
        elif C == "Z":
            if not started:
                raise ValueError("path Z before any M")
            gap = np.linalg.norm(start - cur)
            if gap > 1e-12 * (1.0 + float(np.linalg.norm(start))):
                prims.append(("line", cur, start))
            cur = start
            closed = True
        elif C == "L":
            p = np.array([sc.num(), sc.num()])
            _line_to(cur + p if rel else p)
        elif C == "H":
            x = sc.num()
            _line_to(np.array([cur[0] + x if rel else x, cur[1]]))
        elif C == "V":
            y = sc.num()
            _line_to(np.array([cur[0], cur[1] + y if rel else y]))
        elif C in ("C", "S"):
            if C == "C":
                c1 = np.array([sc.num(), sc.num()])
                if rel:
                    c1 = cur + c1
            else:
                c1 = (
                    2.0 * cur - last_cubic_ctrl
                    if last_cubic_ctrl is not None
                    else cur.copy()
                )
            c2 = np.array([sc.num(), sc.num()])
            p = np.array([sc.num(), sc.num()])
            if rel:
                c2, p = cur + c2, cur + p
            prims.append(("cubic", cur, c1, c2, p))
            new_cubic_ctrl = c2
            cur = p
        elif C in ("Q", "T"):
            if C == "Q":
                c1 = np.array([sc.num(), sc.num()])
                if rel:
                    c1 = cur + c1
            else:
                c1 = (
                    2.0 * cur - last_quad_ctrl
                    if last_quad_ctrl is not None
                    else cur.copy()
                )
            p = np.array([sc.num(), sc.num()])
            if rel:
                p = cur + p
            prims.append(("quad", cur, c1, p))
            new_quad_ctrl = c1
            cur = p
        elif C == "A":
            rx, ry = sc.num(), sc.num()
            phi = math.radians(sc.num())
            large, sweep = sc.flag(), sc.flag()
            p = np.array([sc.num(), sc.num()])
            if rel:
                p = cur + p
            if np.linalg.norm(p - cur) > 0.0:
                if rx == 0.0 or ry == 0.0:
                    prims.append(("line", cur, p))  # per the SVG spec
                else:
                    prims.append(
                        _arc_endpoint_to_center(cur, p, rx, ry, phi, large, sweep)
                    )
            cur = p
        else:  # pragma: no cover — the command regex is exhaustive
            raise AssertionError(cmd)

        if not started:
            raise ValueError("path data must start with M")
        last_cubic_ctrl, last_quad_ctrl = new_cubic_ctrl, new_quad_ctrl

    if not prims:
        raise ValueError("path has no drawable segments")
    return prims, closed


def _arc_endpoint_to_center(
    p0: np.ndarray,
    p1: np.ndarray,
    rx: float,
    ry: float,
    phi: float,
    large: bool,
    sweep: bool,
) -> _Prim:
    """SVG endpoint arc -> center form (W3C F.6.5, with the F.6.6 radius fix)."""
    rx, ry = abs(rx), abs(ry)
    R = _rot(phi)
    p = R.T @ ((p0 - p1) / 2.0)
    lam = (p[0] / rx) ** 2 + (p[1] / ry) ** 2
    if lam > 1.0:
        s = math.sqrt(lam)
        rx, ry = s * rx, s * ry
    num = (rx * ry) ** 2 - (rx * p[1]) ** 2 - (ry * p[0]) ** 2
    den = (rx * p[1]) ** 2 + (ry * p[0]) ** 2
    k = math.sqrt(max(0.0, num / den))
    if large == sweep:
        k = -k
    cp = k * np.array([rx * p[1] / ry, -ry * p[0] / rx])
    center = R @ cp + (p0 + p1) / 2.0

    def ang(v: np.ndarray) -> float:
        return math.atan2(v[1], v[0])

    v0 = (p - cp) / np.array([rx, ry])
    v1 = (-p - cp) / np.array([rx, ry])
    t0 = ang(v0)
    dt = ang(v1) - t0
    if not sweep and dt > 0.0:
        dt -= _TWO_PI
    elif sweep and dt < 0.0:
        dt += _TWO_PI
    return ("arc", center, rx, ry, phi, t0, t0 + dt)


# --------------------------------------------------------------------------
# Exact affine mapping of elliptical arcs.


def _transform_arc(
    A: np.ndarray,
    b: np.ndarray,
    center: np.ndarray,
    rx: float,
    ry: float,
    phi: float,
    t0: float,
    t1: float,
) -> tuple[np.ndarray, float, float, float, float, float]:
    """Map ``C + R(phi) diag(rx, ry) u(t)`` through ``x -> A x + b`` exactly.

    Writes ``A R(phi) diag(rx, ry) = R(alpha) diag(a, b') W`` with W
    orthogonal, so the image is again an ellipse arc
    ``C' + R(alpha) diag(a, b') u(eps t + delta)``. Returns
    ``(center', a, b', alpha, t0', t1')``.
    """
    M = A @ _rot(phi) @ np.diag([rx, ry])
    U, s, _ = np.linalg.svd(M)
    if s[1] <= 1e-12 * s[0]:
        raise ValueError("transform collapses an arc to a degenerate ellipse")
    if np.linalg.det(U) < 0.0:
        U = U @ np.diag([1.0, -1.0])
    alpha = math.atan2(U[1, 0], U[0, 0])
    # W u(t) must equal u(eps t + delta) — rotation or reflection.
    W = np.diag(1.0 / s) @ U.T @ M
    if np.linalg.det(W) > 0.0:
        eps, delta = 1.0, math.atan2(W[1, 0], W[0, 0])
    else:
        eps, delta = -1.0, math.atan2(W[0, 1], W[0, 0])
    return (
        A @ center + b,
        float(s[0]),
        float(s[1]),
        alpha,
        eps * t0 + delta,
        eps * t1 + delta,
    )


def _split_arc_interval(t0: float, t1: float, cut: float) -> list[float]:
    """Breakpoints from t0 to t1: split at ``cut + k*2pi`` and into <= pi/2."""
    lo, hi = min(t0, t1), max(t0, t1)
    eps = 1e-12 * (1.0 + abs(lo) + abs(hi))
    cuts = []
    c = cut + _TWO_PI * math.ceil((lo - cut) / _TWO_PI)
    while c < hi - eps:
        if c > lo + eps:
            cuts.append(c)
        c += _TWO_PI
    pts = [lo, *cuts, hi]
    fine = []
    for a2, b2 in zip(pts, pts[1:]):
        n = max(1, math.ceil((b2 - a2) / (0.5 * math.pi) - 1e-12))
        fine.extend(a2 + (b2 - a2) * i / n for i in range(n))
    fine.append(hi)
    return fine[::-1] if t1 < t0 else fine


def _arc_entities(
    center: np.ndarray, a: float, b: float, alpha: float, t0: float, t1: float
) -> list[Any]:
    """Branch-safe arc entities for one (possibly long) elliptical sweep.

    The C++ projection kernels invert a ``CircleArc`` with ``atan2``
    (branch ``(-pi, pi]``) and an ``EllipseArc`` by Newton over
    ``[0, 2pi]``, so each emitted piece is shifted into that branch and
    the sweep is split at the branch cuts (and into <= 90-degree pieces,
    which also keeps composite nearest-segment selection tight).
    """
    circular = abs(a - b) <= 1e-9 * max(a, b)
    out: list[Any] = []
    if circular:
        # Fold the axis rotation into the angles: C + a u(t + alpha).
        pts = _split_arc_interval(t0 + alpha, t1 + alpha, cut=math.pi)
        for a2, b2 in zip(pts, pts[1:]):
            shift = -_TWO_PI * round(0.5 * (a2 + b2) / _TWO_PI)
            out.append(CircleArc(center, a, a2 + shift, b2 + shift))
    else:
        pts = _split_arc_interval(t0, t1, cut=0.0)
        for a2, b2 in zip(pts, pts[1:]):
            shift = -_TWO_PI * math.floor(0.5 * (a2 + b2) / _TWO_PI)
            out.append(EllipseArc(center, a, b, alpha, a2 + shift, b2 + shift))
    return out


# --------------------------------------------------------------------------
# Element -> entity conversion.


def _prims_to_entity(prims: list[_Prim], A: np.ndarray, b: np.ndarray) -> Any:
    """Transform local primitives and join them into one entity (closing
    segments were already appended by the parser)."""
    segs: list[Any] = []
    for prim in prims:
        kind = prim[0]
        if kind == "line":
            segs.append(LineSegment(A @ prim[1] + b, A @ prim[2] + b))
        elif kind == "quad":
            segs.append(QuadBezier(*(A @ p + b for p in prim[1:])))
        elif kind == "cubic":
            segs.append(CubicBezier(*(A @ p + b for p in prim[1:])))
        else:  # arc
            segs.extend(_arc_entities(*_transform_arc(A, b, *prim[1:])))
    if len(segs) == 1:
        return segs[0]
    return CompositePath(segs)


def _float_attr(el: ElementTree.Element, name: str, default: float = 0.0) -> float:
    v = el.get(name)
    return default if v is None else float(v)


def _shape_prims(el: ElementTree.Element, tag: str) -> tuple[list[_Prim], bool] | None:
    """Local primitives for the basic shape elements (None: not a shape)."""
    if tag == "path":
        d = el.get("d")
        if not d:
            raise ValueError("path element has no d attribute")
        return _parse_path_d(d)
    if tag == "rect":
        if _float_attr(el, "rx") or _float_attr(el, "ry"):
            raise ValueError("rounded <rect> corners are not supported")
        x, y = _float_attr(el, "x"), _float_attr(el, "y")
        w, h = _float_attr(el, "width"), _float_attr(el, "height")
        p = [np.array(q) for q in [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]]
        return [("line", p[i], p[(i + 1) % 4]) for i in range(4)], True
    if tag in ("circle", "ellipse"):
        c = np.array([_float_attr(el, "cx"), _float_attr(el, "cy")])
        if tag == "circle":
            rx = ry = _float_attr(el, "r")
        else:
            rx, ry = _float_attr(el, "rx"), _float_attr(el, "ry")
        return [("arc", c, rx, ry, 0.0, 0.0, _TWO_PI)], True
    if tag == "line":
        p0 = np.array([_float_attr(el, "x1"), _float_attr(el, "y1")])
        p1 = np.array([_float_attr(el, "x2"), _float_attr(el, "y2")])
        return [("line", p0, p1)], False
    if tag in ("polyline", "polygon"):
        nums = [
            float(v) for v in re.split(r"[\s,]+", el.get("points", "").strip()) if v
        ]
        pts = [np.array(nums[i : i + 2]) for i in range(0, len(nums) - 1, 2)]
        if len(pts) < 2:
            raise ValueError(f"<{tag}> needs at least two points")
        prims = [("line", a2, b2) for a2, b2 in zip(pts, pts[1:])]
        if tag == "polygon":
            prims.append(("line", pts[-1], pts[0]))
        return prims, tag == "polygon"
    return None


def _full_period_entity(
    el: ElementTree.Element, tag: str, A: np.ndarray, b: np.ndarray
):
    """Exact periodic entity for full <circle>/<ellipse> elements."""
    c = np.array([_float_attr(el, "cx"), _float_attr(el, "cy")])
    if tag == "circle":
        rx = ry = _float_attr(el, "r")
    else:
        rx, ry = _float_attr(el, "rx"), _float_attr(el, "ry")
    c2, a2, b2, alpha, _, _ = _transform_arc(A, b, c, rx, ry, 0.0, 0.0, _TWO_PI)
    if abs(a2 - b2) <= 1e-9 * max(a2, b2):
        return Circle(c2, a2)
    return EllipseArc(c2, a2, b2, alpha, 0.0, _TWO_PI, closed=True)


# --------------------------------------------------------------------------
# Document walk and the domain API.


@dataclass
class SvgItem:
    """One imported drawing object."""

    label: str  #: ``inkscape:label`` (fallback: ``id``).
    elem_id: str  #: the element's ``id``.
    layer: str  #: enclosing Inkscape layer's label ("" outside layers).
    entity: Any  #: the geometry entity.
    closed: bool  #: whether the source object was a closed contour.
    group: str = ""  #: innermost enclosing ``<g>`` label (layer or plain group).


class SvgDomain:
    """Imported SVG drawing: geometry entities looked up by Inkscape label."""

    def __init__(self, items: list[SvgItem], warnings: list[str]):
        self.items = items
        self.warnings = warnings

    @property
    def labels(self) -> list[str]:
        """All labels, in document order (duplicates kept once)."""
        return list(dict.fromkeys(it.label for it in self.items))

    def all(self, label: str) -> list[Any]:
        """Every entity carrying ``label`` (e.g. several ``wall`` pieces)."""
        return [it.entity for it in self.items if it.label == label]

    def __getitem__(self, label: str) -> Any:
        found = self.all(label)
        if not found:
            raise KeyError(f"no SVG object labeled {label!r}; available: {self.labels}")
        if len(found) > 1:
            raise KeyError(
                f"{len(found)} SVG objects are labeled {label!r} — rename them "
                f"in Inkscape or use .all({label!r})"
            )
        return found[0]

    def get(self, label: str, default: Any = None) -> Any:
        found = self.all(label)
        return found[0] if len(found) == 1 else default

    def edge(self, label: str, arc_length: bool = False, samples: int = 256) -> Edge:
        """The labeled entity wrapped as an :class:`Edge` for node placement."""
        return Edge(self[label], arc_length=arc_length, samples=samples)

    def __contains__(self, label: str) -> bool:
        return any(it.label == label for it in self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __repr__(self) -> str:
        return f"SvgDomain({self.labels})"


def _is_hidden(el: ElementTree.Element) -> bool:
    if el.get("display") == "none":
        return True
    style = el.get("style", "")
    return bool(re.search(r"display\s*:\s*none", style))


def _root_flip(root: ElementTree.Element, y_up: bool, scale: float):
    """The document -> model transform: optional y-flip about the viewBox."""
    A = np.diag([scale, scale])
    b = np.zeros(2)
    if not y_up:
        return A, b
    viewbox = root.get("viewBox")
    if viewbox:
        vals = [float(v) for v in re.split(r"[\s,]+", viewbox.strip()) if v]
        y_top = vals[1] + vals[3]
    else:
        m = re.match(r"([0-9.eE+-]+)", root.get("height", "") or "0")
        y_top = float(m.group(1)) if m else 0.0
    return A @ np.diag([1.0, -1.0]), A @ np.array([0.0, y_top]) + b


def svg_import(
    source: str | Path, *, y_up: bool = True, scale: float = 1.0
) -> SvgDomain:
    """Import an (Inkscape) SVG file as labeled geometry entities.

    Parameters
    ----------
    source : str or Path
        Path to the SVG file, or the SVG document itself (a string
        starting with ``<``).
    y_up : bool, optional
        Flip the y-down SVG drawing about its viewBox into y-up model
        space (default). ``False`` imports raw SVG coordinates.
    scale : float, optional
        Multiply all model coordinates (e.g. document drawn in mm user
        units, model in metres: ``scale=1e-3``).

    Returns
    -------
    SvgDomain
        Lookup by Inkscape label: ``dom["egg"]`` is the entity,
        ``dom.edge("egg")`` wraps it for node placement, ``dom.all("wall")``
        collects same-labeled pieces.
    """
    text = str(source)
    if text.lstrip().startswith("<"):
        root = ElementTree.fromstring(text)
    else:
        root = ElementTree.parse(text).getroot()

    A0, b0 = _root_flip(root, y_up, scale)
    items: list[SvgItem] = []
    warnings: list[str] = []

    def walk(
        el: ElementTree.Element, A: np.ndarray, b: np.ndarray, layer: str, group: str
    ):
        tag = el.tag.split("}")[-1]
        ns = el.tag.split("}")[0].lstrip("{") if "}" in el.tag else _SVG_NS
        if ns != _SVG_NS or tag in _SKIP_TAGS or _is_hidden(el):
            return
        t = el.get("transform")
        if t:
            Ai, bi = _parse_transform(t)
            A, b = A @ Ai, A @ bi + b
        label = el.get(f"{{{_INKSCAPE_NS}}}label") or el.get("id") or ""
        if tag in _GROUP_TAGS:
            if el.get(f"{{{_INKSCAPE_NS}}}groupmode") == "layer":
                layer = label
            # The nearest enclosing group's label (layer or plain <g>); a
            # topology wireframe can live in either, addressed by this name.
            child_group = label if tag == "g" else group
            for child in el:
                walk(child, A, b, layer, child_group)
            return
        try:
            if tag in ("circle", "ellipse"):
                entity, closed = _full_period_entity(el, tag, A, b), True
            else:
                shape = _shape_prims(el, tag)
                if shape is None:
                    warnings.append(f"skipped <{tag}> (id={el.get('id')!r})")
                    return
                prims, closed = shape
                if closed and len(prims) == 1 and prims[0][0] == "arc":
                    # A closed single-arc path is a full ellipse.
                    c2, a2, b2, alpha, _, _ = _transform_arc(A, b, *prims[0][1:])
                    entity = (
                        Circle(c2, a2)
                        if abs(a2 - b2) <= 1e-9 * max(a2, b2)
                        else EllipseArc(c2, a2, b2, alpha, 0.0, _TWO_PI, closed=True)
                    )
                else:
                    entity = _prims_to_entity(prims, A, b)
        except ValueError as e:
            raise ValueError(f"SVG object {label or el.get('id')!r}: {e}") from e
        items.append(
            SvgItem(
                label=label or f"object{len(items)}",
                elem_id=el.get("id") or "",
                layer=layer,
                entity=entity,
                closed=closed,
                group=group,
            )
        )

    walk(root, A0, b0, "", "")
    return SvgDomain(items, warnings)


# --------------------------------------------------------------------------
# Block topology inferred from a wireframe layer/group.
#
# Where svg_import above reads the *geometry* (boundary curves labelled by
# physical name), svg_topology reads the *blocking*: a second group of
# straight line segments whose endpoints are the block corners. The blocks
# are the quadrilateral faces of that planar graph, so a singularity is
# nothing more than a node where a number of edges other than four meet —
# you draw it by drawing the lines that cross there.
#
# (Kept in this module, not a same-named submodule, so ``from egg.geometry
# import svg_topology`` always resolves to the function.)


def _wire_endpoints(entity: Any) -> tuple[np.ndarray, np.ndarray]:
    """The two ends of a straight wireframe segment (model coordinates)."""
    a = np.asarray(entity.eval(0.0), dtype=float)[:2]
    b = np.asarray(entity.eval(1.0), dtype=float)[:2]
    return a, b


def svg_topology(
    dom: SvgDomain,
    *,
    group: str = "topology",
    res: int = 10,
    weld_tol: float | None = None,
    snap_tol: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build a topology from the wireframe on ``group`` of an imported SVG.

    The named layer/group holds a straight-line wireframe whose endpoints are
    the block corners; the blocks are the planar-graph faces. Coincident
    endpoints weld into shared corners (so interfaces conform), the outer face
    and any closed-curve interior (the region meshed *around*, e.g. an egg)
    are dropped, and each remaining quad becomes a block. A block edge whose
    *both* ends lie on a labelled geometry curve is associated to it and
    inherits its label as a boundary tag (a straight schematic edge is allowed
    to follow a curved boundary, ICEM-style); a corner on two or more distinct
    curves is pinned, on one slides, on none is free. Singularities emerge as
    the irregular-degree nodes — nothing about them is authored.

    Parameters
    ----------
    dom : SvgDomain
        The result of :func:`svg_import`.
    group : str
        Label of the Inkscape layer *or* plain ``<g>`` holding the blocking
        wireframe. Everything not on it is treated as geometry.
    res : int
        Uniform cells-per-axis for every block (keeps interfaces conforming);
        per-block / per-loop counts are a refinement layer left to code.
    weld_tol, snap_tol : float, optional
        Distances (model units) for welding coincident corners and for
        deciding a corner/edge lies on a geometry curve. Default to a small
        and a modest fraction of the wireframe's bounding-box diagonal.

    Returns
    -------
    (TopologyBuilder, dict)
        The populated builder (call ``.build()`` after any
        ``set_boundary_layer``) and ``{label: entity}`` for the geometry,
        ready as the pipeline's boundary-marker entities.
    """
    from egg.topology.trace import trace_topology

    wire = [it for it in dom if group in (it.layer, it.group)]
    geom = [it for it in dom if group not in (it.layer, it.group)]
    if not wire:
        present = sorted({it.group for it in dom} | {it.layer for it in dom})
        raise ValueError(
            f"svg_topology: no wireframe found on group {group!r}; "
            f"groups present: {present}"
        )

    # --- gather segments and their endpoints -----------------------------
    segs = [_wire_endpoints(it.entity) for it in wire]
    allpts = np.array([p for seg in segs for p in seg])
    diag = float(np.linalg.norm(allpts.max(0) - allpts.min(0))) or 1.0
    weld = weld_tol if weld_tol is not None else 1e-3 * diag
    snap = snap_tol if snap_tol is not None else 1e-2 * diag

    # --- weld coincident endpoints into shared corners -------------------
    pos: list[np.ndarray] = []

    def node_id(p: np.ndarray) -> int:
        for i, r in enumerate(pos):
            if np.linalg.norm(p - r) <= weld:
                return i
        pos.append(p.copy())
        return len(pos) - 1

    edge_set: set[tuple[int, int]] = set()
    for a, b in segs:
        i, j = node_id(a), node_id(b)
        if i != j:
            edge_set.add((min(i, j), max(i, j)))
    # The graph->topology trace (welding aside) is shared with the interactive
    # editor; a diagnostic here is a malformed wireframe, so it raises.
    gents = [(it.label, it.entity, it.closed) for it in geom]
    b, diags = trace_topology(pos, edge_set, gents, res=res, snap_tol=snap)
    if diags:
        raise ValueError(f"svg_topology: {diags[0].msg}")

    entities = {lbl: ent for lbl, ent, _ in gents}
    return b, entities
