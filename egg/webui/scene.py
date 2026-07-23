# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Script execution + scene harvesting + SVG rendering for the web UI.

"Code as CAD": :func:`build_scene` executes a user geometry script (the
same egg front-end API the examples use), harvests renderable objects out
of the resulting namespace — parametric curve entities, :class:`Edge`
wrappers, :class:`Vector3` / :class:`Node` points, and any built
topology/grid — and renders them to a standalone ``<svg>`` fragment.

:func:`exec_script` / :func:`harvest` are also exposed separately so the
app's smoothing runner can keep the live grid handle and re-render frames
with :func:`render_svg` as the pipeline mutates it in place.

Deliberately fasthtml-free so it can be exercised without the web stack.
"""

from __future__ import annotations

import ast
import io
import json
import os
from typing import TypeGuard
import sys
import threading
import time
import traceback
from contextlib import redirect_stdout, suppress
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from egg import webui as egg_webui

from egg.core.types import MultiBlockGrid
from egg.geometry.base import GeometryEntity
from egg.geometry.frontend2d import Edge, Node, Vector3
from egg.geometry.svg2d import SvgDomain
from egg.topology.block_topology import BlockTopology
from egg.topology.builder import TopologyBuilder
from egg.topology.explicit import ExplicitTopology

CURVE_SAMPLES = 96
COMPOSITE_SAMPLES = 256
MAX_GRID_NODES = 200_000
MAX_GRID_LINES = 129  # per block axis in the preview; see _line_indices
MAX_TANGLED_CELLS = 2_000
HARVEST_DEPTH = 3

# curves cycle through the c0..cN CSS classes (colors live in the
# page theme's --egg-curve* variables, never here)
N_CURVE_CLASSES = 5
N_BLOCK_CLASSES = 6  # b0..b5 fill classes in the theme CSS

Bounds = tuple[np.ndarray, np.ndarray]


@dataclass
class Scene:
    """Harvested world-space primitives, ready for rendering."""

    curves: list[tuple[str, str, np.ndarray, int]] = field(default_factory=list)
    points: list[tuple[str, str, float, float]] = field(default_factory=list)
    # control cages: (name, (n, 2) vertex array, per-vertex kind) — the
    # Vector3s a curve was constructed from, drawn as CAD-style control
    # points joined by dashed tangent lines in the toggleable ctrl layer.
    ctrl_cages: list[tuple[str, np.ndarray, list[str]]] = field(default_factory=list)
    # solved control-net overlay: (block name, (nu, nv, 2) control lattice),
    # drawn as dashed lattice polylines + markers in the toggleable net layer.
    net_blocks: list[tuple[str, np.ndarray]] = field(default_factory=list)
    grid_blocks: list[np.ndarray] = field(default_factory=list)
    grid_block_names: list[str] = field(default_factory=list)
    # block name -> fill-class index; a greedy graph-coloring over shared-face
    # adjacency so no two blocks sharing a boundary get the same colour.
    block_colors: dict[str, int] = field(default_factory=dict)
    # topology-view primitives (straight-edged, from the BlockTopology)
    topo_blocks: list[tuple[str, np.ndarray, list[int]]] = field(default_factory=list)
    topo_corners: list[tuple[str, float, float, bool, int | None, list[int]]] = field(
        default_factory=list
    )
    topo_connections: list[tuple[str, np.ndarray]] = field(default_factory=list)
    topo_associations: list[tuple[str, np.ndarray, int]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def min_det(self) -> float | None:
        """Preview-grade min corner det over all grid cells (sign = validity)."""
        dets = [float(_cell_min_dets(n).min()) for n in self.grid_blocks]
        return min(dets) if dets else None


@dataclass
class Harvest:
    """A harvested scene plus the live handles the view needs."""

    scene: Scene
    grid: MultiBlockGrid | None = None
    topo: BlockTopology | None = None
    editable: ExplicitTopology | None = None
    # topology diagnostics from an ExplicitTopology flatten (empty == valid)
    diagnostics: list = field(default_factory=list)


@dataclass
class SceneResult:
    """Everything the view pane needs for one render."""

    svg: str
    stats: dict[str, int]
    warnings: list[str]
    stdout: str
    error: str | None
    min_det: float | None = None
    elapsed_ms: int = 0
    quality: dict[str, tuple] | None = None  # see grid_quality
    edit_data: dict | None = (
        None  # edit view: the blocking + editability (see edit_data)
    )


_EXEC_LOCK = threading.Lock()


# --- guard parameter panel: a form view over the `a = dict(...)` literal ---


@dataclass
class GuardParam:
    """One editable literal: a guard-dict entry or an ``editable()`` mark."""

    name: str
    text: str  # exact source text of the value (e.g. "5.0e-3")
    kind: str  # bool | int | float | str
    # lineno, col, end_lineno, end_col (1-based lines); end_* are None on some
    # ast nodes.
    span: tuple[int, int, int | None, int | None]
    choices: tuple | None = None  # literal options -> dropdown (editable() only)
    show_if: dict | None = None  # {param: value|[values]} -> shown only when matched


def _is_params_call(node: ast.AST) -> TypeGuard[ast.Call]:
    """``params(...)`` / ``egg_webui.params(...)``."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "params") or (
        isinstance(f, ast.Attribute) and f.attr == "params"
    )


def _params_node(tree: ast.Module):
    """The entries-bearing node of the first ``egg_webui.params(...)`` marker —
    its ``**kwargs`` (the Call itself) or a single ``dict(...)``/``{...}``
    argument — or None if the script declares no run-parameters panel.

    An explicit marker, not "the first dict in the guard": the panel binds only
    to what the script hands to :func:`egg_webui.params`."""
    for node in ast.walk(tree):
        if not _is_params_call(node):
            continue
        if node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Dict):
                return arg
            if (
                isinstance(arg, ast.Call)
                and isinstance(arg.func, ast.Name)
                and arg.func.id == "dict"
            ):
                return arg
        return node  # kwargs form -> the call's own keywords
    return None


def _literal_entries(node) -> list[tuple[str, ast.expr]]:
    if isinstance(node, ast.Call):
        return [(kw.arg, kw.value) for kw in node.keywords if kw.arg]
    return [
        (k.value, v)
        for k, v in zip(node.keys, node.values)
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    ]


def _literal_kind(val: ast.expr) -> str | None:
    """The panel kind of a literal node (unary +/- allowed), or None."""
    if isinstance(val, ast.UnaryOp) and isinstance(val.op, (ast.USub, ast.UAdd)):
        if not isinstance(val.operand, ast.Constant):
            return None
    elif not isinstance(val, ast.Constant):
        return None
    try:
        lit = ast.literal_eval(val)
    except ValueError:
        return None
    if isinstance(lit, bool):
        return "bool"
    if isinstance(lit, int):
        return "int"
    if isinstance(lit, float):
        return "float"
    if isinstance(lit, str):
        return "str"
    return None


def _is_editable_call(node: ast.AST) -> TypeGuard[ast.Call]:
    """``editable(...)`` / ``egg_webui.editable(...)`` with a positional arg."""
    if not (isinstance(node, ast.Call) and node.args):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "editable") or (
        isinstance(f, ast.Attribute) and f.attr == "editable"
    )


def _editable_params(tree: ast.Module, code: str) -> list[GuardParam]:
    """Every ``editable()`` mark in the module, named from its context.

    The editable literal is the call's first positional argument; its exact
    source span is what the panel rewrites (the call wrapper survives). The
    display name is the ``label=`` keyword when given, else the assignment
    target / dict key / keyword argument the call sits in, else the line
    number. ``choices=`` (a list/tuple of literals) becomes a dropdown.
    """
    # Context names, gathered where the parent node is at hand.
    names: dict[int, str] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and _is_editable_call(node.value)
        ):
            names[id(node.value)] = node.targets[0].id
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg and _is_editable_call(kw.value):
                    names[id(kw.value)] = kw.arg
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and _is_editable_call(v):
                    names[id(v)] = str(k.value)

    params = []
    for node in ast.walk(tree):
        if not _is_editable_call(node):
            continue
        val = node.args[0]
        kind = _literal_kind(val)
        if kind is None:
            continue  # only plain literals can be rewritten losslessly
        label, choices, show_if = None, None, None
        for kw in node.keywords:
            if kw.arg == "label" and isinstance(kw.value, ast.Constant):
                label = str(kw.value.value)
            elif kw.arg == "choices" and isinstance(kw.value, (ast.List, ast.Tuple)):
                try:
                    choices = tuple(ast.literal_eval(e) for e in kw.value.elts)
                except ValueError:
                    choices = None
            elif kw.arg == "show_if" and isinstance(kw.value, ast.Dict):
                try:
                    show_if = ast.literal_eval(kw.value)
                except ValueError:
                    show_if = None
        text = ast.get_source_segment(code, val) or ""
        params.append(
            GuardParam(
                label or names.get(id(node)) or f"line {node.lineno}",
                text,
                kind,
                (val.lineno, val.col_offset, val.end_lineno, val.end_col_offset),
                choices=choices,
                show_if=show_if,
            )
        )
    return params


def guard_params(code: str) -> list[GuardParam]:
    """The editable scalar parameters of a script, in source order.

    Two sources: entries of the ``egg_webui.params(...)`` marker call whose
    values are plain literals — bool/int/float/str constants, including negated
    numbers — and ``editable()`` marks anywhere in the module (see
    :func:`egg_webui.params` and :func:`egg_webui.editable`). Non-literal values
    are skipped: the panel edits only what it can rewrite losslessly. Duplicate
    display names get a ``#2``/``#3`` suffix so each maps to exactly one source
    span.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    params = []
    node = _params_node(tree)
    if node is not None:
        for name, val in _literal_entries(node):
            kind = _literal_kind(val)
            if kind is None:
                continue
            text = ast.get_source_segment(code, val) or ""
            params.append(
                GuardParam(
                    name,
                    text,
                    kind,
                    (val.lineno, val.col_offset, val.end_lineno, val.end_col_offset),
                )
            )
    params += _editable_params(tree, code)
    params.sort(key=lambda p: p.span[:2])
    seen: dict[str, int] = {}
    for p in params:
        n = seen[p.name] = seen.get(p.name, 0) + 1
        if n > 1:
            p.name = f"{p.name}#{n}"
    return params


def _param_value(p: GuardParam):
    """The current value of a param, parsed from its source text."""
    try:
        return ast.literal_eval(p.text)
    except (ValueError, SyntaxError):
        return p.text


def visible_params(params: list[GuardParam]) -> list[GuardParam]:
    """Drop params whose ``show_if`` is not met by the other params' values.

    ``show_if`` maps a parameter name to an allowed value (or list of values);
    a param is shown only when every named parameter currently matches. Lets
    the panel show only the parameters that apply to the current choices (e.g.
    node-mode interface knobs disappear when the control-point smoother is
    selected).
    """
    values = {p.name: _param_value(p) for p in params}
    out = []
    for p in params:
        if p.show_if:
            hidden = False
            for key, allowed in p.show_if.items():
                if not isinstance(allowed, (list, tuple, set)):
                    allowed = [allowed]
                if values.get(key) not in allowed:
                    hidden = True
                    break
            if hidden:
                continue
        out.append(p)
    return out


def set_guard_param(code: str, name: str, raw: str) -> str:
    """Rewrite one editable value in ``code``; returns the new source.

    ``raw`` is what the panel input holds; it is coerced by the entry's
    current kind (numbers parsed, strings quoted, bools normalized) and
    spliced over the value's exact source span — surrounding formatting
    and comments are untouched. Raises ValueError if the entry doesn't
    exist, ``raw`` doesn't parse, or it isn't one of the entry's
    ``choices``.
    """
    p = next((p for p in guard_params(code) if p.name == name), None)
    if p is None:
        raise ValueError(f"no editable guard parameter {name!r}")
    raw = raw.strip()
    if p.kind == "bool":
        if raw.lower() in ("true", "1", "on"):
            new = "True"
        elif raw.lower() in ("false", "0", "off", ""):
            new = "False"
        else:
            raise ValueError(f"not a bool: {raw!r}")
    elif p.kind in ("int", "float"):
        try:
            val = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raise ValueError(f"not a number: {raw!r}") from None
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(f"not a number: {raw!r}")
        new = raw  # keep the user's spelling (5e-3 stays 5e-3)
    else:
        new = repr(raw)
    if p.choices is not None and ast.literal_eval(new) not in p.choices:
        raise ValueError(f"{name!r} must be one of {list(p.choices)}, not {raw!r}")
    lines = code.splitlines(keepends=True)
    l0, c0, l1, c1 = p.span
    assert l1 is not None and c1 is not None  # value nodes carry end spans
    # splice over [l0:c0, l1:c1] (1-based lines, 0-based cols)
    start = sum(len(ln) for ln in lines[: l0 - 1]) + c0
    end = sum(len(ln) for ln in lines[: l1 - 1]) + c1
    return code[:start] + new + code[end:]


def _is_explicit_topology_call(node: ast.AST) -> TypeGuard[ast.Call]:
    """``ExplicitTopology(...)`` / ``x.ExplicitTopology(...)``."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "ExplicitTopology") or (
        isinstance(f, ast.Attribute) and f.attr == "ExplicitTopology"
    )


def _node_span(node) -> tuple[int, int, int, int] | None:
    attrs = ("lineno", "col_offset", "end_lineno", "end_col_offset")
    if all(getattr(node, a, None) is not None for a in attrs):
        return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
    return None


def explicit_topology_source(code: str) -> dict | None:
    """How the first ``ExplicitTopology(...)`` in ``code`` authors its blocking.

    Returns ``{"editable": bool, "span": (l0, c0, l1, c1) | None}`` — ``editable``
    is True only when ``connectivity=`` is wrapped in ``editable(...)``, which is
    what opts the edit view into draw tools and commit; ``span`` is the blocking
    literal *inside* the wrapper (what a commit rewrites). ``None`` when the
    script builds no ExplicitTopology.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not _is_explicit_topology_call(node):
            continue
        conn = next(
            (kw.value for kw in node.keywords if kw.arg == "connectivity"), None
        )
        if conn is None:
            return {"editable": False, "span": None}
        if _is_editable_call(conn):  # connectivity=editable({...})
            return {"editable": True, "span": _node_span(conn.args[0])}
        return {"editable": False, "span": _node_span(conn)}
    return None


def editable_block(code: str) -> str | None:
    """Source text of the first ExplicitTopology's ``editable(...)`` connectivity
    call — the block a commit rewrites — so the UI can show it for manual paste
    (a watched file must not be modified without the user's say-so)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not _is_explicit_topology_call(node):
            continue
        conn = next(
            (kw.value for kw in node.keywords if kw.arg == "connectivity"), None
        )
        if conn is not None and _is_editable_call(conn):
            return ast.get_source_segment(code, conn)
        return None
    return None


def _py_literal(v) -> str:
    """Inline Python literal for a JSON value — ``json.dumps`` spacing and
    double-quoted strings, but ``True``/``False``/``None`` instead of the
    JSON spellings (the rendering is spliced into Python source)."""
    if v is True:
        return "True"
    if v is False:
        return "False"
    if v is None:
        return "None"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_py_literal(x) for x in v) + "]"
    if isinstance(v, dict):
        return (
            "{"
            + ", ".join(f"{json.dumps(str(k))}: {_py_literal(x)}" for k, x in v.items())
            + "}"
        )
    return json.dumps(v)


def _format_blocking(blocking: dict, indent: int) -> str:
    """Pretty-print a blocking dict as diff-friendly Python at ``indent`` columns.

    Top-level ``nodes`` / ``edges`` break one entry per line (with trailing
    commas); each node spec and edge stays inline. Deterministic, so
    re-committing an unchanged blocking is a no-op.
    """
    pad, p1, p2 = " " * indent, " " * (indent + 4), " " * (indent + 8)
    out = ["{"]
    out.append(f'{p1}"nodes": {{')
    for nid, spec in (blocking.get("nodes") or {}).items():
        out.append(f"{p2}{json.dumps(str(nid))}: {_py_literal(spec)},")
    out.append(f"{p1}}},")
    out.append(f'{p1}"edges": [')
    for e in blocking.get("edges") or []:
        out.append(f"{p2}{_py_literal(e)},")
    out.append(f"{p1}],")
    # Sections beyond nodes/edges/res (fan_frames, parallel, 3D blocks)
    # survive the rewrite verbatim: dicts and lists break one entry per line,
    # sorted key order.
    for key in sorted(k for k in blocking if k not in ("nodes", "edges", "res")):
        v = blocking[key]
        if isinstance(v, dict):
            out.append(f"{p1}{json.dumps(str(key))}: {{")
            for k2, v2 in v.items():
                out.append(f"{p2}{json.dumps(str(k2))}: {_py_literal(v2)},")
            out.append(f"{p1}}},")
        elif isinstance(v, list):
            out.append(f"{p1}{json.dumps(str(key))}: [")
            for v2 in v:
                out.append(f"{p2}{_py_literal(v2)},")
            out.append(f"{p1}],")
        else:
            out.append(f"{p1}{json.dumps(str(key))}: {_py_literal(v)},")
    if blocking.get("res") is not None:
        out.append(f'{p1}"res": {_py_literal(blocking["res"])},')
    out.append(f"{pad}}}")
    return "\n".join(out)


def set_editable_blocking(code: str, blocking: dict) -> str:
    """Splice ``blocking`` into the ``connectivity=editable({...})`` literal.

    The blocking dict inside the ``editable(...)`` wrapper is replaced with a
    formatted rendering of ``blocking`` (the marker and surrounding call survive);
    returns the new source. Raises ValueError if the script has no editable
    connectivity to write.
    """
    info = explicit_topology_source(code)
    if info is None or not info["editable"] or info["span"] is None:
        raise ValueError("no editable connectivity to commit")
    l0, c0, l1, c1 = info["span"]
    lines = code.splitlines(keepends=True)
    indent = len(lines[l0 - 1]) - len(lines[l0 - 1].lstrip())
    start = sum(len(ln) for ln in lines[: l0 - 1]) + c0
    end = sum(len(ln) for ln in lines[: l1 - 1]) + c1
    return code[:start] + _format_blocking(blocking, indent) + code[end:]


def exec_script(code: str, path: str | None = None) -> tuple[dict, str, str | None]:
    """Run ``code`` in a fresh namespace; capture stdout and any traceback.

    ``path`` — the script's on-disk location, when known — makes the code
    behave like a file being run: ``__file__`` is set and the script's
    directory joins ``sys.path`` for the duration, so sibling imports
    (``from driver import parse_args``) resolve. Modules imported from
    that directory are evicted afterwards: every example ships its own
    ``driver.py``, and the module cache must not serve one example's
    driver to the next. The lock serializes execs — ``sys.path`` and
    ``sys.modules`` are process-global.
    """
    ns: dict = {"__name__": "__egg_webui__"}
    buf = io.StringIO()
    error: str | None = None
    with _EXEC_LOCK:
        script_dir: str | None = None
        modules_before = set(sys.modules)
        egg_webui._reset()
        if path:
            ns["__file__"] = str(path)
            script_dir = str(Path(path).resolve().parent)
            sys.path.insert(0, script_dir)
        try:
            with redirect_stdout(buf):
                try:
                    exec(compile(code, "<script>", "exec"), ns)
                except BaseException:
                    error = traceback.format_exc()
        finally:
            # whatever the script registered via egg_webui.run(...)
            ns["__egg_webui_run__"] = egg_webui._take()
            if script_dir is not None:
                with suppress(ValueError):
                    sys.path.remove(script_dir)
                for name in set(sys.modules) - modules_before:
                    f = getattr(sys.modules[name], "__file__", None)
                    if f and str(Path(f).resolve()).startswith(script_dir + os.sep):
                        del sys.modules[name]
    return ns, buf.getvalue(), error


def build_scene(code: str, mode: str = "grid", path: str | None = None) -> SceneResult:
    """Execute ``code``, harvest its namespace, and render the SVG.

    ``mode="grid"`` (default) renders the structured grid; ``mode="topo"``
    renders the block topology (corners, block outlines, connections)
    instead — and skips harvest-side TFI initialization entirely.
    """
    t0 = time.perf_counter()
    ns, out, error = exec_script(code, path)
    h = harvest(ns, init_grid=(mode == "grid"))
    svg = render_svg(h.scene, mode=mode)
    stats = {
        "blocks": len(h.scene.topo_blocks),
        "curves": len(h.scene.curves),
        "points": len(h.scene.points),
        "grid nodes": sum(int(np.prod(b.shape[:-1])) for b in h.scene.grid_blocks)
        if mode == "grid"
        else 0,
    }
    elapsed = int((time.perf_counter() - t0) * 1000)
    min_det = h.scene.min_det if mode == "grid" else None
    quality = grid_quality(h.scene.grid_blocks) if mode == "grid" else None
    edit_data = edit_data_for(code, h) if mode == "edit" else None
    return SceneResult(
        svg, stats, h.scene.warnings, out, error, min_det, elapsed, quality, edit_data
    )


def _diag_dict(d) -> dict:
    return {
        "kind": d.kind,
        "msg": d.msg,
        "where": list(d.where),
        "xy": list(d.xy) if d.xy else None,
    }


def _debug_blocking(connectivity, diagnostics) -> None:
    """When ``EGG_WEBUI_DEBUG`` is set, dump a drawn topology's diagnostics and
    its full connectivity to the console (stderr) so a broken blocking can be
    inspected verbatim.

    ``diagnostics`` may hold :class:`~egg.topology.trace.Diagnostic` objects or
    plain strings; a no-op unless the env var is set and there is something to
    report. Emits to stderr because the render/run workers reserve stdout as
    their response channel — stderr is what reaches the terminal.
    """
    if not (diagnostics and os.environ.get("EGG_WEBUI_DEBUG")):
        return
    lines = ["[egg-webui] topology issue(s) in the drawn blocking:"]
    for d in diagnostics:
        if isinstance(d, str):
            lines.append(f"  - {d}")
        else:
            where = f" at {tuple(d.where)}" if getattr(d, "where", ()) else ""
            lines.append(f"  - {d.kind}: {d.msg}{where}")
    lines.append("[egg-webui] full connectivity:")
    try:
        lines.append(json.dumps(connectivity, indent=2, default=str))
    except (TypeError, ValueError) as exc:
        lines.append(f"  (could not serialise connectivity: {exc!r})")
    print("\n".join(lines), file=sys.stderr, flush=True)


def edit_data_for(code: str, h: Harvest) -> dict | None:
    """The edit view's structured payload for the client, or None.

    Carries the JSON-able blocking structure, whether it is editable (the
    ``editable(...)`` wrap), the geometry labels bindings can name, and the
    current flatten diagnostics (empty == green/committable).
    """
    if h.editable is None:
        return None
    src = explicit_topology_source(code) or {}
    out = {
        "editable": bool(src.get("editable")),
        "blocking": h.editable.connectivity,
        "geometry": _geometry_polylines(h.editable.geometry),
        "base": h.editable.base_graph(),
        "diagnostics": [_diag_dict(d) for d in h.diagnostics],
    }
    # A valid flatten's per-edge effective resolutions + must-share classes,
    # so the edit view shows loop-propagated values from the first paint
    # (validate refreshes them after each edit).
    errors = [d for d in h.diagnostics if not d.kind.startswith("warn")]
    if h.topo is not None and not errors and getattr(h.topo, "d", 2) == 2:
        eff, classes = _edge_res_classes(h.topo)
        out["edge_res"] = eff
        out["res_classes"] = classes
    return out


def _geometry_polylines(geometry: dict, n: int = 64) -> list[dict]:
    """Each named bindable entity sampled to a world-space polyline, so the edit
    view can hit-test a click against it and snap/bind a node onto the curve."""
    out = []
    for label, ent in geometry.items():
        e = getattr(ent, "entity", ent)  # unwrap an Edge to its entity
        if not hasattr(e, "eval_frac"):
            continue
        try:
            pts = np.array(
                [
                    np.asarray(e.eval_frac(t), dtype=float)[:2]
                    for t in np.linspace(0, 1, n)
                ]
            )
        except Exception:
            continue
        if np.isfinite(pts).all():
            out.append({"label": label, "points": pts.tolist()})
    return out


def _edge_res_classes(topo) -> tuple[dict[str, int], list[list[str]]]:
    """Per-edge effective resolution plus the classes of edges that must share
    one, from a flattened 2D topology.

    Opposite faces of a block share a resolution and a shared face links two
    blocks, so one explicit setting drives its whole loop (the same union-find
    the flatten's ``_propagate_loop_res`` runs). Keys match the client's edge
    keys: the sorted corner-name pair joined by ``|``.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def key(a: str, b: str) -> str:
        return f"{a}|{b}" if a < b else f"{b}|{a}"

    eff: dict[str, int] = {}
    for spec in topo.block_specs.values():
        for axis in (0, 1):
            ks = []
            for side in (0, 1):
                a, b = spec.face_corner_names(axis, side, 2)
                if a == b:
                    continue
                k = key(a, b)
                eff[k] = int(spec.resolutions[1 - axis])
                ks.append(k)
            if len(ks) == 2:
                parent[find(ks[0])] = find(ks[1])
    classes: dict[str, list[str]] = {}
    for k in eff:
        classes.setdefault(find(k), []).append(k)
    return eff, [sorted(v) for v in classes.values()]


def validate_blocking(code: str, blocking: dict, path: str | None = None) -> dict:
    """Validation payload for a candidate blocking against the script's base +
    geometry: ``diagnostics`` (empty means green/committable) plus, when the
    flatten succeeds in 2D, ``edge_res`` (edge key -> effective cell count)
    and ``res_classes`` (groups of edge keys that must share a resolution), so
    the edit view can show the propagated resolution on every loop edge.

    Execs ``code`` to recover the ExplicitTopology's base and geometry, then
    flattens ``blocking`` in their context — without touching the source — so the
    edit view can gate a commit on a green (empty) result. Execs the script;
    call it in a worker process.
    """
    ns, _out, err = exec_script(code, path)
    if err is not None:
        return {
            "diagnostics": [
                {
                    "kind": "script_error",
                    "msg": "script error — fix it first",
                    "where": [],
                    "xy": None,
                }
            ]
        }
    h = harvest(ns, init_grid=False)
    if h.editable is None:
        return {
            "diagnostics": [
                {
                    "kind": "no_editable",
                    "msg": "script builds no ExplicitTopology",
                    "where": [],
                    "xy": None,
                }
            ]
        }
    et = ExplicitTopology(
        base=h.editable.base,
        geometry=h.editable.geometry,
        connectivity=blocking,
        relax_orthogonality=getattr(h.editable, "relax_orthogonality", ()),
        fan_frames=getattr(h.editable, "fan_frames", None),
        parallel=getattr(h.editable, "parallel", None),
    )
    topo, diags = et.flatten()
    _debug_blocking(blocking, diags)
    out: dict = {"diagnostics": [_diag_dict(d) for d in diags]}
    if topo is not None and getattr(topo, "d", 2) == 2:
        eff, classes = _edge_res_classes(topo)
        out["edge_res"] = eff
        out["res_classes"] = classes
    return out


def webui_block_suggestion(code: str, path: str | None = None) -> str | None:
    """For a library-style script that draws nothing in the UI (defines a
    single parameterless ``build*`` function, builds nothing at top level),
    a ready-to-append ``__egg_webui__`` block that builds the topology and
    registers a run via ``egg_webui.run``. Never applied silently: the
    client shows a confirmation prompt before appending it to the file.

    Execs the script — call it in a worker process, not the server.
    """
    import inspect

    if "__egg_webui__" in code:
        return None
    ns, _out, err = exec_script(code, path)
    if err is not None or not isinstance(ns, dict):
        return None
    h = harvest(ns, init_grid=False)
    s = h.scene
    if h.grid is not None or h.topo is not None or s.curves or s.points:
        return None
    cands = [
        name
        for name, f in ns.items()
        if inspect.isfunction(f)
        and name.startswith("build")
        and f.__globals__ is ns  # defined by the script, not imported
        and all(
            p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            for p in inspect.signature(f).parameters.values()
        )
    ]
    if len(cands) != 1:
        return None
    return (
        f'\n\nif __name__ == "__egg_webui__":  # running inside the egg web UI\n'
        f"    import egg.webui as egg_webui\n"
        f"    from egg.pipeline import PipelineConfig, generate_steps\n\n"
        f"    topo, ents = {cands[0]}()\n"
        f"    grid = topo.initialize_grid()\n"
        f"    cfg = PipelineConfig()\n"
        f"    egg_webui.run(grid, generate_steps(grid, None, cfg, "
        f"untangle_direct=False))\n"
    )


def grid_to_su2_text(grid) -> str:
    """The grid as SU2 file text (markers from ``tag_boundary``)."""
    import tempfile

    from egg.io.su2 import export_su2

    fd, p = tempfile.mkstemp(suffix=".su2")
    try:
        os.close(fd)
        export_su2(grid, p)
        return Path(p).read_text()
    finally:
        os.unlink(p)


def su2_export_text(code: str, path: str | None = None) -> tuple[str | None, str]:
    """Exec ``code`` and export its grid as SU2 text.

    Returns ``(text, "")`` on success, ``(None, reason)`` otherwise. The
    registered run's grid wins over a harvested one, mirroring what the
    view shows. Execs the script — call it in a worker process.
    """
    ns, _out, err = exec_script(code, path)
    if err is not None:
        return None, f"script error:\n{err}"
    h = harvest(ns)
    if (reg := ns.get("__egg_webui_run__")) is not None:
        h.grid = reg[0]
    if h.grid is None:
        return None, "no grid to export"
    try:
        return grid_to_su2_text(h.grid), ""
    except Exception as exc:
        return None, f"export failed: {exc}"


def lmr_export_to_dir(
    code: str, out_dir: str, path: str | None = None
) -> tuple[dict | None, str]:
    """Exec ``code`` and export its grid as lmr structured blocks + grid.lua.

    Writes the per-block grid files and ``grid.lua`` into ``out_dir`` and
    returns ``({"written": [...], "untagged": [...]}, "")`` on success,
    ``(None, reason)`` otherwise. ``untagged`` reports the ``egg-untagged-N``
    marker groups (empty when every external face is tagged) so the UI can warn.
    Multi-file, so it writes directly rather than returning text; the registered
    run's grid wins over a harvested one, mirroring what the view shows. Execs
    the script — call it in a worker process.
    """
    ns, _out, err = exec_script(code, path)
    if err is not None:
        return None, f"script error:\n{err}"
    h = harvest(ns)
    if (reg := ns.get("__egg_webui_run__")) is not None:
        h.grid = reg[0]
    if h.grid is None:
        return None, "no grid to export"
    try:
        from egg.io.lmr import export_lmr, untagged_external_faces

        # The /export/lmr route already gates the overwrite conflict, so write
        # unconditionally here.
        written = export_lmr(h.grid, out_dir, overwrite=True)
        return {"written": written, "untagged": untagged_external_faces(h.grid)}, ""
    except Exception as exc:
        return None, f"export failed: {exc}"


def _iter_named(ns: dict | list, depth: int = HARVEST_DEPTH, prefix: str = ""):
    """Yield (name, obj) for values, recursing ``depth`` levels into containers."""
    items = (
        ns.items()
        if isinstance(ns, dict)
        else ((f"[{i}]", v) for i, v in enumerate(ns))
    )
    for name, obj in items:
        if isinstance(name, str) and not prefix and name.startswith("_"):
            continue
        label = f"{prefix}[{name!r}]" if prefix else str(name)
        if isinstance(obj, (list, tuple, set, dict)) and depth > 0:
            inner = obj if isinstance(obj, dict) else list(obj)
            yield from _iter_named(inner, depth - 1, prefix=label)
        else:
            yield label, obj


def harvest(ns: dict, init_grid: bool = True) -> Harvest:
    """Collect renderables + live handles from an executed script namespace.

    ``init_grid=False`` skips the harvest-side TFI initialization (used by
    the topology view, which never draws the grid).
    """
    scene = Scene()
    h = Harvest(scene)
    seen: set[int] = set()

    for name, obj in _iter_named(ns):
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, Edge):
            _add_curve(scene, name, obj.entity, seen)
        elif isinstance(obj, SvgDomain):
            # an imported Inkscape file: draw every labeled path under the
            # name it carries in the Objects panel (dom['egg'], ...)
            for item in obj:
                if id(item.entity) not in seen:
                    _add_curve(scene, f"{name}[{item.label!r}]", item.entity, seen)
        elif isinstance(obj, GeometryEntity):
            _add_curve(scene, name, obj, seen)
        elif isinstance(obj, Vector3):
            kind = (
                "node" if isinstance(obj, Node) else ("fixed" if obj.fixed else "point")
            )
            scene.points.append((name, kind, obj.x, obj.y))
        elif isinstance(obj, MultiBlockGrid) and h.grid is None:
            h.grid = obj
        elif isinstance(obj, BlockTopology) and h.topo is None:
            h.topo = obj
        elif isinstance(obj, TopologyBuilder) and h.topo is None and h.grid is None:
            try:
                h.topo = obj.build()
            except Exception as exc:
                scene.warnings.append(f"builder.build() failed: {exc}")
        elif isinstance(obj, ExplicitTopology) and h.editable is None:
            # A drawn topology is authoritative for the edit/topo layer: a valid
            # flatten becomes the topo (overriding a programmatic one and even
            # coexisting with a run grid), an invalid one surfaces its located
            # reasons as warnings.
            h.editable = obj
            try:
                topo, h.diagnostics = obj.flatten()
                if topo is not None:
                    h.topo = topo
                scene.warnings += [f"topology: {d.msg}" for d in h.diagnostics]
                _debug_blocking(obj.connectivity, h.diagnostics)
            except Exception as exc:
                scene.warnings.append(f"ExplicitTopology.flatten() failed: {exc}")
                _debug_blocking(obj.connectivity, [f"flatten() raised: {exc}"])

    if init_grid and h.grid is None and h.topo is not None:
        try:
            h.grid = h.topo.initialize_grid()
        except Exception as exc:
            scene.warnings.append(f"initialize_grid() failed: {exc}")
    if h.grid is not None and h.topo is None:
        h.topo = h.grid.topology
    # Guardrail: run() relaxes the grid it is handed. If an editable topology is
    # in scope but that grid was not built from it, the edits are drawn but never
    # smoothed — a silent footgun, so say so.
    _errs = [d for d in h.diagnostics if not d.kind.startswith("warn")]
    if h.editable is not None and h.grid is not None and not _errs:
        gtopo = getattr(h.grid, "topology", None)
        if (
            gtopo is not None
            and h.topo is not None
            and not _topologies_match(gtopo, h.topo)
        ):
            scene.warnings.append(
                "the grid handed to run() was not built from the edited topology "
                "— its edits are shown here but will NOT be smoothed. Rebuild "
                "`grid = editable_topo.build().initialize_grid()` (and its target) "
                "before run()."
            )
    refresh_grid_layer(scene, h.grid)
    refresh_net_layer(scene, h.grid)
    _harvest_topology(scene, h.topo)
    if h.topo is not None:
        scene.block_colors = _color_blocks(h.topo)
    return h


def _color_blocks(topo) -> dict[str, int]:
    """Greedy graph-colouring of blocks by shared-face adjacency: no two blocks
    that share an interface get the same colour index. Block faces cap the degree
    at 2*d, so a handful of classes always suffices. Deterministic (block order)."""
    names = list(topo.block_specs)
    adj: dict[str, set] = {n: set() for n in names}
    for conn in getattr(topo, "interface_connections", []):
        a, b = conn.face_a.block_name, conn.face_b.block_name
        if a in adj and b in adj and a != b:
            adj[a].add(b)
            adj[b].add(a)
    color: dict[str, int] = {}
    for n in names:
        used = {color[m] for m in adj[n] if m in color}
        k = 0
        while k in used:
            k += 1
        color[n] = k
    return color


def _topologies_match(a, b) -> bool:
    """Same corner names at the same positions — a cheap check for whether a
    grid was built from a given topology (identity is lost across a re-flatten)."""
    ca, cb = a.corners, b.corners
    if set(ca) != set(cb):
        return False
    return all(
        np.allclose(
            np.asarray(ca[k].position)[:2], np.asarray(cb[k].position)[:2], atol=1e-9
        )
        for k in ca
    )


def _harvest_topology(scene: Scene, topo) -> None:
    """Fill the topology-view primitives from a 2D BlockTopology.

    Entity ids (``id(entity)``) link topology elements to the geometry
    curves that will constrain them, for the click-to-highlight UI. Corner
    styling mirrors :func:`egg.io.visualize.plot_topology`: black block
    edges, blue fixed / green free corners, singular valences called out.
    """
    if topo is None or getattr(topo, "d", 0) != 2:
        return
    pos = {name: c.position[:2] for name, c in topo.corners.items()}

    # face -> constraining entity id; corner/block -> ids of face entities.
    corner_eids: dict[str, set[int]] = {}
    block_eids: dict[str, set[int]] = {}
    for assoc in topo.associations:
        f = assoc.face
        spec = topo.block_specs[f.block_name]
        eid = id(assoc.entity)
        block_eids.setdefault(f.block_name, set()).add(eid)
        a_name, b_name = spec.face_corner_names(f.axis, f.side, 2)
        for cname in (a_name, b_name):
            corner_eids.setdefault(cname, set()).add(eid)
        a, b = pos[a_name], pos[b_name]
        label = f"{f.block_name} face -> {type(assoc.entity).__name__}"
        scene.topo_associations.append((label, np.array([a, b]), eid))

    # corner name -> singular valence (regular interior valence is 2*d).
    singular: dict[str, int] = {}
    for s in topo.singularities:
        spec = topo.block_specs[s.block_name]
        shape = spec.logical_shape
        for ci, cname in zip(spec._corner_logical_indices(2), spec.corner_names):
            actual = tuple(0 if c == 0 else shape[dim] - 1 for dim, c in enumerate(ci))
            if actual == s.logical_idx:
                singular[cname] = s.valence

    for name, spec in topo.block_specs.items():
        # corner_names is in product((0,1), repeat=2) order: sw, nw, se, ne.
        sw, nw, se, ne = (pos[c] for c in spec.corner_names)
        scene.topo_blocks.append(
            (name, np.array([sw, se, ne, nw]), sorted(block_eids.get(name, ())))
        )
    for name, c in topo.corners.items():
        scene.topo_corners.append(
            (
                name,
                float(c.position[0]),
                float(c.position[1]),
                c.fixed,
                singular.get(name),
                sorted(corner_eids.get(name, ())),
            )
        )
    for conn in topo.interface_connections:
        fa = conn.face_a
        spec = topo.block_specs[fa.block_name]
        a, b = (pos[c] for c in spec.face_corner_names(fa.axis, fa.side, 2))
        label = f"{fa.block_name} <-> {conn.face_b.block_name}"
        scene.topo_connections.append((label, np.array([a, b])))


def control_net_state(grid) -> list[np.ndarray] | None:
    """Per-block control lattices of the grid's live control net (2D only).

    The streamable form of the net overlay: the run worker ships this with
    each step frame so the lattice animates alongside the grid while the
    control phase moves it (``None`` when there is no 2D net yet).
    """
    topo = getattr(grid, "control_net", None) if grid is not None else None
    if topo is None or getattr(topo, "d", 0) != 2:
        return None
    C = np.asarray(topo.R @ topo.q, dtype=float)
    return [
        C[topo.block_offsets[bi] : topo.block_offsets[bi + 1]]
        .reshape(tuple(topo.ctrl_shapes[bi]) + (topo.d,))
        .copy()
        for bi in range(len(topo.ctrl_shapes))
    ]


def refresh_net_layer(
    scene: Scene, grid: MultiBlockGrid | None, blocks: list[np.ndarray] | None = None
) -> None:
    """(Re)build the control-net overlay from the grid's control net.

    Populated when the grid carries a ``control_net`` (a solved
    :class:`~egg.smoothing.control_topology.ControlTopology`, either from a
    ``tmop_smoother="control_point"`` run or loaded from a saved ``.npz``).
    ``blocks`` supplies the per-block lattices directly (the run reader's
    streamed :func:`control_net_state` — the server-side grid carries no net
    mid-run). 2D scenes only — the 3D view has its own renderer.
    """
    scene.net_blocks = []
    if blocks is None:
        blocks = control_net_state(grid)
        if blocks is None:
            return
    if not blocks or np.asarray(blocks[0]).shape[-1] != 2:
        return
    specs = getattr(getattr(grid, "topology", None), "block_specs", {})
    names = list(specs) if len(specs) == len(blocks) else []
    for bi, C in enumerate(blocks):
        name = names[bi] if names else f"block {bi}"
        scene.net_blocks.append((name, np.asarray(C, dtype=float)[..., :2]))


def refresh_grid_layer(scene: Scene, grid: MultiBlockGrid | None) -> None:
    """(Re)point the scene's grid layer at the grid's per-block node arrays."""
    scene.grid_blocks = []
    scene.grid_block_names = []
    if grid is None:
        return
    n = sum(int(np.prod(b.nodes.shape[:-1])) for b in grid.blocks)
    if n > MAX_GRID_NODES:
        scene.warnings.append(f"grid too large to preview ({n} nodes)")
        return
    # initialize_grid builds blocks in block_specs order, so names align.
    specs = getattr(getattr(grid, "topology", None), "block_specs", {})
    names = list(specs) if len(specs) == len(grid.blocks) else []
    for bi, b in enumerate(grid.blocks):
        if b.nodes.ndim == 3 and not np.isnan(b.nodes).any():
            scene.grid_blocks.append(b.nodes)
            scene.grid_block_names.append(names[bi] if names else f"block {bi}")


def _add_curve(scene: Scene, name: str, ent: GeometryEntity, seen: set[int]) -> None:
    """Sample a parametric entity into a world-space polyline."""
    seen.add(id(ent))
    if not (hasattr(ent, "eval") and hasattr(ent, "t0")):
        return
    kind = type(ent).__name__
    n = COMPOSITE_SAMPLES if hasattr(ent, "segments") else CURVE_SAMPLES
    try:
        pts = np.array([ent.eval_frac(t)[:2] for t in np.linspace(0.0, 1.0, n)])
    except Exception as exc:
        scene.warnings.append(f"could not sample {name}: {exc}")
        return
    if np.isfinite(pts).all():
        scene.curves.append((name, kind, pts, id(ent)))
    _add_ctrl_cage(scene, name, ent)


def _pt_kind(v: Vector3) -> str:
    return "node" if isinstance(v, Node) else ("fixed" if v.fixed else "point")


def _ctrl_points_of(ent) -> list[Vector3]:
    """The Vector3s ``ent`` was constructed from, in construction order.

    The 2D front-end retains its constructor arguments gdtk-style:
    ``Spline``/``Bezier`` keep ``.points``, ``Line`` keeps ``p0``/``p1``,
    ``Arc`` keeps ``p0``/``centre``/``p1`` (centre in the middle so the
    cage draws the two dashed radius lines).
    """
    pts = getattr(ent, "points", None)
    if isinstance(pts, (list, tuple)) and all(isinstance(p, Vector3) for p in pts):
        return list(pts)
    named = [getattr(ent, a, None) for a in ("p0", "centre", "p1")]
    return [p for p in named if isinstance(p, Vector3)]


def _add_ctrl_cage(scene: Scene, name: str, ent) -> None:
    """Collect a curve's construction points as a CAD-style control cage.

    One cage per curve (or per segment of a composite that doesn't retain
    its own through-points): the ordered construction ``Vector3``\\ s plus
    a per-vertex kind. A closed composite whose retained points don't
    repeat the first one gets the closing chord back so the dashed cage
    wraps like the curve does.
    """
    pts = _ctrl_points_of(ent)
    if pts:
        kinds = [_pt_kind(p) for p in pts]
        xy = np.array([[p.x, p.y] for p in pts], dtype=float)
        if xy.shape[0] >= 3 and not np.allclose(xy[0], xy[-1]):
            try:
                end = np.asarray(ent.eval_frac(1.0), dtype=float)[:2]
                if np.allclose(end, xy[0], atol=1e-12):
                    xy = np.vstack([xy, xy[0:1]])
                    kinds.append(kinds[0])
            except Exception:
                pass
        scene.ctrl_cages.append((name, xy, kinds))
        return
    # Composite without retained points (e.g. a Polyline of Arc/Line
    # segments): each segment contributes its own cage.
    for i, seg in enumerate(getattr(ent, "segments", ())):
        seg_pts = _ctrl_points_of(seg)
        if seg_pts:
            scene.ctrl_cages.append(
                (
                    f"{name}[{i}]",
                    np.array([[p.x, p.y] for p in seg_pts], dtype=float),
                    [_pt_kind(p) for p in seg_pts],
                )
            )


def grid_quality(grid_blocks: list[np.ndarray]) -> dict[str, tuple] | None:
    """Per-cell quality statistics over all blocks.

    Three standard structured-quad measures plus a size-uniformity one,
    each as ``(mean, std, worst)`` over every cell of every block:

    - ``orthogonality`` — the cell's smallest interior corner angle in
      degrees (90 is a perfect quad; small values mean sheared cells);
      worst = min.
    - ``skewness`` — equiangular skew ``max(θmax−90, 90−θmin)/90``
      (0 perfect, 1 degenerate); worst = max.
    - ``aspect ratio`` — mean i-edge length over mean j-edge length,
      folded to ≥ 1; worst = max.
    - ``cell size`` — signed cell areas, reported as (CV, std, max/min);
      the shape metrics don't control this (scale-invariant), only
      ``tmop_metric="shape_size"`` or resolution allocation do.
    """
    ortho, skew, aspect, area = [], [], [], []
    for nodes in grid_blocks:
        p00 = nodes[:-1, :-1, :2]
        p10 = nodes[1:, :-1, :2]
        p01 = nodes[:-1, 1:, :2]
        p11 = nodes[1:, 1:, :2]

        def angle(a, b):
            num = (a * b).sum(-1)
            den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
            return np.degrees(np.arccos(np.clip(num / np.maximum(den, 1e-300), -1, 1)))

        th = np.stack(
            [
                angle(p10 - p00, p01 - p00),
                angle(p11 - p10, p00 - p10),
                angle(p01 - p11, p10 - p11),
                angle(p00 - p01, p11 - p01),
            ]
        )
        ortho.append(th.min(axis=0).ravel())
        skew.append(
            (np.maximum(th.max(axis=0) - 90.0, 90.0 - th.min(axis=0)) / 90.0).ravel()
        )
        li = 0.5 * (
            np.linalg.norm(p10 - p00, axis=-1) + np.linalg.norm(p11 - p01, axis=-1)
        )
        lj = 0.5 * (
            np.linalg.norm(p01 - p00, axis=-1) + np.linalg.norm(p11 - p10, axis=-1)
        )
        r = li / np.maximum(lj, 1e-300)
        aspect.append(np.maximum(r, 1.0 / np.maximum(r, 1e-300)).ravel())

        def cross2(u, v):
            return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

        area.append(
            (
                0.5 * (cross2(p10 - p00, p01 - p00) + cross2(p01 - p11, p10 - p11))
            ).ravel()
        )
    if not ortho:
        return None

    def stats(parts, worst):
        v = np.concatenate(parts)
        return (float(v.mean()), float(v.std()), float(worst(v)))

    a = np.concatenate(area)
    size_cv = float(a.std() / max(abs(float(a.mean())), 1e-300))
    size_ratio = float(a.max() / a.min()) if float(a.min()) > 0.0 else float("inf")
    return {
        "orthogonality": stats(ortho, np.min),
        "skewness": stats(skew, np.max),
        "aspect ratio": stats(aspect, np.max),
        "cell size": (size_cv, float(a.std()), size_ratio),
    }


def _line_indices(n: int, cap: int | None = None) -> np.ndarray:
    """Grid-line indices to draw for a block axis of ``n`` lines.

    Dense blocks are decimated to ``MAX_GRID_LINES`` evenly spaced lines
    (both boundary lines always kept), so the per-frame SVG stays bounded
    on large runs — beyond ~a hundred lines per block they'd be
    sub-pixel anyway. The exported/streamed *grid* is never touched,
    only its preview.
    """
    cap = MAX_GRID_LINES if cap is None else cap
    if n <= cap:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, cap)).astype(int))


def _cell_min_dets(nodes: np.ndarray) -> np.ndarray:
    """Min corner-Jacobian det per cell (edge-based preview, not TMOP samples)."""

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    dxi = nodes[1:, :, :2] - nodes[:-1, :, :2]
    dxj = nodes[:, 1:, :2] - nodes[:, :-1, :2]
    return np.minimum(
        np.minimum(cross(dxi[:, :-1], dxj[:-1, :]), cross(dxi[:, 1:], dxj[:-1, :])),
        np.minimum(cross(dxi[:, :-1], dxj[1:, :]), cross(dxi[:, 1:], dxj[1:, :])),
    )


_SPARK_SLUGS = {"energy": "energy", "min det": "mindet"}


def render_sparkline(
    history: dict[str, list[float]], height=118, prev: dict | None = None
) -> str:
    """Convergence chart: normalized polylines plus axes labels + legend.

    The polylines live in a nested SVG that stretches to the pane width
    (``preserveAspectRatio="none"``); all text sits in the outer,
    viewBox-free SVG (pixel/percent units), so stretching never distorts
    it. Each series is min-max normalized to its *own* range, so there is
    no shared y-axis — instead the first series' range is labelled on the
    left edge and the second's on the right, coloured to match its line.
    The x axis counts pipeline steps. Styling lives in CSS
    (``.spark-<slug>``, ``.spark-lab``, ``.spark-leg``, ``.spark-tick``).

    ``prev`` — the previous run's history — draws each matching series
    faded underneath the live one, normalized over both runs together so
    the two are directly comparable on the same scale (the edge labels
    then span the combined range).
    """
    series = {k: v for k, v in history.items() if len(v) >= 2}
    if not series:
        return ""
    prev = {k: v for k, v in (prev or {}).items() if len(v) >= 2}
    plot_h, vb_w, pad = 100.0, 600.0, 6.0

    def _range(name, vals):
        allv = list(vals) + list(prev.get(name, []))
        return min(allv), max(allv)

    out = [
        f'<svg class="sparkchart" width="100%" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg">',
        f'<svg width="100%" height="{plot_h:g}" viewBox="0 0 {vb_w:g} {plot_h:g}" '
        f'preserveAspectRatio="none">',
    ]
    for name, vals in series.items():
        lo, hi = _range(name, vals)
        span = max(hi - lo, 1e-30)

        def y(v):
            return pad + (1.0 - (v - lo) / span) * (plot_h - 2 * pad)

        def line(values):
            return " ".join(
                f"{i / (len(values) - 1) * vb_w:.1f},{y(v):.1f}"
                for i, v in enumerate(values)
            )

        slug = _SPARK_SLUGS.get(name, "other")
        if lo < 0.0 < hi:  # min det crossing zero: mark the fold boundary
            out.append(
                f'<line x1="0" y1="{y(0.0):.1f}" x2="{vb_w:g}" y2="{y(0.0):.1f}" '
                f'class="spark-zero spark-{slug}"/>'
            )
        if name in prev:
            out.append(
                f'<polyline points="{line(prev[name])}" '
                f'class="spark spark-{slug} spark-prev">'
                f"<title>{_esc(name)} (previous run)</title></polyline>"
            )
        out.append(
            f'<polyline points="{line(vals)}" class="spark spark-{slug}">'
            f"<title>{_esc(name)}</title></polyline>"
        )
    out.append("</svg>")
    # Per-series y-range labels: first series left edge, second right edge.
    for i, (name, vals) in enumerate(series.items()):
        if i > 1:
            break
        slug = _SPARK_SLUGS.get(name, "other")
        x, anchor = ("4", "start") if i == 0 else ("99.5%", "end")
        lo, hi = _range(name, vals)
        for val, ty in ((hi, 13.0), (lo, plot_h - 4.0)):
            out.append(
                f'<text x="{x}" y="{ty:g}" text-anchor="{anchor}" '
                f'class="spark-lab spark-{slug}">{val:.3g}</text>'
            )
    # Legend, centered-ish along the top (percent positions: no text
    # metrics server-side, so entries get fixed slots).
    for i, name in enumerate(series):
        slug = _SPARK_SLUGS.get(name, "other")
        x0 = 32 + 22 * i
        out.append(
            f'<line x1="{x0}%" y1="9" x2="{x0 + 3}%" y2="9" '
            f'class="spark spark-{slug}"/>'
        )
        out.append(
            f'<text x="{x0 + 3.7}%" y="13" class="spark-leg">{_esc(name)}</text>'
        )
    # x axis: pipeline steps (series with fewer samples still stretch to
    # the full width, so the count is exact only for the longest one).
    n = max(len(v) for v in series.values())
    ticks = (
        ("0.5%", "start", "0"),
        ("50%", "middle", f"{(n - 1) // 2}"),
        ("99.5%", "end", f"{n - 1} steps"),
    )
    for tx, _, _ in (("0%", "", ""), ("50%", "", ""), ("100%", "", "")):
        out.append(
            f'<line x1="{tx}" y1="{plot_h:g}" x2="{tx}" y2="{plot_h + 4:g}" '
            f'class="spark-tick"/>'
        )
    for tx, anchor, lab in ticks:
        out.append(
            f'<text x="{tx}" y="{height - 4}" text-anchor="{anchor}" '
            f'class="spark-xlab">{lab}</text>'
        )
    out.append("</svg>")
    return "".join(out)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def scene_bounds(scene: Scene) -> Bounds | None:
    """Padded world-space bbox of everything drawable, or None if empty."""
    # NB: ctrl cages are excluded on purpose — the layer toggle is
    # client-side, and showing/hiding construction points must not
    # reshape the frame (a far-away arc centre may clip; use zoom).
    all_pts = [c[2] for c in scene.curves]
    all_pts += [np.array([[x, y]]) for _, _, x, y in scene.points]
    all_pts += [b.reshape(-1, b.shape[-1])[:, :2] for b in scene.grid_blocks]
    all_pts += [quad for _, quad, _eids in scene.topo_blocks]
    if not all_pts:
        return None
    stacked = np.vstack(all_pts)
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    pad = 0.05 * float(np.maximum(hi - lo, 1e-12).max())
    return lo - pad, hi + pad


def render_svg(
    scene: Scene,
    width: int = 1200,
    height: int = 900,
    bounds: Bounds | None = None,
    mode: str = "grid",
) -> str:
    """Render the scene; pass ``bounds`` to keep a fixed frame across renders.

    ``mode="grid"`` draws the structured grid + tangled-cell overlay;
    ``mode="topo"`` draws the block topology (outlines, connections,
    associated faces, corners) instead. Curves and points draw in both.
    """
    if bounds is None:
        bounds = scene_bounds(scene)
    if bounds is None:
        return (
            f'<svg id="scene-svg" class="mode-{mode}" viewBox="0 0 {width} {height}" '
            f'data-fit="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width / 2}" y="{height / 2}" text-anchor="middle" '
            f'class="empty-note">nothing to draw yet</text></svg>'
        )
    lo, hi = bounds
    s = min(width / (hi[0] - lo[0]), height / (hi[1] - lo[1]))
    ox = (width - s * (hi[0] - lo[0])) / 2.0
    oy = (height - s * (hi[1] - lo[1])) / 2.0

    def T(p) -> tuple[float, float]:
        return (ox + s * (p[0] - lo[0]), oy + s * (hi[1] - p[1]))

    def poly(pts: np.ndarray) -> str:
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in (T(p) for p in pts))

    # whole-grid cell count, shown when a click lands outside every block
    total = ""
    if scene.grid_blocks:
        n_cells = sum((n.shape[0] - 1) * (n.shape[1] - 1) for n in scene.grid_blocks)
        total = (
            f'data-cells="grid: {n_cells} cells in {len(scene.grid_blocks)} blocks" '
        )
    out: list[str] = [
        f'<svg id="scene-svg" class="mode-{mode}" viewBox="0 0 {width} {height}" '
        f'data-fit="0 0 {width} {height}" {total}'
        # inverse world transform for the client-side coordinate readout
        f'data-sx="{s:.10g}" data-ox="{ox:.6f}" data-oy="{oy:.6f}" '
        f'data-lox="{lo[0]:.10g}" data-hiy="{hi[1]:.10g}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]

    # NB: all styling (color, width, dash) lives in CSS behind --egg-*
    # custom properties — elements carry semantic classes only, so the
    # whole view is user-themeable. Keep geometry-only attributes here.
    if mode == "grid":
        # Grid layer per block: a soft per-block color fill under the
        # lines, then one polyline per structured line, outline on top.
        out.append('<g class="layer-grid">')
        seen_edges: set[frozenset] = set()
        for bi, nodes in enumerate(scene.grid_blocks):
            ring = np.concatenate(
                [
                    nodes[:, 0, :2],
                    nodes[-1, :, :2],
                    nodes[::-1, -1, :2],
                    nodes[0, ::-1, :2],
                ]
            )
            name = (
                scene.grid_block_names[bi]
                if bi < len(scene.grid_block_names)
                else f"block {bi}"
            )
            # adjacency-aware colour (falls back to index if the block is unknown)
            b = f"b{scene.block_colors.get(name, bi) % N_BLOCK_CLASSES}"
            ni, nj = nodes.shape[0] - 1, nodes.shape[1] - 1
            label = f"{name}: {ni} × {nj} cells = {ni * nj}"
            out.append(f'<g class="grid-block" data-label="{_esc(label)}">')
            out.append(f'<polygon points="{poly(ring)}" class="block-fill {b}"/>')
            out.append(f'<g class="grid-lines {b}">')
            n0, n1 = nodes.shape[0], nodes.shape[1]
            for i in _line_indices(n0):
                if i in (0, n0 - 1):
                    key = frozenset((tuple(nodes[i, 0, :2]), tuple(nodes[i, -1, :2])))
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                out.append(f'<polyline points="{poly(nodes[i, :, :2])}"/>')
            for j in _line_indices(n1):
                if j in (0, n1 - 1):
                    key = frozenset((tuple(nodes[0, j, :2]), tuple(nodes[-1, j, :2])))
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                out.append(f'<polyline points="{poly(nodes[:, j, :2])}"/>')
            out.append("</g>")
            out.append(f'<polyline points="{poly(ring)}" class="grid-outline {b}"/>')
            out.append("</g>")
        out.append("</g>")

        # Tangled cells (min corner det <= 0), translucent fill via CSS.
        out.append('<g class="layer-tangled">')
        shown = 0
        for nodes in scene.grid_blocks:
            if shown >= MAX_TANGLED_CELLS:
                break
            bad = np.argwhere(_cell_min_dets(nodes) <= 0.0)
            for i, j in bad[: MAX_TANGLED_CELLS - shown]:
                quad = nodes[[i, i + 1, i + 1, i], [j, j, j + 1, j + 1], :2]
                out.append(f'<polygon points="{poly(quad)}"/>')
            shown += len(bad)
        out.append("</g>")
    out.append('<g class="layer-curves">')
    for k, (name, kind, pts, eid) in enumerate(scene.curves):
        out.append(
            f'<polyline points="{poly(pts)}" '
            f'class="curve c{k % N_CURVE_CLASSES}" data-eid="{eid}">'
            f"<title>{_esc(name)}: {_esc(kind)}</title></polyline>"
        )
    out.append("</g>")

    out.append('<g class="layer-points">')
    for name, kind, x, y in scene.points:
        cx, cy = T((x, y))
        title = f"<title>{_esc(name)} ({x:g}, {y:g})</title>"
        if kind == "fixed":
            out.append(
                f'<rect x="{cx - 3.5:.2f}" y="{cy - 3.5:.2f}" width="7" height="7" '
                f'class="pt pt-fixed">{title}</rect>'
            )
        elif kind == "node":
            out.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4" '
                f'class="pt pt-node">{title}</circle>'
            )
        else:
            out.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3.2" '
                f'class="pt">{title}</circle>'
            )
    out.append("</g>")

    # Construction points + dashed control polygon (toggleable ctrl layer).
    out.append('<g class="layer-ctrl">')
    for name, cage, kinds in scene.ctrl_cages:
        if cage.shape[0] >= 3:
            out.append(
                f'<polyline points="{poly(cage)}" class="ctrl-cage">'
                f"<title>{_esc(name)}</title></polyline>"
            )
        for i, ((x, y), kind) in enumerate(zip(cage, kinds)):
            if cage.shape[0] >= 3 and i == cage.shape[0] - 1 and kinds[0] == kind:
                if np.allclose(cage[0], cage[-1]):
                    continue  # closing chord vertex, marker already drawn
            cx, cy = T((x, y))
            title = f"<title>{_esc(name)}[{i}] ({x:g}, {y:g})</title>"
            if kind == "fixed":
                out.append(
                    f'<rect x="{cx - 3:.2f}" y="{cy - 3:.2f}" width="6" height="6" '
                    f'class="ctrl-pt fixed">{title}</rect>'
                )
            else:
                cls = "ctrl-pt node" if kind == "node" else "ctrl-pt"
                out.append(
                    f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="3" '
                    f'class="{cls}">{title}</circle>'
                )
    out.append("</g>")

    # Solved control-net overlay (toggleable net layer): the B-spline control
    # lattice per block as dashed polylines with point markers.
    if mode == "grid" and scene.net_blocks:
        out.append('<g class="layer-net">')
        for name, C in scene.net_blocks:
            for i in range(C.shape[0]):
                out.append(
                    f'<polyline points="{poly(C[i])}" class="net-line">'
                    f"<title>{_esc(name)} control net</title></polyline>"
                )
            for j in range(C.shape[1]):
                out.append(
                    f'<polyline points="{poly(C[:, j])}" class="net-line">'
                    f"<title>{_esc(name)} control net</title></polyline>"
                )
            for (i, j), _ in np.ndenumerate(C[..., 0]):
                cx, cy = T(C[i, j])
                out.append(
                    f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.5" '
                    f'class="net-pt"><title>{_esc(name)} C[{i},{j}] '
                    f"({C[i, j, 0]:g}, {C[i, j, 1]:g})</title></circle>"
                )
        out.append("</g>")

    if mode in ("topo", "edit"):
        # Topology lines ON TOP of the (dimmed) geometry, styled like
        # egg.io.visualize.plot_topology: all topology lines solid black;
        # clicking an element highlights it plus the geometry (data-eid
        # curves) that will constrain it. The edit view reuses this layout.
        out.append('<g class="layer-blocks">')
        for name, quad, eids in scene.topo_blocks:
            eid_attr = f' data-eids="{",".join(map(str, eids))}"' if eids else ""
            out.append(
                f'<polygon points="{poly(quad)}" class="topo-sel topo-block"'
                f"{eid_attr}><title>block {_esc(name)}</title></polygon>"
            )
        for label, seg in scene.topo_connections:
            out.append(
                f'<polyline points="{poly(seg)}" class="topo-sel topo-conn">'
                f"<title>{_esc(label)}</title></polyline>"
            )
        # Associated faces: invisible wide hit-targets over the block edge
        # carrying the entity link (visible via hover title / click).
        for label, seg, eid in scene.topo_associations:
            out.append(
                f'<polyline points="{poly(seg)}" class="topo-sel topo-assoc" '
                f'data-eids="{eid}"><title>{_esc(label)}</title></polyline>'
            )
        out.append("</g>")

        # Topology corners on top of everything. Colors mirror
        # plot_topology: blue = fixed, green = free; singular corners get
        # a red marker box and the valence in the tooltip.
        out.append('<g class="layer-corners">')
        for name, x, y, fixed, valence, eids in scene.topo_corners:
            cx, cy = T((x, y))
            eid_attr = f' data-eids="{",".join(map(str, eids))}"' if eids else ""
            label = f"{name} ({x:g}, {y:g})" + (
                f" — singular, valence {valence}" if valence is not None else ""
            )
            title = f"<title>{_esc(label)}</title>"
            if valence is not None:
                out.append(
                    f'<rect x="{cx - 8:.2f}" y="{cy - 8:.2f}" width="16" height="16" '
                    f'class="singular-box"/>'
                )
            kind = "fixed" if fixed else "free"
            out.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="5" '
                f'class="topo-sel corner {kind}"{eid_attr}>{title}</circle>'
            )
        out.append("</g>")

    out.append("</svg>")
    return "".join(out)
