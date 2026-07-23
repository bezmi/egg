# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""FastHTML "code as CAD" prototype: live SVG view of an egg geometry script.

Run (after ``uv sync --group webui``)::

    egg-webui [path/to/script.py]

Left pane: a Python script using the egg 2D front-end. Right pane: an SVG
render of whatever the script defines (curves, points, and — if it builds
a topology — the TFI-initialized grid), re-rendered ~0.5 s after you stop
typing. "run" hands the script to a separate worker process
(``worker.py``) that consumes the pipeline the script *explicitly*
registered via ``egg_webui.run(grid, steps)`` in its
``if __name__ == "__egg_webui__":`` block and streams node updates back
over a pipe; the server renders them into frames and fans them out over
a websocket. No registration → the run button does nothing: the UI never
invents a pipeline behind the script's back. The examples' guards call
their ``setup(parse_args([]))``, so a UI run matches the CLI defaults
exactly.

The script runs with full interpreter privileges (in a persistent render
worker for editor renders — ``render_worker.py`` — and a per-run worker
for runs): local single-user tool, not a deployable service.
"""

from __future__ import annotations

import asyncio
import io
import json
import mimetypes
import os
import pickle
import re
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import time
from typing import Any
from pathlib import Path

from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from ._assets import MISSING_MSG, VENDOR_DIR, vendor_ready
from .lsp import DOC_URI, ROOT_URI, LspBridge, lsp_available

from fasthtml.common import (
    A,
    Button,
    Details,
    Div,
    Header,
    Input,
    Label,
    Link,
    NotStr,
    Option,
    Pre,
    Script,
    Select,
    Span,
    Style,
    Summary,
    Textarea,
    Title,
    fast_app,
    serve,
    to_xml,
)
from .render_worker import RenderWorker
from .scene import (
    SceneResult,
    exec_script,
    grid_to_su2_text,
    grid_quality,
    guard_params,
    harvest,
    editable_block,
    refresh_grid_layer,
    refresh_net_layer,
    render_sparkline,
    render_svg,
    scene_bounds,
    set_editable_blocking,
    set_guard_param,
    visible_params,
)

from egg.pipeline import PipelineConfig


def _find_repo_root(start: Path) -> Path | None:
    """The source-checkout root (holds ``pyproject.toml``/``.git``) above the
    installed ``egg`` package, or ``None`` when running from a packaged wheel.
    Checkout-only features (examples browser, docs) key off this."""
    for p in (start, *start.parents):
        if (p / "pyproject.toml").is_file() or (p / ".git").exists():
            return p
    return None


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())
EXAMPLES_DIR = _REPO_ROOT / "examples" / "2D" if _REPO_ROOT else None
DEFAULT_SCRIPT = EXAMPLES_DIR / "egg" / "egg.py" if EXAMPLES_DIR else None

# The page's CSS / JS / static SVG live under static/ (loaded here, embedded
# inline in the head); keep large front-end assets out of this module.
_STATIC = Path(__file__).resolve().parent / "static"


def _static(name: str) -> str:
    return (_STATIC / name).read_text()


# egg mark: an egg outline with a 3x3 "#" grid clipped inside it. Used inline
# in the header (currentColor, tinted by --ctp-* so it re-themes) and as the
# favicon (app.js recolors that per flavor). Keep the shape in one place.
_EGG_D = "M50 7C65 7 83 35 83 59 83 81 68 93 50 93 32 93 17 81 17 59 17 35 35 7 50 7Z"


def _egg_svg(stroke: str, attrs: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" fill="none" '
        f'stroke="{stroke}" stroke-width="7" stroke-linecap="round" '
        f'stroke-linejoin="round"{attrs}>'
        f'<clipPath id="egg-clip"><path d="{_EGG_D}"/></clipPath>'
        f'<path d="{_EGG_D}"/>'
        f'<g clip-path="url(#egg-clip)" stroke-width="6">'
        f'<path d="M40 15V95M60 15V95M10 48H90M10 66H90"/></g></svg>'
    )


EGG_LOGO = _egg_svg(
    "currentColor",
    ' class="egg-logo" width="26" height="26" role="img" aria-label="egg"',
)
# Default favicon (mocha yellow); app.js re-tints it to the active flavor.
FAVICON = "data:image/svg+xml," + urllib.parse.quote(_egg_svg("#f9e2af"))

# Static axis orientation gizmo (2D pan/zoom never rotates the frame).
AXES_SVG = _static("axes.svg")

CSS = _static("app.css")

# Catppuccin theme for the prism editor (token colors + chrome), driven off the
# same --ctp-* variables as the rest of the UI so it re-themes with the flavor.
CATPPUCCIN = _static("catppuccin.css")

# Tab key in the editor; viewBox-based pan/zoom on the SVG (kept across
# HTMX swaps once the user has interacted, reset by the fit button);
# layer visibility toggles; example loader.
JS = _static("app.js")

# prism-code-editor upgrade, loaded as ES modules from the local /vendor
# mount (never a CDN). Completions come from based-pyright over /lsp, with
# the introspected /api/completions list as a fallback source.
EDITOR_JS = _static("editor.js")

# The user's deep config (delays, keybinds, auto-run policy) exposed to the
# browser as window.eggConfig, read at import so it is set before app.js runs.
# app.js / editor.js read it with defaults, so a missing key changes nothing.
# Config changes take effect on the next server start.
from .config import client_config as _client_config

CONFIG_JS = f"window.eggConfig = {json.dumps(_client_config())};"

# All browser assets are served locally from /vendor (no CDN fallback, by
# design). They are produced offline by tools/vendor_webui.py — at wheel-build
# time, or via `egg-webui --dev` in a checkout. If they are absent there is no
# safe way to serve the UI, so fail fast with an actionable message rather than
# reaching out to the network.
if not vendor_ready():
    raise SystemExit(MISSING_MSG)

_SPLIT_SRC = "/vendor/split.min.js"

# prism-code-editor stylesheets (served locally from /vendor): the required
# layout plus the enabled extensions. The catppuccin token/chrome theme lives
# in app.css (Style(CSS)), loaded after these so it wins.
_PCE_CSS = (
    "layout.css",
    "search.css",
    "autocomplete.css",
    "autocomplete-icons.css",
    "cursor.css",
)

app, rt = fast_app(
    pico=False,
    exts="ws",
    hdrs=(
        Link(rel="icon", type="image/svg+xml", href=FAVICON, id="favicon"),
        *(Link(rel="stylesheet", href=f"/vendor/{c}") for c in _PCE_CSS),
        Style(CSS),
        Style(CATPPUCCIN),
        Script(src=_SPLIT_SRC),
        Script(NotStr(CONFIG_JS)),  # window.eggConfig, before app.js reads it
        Script(JS),
        Script(EDITOR_JS, type="module"),
    ),
)

mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/javascript", ".js")
# Insert FIRST: fasthtml's built-in root static route matches any path with a
# known extension (including .js) and would 404 /vendor/*.js before a
# normally-appended mount is ever consulted.
app.router.routes.insert(
    0, Mount("/vendor", app=StaticFiles(directory=VENDOR_DIR), name="vendor")
)

# The Sphinx site. In an installed wheel it ships under egg/webui/docs
# (built at wheel time by tools/vendor_webui.py --docs). In a checkout it is
# whatever `egg-webui` refreshed into docs/_build/html at startup (also
# `uv run --group docs sphinx-build -b html docs docs/_build/html`).
_PACKAGED_DOCS = _STATIC.parent / "docs"
if _PACKAGED_DOCS.is_dir():
    DOCS_DIR = _PACKAGED_DOCS
elif _REPO_ROOT:
    DOCS_DIR = _REPO_ROOT / "docs" / "_build" / "html"
else:
    DOCS_DIR = None
if DOCS_DIR and DOCS_DIR.is_dir():
    app.router.routes.insert(
        0, Mount("/docs", app=StaticFiles(directory=DOCS_DIR, html=True), name="docs")
    )

# --- websocket fan-out (server -> client frames pushed by the worker) ---

_clients: dict[int, tuple] = {}
# "svg"/"quality": the latest streamed frame, so editor renders during a
# run can keep the relaxing mesh in the canvas instead of flashing back
# to a static TFI render. "log": the script's egg.webui_print lines this
# run (streamed to the #runlog panel; bounded to the last LOG_MAX lines).
_run: dict = {
    "proc": None,
    "reader": None,
    "stop": False,
    "tmp": None,
    "svg": None,
    "quality": None,
    "log": [],
}
# Last completed smoothing run: {"code": str, "harvest": Harvest, "hist":
# convergence series}. The SU2 export uses it so "what you see is what you
# export" after a run; "hist" draws faded under the next run's chart.
_last: dict = {"code": None, "harvest": None, "hist": None, "net": None}

# Resume cache: the node coordinates streamed by the latest *stopped*
# run, plus the code that produced them. While present, the run button shows
# "resume": a fresh run re-execs the (possibly edited) script and seeds the grid
# to this cache before running its stages, so a stopped solve picks up from where
# it left off instead of restarting. Cleared by a full completion or the reset
# button. Survives across runs (unlike ``_run``, which resets every run).
_resume: dict = {"X": None, "code": None}


def _is_resumable() -> bool:
    return _resume["X"] is not None


def _set_resume(X, code: str) -> None:
    _resume.update(X=X, code=code)


def _clear_resume() -> None:
    _resume.update(X=None, code=None)

# Editor renders exec the script in this persistent worker process, not in
# the server: the event loop only awaits, a script stuck in a loop is
# killed at the render timeout, and a native crash in a script costs one
# render instead of the server. Spawned eagerly so the first page load
# finds it warm.
_render_worker = RenderWorker()


async def _render_scene(code: str, mode: str, path: str) -> SceneResult:
    return await asyncio.wrap_future(_render_worker.submit(code, mode, path))


# NB: async on purpose — fasthtml runs sync handlers in a threadpool,
# where get_running_loop() has nothing to return.
async def _on_conn(ws):
    _clients[id(ws)] = (ws, asyncio.get_running_loop())


async def _on_disconn(ws):
    _clients.pop(id(ws), None)


@app.ws("/ws", conn=_on_conn, disconn=_on_disconn)
async def ws(msg: str):
    pass  # server-push only


# --- language server bridge (based-pyright over /lsp) ---
# One bridge (one basedpyright subprocess) per editor connection. The browser
# drives the LSP flow; the reader thread forwards server frames back over the
# same socket. Kept separate from /ws, which is bound to run-frame HTML
# fan-out. Degrades gracefully: if the language server isn't installed we tell
# the client and the editor keeps working without language features.
_lsp_bridges: dict[int, LspBridge] = {}


async def _lsp_conn(ws):
    loop = asyncio.get_running_loop()

    def to_client(msg: dict) -> None:
        asyncio.run_coroutine_threadsafe(ws.send_text(json.dumps(msg)), loop)

    if not lsp_available():
        await ws.send_text(json.dumps({"jsonrpc": "2.0", "method": "egg/unavailable"}))
        return
    _lsp_bridges[id(ws)] = LspBridge(to_client)
    # Tell the browser which workspace/document URIs to use for its LSP flow.
    await ws.send_text(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "egg/ready",
                "params": {"rootUri": ROOT_URI, "docUri": DOC_URI},
            }
        )
    )


async def _lsp_disconn(ws):
    bridge = _lsp_bridges.pop(id(ws), None)
    if bridge:
        bridge.close()


@app.ws("/lsp", conn=_lsp_conn, disconn=_lsp_disconn)
async def lsp_ws(ws, data):
    # fasthtml JSON-parses the incoming frame; `data` is the JSON-RPC message.
    bridge = _lsp_bridges.get(id(ws))
    if bridge and data:
        bridge.from_client(data)


def _broadcast(html: str) -> None:
    """Push pre-rendered HTML (OOB fragments) to every connected client."""
    if not _clients:
        print("webui: no websocket clients connected, frame dropped", flush=True)
    for ws, loop in list(_clients.values()):
        fut = asyncio.run_coroutine_threadsafe(ws.send_text(html), loop)
        fut.add_done_callback(
            lambda f: (
                f.exception()
                and print(f"webui: frame send failed: {f.exception()!r}", flush=True)
            )
        )


def _running() -> bool:
    p = _run["proc"]
    return p is not None and p.poll() is None


# --- view fragments ---


def _fmt_chip(k: str, v) -> str:
    if isinstance(v, float):
        return f"{k} {v:.4g}"
    return f"{k} {v}"


# Lucide "refresh-cw": a reset-to-unsmoothed affordance next to run.
_REFRESH_SVG = (
    "<svg width='14' height='14' viewBox='0 0 24 24' fill='none' "
    "stroke='currentColor' stroke-width='2' stroke-linecap='round' "
    "stroke-linejoin='round'><path d='M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 "
    "2.74L3 8'/><path d='M3 3v5h5'/><path d='M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 "
    "6.74-2.74L21 16'/><path d='M16 16h5v5'/></svg>"
)


def view_bar(*chips, running: bool, oob: bool = False, mode: str = "grid"):
    resumable = _is_resumable()
    runner = (
        [
            Button(
                "resume" if resumable else "run",
                id="run-btn",
                cls="primary",
                hx_post="/run",
                hx_include="[name='code'],[name='path']",
                hx_target="#viewbar",
                hx_swap="outerHTML",
                # A resume passes resume=1 so the worker seeds the grid from the
                # cache; JS warns first if the file changed (see htmx:confirm).
                hx_vals='{"resume": "1"}' if resumable else None,
                data_resume="1" if resumable else None,
                disabled=running or None,
                title=(
                    "resume from the stopped result (Ctrl+Enter)"
                    if resumable
                    else "run (Ctrl+Enter)"
                ),
            ),
            Button(
                "stop",
                cls="danger",
                hx_post="/stop",
                hx_swap="none",
                # Serialize presses so a fast double-click escalates in order:
                # the first request sets the stop flag, the second (queued
                # behind it) then sees it and hard-kills. Without this the two
                # presses can race, both take the "first press" branch, and
                # nothing kills.
                hx_sync="this:queue all",
                disabled=(not running) or None,
                title="stop (Ctrl+Enter); double-click to force kill",
            ),
            Button(
                NotStr(_REFRESH_SVG),
                cls="icon",
                hx_post="/reset",
                hx_include="[name='code'],[name='path'],[name='view']",
                hx_target="#view",
                disabled=running or None,
                title="reset the grid to the unsmoothed state",
                aria_label="reset grid",
            ),
        ]
        if mode == "grid"
        else []
    )
    return Div(
        # Chips live in their own #viewchips box (display:contents, so the flex
        # layout is unchanged). Run frames OOB-swap ONLY #viewchips, never the
        # whole bar — the control buttons keep a stable DOM node, so a stop
        # click always lands on an htmx-bound button instead of one that the
        # last frame just replaced (the old whole-bar swap dropped stop clicks).
        Div(*chips, id="viewchips"),
        Div(
            Select(
                Option("grid view", value="grid", selected=(mode == "grid")),
                Option("topology view", value="topo", selected=(mode == "topo")),
                Option("edit view", value="edit", selected=(mode == "edit")),
                name="view",
                id="viewmode",
                # the browser restores a stale value across reloads; keep it from
                # desyncing with the server-rendered view (JS also resets on load)
                autocomplete="off",
                hx_post="/render",
                hx_include="[name='code'],[name='path']",
                hx_target="#view",
                # run frames render grid-mode; switching views mid-run
                # would be silently undone by the next frame
                disabled=running or None,
                title="stop the run to switch views" if running else None,
            ),
            *runner,
            cls="btns",
        ),
        id="viewbar",
        cls="bar",
        hx_swap_oob="true" if oob else None,
    )


def view_chips(*chips, oob: bool = False):
    """Just the status-chip box inside the view bar. Run frames swap this (not
    the whole bar) so the run/stop/reset controls keep a stable, htmx-bound
    DOM node across the ~16 fps frame stream."""
    return Div(*chips, id="viewchips", hx_swap_oob="true" if oob else None)


def _mindet_chip(md: float | None):
    if md is None:
        return None
    return Span(f"min det {md:.3g}", cls="bad" if md <= 0 else None)


LOG_MAX = 500  # keep the last N webui_print lines; older ones scroll off


def _log_panel(text: str, oob: bool = False):
    """The run's live log (script ``egg.webui_print`` output).

    Empty when the script printed nothing this run, so ``#runlog:empty``
    collapses it out of the layout; otherwise a scrolling ``<pre>``.
    """
    return Div(
        Pre(text) if text else None,
        id="runlog",
        cls="runlog",
        hx_swap_oob="true" if oob else None,
    )


# str-valued guard params with a known closed set of values get a select.
_PARAM_CHOICES = {"smoother": ("jacobi", "fas"), "device": ("cpu", "gpu", "auto")}


def _quality_panel(quality: dict | None, oob: bool = False):
    """Grid-quality strip: one summary line, mean and worst per metric.
    Worst direction is min for orthogonality (degrees, 90 ideal), max for
    skewness and aspect ratio."""
    if not quality:
        return Div(id="quality", hx_swap_oob="true" if oob else None)
    fmt = {
        "orthogonality": ("min angle", "°"),
        "skewness": ("skew", ""),
        "aspect ratio": ("AR", ""),
    }
    chips = []
    for metric, (mean, _std, worst) in quality.items():
        if metric == "cell size":
            # (CV, std, max/min) — uniformity, not a per-cell worst.
            chips.append(Span(f"size CV {mean:.3g} (max/min {worst:.3g})"))
            continue
        short, unit = fmt.get(metric, (metric, ""))
        chips.append(Span(f"{short} {mean:.3g}{unit} (worst {worst:.3g}{unit})"))
    return Div(
        Span("quality", cls="q-tag"),
        *chips,
        id="quality",
        hx_swap_oob="true" if oob else None,
    )


def _params_panel(code: str):
    """A form view over the guard dict's literal entries (collapsed strip
    between the view bar and the canvas). Edits post to /api/param, which
    rewrites the value's exact source span in the code buffer — the code
    stays the single source of truth, the panel is just a lens on it."""
    params = visible_params(guard_params(code))
    if not params:
        return Div(id="params")
    fields = []
    for p in params:
        # A bag of dynamic DOM/htmx attributes splatted into the FT components.
        common: dict[str, Any] = dict(cls="param-input", data_param=p.name)
        cur = (
            p.text[1:-1] if p.kind == "str" and len(p.text) >= 2 else p.text
        )  # unquoted current value, as posted back
        # editable(choices=[...]) marks and the built-in smoother/device
        # entries render dropdowns; option text is what /api/param receives.
        choices = p.choices or (_PARAM_CHOICES.get(p.name) if p.kind == "str" else None)
        if choices is not None and any(str(c) == cur for c in choices):
            inp = Select(
                *[
                    Option(str(c), value=str(c), selected=(str(c) == cur))
                    for c in choices
                ],
                **common,
            )
        elif p.kind == "bool":
            inp = Input(type="checkbox", checked=(p.text == "True") or None, **common)
        elif p.kind == "str":
            inp = Input(type="text", value=cur, **common)
        else:
            # text (not type=number) keeps spellings like 5.0e-3 intact
            inp = Input(type="text", inputmode="decimal", value=p.text, **common)
        fields.append(Label(Span(p.name), inp))
    return Details(
        Summary(f"run parameters ({len(params)})"),
        Div(*fields, cls="param-grid"),
        id="params",
    )


def _edit_disabled_note(r: SceneResult):
    """Banner for edit view when there is nothing to edit, else None.

    Editing needs an :class:`ExplicitTopology` whose ``connectivity`` is wrapped
    in :func:`editable`. Absent that, the edit view still draws the layout but
    the draw tools are off (client-side); this says so, and why.
    """
    if r.edit_data is not None and r.edit_data["editable"]:
        return None  # editable: the draw tools are live, no note
    return Div(
        Span("edit disabled", cls="ed-off-tag"),
        Span(
            "your script needs an ExplicitTopology with an editable "
            "connectivity field to enable editing"
        ),
        id="editnote",
        cls="editnote",
    )


def view_fragment(r: SceneResult, code: str, mode: str = "grid"):
    """Build the #view contents from a worker-rendered scene."""
    running = _running()
    chips = []
    if running:
        # page loaded mid-run: read as running right away — the next
        # frame's OOB view bar brings the real phase chips
        chips.append(Span("running…", cls="phase"))
    chips += [Span(f"{v} {k}") for k, v in r.stats.items() if v]
    if (c := _mindet_chip(r.min_det)) is not None:
        chips.append(c)
    chips.append(Span(f"{r.elapsed_ms} ms"))
    chips += [Span(w, cls="warn") for w in r.warnings]
    # edit view: a green/red validity chip for an editable topology
    if r.edit_data is not None and r.edit_data["editable"]:
        n = len(r.edit_data["diagnostics"])
        chips.append(
            Span(
                "valid" if n == 0 else f"{n} issue{'' if n == 1 else 's'}",
                cls="q-tag" if n == 0 else "bad",
            )
        )
    extras = []
    if r.stdout:
        extras.append(Details(Summary("stdout"), Pre(r.stdout), cls="out"))
    if r.error:
        lines = re.findall(r'File "<script>", line (\d+)', r.error)
        if lines:
            chips.append(
                Span(
                    f"error at line {lines[-1]}", cls="errline bad", data_line=lines[-1]
                )
            )
        extras.append(Pre(r.error, cls="err"))
    # Editing (or reloading) mid-run must not flash the canvas back to a
    # static TFI render: keep the streaming mesh; the fresh exec still
    # supplies the chips, errors, and stdout above/below it.
    run_svg = _run["svg"] if running else None
    svg = run_svg or r.svg
    quality = _run["quality"] if running and run_svg else r.quality
    edit_note = _edit_disabled_note(r) if mode == "edit" else None
    parts = [
        view_bar(*chips, running=running, mode=mode),
        # edit view with nothing editable: say so (draw tools stay hidden)
        *([edit_note] if edit_note is not None else []),
        # run parameters only make sense in grid view (where the run happens);
        # an empty placeholder keeps the id stable and collapses via #params:empty
        _params_panel(code) if mode == "grid" else Div(id="params"),
        _quality_panel(quality),
        Div(NotStr(svg), id="canvas", cls="canvas"),
        Div(id="chart"),
        # live script output (egg.webui_print) — populated by the run reader's
        # OOB frames; seeded here so a mid-run reload keeps what was printed
        _log_panel("".join(_run["log"]) if running else ""),
        Div(*extras, id="viewextra"),
    ]
    # the edit view's structured blocking, for the client draw tools to read
    if r.edit_data is not None:
        parts.append(
            Script(
                # fasthtml's Script accepts a NotStr child (raw, unescaped) at
                # runtime, but its typed signature only declares str.
                NotStr(json.dumps(r.edit_data)),  # type: ignore[arg-type]
                type="application/json",
                id="egg-edit-data",
            )
        )
    return tuple(parts)


# --- smoothing: the pipeline runs in a separate worker process (worker.py);
# --- the server only renders the frames it streams back and fans them out ---


# Run the solver worker as a package module so its `from .scene import …`
# resolves from the installed package (the webui now lives at egg/webui/).
WORKER_MODULE = "egg.webui.worker"

FRAME_INTERVAL = 0.06  # s; skip intermediate frames arriving faster than this
QUALITY_INTERVAL = 1.0  # s; quality stats recompute at most this often mid-run


def _frame(
    h, chips: list, running: bool, bounds, hist=None, prev_hist=None, quality=True
) -> None:
    refresh_grid_layer(h.scene, h.grid)
    svg = _run["svg"] = render_svg(h.scene, bounds=bounds)
    # Mid-run frames refresh only the chips so the control buttons stay put; the
    # terminal frame (running=False) swaps the whole bar to re-enable run / grey
    # out stop.
    bar = (
        view_chips(*chips, oob=True)
        if running
        else view_bar(*chips, running=False, oob=True)
    )
    msg = to_xml(bar) + to_xml(
        Div(NotStr(svg), id="canvas", cls="canvas", hx_swap_oob="true")
    )
    if quality:
        q = _run["quality"] = grid_quality(h.scene.grid_blocks)
        msg += to_xml(_quality_panel(q, oob=True))
    if hist is not None:
        msg += to_xml(
            Div(
                NotStr(render_sparkline(hist, prev=prev_hist)),
                id="chart",
                hx_swap_oob="true",
            )
        )
    _broadcast(msg)


def _fail_frame(msg: str, detail: str | None = None) -> None:
    parts = to_xml(view_bar(Span(msg, cls="warn"), running=False, oob=True))
    if detail:
        parts += to_xml(Div(Pre(detail, cls="err"), id="viewextra", hx_swap_oob="true"))
    _broadcast(parts)


NO_RUN_MSG = (
    "script registered no run — call egg_webui.run(grid, steps) "
    'in its `if __name__ == "__egg_webui__":` block'
)


def _run_reader(code: str, path: str, proc: subprocess.Popen) -> None:
    """Server-side half of a run: render + fan out the frames the worker
    process streams back; keep the harvest for the SU2 export at the end."""
    import traceback as tb

    done = False
    try:
        # The server keeps its own exec of the same script purely for
        # rendering (scene, bounds, grid layout to scatter nodes into).
        # The grid is the one the script registered via egg_webui.run —
        # never one the UI invented.
        ns, _out, err = exec_script(code, path or None)
        reg = ns.get("__egg_webui_run__")
        if err is not None or reg is None:
            _fail_frame("script error — fix it first" if err else NO_RUN_MSG)
            proc.kill()
            return
        h = harvest(ns, init_grid=False)
        h.grid = reg[0]
        assert h.grid is not None  # a registered run always carries a grid
        bounds = scene_bounds(h.scene)
        hist: dict[str, list[float]] = {"energy": [], "min det": []}
        prev_hist = _last["hist"]  # previous run, faded under the chart
        net_bytes: bytes | None = None
        last_emit = 0.0
        last_q = 0.0  # quality stats debounce off the frame hot path
        last_phase = None
        last_stage = None
        latest_X = None  # newest streamed node coords, for the resume cache
        chips: list = []
        assert proc.stdout is not None  # spawned with stdout=PIPE
        while not done:
            try:
                msg = pickle.load(proc.stdout)
            except (EOFError, pickle.UnpicklingError):
                break
            match msg[0]:
                case "fatal":
                    _fail_frame(msg[1])
                    done = True
                case "error":
                    _fail_frame("pipeline failed", detail=msg[1])
                    done = True
                case "print":
                    # a script/pipeline egg.webui_print — append and push the
                    # (bounded) log to #runlog; interleaves with step frames
                    log = _run["log"]
                    log.append(msg[1])
                    if len(log) > LOG_MAX:
                        del log[: len(log) - LOG_MAX]
                    _broadcast(to_xml(_log_panel("".join(log), oob=True)))
                case "net_state":
                    # Streamed per-step control lattice: update the overlay so
                    # the net animates with the grid (the paired step frame
                    # renders right after this message).
                    refresh_net_layer(h.scene, h.grid, blocks=msg[1])
                case "net":
                    # The worker's solved control net (persisted npz bytes):
                    # keep for the export and attach to the server-side grid.
                    # It arrives after the final step frame, so re-emit one
                    # frame carrying the net overlay.
                    net_bytes = msg[1]
                    assert isinstance(net_bytes, bytes)
                    try:
                        from egg.io import load_control_net

                        h.grid.control_net = load_control_net(
                            h.grid, io.BytesIO(net_bytes)
                        )
                        refresh_net_layer(h.scene, h.grid)
                    except Exception:
                        pass  # export still works from the raw bytes
                    else:
                        _frame(
                            h,
                            chips,
                            running=False,
                            bounds=bounds,
                            hist=hist,
                            prev_hist=prev_hist,
                            quality=False,
                        )
                case "done":
                    _last["code"], _last["harvest"] = code, h
                    _last["net"] = net_bytes
                    if any(len(v) >= 2 for v in hist.values()):
                        _last["hist"] = {k: list(v) for k, v in hist.items()}
                    done = True
                case "step":
                    phase, info, X = msg[1], msg[2], msg[3]
                    latest_X = X  # newest result, seeds a resume
                    # Mirror of pipeline._sync on the server-side grid.
                    h.grid.global_nodes = X
                    for bi, blk in enumerate(h.grid.blocks):
                        blk.nodes[...] = X[h.grid.block_dof_maps[bi]]
                    if (e := info.get("energy")) is not None:
                        hist["energy"].append(e)
                    if (m := info.get("min_det")) is not None:
                        hist["min det"].append(m)
                    stopped = _run["stop"]
                    # The named stage is the headline chip; the raw phase is
                    # the fallback for events with no stage label.
                    stage_label = info.get("stage") or phase
                    chips = [Span(stage_label, cls="phase")]
                    chips += [
                        Span(_fmt_chip(k, v))
                        for k, v in info.items()
                        # scalar chips only (diagnostics vectors like the
                        # control phase's frame_jumps stay off the bar)
                        if k not in ("min_det", "stage")
                        and not isinstance(v, (list, tuple))
                    ]
                    if (c := _mindet_chip(info.get("min_det"))) is not None:
                        chips.append(c)
                    if stopped:
                        chips.append(Span("stopped", cls="warn"))
                    final = stopped or phase == "final"
                    # Decide the resume cache before rendering the terminal bar
                    # so the run button in that frame reflects it: a stop caches
                    # the latest result (button -> "resume"); a full completion
                    # clears it (button -> "run").
                    if final:
                        if stopped and latest_X is not None:
                            _set_resume(latest_X, code)
                        elif not stopped:
                            _clear_resume()
                    now = time.perf_counter()
                    # Emit on any phase OR named-stage transition, so fast runs
                    # show each step (two stages can share a phase, e.g. two
                    # smoothers both emitting "tmop").
                    if (
                        final
                        or phase != last_phase
                        or stage_label != last_stage
                        or now - last_emit >= FRAME_INTERVAL
                    ):
                        with_q = final or now - last_q >= QUALITY_INTERVAL
                        _frame(
                            h,
                            chips,
                            running=not final,
                            bounds=bounds,
                            hist=hist,
                            prev_hist=prev_hist,
                            quality=with_q,
                        )
                        last_emit = now
                        if with_q:
                            last_q = now
                    last_phase = phase
                    last_stage = stage_label
        if not done:
            # Channel closed without a terminal message: crash or hard kill.
            rc = proc.wait()
            if _run["stop"]:
                # A hard kill (double-click stop) can end mid-chunk with no
                # terminal frame; still cache the last result for resume.
                if latest_X is not None:
                    _set_resume(latest_X, code)
                _fail_frame("stopped")
            else:
                _clear_resume()
                _fail_frame(f"pipeline failed (worker exited {rc})")
    except Exception:
        _clear_resume()
        _fail_frame("pipeline failed", detail=tb.format_exc())
    finally:
        for pipe in (proc.stdout, proc.stdin):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if _run["tmp"]:
            Path(_run["tmp"]).unlink(missing_ok=True)
        if _run.get("resume_tmp"):
            Path(_run["resume_tmp"]).unlink(missing_ok=True)
        _run.update(
            proc=None,
            reader=None,
            stop=False,
            tmp=None,
            resume_tmp=None,
            svg=None,
            quality=None,
            log=[],
        )


@rt("/run", methods=["post"])
def run_route(code: str, path: str = "", resume: int = 0):
    if _running():
        return view_bar(Span("already running", cls="warn"), running=True)
    fd, tmp = tempfile.mkstemp(suffix=".py", prefix="egg-webui-run-")
    with os.fdopen(fd, "w") as f:
        f.write(code)
    # Resume: hand the worker the cached node coordinates (a .npy) so it seeds
    # the grid before running the freshly re-exec'd stages. A plain run drops any
    # stale cache so the next stop starts a fresh resume point.
    resume_tmp = None
    argv = [sys.executable, "-m", WORKER_MODULE, tmp, path]
    if resume and _is_resumable():
        import numpy as np

        rfd, resume_tmp = tempfile.mkstemp(suffix=".npy", prefix="egg-webui-resume-")
        os.close(rfd)
        np.save(resume_tmp, _resume["X"])
        argv.append(resume_tmp)
    else:
        _clear_resume()
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,  # the frame channel; stderr stays on the console
    )
    t = threading.Thread(target=_run_reader, args=(code, path, proc), daemon=True)
    _run.update(proc=proc, reader=t, stop=False, tmp=tmp, resume_tmp=resume_tmp, log=[])
    t.start()
    label = "resuming…" if resume_tmp else "starting…"
    return view_bar(Span(label, cls="phase"), running=True)


@rt("/stop", methods=["post"])
def stop():
    proc = _run["proc"]
    if proc is None or proc.poll() is not None:
        return ""
    if _run["stop"]:
        proc.kill()  # second press: hard kill (worker hung mid-chunk)
        return ""
    _run["stop"] = True
    try:
        proc.stdin.write(b"stop\n")
        proc.stdin.flush()
    except Exception:
        proc.kill()
    return ""


def _api_completions() -> list[dict]:
    """Editor completions introspected from the real egg front-end API."""
    import inspect
    from dataclasses import fields as dc_fields

    import egg.geometry as geom
    from egg.topology.builder import TopologyBuilder

    items: list[dict] = []

    def add(label: str, type_: str, info: str = "") -> None:
        items.append({"label": label, "type": type_, "info": info})

    for name in sorted(set(getattr(geom, "__all__", [])) | set(dir(geom))):
        obj = getattr(geom, name, None)
        if name.startswith("_") or not (
            inspect.isclass(obj) or inspect.isfunction(obj)
        ):
            continue
        doc = (inspect.getdoc(obj) or "").split("\n")[0]
        add(name, "class" if inspect.isclass(obj) else "function", doc)

    for cls in (TopologyBuilder,):
        for name, m in inspect.getmembers(cls, inspect.isfunction):
            if name.startswith("_"):
                continue
            try:
                sig = (
                    str(inspect.signature(m)).replace("self, ", "").replace("self", "")
                )
            except (ValueError, TypeError):
                sig = "(...)"
            add(name, "method", f"{cls.__name__}.{name}{sig}")

    for f in dc_fields(PipelineConfig):
        add(f"{f.name}=", "property", f"PipelineConfig.{f.name}: {f.type}")

    for line in (
        "from egg.geometry import Arc, Bezier, Edge, Line, Polyline, Spline, Vector3",
        "from egg.pipeline import PipelineConfig",
        "from egg.topology.builder import TopologyBuilder",
    ):
        add(line, "text")
    return items


_completions_cache: list[dict] | None = None


@rt("/api/completions")
def completions():
    global _completions_cache
    if _completions_cache is None:
        _completions_cache = _api_completions()
    return JSONResponse(_completions_cache)


def _write_out(out: str, data: bytes | str):
    """Write an export to a chosen destination path (this is a local tool, so
    the picker's path is a real filesystem path). Returns a JSONResponse."""
    if not out:
        return JSONResponse({"error": "no destination chosen"}, status_code=400)
    p = Path(out).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            p.write_text(data)
        else:
            p.write_bytes(data)
    except OSError as exc:
        return JSONResponse({"error": f"write failed: {exc}"}, status_code=400)
    return JSONResponse({"path": str(p.resolve())})


@rt("/export/su2", methods=["post"])
async def export_su2_route(code: str, path: str = "", out: str = ""):
    """Write the current mesh as SU2 to ``out`` — the last smoothed grid if the
    script hasn't changed since that run (straight from the resident grid, no
    exec), else a fresh TFI-initialized one built in the render worker."""
    if _last["harvest"] is not None and _last["code"] == code:
        h = _last["harvest"]
        if h.grid is None:
            return JSONResponse({"error": "no grid to export"}, status_code=400)
        try:
            text = grid_to_su2_text(h.grid)
        except Exception as exc:
            return JSONResponse({"error": f"export failed: {exc}"}, status_code=400)
    else:
        # big grids TFI-init in the worker; give them room beyond the
        # editor-render timeout
        text, why = await asyncio.wrap_future(
            _render_worker.su2(code, path, timeout=120.0)
        )
        if text is None:
            return JSONResponse({"error": why}, status_code=400)
    return _write_out(out, text)


@rt("/export/svg", methods=["post"])
def export_svg_route(svg: str, out: str = ""):
    """Write the client-serialized scene SVG to ``out`` (the SVG is built in the
    browser from the live DOM, so it arrives as text here)."""
    return _write_out(out, svg)


@rt("/export/net", methods=["post"])
def export_net_route(code: str = "", path: str = "", out: str = ""):
    """Write the resident grid's control net as an ``.npz`` to ``out`` — the
    escape hatch for a run whose pipeline had no :class:`~egg.pipeline.Save`
    stage. Uses the last smoothed grid's net (no re-run) when the script is
    unchanged; 400 when there is no net (a nodal run with no Refit produces none).
    """
    if _last["harvest"] is None or _last["code"] != code:
        return JSONResponse(
            {"error": "run the pipeline first, then save the net"}, status_code=400
        )
    h = _last["harvest"]
    net = getattr(h.grid, "control_net", None) if h.grid is not None else None
    if net is None:
        return JSONResponse(
            {
                "error": "this run produced no control net — use the control-point "
                "smoother or add a Refit stage, then run again"
            },
            status_code=400,
        )
    try:
        from egg.io import save_control_net

        buf = io.BytesIO()
        save_control_net(net, buf)
    except Exception as exc:
        return JSONResponse({"error": f"export failed: {exc}"}, status_code=400)
    return _write_out(out, buf.getvalue())


@rt("/open/eggy", methods=["post"])
async def open_eggy_route(archive: str, dest: str, name: str):
    """Extract a ``.eggy`` case archive (a path on this machine) into
    ``dest/name`` and load its entry script.

    Nothing runs: the unpacked script + its saved net and assets sit on disk
    (a regrid script can resample from them). Returns ``{path, code}``.
    """

    def work():
        from egg.io import eggy

        arc = str(Path(archive).expanduser())
        if not eggy.is_eggy(arc):
            raise ValueError("not a .eggy archive (a zip holding a script)")
        target = Path(dest).expanduser() / name
        target.mkdir(parents=True, exist_ok=True)
        eggy.unpack(arc, str(target))
        script = eggy.entry_script(str(target))
        return script, Path(script).read_text()

    try:
        script, code = await asyncio.to_thread(work)
    except Exception as exc:
        return JSONResponse({"error": f"open failed: {exc}"}, status_code=400)
    return JSONResponse({"path": script, "code": code})


def _open_in_file_manager(p: Path) -> None:
    """Open a directory in the OS file manager. Runs on the machine hosting the
    server (localhost in a browser, the user's session in the desktop app)."""
    p.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(p)])
    elif os.name == "nt":
        os.startfile(str(p))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(p)])


@rt("/open/dir", methods=["post"])
def open_dir_route(which: str = "config"):
    """Open the egg config directory (``which=config``) or the logs directory
    (``which=logs``) in the OS file manager. The path is resolved server-side
    (never taken from the client) so this can't open an arbitrary directory."""
    from .config import config_dir, logs_dir

    target = logs_dir() if which == "logs" else config_dir()
    try:
        _open_in_file_manager(Path(target))
    except Exception as exc:
        return JSONResponse({"error": f"could not open: {exc}"}, status_code=400)
    return JSONResponse({"ok": True, "path": str(target)})


@rt("/save/eggy", methods=["post"])
def save_eggy_route(code: str, path: str = "", out: str = ""):
    """Pack the current case (script + its folder) into the ``.eggy`` at ``out``.

    The editor's script is written to ``path`` first, then that script's whole
    folder (assets and the net a Save stage wrote) is zipped verbatim. Save the
    script to a file before saving the archive, so it has a folder to pack.
    """
    if not path:
        return JSONResponse(
            {"error": "save the script to a file first, then save the .eggy"},
            status_code=400,
        )
    if not out:
        return JSONResponse({"error": "no destination chosen"}, status_code=400)

    def work():
        from egg.io import deps

        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code)
        outp = Path(out).expanduser()
        outp.parent.mkdir(parents=True, exist_ok=True)
        # pack the case folder AND bundle any egg.file_import() deps of the
        # script (copied under deps/, their paths rewritten in the archive).
        deps.pack_case(str(outp), str(p.parent), str(p))
        return str(outp.resolve())

    try:
        outp = work()
    except Exception as exc:
        return JSONResponse({"error": f"save failed: {exc}"}, status_code=400)
    return JSONResponse({"path": outp})


# --- open/save: the server's filesystem IS the user's filesystem (local
# --- single-user tool — the script already execs with full privileges) ---


@rt("/api/files")
def api_files(dir: str = "", ext: str = ".py"):
    """List one directory: subdirs + files matching ``ext`` (``*`` for all;
    hidden/__pycache__ skipped)."""
    d = (Path(dir).expanduser() if dir else (_REPO_ROOT or Path.home())).resolve()
    if not d.is_dir():
        return JSONResponse({"error": f"not a directory: {d}"}, status_code=400)
    try:
        entries = sorted(d.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        return JSONResponse({"error": f"permission denied: {d}"}, status_code=400)
    skip = lambda n: n.startswith(".") or n == "__pycache__"  # noqa: E731
    keep = lambda p: ext == "*" or p.suffix == ext  # noqa: E731
    return JSONResponse(
        {
            "dir": str(d),
            "parent": str(d.parent),
            "dirs": [p.name for p in entries if p.is_dir() and not skip(p.name)],
            "files": [p.name for p in entries if p.is_file() and keep(p)],
        }
    )


@rt("/api/file")
async def api_file_read(path: str, check: str = "1"):
    p = Path(path).expanduser()
    try:
        code = p.read_text()
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    resp = {"path": str(p.resolve()), "code": code}
    if check == "1":
        # the probe execs whatever file was clicked — in the worker,
        # never in the server (scene.webui_block_suggestion)
        suggest = await asyncio.wrap_future(_render_worker.suggest(code, str(p)))
        if suggest:
            resp["suggest"] = suggest
    return JSONResponse(resp)


@rt("/api/syntax", methods=["post"])
def api_syntax(code: str):
    """Whether ``code`` is syntactically valid Python (compile only, no exec).

    The editor's auto-run gates on this so a half-typed line never re-execs into
    an error — but only real syntax errors block it; type-checker complaints
    (which don't stop the code running) do not reach here.
    """
    try:
        compile(code, "<editor>", "exec")
        return JSONResponse({"ok": True})
    except SyntaxError as exc:
        return JSONResponse({"ok": False, "error": str(exc)})


@rt("/api/exists")
def api_exists(path: str):
    """Whether a path exists (and is a dir) — the picker uses this to prompt
    before overwriting a file."""
    p = Path(path).expanduser()
    try:
        return JSONResponse({"exists": p.exists(), "is_dir": p.is_dir()})
    except OSError:
        return JSONResponse({"exists": False, "is_dir": False})


@rt("/api/file/save", methods=["post"])
def api_file_save(path: str, code: str):
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code)
        return JSONResponse({"path": str(p.resolve())})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


# --- file-picker sidebar: quick locations + persistent favourites ---

# Favourites live in the user's config dir (XDG-aware) so they persist across
# sessions and installs. One small JSON: {"favourites": ["/abs/path", ...]}.
_CONFIG_DIR = (
    Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "egg"
)
_FAVOURITES_FILE = _CONFIG_DIR / "filepicker.json"


def _xdg_user_dir(key: str, default_rel: str) -> Path:
    """A freedesktop user dir (e.g. ``DESKTOP`` -> ~/Desktop), honouring
    ~/.config/user-dirs.dirs when present, else the conventional ~ subdir."""
    home = Path.home()
    cfg = (
        Path(os.environ.get("XDG_CONFIG_HOME") or (home / ".config"))
        / "user-dirs.dirs"
    )
    if cfg.is_file():
        try:
            for line in cfg.read_text().splitlines():
                line = line.strip()
                if line.startswith(f"XDG_{key}_DIR"):
                    val = line.split("=", 1)[1].strip().strip('"')
                    return Path(val.replace("$HOME", str(home)))
        except Exception:
            pass
    return home / default_rel


def _quick_places() -> list[dict]:
    """Sidebar shortcuts: home + a few XDG user dirs, the egg examples (in a
    checkout), and the filesystem root — only those that exist, de-duplicated.
    Each carries an ``icon`` key the client maps to a small inline SVG."""
    cand: list[tuple[str, str, Path | None]] = [
        ("Home", "home", Path.home()),
        ("Desktop", "desktop", _xdg_user_dir("DESKTOP", "Desktop")),
        ("Documents", "documents", _xdg_user_dir("DOCUMENTS", "Documents")),
        ("Downloads", "downloads", _xdg_user_dir("DOWNLOAD", "Downloads")),
        ("egg examples", "folder", EXAMPLES_DIR),
        ("/", "drive", Path("/")),
    ]
    out: list[dict] = []
    seen: set[str] = set()
    for name, icon, p in cand:
        if p is None:
            continue
        try:
            if p.is_dir():
                rp = str(p.resolve())
                if rp not in seen:
                    seen.add(rp)
                    out.append({"name": name, "path": rp, "icon": icon})
        except OSError:
            pass
    return out


def _load_favourites() -> list[str]:
    try:
        data = json.loads(_FAVOURITES_FILE.read_text())
        return [f for f in data.get("favourites", []) if isinstance(f, str)]
    except Exception:
        return []


def _save_favourites(favs: list[str]) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _FAVOURITES_FILE.write_text(json.dumps({"favourites": favs}, indent=2))


# Recently opened files / visited directories, most-recent-first, capped.
_RECENT_FILE = _CONFIG_DIR / "recent.json"
_RECENT_MAX = 25

# Per-directory visit counts drive "automatic" favourites: a directory visited
# at least _AUTOFAV_MIN times is promoted to a favourite (clock icon, still
# removable); at most _AUTOFAV_MAX are kept, the oldest promotion dropped.
_USAGE_FILE = _CONFIG_DIR / "usage.json"
_AUTOFAV_MIN = 5
_AUTOFAV_MAX = 5


def _load_recent() -> list[str]:
    try:
        data = json.loads(_RECENT_FILE.read_text())
        return [f for f in data.get("recent", []) if isinstance(f, str)]
    except Exception:
        return []


def _save_recent(items: list[str]) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _RECENT_FILE.write_text(json.dumps({"recent": items[:_RECENT_MAX]}, indent=2))


def _recent_entries() -> list[dict]:
    """Recent paths that still exist, resolved to {path, name, is_dir}; stale
    entries are silently dropped from the view (kept in the file until pushed
    out by newer ones)."""
    out: list[dict] = []
    for p in _load_recent():
        try:
            pp = Path(p)
            if pp.exists():
                out.append({"path": p, "name": pp.name or p, "is_dir": pp.is_dir()})
        except OSError:
            pass
    return out


def _load_usage() -> dict:
    try:
        data = json.loads(_USAGE_FILE.read_text())
        return {
            "counts": {
                k: v for k, v in data.get("counts", {}).items() if isinstance(v, int)
            },
            "auto": [a for a in data.get("auto", []) if isinstance(a, str)],
        }
    except Exception:
        return {"counts": {}, "auto": []}


def _save_usage(usage: dict) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _USAGE_FILE.write_text(json.dumps(usage, indent=2))


def _record_visit(path: str) -> None:
    """Front the recents with ``path``; for a directory, bump its visit count
    and auto-promote it to a favourite past the threshold (unless it is already
    a manual favourite)."""
    try:
        p = str(Path(path).expanduser().resolve())
    except OSError:
        return
    _save_recent([p] + [x for x in _load_recent() if x != p])
    try:
        is_dir = Path(p).is_dir()
    except OSError:
        is_dir = False
    if not is_dir:
        return
    usage = _load_usage()
    counts, auto = usage["counts"], usage["auto"]
    counts[p] = counts.get(p, 0) + 1
    if counts[p] >= _AUTOFAV_MIN and p not in auto and p not in _load_favourites():
        auto.append(p)
        while len(auto) > _AUTOFAV_MAX:
            auto.pop(0)
    _save_usage({"counts": counts, "auto": auto})


def _favourite_entries() -> list[dict]:
    """Manual favourites (``auto: false``) then automatic ones (``auto: true``),
    existing directories only, de-duplicated."""
    manual = _load_favourites()
    out = [{"path": p, "name": Path(p).name or p, "auto": False} for p in manual]
    for p in _load_usage()["auto"]:
        if p in manual:
            continue
        try:
            if Path(p).is_dir():
                out.append({"path": p, "name": Path(p).name or p, "auto": True})
        except OSError:
            pass
    return out


@rt("/api/places")
def api_places():
    return JSONResponse(
        {
            "quick": _quick_places(),
            "recent": _recent_entries(),
            "favourites": _favourite_entries(),
        }
    )


@rt("/api/recent", methods=["post"])
def api_recent(path: str):
    """Record a visit (recents + usage/auto-favourite); returns both lists."""
    _record_visit(path)
    return JSONResponse(
        {"recent": _recent_entries(), "favourites": _favourite_entries()}
    )


@rt("/api/favourites", methods=["post"])
def api_favourites(path: str, action: str = "add"):
    """Add/remove one directory from the favourites; returns the updated
    favourite entries (manual + automatic) for the sidebar. Removing an
    automatic favourite also resets its usage count so it isn't re-promoted at
    once; manually adding one that was automatic converts it to manual."""
    try:
        p = str(Path(path).expanduser().resolve())
    except OSError:
        return JSONResponse({"error": "bad path"}, status_code=400)
    favs = _load_favourites()
    usage = _load_usage()
    if action == "remove":
        favs = [f for f in favs if f != p]
        if p in usage["auto"]:
            usage["auto"] = [a for a in usage["auto"] if a != p]
            usage["counts"][p] = 0
            _save_usage(usage)
    elif p not in favs:
        favs.append(p)
        if p in usage["auto"]:  # promoted to a manual favourite: de-dup
            usage["auto"] = [a for a in usage["auto"] if a != p]
            _save_usage(usage)
    _save_favourites(favs)
    return JSONResponse({"favourites": _favourite_entries()})


# --- file-picker recursive fuzzy search ---

# A search bails to a confirm prompt once it has visited this many entries
# without a prior go-ahead (starting a walk at $HOME is otherwise a foot-gun);
# results are capped so a broad query can't return an unbounded list.
_SEARCH_CONFIRM_AFTER = 10_000
_SEARCH_MAX_RESULTS = 300
# In-flight searches, keyed by the client-supplied id, so /api/search/cancel
# (running on a separate threadpool thread) can signal the walk to stop.
_searches: dict[str, threading.Event] = {}
# Live counts per in-flight search ({scanned, matches}) that the client polls
# via /api/search/progress to drive the "searching…" counter.
_search_progress: dict[str, dict] = {}


def _fuzzy_score(query: str, name: str) -> int | None:
    """Case-insensitive subsequence match of ``query`` in ``name``. Returns a
    score (higher is a better match) or None when it doesn't match at all."""
    if not query:
        return 0
    q = query.lower()
    n = name.lower()
    qi = 0
    score = 0
    prev = -2
    for i, ch in enumerate(n):
        if qi < len(q) and ch == q[qi]:
            score += 6 if i == prev + 1 else 1  # reward contiguous runs
            if i == 0:
                score += 5  # and a match at the very start
            prev = i
            qi += 1
    return score if qi == len(q) else None


@rt("/api/search")
def api_search(dir: str, q: str, id: str, confirm: str = "0"):
    """Recursively fuzzy-match .py files and folders under ``dir``.

    Runs synchronously in fasthtml's threadpool (so the walk never blocks the
    event loop) and checks a per-id cancel flag that ``/api/search/cancel``
    sets. With ``confirm != "1"`` it stops and returns ``needs_confirm`` once
    it has scanned :data:`_SEARCH_CONFIRM_AFTER` entries, so a huge tree
    (e.g. $HOME) needs an explicit go-ahead before the full walk.
    """
    root = Path(dir).expanduser()
    if not root.is_dir():
        return JSONResponse({"error": f"not a directory: {root}"}, status_code=400)
    root = str(root.resolve())
    confirmed = confirm == "1"
    cancel = threading.Event()
    _searches[id] = cancel
    _search_progress[id] = {"scanned": 0, "matches": 0}
    matches: list[tuple[int, str, bool, str]] = []
    scanned = 0
    try:
        stack = [root]
        while stack:
            if cancel.is_set():
                break
            cur = stack.pop()
            try:
                it = os.scandir(cur)
            except OSError:
                continue
            with it:
                for e in it:
                    if cancel.is_set():
                        break
                    name = e.name
                    if name.startswith(".") or name == "__pycache__":
                        continue
                    scanned += 1
                    if scanned % 512 == 0:  # cheap live counter for the client
                        _search_progress[id] = {
                            "scanned": scanned,
                            "matches": len(matches),
                        }
                    if not confirmed and scanned > _SEARCH_CONFIRM_AFTER:
                        return JSONResponse({"needs_confirm": True, "scanned": scanned})
                    try:
                        is_dir = e.is_dir()
                    except OSError:
                        continue
                    if not is_dir and not (e.is_file() and name.endswith(".py")):
                        continue
                    if is_dir:
                        stack.append(e.path)
                    sc = _fuzzy_score(q, name)
                    if sc is not None:
                        matches.append((sc, e.path, is_dir, name))
        matches.sort(key=lambda m: (-m[0], len(m[3]), m[1]))
        truncated = len(matches) > _SEARCH_MAX_RESULTS
        results = [
            {
                "path": p,
                "name": nm,
                "is_dir": d,
                "rel": os.path.relpath(p, root),
            }
            for _, p, d, nm in matches[:_SEARCH_MAX_RESULTS]
        ]
        return JSONResponse(
            {
                "results": results,
                "truncated": truncated,
                "scanned": scanned,
                "cancelled": cancel.is_set(),
            }
        )
    finally:
        _searches.pop(id, None)
        _search_progress.pop(id, None)


@rt("/api/search/cancel", methods=["post"])
def api_search_cancel(id: str):
    ev = _searches.get(id)
    if ev is not None:
        ev.set()
    return JSONResponse({"ok": True})


@rt("/api/search/progress")
def api_search_progress(id: str):
    """Live {scanned, matches} for an in-flight search (client counter)."""
    return JSONResponse(_search_progress.get(id, {}))


# --- pages ---


@rt("/api/param", methods=["post"])
def set_param(code: str, name: str, value: str):
    """Rewrite one guard-dict value; returns the whole new script text."""
    try:
        return PlainTextResponse(set_guard_param(code, name, value))
    except ValueError as e:
        return PlainTextResponse(str(e), status_code=422)


@rt("/render", methods=["post"])
async def render(code: str, view: str = "grid", path: str = ""):
    mode = view if view in ("grid", "topo", "edit") else "grid"
    return view_fragment(await _render_scene(code, mode, path), code, mode=mode)


@rt("/api/topo/validate", methods=["post"])
async def topo_validate(code: str, blocking: str = "{}", path: str = ""):
    """Diagnostics for a candidate blocking (JSON) against the script's base +
    geometry; an empty list means green/committable. Runs in the render worker
    (it execs the script), never in the server."""
    try:
        f = json.loads(blocking)
    except json.JSONDecodeError:
        return JSONResponse({"error": "bad blocking json"}, status_code=400)
    out = await asyncio.wrap_future(_render_worker.validate(code, f, path))
    return JSONResponse(out)


@rt("/api/topo/commit", methods=["post"])
async def topo_commit(code: str, blocking: str = "{}", path: str = ""):
    """Write a blocking back into the ``editable({...})`` source span — but only
    when it has no errors. Validates in the worker; advisory ``warn_*``
    diagnostics don't block, real errors refuse with the reasons."""
    try:
        f = json.loads(blocking)
    except json.JSONDecodeError:
        return JSONResponse({"error": "bad blocking json"}, status_code=400)
    val = await asyncio.wrap_future(_render_worker.validate(code, f, path))
    diags = val["diagnostics"]
    errors = [d for d in diags if not d.get("kind", "").startswith("warn")]
    if errors:
        return JSONResponse(
            {
                "error": "topology not valid — fix the issues first",
                "diagnostics": errors,
            },
            status_code=422,
        )
    try:
        new_code = set_editable_blocking(code, f)  # pure AST/string splice
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"code": new_code, "block": editable_block(new_code)})


@rt("/reset", methods=["post"])
async def reset(code: str, view: str = "grid", path: str = ""):
    """Discard the last smoothed result and show the fresh unsmoothed grid.

    A completed run leaves the smoothed mesh on the canvas (and in _last for
    export / _run for the mid-render keep). Reset drops those so a plain
    re-exec of the script — TFI-initialized, unsmoothed — is what shows and
    what exports. Disabled while a run streams (it owns the canvas).
    """
    mode = view if view in ("grid", "topo", "edit") else "grid"
    if not _running():
        _last.update(code=None, harvest=None, hist=None)
        _run["svg"] = _run["quality"] = None
        _clear_resume()  # reset also discards the resume cache
    return view_fragment(await _render_scene(code, mode, path), code, mode=mode)


def _menu(label: str, *items):
    """Header dropdown: a button + absolutely-positioned item panel."""
    return Div(Button(label, cls="menu-btn"), Div(*items, cls="menu-items"), cls="menu")


def _window_controls():
    """The min / maximize / close buttons for the native-app titlebar. They
    call the window controls exposed through ``js_api`` under
    ``window.pywebview.api`` (see egg/_desktop.py)."""
    return Div(
        Button(
            "−",  # minus sign
            type="button",
            aria_label="Minimize",
            cls="desktop-titlebar__btn",
            onclick="window.pywebview.api.minimize()",
        ),
        Button(
            "□",  # white square
            type="button",
            aria_label="Maximize",
            cls="desktop-titlebar__btn",
            onclick="window.pywebview.api.toggle_maximize()",
        ),
        Button(
            "✕",  # multiplication x
            type="button",
            aria_label="Close",
            cls="desktop-titlebar__btn desktop-titlebar__close",
            onclick="window.pywebview.api.close()",
        ),
        cls="desktop-titlebar__controls",
    )


def _titlebar_or_header(desktop: object, *menu_items):
    """The top toolbar: the same logo + file/view/help dropdowns either way,
    followed by the filename chip and the pan/zoom hint.

    In an ordinary browser this is the plain app header. In the native
    (pywebview) app — ``/?desktop=1``, opened by the ``egg-desktop``
    launcher — it becomes the window titlebar: the menus move into it, and
    it also carries a drag handle and the window controls. The launcher runs
    a *frameless* window; dragging the spacer starts a compositor move via
    ``window.pywebview.api.start_drag`` (wired in app.js), so it works on
    Wayland too, while the menus and buttons keep their own clicks.
    """
    filechip = Span(id="filechip")
    if not desktop:
        return Header(*menu_items, filechip)
    return Header(
        *menu_items,
        filechip,
        # Flex spacer between the menus and the window controls; mousedown
        # here starts the window drag (see app.js / start_drag).
        Div(cls="desktop-titlebar__drag"),
        _window_controls(),
        cls="desktop-titlebar",
    )


@rt("/")
async def get(view: str = "grid", desktop: int = 0):
    mode = view if view in ("grid", "topo", "edit") else "grid"
    code, path = _initial_code(), _initial_path()
    scene_r = await _render_scene(code, mode, path)
    return (
        Title("egg webui"),
        # The top toolbar: the ordinary browser header, or — at /?desktop=1,
        # opened by the egg-desktop launcher — the native-app titlebar with
        # the same menus plus window controls.
        _titlebar_or_header(
            desktop,
            NotStr(EGG_LOGO),
            _menu(
                "file",
                Button("open…", id="file-open"),
                Button(
                    "examples…", id="file-examples", data_dir=str(EXAMPLES_DIR or "")
                ),
                Button("save", id="file-save"),
                Button("save as…", id="file-saveas"),
                Label(
                    Input(type="checkbox", id="watch-toggle"),
                    "watch file",
                    title="follow the opened file on disk (edit in your own "
                    "editor); hides the editor pane",
                ),
                Label(
                    Input(type="checkbox", id="autosave-toggle"),
                    "auto-save",
                    title="write edits to the open file automatically, about a "
                    "second after you stop typing",
                ),
                Div(cls="menu-sep"),
                Div(
                    Button("export", cls="menu-sub-btn"),
                    Div(
                        Button("SVG", id="file-dl-svg"),
                        Button("SU2", id="file-dl-su2"),
                        Button(
                            "control net (npz)",
                            id="file-dl-net",
                            title="save the last run's control net as an .npz — the "
                            "escape hatch if the pipeline had no Save stage; a regrid "
                            "script can resample from it",
                        ),
                        cls="menu-sub-items",
                    ),
                    cls="menu-sub",
                ),
                # populated + shown by JS after the first export (overwrite in place)
                Button("export as…", id="file-export-as", style="display:none"),
                Div(cls="menu-sep"),
                Button(
                    "open .eggy archive…",
                    id="file-open-eggy",
                    title="open a case archive (.eggy): pick it, then choose a "
                    "folder to extract it into; a regrid script can resample "
                    "from the packed net",
                ),
                Button(
                    "save .eggy archive…",
                    id="file-save-eggy",
                    title="pack this case (script + its folder, including the "
                    "saved net) into a .eggy archive to share or regrid",
                ),
                Div(cls="menu-sep"),
                Button(
                    "open config directory",
                    id="file-config-dir",
                    title="open ~/.config/egg (config.toml, favourites, recent) "
                    "in the system file manager",
                ),
            ),
            _menu(
                "view",
                Button("fit", id="fit"),
                Div(cls="menu-sep"),
                *[
                    Label(
                        Input(
                            type="checkbox",
                            checked=True,
                            cls="layer-toggle",
                            data_layer=layer,
                        ),
                        layer,
                    )
                    for layer in ("grid", "tangled", "curves", "points")
                ],
                Label(
                    Input(type="checkbox", cls="layer-toggle", data_layer="ctrl"),
                    "control points",
                    title="construction points with their dashed control cage "
                    "(spline through-points, Bézier control points, arc "
                    "endpoints and centres)",
                ),
                Label(
                    Input(type="checkbox", cls="layer-toggle", data_layer="net"),
                    "control net",
                    title="the solved B-spline control lattice of a "
                    'tmop_smoother="control_point" run',
                ),
                Div(cls="menu-sep"),
                Label(
                    Input(
                        type="checkbox",
                        checked=True,
                        cls="scene-toggle",
                        data_toggle="block-fill",
                    ),
                    "block colours",
                    title="soft per-block colour fill and line tint",
                ),
                Label(
                    Input(
                        type="checkbox",
                        checked=True,
                        cls="scene-toggle",
                        data_toggle="block-outline",
                    ),
                    "block boundaries",
                    title="thicker outlines around each grid block",
                ),
                Div(cls="menu-sep"),
                Label(
                    Input(type="checkbox", id="wrap-toggle"),
                    "wrap long lines",
                    title="wrap editor lines that don't fit the panel width",
                ),
                Div(cls="menu-sep"),
                Label(
                    "theme",
                    Select(
                        # catppuccin flavors, dark → light; JS restores the
                        # persisted choice and re-themes CodeMirror
                        Option("mocha", value="mocha"),
                        Option("macchiato", value="macchiato"),
                        Option("frappé", value="frappe"),
                        Option("latte", value="latte", selected=True),
                        id="theme-select",
                    ),
                ),
            ),
            _menu(
                "help",
                A("documentation", href="/docs/", target="_blank")
                if DOCS_DIR and DOCS_DIR.is_dir()
                else Span(
                    "docs not built — uv sync --group docs, then restart egg-webui",
                    cls="menu-note",
                ),
                Button(
                    "report a problem",
                    id="help-report",
                    data_url="https://github.com/bezmi/egg/issues",
                ),
                Button(
                    "view logs",
                    id="help-logs",
                    title="open the log directory (default ~/.config/egg/logs) "
                    "in the system file manager",
                ),
            ),
        ),
        Div(
            Div(
                Textarea(
                    code,
                    name="code",
                    spellcheck="false",
                    # Firefox restores a textarea's value from session history on
                    # reload (Chrome does not); that stale value desyncs from the
                    # server-rendered canvas. Same defense as the view <select>.
                    autocomplete="off",
                    data_persist="0" if _script_arg() is not None else "1",
                    data_file=str(_script_arg() or ""),
                    data_watch="1"
                    if os.environ.get("EGG_WEBUI_WATCH") and _script_arg()
                    else "0",
                    # No input-driven render: the preview re-executes when an
                    # editable item is used (params/topology/loading a file) or,
                    # for typed edits, a couple of seconds after typing stops and
                    # only when the LSP reports no errors — so a half-typed line
                    # never re-execs into a runtime error. Renders are driven
                    # explicitly via eggForceRender() in app.js.
                ),
                Input(type="hidden", name="path", id="scriptpath", value=path),
                cls="editor",
            ),
            Div(
                Div(*view_fragment(scene_r, code, mode=mode), id="view"),
                Div(id="coords"),
                Div(NotStr(AXES_SVG), id="axes"),
                Div(id="selinfo"),
                Div(
                    Button(
                        "split node",
                        id="tool-split",
                        title="un-weld the selected node: one node per edge",
                    ),
                    Button(
                        "join", id="tool-join", title="weld the selected nodes into one"
                    ),
                    Button(
                        "coincident",
                        id="tool-coincident",
                        title="snap the selected node onto the selected edge",
                    ),
                    Button(
                        "set res",
                        id="tool-res",
                        title="set the cell count along the selected edge — "
                        "propagates to every edge that must stay consistent with it",
                    ),
                    id="ed-tools",
                ),
                cls="viewer",
            ),
            cls="panes",
            hx_ext="ws",
            ws_connect="/ws",
        ),
        Div(
            Div(
                Div(
                    Span("open", id="fs-title"),
                    Button("cancel", id="fs-cancel"),
                    cls="fs-head",
                ),
                # editable current-path bar + favourite toggle for this folder
                Div(
                    Input(
                        id="fs-path",
                        cls="fs-path-input",
                        autocomplete="off",
                        spellcheck="false",
                        placeholder="/path/to/folder",
                        title="type a path and press Enter to jump there",
                    ),
                    Button(
                        "☆",
                        id="fs-fav",
                        cls="fs-fav",
                        title="add this folder to favourites",
                    ),
                    cls="fs-pathrow",
                ),
                Div(
                    Input(
                        id="fs-search",
                        cls="fs-search",
                        autocomplete="off",
                        spellcheck="false",
                        placeholder="Search this directory",
                    ),
                    Select(
                        Option("name A–Z", value="az"),
                        Option("name Z–A", value="za"),
                        id="fs-sort",
                        title="sort order",
                    ),
                    cls="fs-searchrow",
                ),
                # deep-search confirm prompt (shown once a walk gets large)
                Div(
                    Span(id="fs-confirm-text"),
                    Div(
                        Button("keep searching", id="fs-confirm-go", cls="primary"),
                        Button("stop", id="fs-confirm-stop"),
                        cls="fs-actions",
                    ),
                    id="fs-confirm",
                    cls="fs-note",
                ),
                # live "searching…" indicator with a cancel button
                Div(
                    Span("searching…", id="fs-searching-text"),
                    Button("stop", id="fs-search-stop"),
                    id="fs-searching",
                    cls="fs-note",
                ),
                Div(
                    Div(id="fs-sidebar", cls="fs-sidebar"),
                    Div(id="fs-list"),
                    cls="fs-main",
                ),
                Div(
                    Input(id="fs-name", placeholder="filename.py", autocomplete="off"),
                    Button("save", id="fs-do-save", cls="primary"),
                    cls="fs-saverow",
                ),
                cls="fs-box",
            ),
            id="fsmodal",
        ),
        # generic yes/no confirm (overwrite prompts); JS drives it as a promise
        Div(
            Div(
                Div(Span(id="cf-text"), cls="cf-body"),
                Div(
                    Button("cancel", id="cf-no"),
                    Button("ok", id="cf-yes", cls="primary"),
                    cls="fs-actions",
                ),
                cls="fs-box cf-box",
            ),
            id="cfmodal",
        ),
        Div(
            Div(
                Div(
                    Span("add this to your file"),
                    Button("close", id="sug-close"),
                    cls="fs-head",
                ),
                Div(
                    "the script registers no web-UI run — paste this block at "
                    "the end of the file (the UI won't modify a watched file):",
                    cls="fs-path",
                ),
                Pre(id="sug-text"),
                Div(Button("copy", id="sug-copy"), cls="fs-actions"),
                cls="fs-box",
            ),
            id="sugmodal",
        ),
        Div(
            Div(
                Div(
                    Span("apply edits"),
                    Button("close", id="save-close"),
                    cls="fs-head",
                ),
                Div(
                    "replace the editable({…}) connectivity in your file with "
                    "this — or write it directly:",
                    cls="fs-path",
                ),
                Pre(id="save-text"),
                Label(
                    Input(type="checkbox", id="save-remember"),
                    "keep writing to the file this session (even when watching)",
                    id="save-remember-row",
                    cls="fs-path",
                ),
                Div(
                    Button("copy", id="save-copy"),
                    Button("write to file", id="save-write", cls="primary"),
                    cls="fs-actions",
                ),
                cls="fs-box",
            ),
            id="savemodal",
        ),
        Div(
            Div(
                Div(
                    Span("set resolution"),
                    Button("close", id="res-close"),
                    cls="fs-head",
                ),
                Label(
                    "cells along the selected edge",
                    Input(type="number", id="res-input", min="1", step="1"),
                    cls="res-row",
                ),
                Div(
                    "propagates to every edge that must stay consistent with it",
                    cls="fs-path",
                ),
                Div(
                    Button("cancel", id="res-cancel"),
                    Button("set", id="res-ok", cls="primary"),
                    cls="fs-actions",
                ),
                cls="fs-box",
            ),
            id="resmodal",
        ),
    )


def _script_arg() -> Path | None:
    """Optional script to open: $EGG_WEBUI_SCRIPT, or argv[1] when run
    directly (under uvicorn argv holds uvicorn's own arguments)."""
    if p := os.environ.get("EGG_WEBUI_SCRIPT"):
        return Path(p)
    if len(sys.argv) > 1 and sys.argv[0].endswith("app.py"):
        return Path(sys.argv[1])
    return None


def _initial_code() -> str:
    p = _script_arg() or DEFAULT_SCRIPT
    return p.read_text() if p else ""


def _initial_path() -> str:
    p = _script_arg() or DEFAULT_SCRIPT
    return str(p) if p else ""


if __name__ == "__main__":
    serve(port=5001, reload=False)
