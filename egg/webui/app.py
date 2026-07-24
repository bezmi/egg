# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""FastHTML "code as CAD" prototype: live SVG view of an egg geometry script.

Run (after ``uv sync --group ui``)::

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

from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from ._assets import MISSING_MSG, VENDOR_DIR, vendor_ready
from .lsp import DOC_URI, ROOT_URI, LspBridge, lsp_available

from fasthtml.common import (
    Body,
    Button,
    Details,
    Div,
    Head,
    Header,
    Html,
    Iframe,
    Input,
    Label,
    Link,
    Meta,
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
# Examples root (2D + 3D); the landing "examples" shortcut opens here.
EXAMPLES_ROOT = _REPO_ROOT / "examples" if _REPO_ROOT else None
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
from .config import load_config
from .security import SecurityMiddleware, canonical_path, configured_token

CONFIG_JS = f"window.eggConfig = {json.dumps(_client_config())};"


def _font_css() -> str:
    """A ``:root`` override for the interface/editor fonts and base sizes from
    ``[fonts]`` in config.toml: ``--font-ui`` / ``--font-editor`` (families) and
    ``--fs-ui`` / ``--fs-editor`` (base px sizes that every other UI/editor size
    is a factor of). Each configured family name is quoted and placed before
    ``--font-mono-fallback`` so a font that is not installed falls back to the
    built-in stack. Emitted into the head after app.css (which defines the
    defaults) so it wins. Only a plain family name is honored (``[A-Za-z0-9 ._-]``)
    and sizes must be a number in ``[6, 40]`` px, so nothing can break out of the
    CSS string; unset or invalid values are skipped, and an empty string is
    returned when nothing is set."""
    fonts = load_config().get("fonts", {})
    rules = []
    for key, var in (("interface", "--font-ui"), ("editor", "--font-editor")):
        name = str(fonts.get(key, "")).strip()
        if name and re.fullmatch(r"[A-Za-z0-9 ._-]+", name):
            rules.append(f'{var}: "{name}", var(--font-mono-fallback);')
    for key, var in (("interface_size", "--fs-ui"), ("editor_size", "--fs-editor")):
        size = fonts.get(key)
        if isinstance(size, (int, float)) and not isinstance(size, bool) and 6 <= size <= 40:
            rules.append(f"{var}: {size:g}px;")
    return f":root {{ {' '.join(rules)} }}" if rules else ""


FONT_CSS = _font_css()
# The launch auth token, exposed to the page so JS can append it when it opens a
# local URL in a separate browser (documentation in the system browser has no
# cookie yet). Only an already-authenticated client can load this page, so this
# does not widen exposure. Empty string when auth is off.
TOKEN_JS = f"window.eggToken = {json.dumps(configured_token() or '')};"

# All browser assets are served locally from /vendor (no CDN fallback, by
# design). They are produced offline by tools/vendor_webui.py — at wheel-build
# time, or via `egg-webui --dev` in a checkout. If they are absent there is no
# safe way to serve the UI, so fail fast with an actionable message rather than
# reaching out to the network.
if not vendor_ready():
    raise SystemExit(MISSING_MSG)

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

# default_hdrs=False: FastHTML's defaults pull htmx / fasthtml.js / surreal /
# css-scope-inline from a CDN. We serve vendored copies from /vendor instead, so
# the page never reaches the network at runtime and the CSP can stay 'self'.
# htmx loads before the inline app.js that uses it. The ws htmx extension is not
# used (the frame socket is a manual WebSocket), so it is not included.
app, rt = fast_app(
    default_hdrs=False,
    hdrs=(
        Meta(charset="utf-8"),
        Meta(name="viewport",
             content="width=device-width, initial-scale=1, viewport-fit=cover"),
        Script(src="/vendor/htmx.min.js"),
        Script(src="/vendor/fasthtml.js"),
        Script(src="/vendor/surreal.js"),
        Script(src="/vendor/scope.js"),
        Link(rel="icon", type="image/svg+xml", href=FAVICON, id="favicon"),
        *(Link(rel="stylesheet", href=f"/vendor/{c}") for c in _PCE_CSS),
        Style(CSS),
        Style(CATPPUCCIN),
        Style(NotStr(FONT_CSS)),  # [fonts] family/size overrides; inert when unset
        Script(CONFIG_JS),  # window.eggConfig, before app.js reads it (Script leaves JS unescaped)
        Script(TOKEN_JS),  # window.eggToken, for local links opened elsewhere
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
# `uv run sphinx-build -b html docs docs/_build/html`).
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

# Treat the client/server link as privileged local IPC: enforce the launch auth
# token, the Host/Origin allowlists, and the CSP on every request (HTTP and
# WebSocket). Inert when no token is configured (a bare import or the tests).
app.add_middleware(SecurityMiddleware)

# --- per-session state (server pushes frames to the client) ---
# A solver run's state is keyed by a session id: one per open UI (a browser tab,
# the desktop window, another browser). The client makes the id and sends it with
# every request and on the frame socket. So sessions stay separate: a run's frames
# go only to the UI that started it, and several runs can go at once, one worker
# each. The id is an opaque token, so any frontend that keeps a stable id works.

_clients: dict[str, tuple] = {}  # sid -> (ws, loop): the frame socket
_ws_sid: dict[int, str] = {}     # id(ws) -> sid, so a disconnect knows its session
_sessions: dict[str, dict] = {}

# A run survives its client disconnecting this long (a page reload reconnects
# with the same sid inside the window); a closed tab is cleaned up after it.
_ORPHAN_GRACE = 20.0


def _new_run_state() -> dict:
    # "svg"/"quality": the latest streamed frame, so editor renders during a run
    # keep the relaxing mesh instead of flashing back to a static render. "log":
    # this run's egg.webui_print lines (streamed to #runlog, bounded to LOG_MAX).
    return {
        "proc": None,
        "reader": None,
        "stop": False,
        "tmp": None,
        "resume_tmp": None,
        "svg": None,
        "quality": None,
        "log": [],
    }


def _session(sid: str) -> dict:
    """The state bag for a session id, created on first use."""
    s = _sessions.get(sid)
    if s is None:
        s = _sessions[sid] = {
            "run": _new_run_state(),
            # Last completed run: {code, harvest, hist, net}. The SU2/net export
            # uses it ("what you see is what you export"); hist draws faded under
            # the next run's chart.
            "last": {"code": None, "harvest": None, "hist": None, "net": None},
            # Resume cache: node coords + code of the latest *stopped* run. While
            # present the run button shows "resume" (a fresh run re-execs the
            # script and seeds the grid to this cache). Cleared by a full
            # completion or the reset button.
            "resume": {"X": None, "code": None},
        }
    return s


def _is_resumable(sess: dict | None) -> bool:
    return bool(sess and sess["resume"]["X"] is not None)


def _set_resume(sess: dict, X, code: str) -> None:
    sess["resume"].update(X=X, code=code)


def _clear_resume(sess: dict) -> None:
    sess["resume"].update(X=None, code=None)


def _kill_run(run: dict) -> None:
    """Stop a session's worker if it is still alive."""
    proc = run.get("proc")
    if proc is not None and proc.poll() is None:
        run["stop"] = True
        try:
            proc.kill()
        except Exception:
            pass


def _cleanup_session(sid: str) -> None:
    """Drop a session (and kill its run) once its client is gone for good."""
    if sid in _clients:
        return  # reconnected within the grace window
    sess = _sessions.pop(sid, None)
    if sess:
        _kill_run(sess["run"])

# Editor renders exec the script in this persistent worker process, not in
# the server: the event loop only awaits, a script stuck in a loop is
# killed at the render timeout, and a native crash in a script costs one
# render instead of the server. Spawned eagerly so the first page load
# finds it warm.
_render_worker = RenderWorker()


async def _render_scene(code: str, mode: str, path: str, sid: str = "") -> SceneResult:
    return await asyncio.wrap_future(_render_worker.submit(code, mode, path, sid=sid))


def _ws_sid_of(ws) -> str:
    """The session id a frame socket connected with (``/ws?sid=...``)."""
    try:
        return ws.query_params.get("sid") or ""
    except Exception:
        return ""


# NB: async on purpose — fasthtml runs sync handlers in a threadpool,
# where get_running_loop() has nothing to return.
async def _on_conn(ws):
    sid = _ws_sid_of(ws)
    _ws_sid[id(ws)] = sid
    _clients[sid] = (ws, asyncio.get_running_loop())


async def _on_disconn(ws):
    sid = _ws_sid.pop(id(ws), None)
    if sid is None:
        return
    # Only drop the client if this exact socket is still the current one (a fast
    # reconnect may already have replaced it).
    cur = _clients.get(sid)
    if cur and cur[0] is ws:
        _clients.pop(sid, None)
    # A closed instance must not leave its worker running forever; a reload
    # reconnects with the same sid inside the grace window and is spared.
    threading.Timer(_ORPHAN_GRACE, _cleanup_session, args=(sid,)).start()


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


def _broadcast(sid: str, html: str) -> None:
    """Push pre-rendered HTML (OOB fragments) to one session's frame socket."""
    client = _clients.get(sid)
    if client is None:
        print(f"webui: no client for session {sid!r}, frame dropped", flush=True)
        return
    ws, loop = client
    fut = asyncio.run_coroutine_threadsafe(ws.send_text(html), loop)
    fut.add_done_callback(
        lambda f: (
            f.exception()
            and print(f"webui: frame send failed: {f.exception()!r}", flush=True)
        )
    )


def _running(sess: dict) -> bool:
    p = sess["run"]["proc"]
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


def view_bar(*chips, running: bool, oob: bool = False, mode: str = "grid",
             resumable: bool = False):
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


def _err_box(text: str):
    """The runtime-error overlay: a red egg-pane fixed across the whole bottom of
    the app (the same system as the docs / warning panes), with a one-click copy
    button. The text stays a selectable <pre>. A docs/warning pane covers it
    while open (body.egg-pane-open in app.css)."""
    return Div(
        Div(
            Span("error", cls="egg-pane-title"),
            Button("copy", cls="copybtn", type="button", title="copy the error text"),
            cls="egg-pane-head",
        ),
        Pre(text, cls="egg-pane-body copytext egg-pane-pre"),
        cls="egg-pane egg-pane-error copybox",
    )


def view_fragment(r: SceneResult, code: str, mode: str = "grid", sess: dict | None = None):
    """Build the #view contents from a worker-rendered scene."""
    run = sess["run"] if sess else None
    running = _running(sess) if sess else False
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
        # id ed-validity: the client updates this one chip in place as the user
        # edits (eggShowValidity), so no second validity chip is created.
        chips.append(
            Span(
                "valid" if n == 0 else f"{n} issue{'' if n == 1 else 's'}",
                id="ed-validity",
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
        extras.append(_err_box(r.error))
    # Editing (or reloading) mid-run must not flash the canvas back to a
    # static TFI render: keep the streaming mesh; the fresh exec still
    # supplies the chips, errors, and stdout above/below it.
    run_svg = run["svg"] if (running and run) else None
    svg = run_svg or r.svg
    quality = run["quality"] if (running and run and run_svg) else r.quality
    edit_note = _edit_disabled_note(r) if mode == "edit" else None
    parts = [
        view_bar(*chips, running=running, mode=mode, resumable=_is_resumable(sess)),
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
        _log_panel("".join(run["log"]) if (running and run) else ""),
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
    sid, sess, h, chips: list, running: bool, bounds, hist=None, prev_hist=None,
    quality=True
) -> None:
    refresh_grid_layer(h.scene, h.grid)
    svg = sess["run"]["svg"] = render_svg(h.scene, bounds=bounds)
    # Mid-run frames refresh only the chips so the control buttons stay put; the
    # terminal frame (running=False) swaps the whole bar to re-enable run / grey
    # out stop.
    bar = (
        view_chips(*chips, oob=True)
        if running
        else view_bar(*chips, running=False, oob=True, resumable=_is_resumable(sess))
    )
    msg = to_xml(bar) + to_xml(
        Div(NotStr(svg), id="canvas", cls="canvas", hx_swap_oob="true")
    )
    if quality:
        q = sess["run"]["quality"] = grid_quality(h.scene.grid_blocks)
        msg += to_xml(_quality_panel(q, oob=True))
    if hist is not None:
        msg += to_xml(
            Div(
                NotStr(render_sparkline(hist, prev=prev_hist)),
                id="chart",
                hx_swap_oob="true",
            )
        )
    _broadcast(sid, msg)


def _fail_frame(sid, sess, msg: str, detail: str | None = None) -> None:
    parts = to_xml(
        view_bar(Span(msg, cls="warn"), running=False, oob=True,
                 resumable=_is_resumable(sess))
    )
    if detail:
        parts += to_xml(Div(_err_box(detail), id="viewextra", hx_swap_oob="true"))
    _broadcast(sid, parts)


NO_RUN_MSG = (
    "script registered no run — call egg_webui.run(grid, steps) "
    'in its `if __name__ == "__egg_webui__":` block'
)


def _run_reader(sid: str, code: str, path: str, proc: subprocess.Popen) -> None:
    """Server-side half of a run: render + fan out the frames the worker
    process streams back to this session; keep the harvest for the SU2 export."""
    import traceback as tb

    sess = _session(sid)
    run = sess["run"]
    done = False
    try:
        # The server keeps its own exec of the same script purely for
        # rendering (scene, bounds, grid layout to scatter nodes into).
        # The grid is the one the script registered via egg_webui.run —
        # never one the UI invented.
        ns, _out, err = exec_script(code, path or None)
        reg = ns.get("__egg_webui_run__")
        if err is not None or reg is None:
            _fail_frame(sid, sess, "script error, fix it first" if err else NO_RUN_MSG)
            proc.kill()
            return
        h = harvest(ns, init_grid=False)
        h.grid = reg[0]
        assert h.grid is not None  # a registered run always carries a grid
        bounds = scene_bounds(h.scene)
        hist: dict[str, list[float]] = {"energy": [], "min det": []}
        prev_hist = sess["last"]["hist"]  # previous run, faded under the chart
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
                    _fail_frame(sid, sess, msg[1])
                    done = True
                case "error":
                    _fail_frame(sid, sess, "pipeline failed", detail=msg[1])
                    done = True
                case "print":
                    # a script/pipeline egg.webui_print — append and push the
                    # (bounded) log to #runlog; interleaves with step frames
                    log = run["log"]
                    log.append(msg[1])
                    if len(log) > LOG_MAX:
                        del log[: len(log) - LOG_MAX]
                    _broadcast(sid, to_xml(_log_panel("".join(log), oob=True)))
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
                            sid,
                            sess,
                            h,
                            chips,
                            running=False,
                            bounds=bounds,
                            hist=hist,
                            prev_hist=prev_hist,
                            quality=False,
                        )
                case "done":
                    sess["last"]["code"], sess["last"]["harvest"] = code, h
                    sess["last"]["net"] = net_bytes
                    if any(len(v) >= 2 for v in hist.values()):
                        sess["last"]["hist"] = {k: list(v) for k, v in hist.items()}
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
                    stopped = run["stop"]
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
                            _set_resume(sess, latest_X, code)
                        elif not stopped:
                            _clear_resume(sess)
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
                            sid,
                            sess,
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
            if run["stop"]:
                # A hard kill (double-click stop) can end mid-chunk with no
                # terminal frame; still cache the last result for resume.
                if latest_X is not None:
                    _set_resume(sess, latest_X, code)
                _fail_frame(sid, sess, "stopped")
            else:
                _clear_resume(sess)
                _fail_frame(sid, sess, f"pipeline failed (worker exited {rc})")
    except Exception:
        _clear_resume(sess)
        _fail_frame(sid, sess, "pipeline failed", detail=tb.format_exc())
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
        if run["tmp"]:
            Path(run["tmp"]).unlink(missing_ok=True)
        if run.get("resume_tmp"):
            Path(run["resume_tmp"]).unlink(missing_ok=True)
        run.update(
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
def run_route(code: str, path: str = "", resume: int = 0, sid: str = ""):
    sess = _session(sid)
    run = sess["run"]
    if _running(sess):
        return view_bar(Span("already running", cls="warn"), running=True)
    fd, tmp = tempfile.mkstemp(suffix=".py", prefix="egg-webui-run-")
    with os.fdopen(fd, "w") as f:
        f.write(code)
    # Resume: hand the worker the cached node coordinates (a .npy) so it seeds
    # the grid before running the freshly re-exec'd stages. A plain run drops any
    # stale cache so the next stop starts a fresh resume point.
    resume_tmp = None
    argv = [sys.executable, "-m", WORKER_MODULE, tmp, path]
    if resume and _is_resumable(sess):
        import numpy as np

        rfd, resume_tmp = tempfile.mkstemp(suffix=".npy", prefix="egg-webui-resume-")
        os.close(rfd)
        np.save(resume_tmp, sess["resume"]["X"])
        argv.append(resume_tmp)
    else:
        _clear_resume(sess)
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,  # the frame channel; stderr stays on the console
    )
    t = threading.Thread(target=_run_reader, args=(sid, code, path, proc), daemon=True)
    run.update(proc=proc, reader=t, stop=False, tmp=tmp, resume_tmp=resume_tmp, log=[])
    t.start()
    label = "resuming…" if resume_tmp else "starting…"
    return view_bar(Span(label, cls="phase"), running=True)


@rt("/stop", methods=["post"])
def stop(sid: str = ""):
    run = _session(sid)["run"]
    proc = run["proc"]
    if proc is None or proc.poll() is not None:
        return ""
    if run["stop"]:
        proc.kill()  # second press: hard kill (worker hung mid-chunk)
        return ""
    run["stop"] = True
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
    try:
        p = canonical_path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            p.write_text(data)
        else:
            p.write_bytes(data)
    except (OSError, ValueError) as exc:
        return JSONResponse({"error": f"write failed: {exc}"}, status_code=400)
    return JSONResponse({"path": str(p)})


@rt("/export/su2", methods=["post"])
async def export_su2_route(code: str, path: str = "", out: str = "", sid: str = ""):
    """Write the current mesh as SU2 to ``out`` — the last smoothed grid if the
    script hasn't changed since that run (straight from the resident grid, no
    exec), else a fresh TFI-initialized one built in the render worker."""
    last = _session(sid)["last"]
    if last["harvest"] is not None and last["code"] == code:
        h = last["harvest"]
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
            _render_worker.su2(code, path, timeout=120.0, sid=sid)
        )
        if text is None:
            return JSONResponse({"error": why}, status_code=400)
    return _write_out(out, text)


@rt("/export/lmr", methods=["post"])
async def export_lmr_route(
    code: str, path: str = "", out: str = "", sid: str = "", overwrite: str = ""
):
    """Write the current mesh as gdtk/Eilmer lmr structured blocks + grid.lua into
    the chosen directory ``out`` — the last smoothed grid if the script hasn't
    changed since that run (straight from the resident grid, no exec), else a
    fresh TFI-initialized one built in the render worker. Multi-file, so the
    export writes the files itself rather than returning text.

    Refuses (409 ``conflict``) to overwrite a directory that already holds an
    export unless ``overwrite`` is set, so a hand-edited grid.lua is not
    silently clobbered."""
    if not out:
        return JSONResponse({"error": "no destination chosen"}, status_code=400)
    try:
        out_dir = str(canonical_path(out))
    except (OSError, ValueError) as exc:
        return JSONResponse({"error": f"bad destination: {exc}"}, status_code=400)
    force = overwrite.strip().lower() in ("1", "true", "yes", "on")
    if not force and os.path.exists(os.path.join(out_dir, "grid.lua")):
        return JSONResponse(
            {
                "conflict": True,
                "message": "This folder already contains an exported grid "
                "(grid.lua).",
            },
            status_code=409,
        )
    last = _session(sid)["last"]
    if last["harvest"] is not None and last["code"] == code:
        h = last["harvest"]
        if h.grid is None:
            return JSONResponse({"error": "no grid to export"}, status_code=400)
        try:
            from egg.io.lmr import export_lmr, untagged_external_faces

            written = export_lmr(h.grid, out_dir, overwrite=True)
            untagged = untagged_external_faces(h.grid)
        except Exception as exc:
            return JSONResponse({"error": f"export failed: {exc}"}, status_code=400)
    else:
        payload, why = await asyncio.wrap_future(
            _render_worker.lmr(code, out_dir, path, timeout=120.0, sid=sid)
        )
        if payload is None:
            return JSONResponse({"error": why}, status_code=400)
        written, untagged = payload["written"], payload["untagged"]
    return JSONResponse({"path": out_dir, "files": len(written), "untagged": untagged})


@rt("/export/svg", methods=["post"])
def export_svg_route(svg: str, out: str = ""):
    """Write the client-serialized scene SVG to ``out`` (the SVG is built in the
    browser from the live DOM, so it arrives as text here)."""
    return _write_out(out, svg)


@rt("/export/net", methods=["post"])
def export_net_route(code: str = "", path: str = "", out: str = "", sid: str = ""):
    """Write the resident grid's control net as an ``.npz`` to ``out`` — the
    escape hatch for a run whose pipeline had no :class:`~egg.pipeline.Save`
    stage. Uses the last smoothed grid's net (no re-run) when the script is
    unchanged; 400 when there is no net (a nodal run with no Refit produces none).
    """
    last = _session(sid)["last"]
    if last["harvest"] is None or last["code"] != code:
        return JSONResponse(
            {"error": "run the pipeline first, then save the net"}, status_code=400
        )
    h = last["harvest"]
    net = getattr(h.grid, "control_net", None) if h.grid is not None else None
    if net is None:
        return JSONResponse(
            {
                "error": "this run produced no control net; use the control-point "
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

        arc = str(canonical_path(archive))
        if not eggy.is_eggy(arc):
            raise ValueError("not a .eggy archive (a zip holding a script)")
        leaf = Path(name).name
        if not leaf or leaf in (".", ".."):
            raise ValueError("invalid destination name")
        target = canonical_path(dest) / leaf
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


# The starter script a new project is seeded with: a tiny self-contained
# single-block topology that untangles + smooths, so a fresh project runs at once.
_NEW_PROJECT_STARTER = '''\
"""A new egg project. Build a topology, then register a run for the web UI."""

from egg.geometry import Vector3
from egg.topology.builder import TopologyBuilder


def build():
    b = TopologyBuilder(d=2)
    b.add_block(
        "square",
        sw=Vector3(0, 0, 0), se=Vector3(1, 0, 0),
        nw=Vector3(0, 1, 0), ne=Vector3(1, 1, 0),
        res=(21, 21),
    )
    return b.build()


topology = build()

if __name__ == "__egg_webui__":
    import egg.webui as egg_webui
    from egg.pipeline import Untangle, JacobiSmoother, generate_steps

    grid = topology.initialize_grid()
    egg_webui.run(
        grid,
        generate_steps(
            grid,
            stages=[Untangle(name="untangle folds"),
                    JacobiSmoother(name="shape smoothing")],
        ),
    )
'''


@rt("/new/project", methods=["post"])
def new_project_route(dest: str, name: str):
    """Create a new project directory ``dest/name`` with a starter script and
    return ``{path, code}`` so the client can open it. The script is the bundled
    default example when present, otherwise a minimal runnable topology."""

    def work():
        # name is one directory component; strip any path separators so it can
        # not escape dest with a slash or "..".
        leaf = Path(name).name
        if not leaf or leaf in (".", ".."):
            raise ValueError("invalid project name")
        base = canonical_path(dest) / leaf
        base.mkdir(parents=True, exist_ok=True)
        script = base / f"{leaf}.py"
        if script.exists():
            raise ValueError(f"{script} already exists")
        script.write_text(_NEW_PROJECT_STARTER)
        return str(script), _NEW_PROJECT_STARTER

    try:
        script, code = work()
    except Exception as exc:
        return JSONResponse({"error": f"could not create project: {exc}"}, status_code=400)
    return JSONResponse({"path": script, "code": code})


# --- documentation lookup: map the symbol at point to its Sphinx section ---
# The editor's "show documentation" prefers the already-built Sphinx HTML (rich
# signatures/params) over the raw LSP hover. objects.inv gives FQN -> page#anchor;
# the object's <dl> is lifted out of the page and embedded in the docs pane.

_DOC_ROLES = {
    "class", "function", "method", "attribute", "data",
    "exception", "property", "module",
}
_inventory_cache: list[tuple[str, str, str]] | None = None


def _load_inventory() -> list[tuple[str, str, str]]:
    """Parse Sphinx ``objects.inv`` into ``(fqn, role, uri)`` for py objects.
    Cached; empty when docs (or the inventory) are absent."""
    global _inventory_cache
    if _inventory_cache is not None:
        return _inventory_cache
    import zlib

    out: list[tuple[str, str, str]] = []
    try:
        data = (DOCS_DIR / "objects.inv").read_bytes()  # type: ignore[operator]
        body = zlib.decompress(data.split(b"\n", 4)[4]).decode()
        for line in body.splitlines():
            m = re.match(r"(.+?)\s+py:(\S+)\s+-?\d+\s+(\S+)\s+", line)
            if m:
                out.append((m.group(1), m.group(2), m.group(3)))
    except Exception:
        out = []
    _inventory_cache = out
    return out


def _find_doc_object(name: str) -> tuple[Path, str] | None:
    """Map a bare identifier (or FQN) to ``(html_file, anchor)``: the shortest
    documented py object whose last name part matches."""
    cands = [
        e for e in _load_inventory()
        if e[1] in _DOC_ROLES and (e[0] == name or e[0].rsplit(".", 1)[-1] == name)
    ]
    if not cands:
        return None
    # Prefer a real object over a bare module, then the shortest FQN.
    cands.sort(key=lambda e: (e[1] == "module", len(e[0])))
    fqn, _role, uri = cands[0]
    file_part, _, anchor = uri.partition("#")
    anchor = anchor.replace("$", fqn)  # Sphinx abbreviates the anchor as "$"
    return DOCS_DIR / file_part, anchor  # type: ignore[operator]


def _extract_dl_fragment(html_path: Path, anchor: str) -> str | None:
    """Return the ``<dl>…</dl>`` describing ``anchor`` (depth-matched so a class
    keeps its methods), or ``None`` if the page/anchor is missing."""
    try:
        s = html_path.read_text()
    except OSError:
        return None
    idpos = s.find(f'id="{anchor}"')
    if idpos < 0:
        return None
    start = s.rfind("<dl", 0, idpos)
    if start < 0:
        return None
    depth, i, n = 0, start, len(s)
    while i < n:
        no = s.find("<dl", i)
        nc = s.find("</dl>", i)
        if nc < 0:
            return None
        if no != -1 and no < nc:
            depth += 1
            i = no + 3
        else:
            depth -= 1
            i = nc + 5
            if depth == 0:
                return s[start:i]
    return None


@rt("/api/docsym")
def docsym_route(name: str):
    """The Sphinx documentation fragment for the symbol ``name`` (a bare
    identifier), or ``{found: False}`` to let the client fall back to the LSP
    hover. Serves only from the locally built docs."""
    if not name or not DOCS_DIR or not DOCS_DIR.is_dir():
        return JSONResponse({"found": False})
    try:
        found = _find_doc_object(name)
        if not found:
            return JSONResponse({"found": False})
        html = _extract_dl_fragment(found[0], found[1])
        if not html:
            return JSONResponse({"found": False})
        return JSONResponse({"found": True, "html": html, "fqn": found[1]})
    except Exception:
        return JSONResponse({"found": False})


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

        p = canonical_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code)
        outp = canonical_path(out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        # pack the case folder AND bundle any egg.file_import() deps of the
        # script (copied under deps/, their paths rewritten in the archive).
        deps.pack_case(str(outp), str(p.parent), str(p))
        return str(outp)

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
    try:
        d = canonical_path(dir) if dir else (_REPO_ROOT or Path.home()).resolve()
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
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
    try:
        p = canonical_path(path)
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
    an error. Only real syntax errors block it; type-checker complaints (which
    do not stop the code running) never reach here.
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
    try:
        p = canonical_path(path)
        return JSONResponse({"exists": p.exists(), "is_dir": p.is_dir()})
    except Exception:
        return JSONResponse({"exists": False, "is_dir": False})


@rt("/api/file/save", methods=["post"])
def api_file_save(path: str, code: str):
    try:
        p = canonical_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code)
        return JSONResponse({"path": str(p)})
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


@rt("/api/clientlog", methods=["post"])
def api_clientlog(level: str = "error", msg: str = "", src: str = ""):
    """Log a browser-side error to this process's stderr, so the tee writes it
    to the logfile. JS console errors (browser devtools or the desktop webview)
    are otherwise lost. Truncated to keep a runaway page from flooding the log."""
    tag = level.strip()[:16] or "error"
    text = (msg or "").replace("\r", " ")[:2000]
    where = (" @ " + src[:300]) if src else ""
    try:
        print(f"js {tag}: {text}{where}", file=sys.stderr, flush=True)
    except Exception:
        pass
    return JSONResponse({"ok": True})


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
async def render(code: str, view: str = "grid", path: str = "", sid: str = ""):
    mode = view if view in ("grid", "topo", "edit") else "grid"
    return view_fragment(
        await _render_scene(code, mode, path, sid=sid), code, mode=mode,
        sess=_session(sid)
    )


@rt("/api/topo/validate", methods=["post"])
async def topo_validate(code: str, blocking: str = "{}", path: str = "", sid: str = ""):
    """Diagnostics for a candidate blocking (JSON) against the script's base +
    geometry; an empty list means green/committable. Runs in the render worker
    (it execs the script), never in the server."""
    try:
        f = json.loads(blocking)
    except json.JSONDecodeError:
        return JSONResponse({"error": "bad blocking json"}, status_code=400)
    out = await asyncio.wrap_future(_render_worker.validate(code, f, path, sid=sid))
    return JSONResponse(out)


@rt("/api/topo/commit", methods=["post"])
async def topo_commit(code: str, blocking: str = "{}", path: str = "", sid: str = ""):
    """Write a blocking back into the ``editable({...})`` source span — but only
    when it has no errors. Validates in the worker; advisory ``warn_*``
    diagnostics don't block, real errors refuse with the reasons."""
    try:
        f = json.loads(blocking)
    except json.JSONDecodeError:
        return JSONResponse({"error": "bad blocking json"}, status_code=400)
    val = await asyncio.wrap_future(_render_worker.validate(code, f, path, sid=sid))
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
async def reset(code: str, view: str = "grid", path: str = "", sid: str = ""):
    """Drop the last smoothed result and show the fresh unsmoothed grid.

    A finished run leaves its smoothed mesh on the canvas (and in the session
    state). Reset clears that, so a plain re-exec of the script (TFI-initialized,
    unsmoothed) is what shows and what exports. Off while a run streams (the run
    owns the canvas).
    """
    mode = view if view in ("grid", "topo", "edit") else "grid"
    sess = _session(sid)
    if not _running(sess):
        sess["last"].update(code=None, harvest=None, hist=None)
        sess["run"]["svg"] = sess["run"]["quality"] = None
        _clear_resume(sess)  # reset also discards the resume cache
    return view_fragment(
        await _render_scene(code, mode, path, sid=sid), code, mode=mode, sess=sess
    )


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
            # prompt on unsaved changes / unapplied topology edits, then close
            onclick="window.eggDesktopClose()",
        ),
        cls="desktop-titlebar__controls",
    )


# The docs window (help > documentation) is a SEPARATE pywebview PROCESS (see
# open_docs in egg/_desktop.py: a second window inside the main process is
# unreliable on this Qt build). That process loads this /docs-view shell, an
# egg-themed titlebar (back/forward + the window controls) wrapping the built
# Sphinx site in a same-origin iframe. It reuses app.css (CSS) for the theme +
# .desktop-titlebar rules and adds only the nav-button / iframe layout below.
_DOCS_VIEW_CSS = (
    ".docs-nav { display: flex; align-self: stretch; }\n"
    ".docs-nav .desktop-titlebar__btn { width: 38px; }\n"
    ".docs-title { font-size: calc(var(--fs-ui) * 12 / 13); "
    "color: var(--ctp-subtext0); padding: 0 8px; }\n"
    ".docs-frame { flex: 1 1 auto; min-height: 0; width: 100%; border: 0; "
    "background: var(--ctp-base); }\n"
)

# Runs in the shell page. Matches the app's current catppuccin flavor (the docs
# process shares the app's same-origin localStorage), wires the nav buttons to
# the iframe's own history, and starts a frameless window drag from the titlebar
# spacer (same start_drag bridge call as app.js).
_DOCS_VIEW_JS = (
    "(function () {\n"
    "  var THEMES = ['mocha', 'macchiato', 'frappe', 'latte'];\n"
    "  var t = localStorage.getItem('egg-webui-theme');\n"
    "  if (THEMES.indexOf(t) < 0)\n"
    "    t = window.matchMedia('(prefers-color-scheme: dark)').matches"
    " ? 'mocha' : 'latte';\n"
    "  document.documentElement.dataset.theme = t;\n"
    "  var frame = document.getElementById('docs-frame');\n"
    "  function onFrame(fn) {\n"
    "    return function () { try { fn(frame.contentWindow); } catch (e) {} };\n"
    "  }\n"
    "  document.getElementById('docs-back').onclick ="
    " onFrame(function (w) { w.history.back(); });\n"
    "  document.getElementById('docs-fwd').onclick ="
    " onFrame(function (w) { w.history.forward(); });\n"
    "  document.getElementById('docs-reload').onclick ="
    " onFrame(function (w) { w.location.reload(); });\n"
    "  document.addEventListener('mousedown', function (e) {\n"
    "    if (e.button === 0 && e.target.closest('.desktop-titlebar__drag'))\n"
    "      window.pywebview && window.pywebview.api"
    " && window.pywebview.api.start_drag && window.pywebview.api.start_drag();\n"
    "  });\n"
    "})();\n"
)


def _docs_window_controls():
    """Min / maximize / close for the docs window titlebar. Like
    :func:`_window_controls`, but close just closes the docs window (its own
    process; no unsaved-changes prompt)."""
    return Div(
        Button(
            "−",
            type="button",
            aria_label="Minimize",
            cls="desktop-titlebar__btn",
            onclick="window.pywebview.api.minimize()",
        ),
        Button(
            "□",
            type="button",
            aria_label="Maximize",
            cls="desktop-titlebar__btn",
            onclick="window.pywebview.api.toggle_maximize()",
        ),
        Button(
            "✕",
            type="button",
            aria_label="Close",
            cls="desktop-titlebar__btn desktop-titlebar__close",
            onclick="window.pywebview.api.close()",
        ),
        cls="desktop-titlebar__controls",
    )


@rt("/docs-view")
def docs_view():
    """The docs window shell (loaded by the separate docs process, see open_docs
    in egg/_desktop.py). A standalone document (not the app's ``hdrs``, so app.js
    / the editor never load here) that frames the built Sphinx site under an
    egg-themed titlebar. A plain browser never hits this; it opens /docs/ in a
    new tab instead."""
    if not (DOCS_DIR and DOCS_DIR.is_dir()):
        return PlainTextResponse("docs not built", status_code=404)

    def _nav(sym, id_, label):
        return Button(
            sym,
            type="button",
            id=id_,
            aria_label=label,
            title=label,
            cls="desktop-titlebar__btn",
        )

    doc = Html(
        Head(
            Meta(charset="utf-8"),
            Meta(name="viewport", content="width=device-width, initial-scale=1"),
            Title("egg documentation"),
            Style(CSS),
            Style(_DOCS_VIEW_CSS),
        ),
        Body(
            Header(
                NotStr(EGG_LOGO),
                Div(
                    _nav("←", "docs-back", "Back"),
                    _nav("→", "docs-fwd", "Forward"),
                    _nav("⟳", "docs-reload", "Reload"),
                    cls="docs-nav",
                ),
                Span("documentation", cls="docs-title"),
                Div(cls="desktop-titlebar__drag"),
                _docs_window_controls(),
                cls="desktop-titlebar",
            ),
            Iframe(src="/docs/index.html", id="docs-frame", cls="docs-frame"),
            Script(_DOCS_VIEW_JS),
        ),
    )
    # Html(doctype=True) already prepends <!DOCTYPE html>.
    return HTMLResponse(to_xml(doc))


def _landing_titlebar():
    """A minimal window titlebar for the landing overlay in the native app: just
    the drag handle and the min/maximize/close controls (no menus). The landing
    overlay covers the real titlebar, so without this the window controls would
    be unreachable while the landing page is up."""
    return Header(
        Div(cls="desktop-titlebar__drag"),
        _window_controls(),
        cls="desktop-titlebar landing-titlebar",
    )


def _titlebar_or_header(desktop: object, *menu_items):
    """The top toolbar: the same logo + file/view/help dropdowns either way,
    followed by the filename chip and the pan/zoom hint.

    In an ordinary browser this is the plain app header. In the native
    (pywebview) app (``/?desktop=1``, opened by the ``egg-desktop`` launcher)
    it becomes the window titlebar: the menus move into it, and it also carries
    a drag handle and the window controls. The launcher runs a *frameless*
    window; dragging the spacer starts a compositor move via
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
        # The top toolbar: the ordinary browser header, or (at /?desktop=1,
        # opened by the egg-desktop launcher) the native-app titlebar with
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
                            "gdtk grid (Eilmer)",
                            id="file-dl-lmr",
                            title="write the current mesh as gdtk/Eilmer lmr "
                            "structured blocks plus a grid.lua for prep-grid; pick "
                            "a folder to write them into",
                        ),
                        Button(
                            "control net (npz)",
                            id="file-dl-net",
                            title="save the last run's control net as an .npz, the "
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
                # "grid" hides only the interior grid lines (a scene-toggle on
                # #view), so block boundaries and fills stay independently
                # toggleable below even with the grid off.
                Label(
                    Input(
                        type="checkbox",
                        checked=True,
                        cls="scene-toggle",
                        data_toggle="grid-lines",
                    ),
                    "grid",
                ),
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
                    for layer in ("tangled", "curves", "points")
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
                # In the desktop app this opens the in-app docs viewer (the
                # overlay below); in a browser app.js opens /docs/ in a new tab.
                Button("documentation", id="help-docs")
                if DOCS_DIR and DOCS_DIR.is_dir()
                else Span(
                    "docs not built; restart the app to build them",
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
                    # only when the LSP reports no errors, so a half-typed line
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
            # The run-frame socket is opened manually in app.js (with the session
            # id: /ws?sid=...), so server-pushed OOB frames reach only this
            # instance. (Not htmx-ws, which can't carry the per-instance sid.)
        ),
        # Landing page: shown by JS at startup when no script is open (no default
        # example is loaded). A splash plus the entry points; each option drives
        # an existing flow and hides the landing once a project is opened.
        Div(
            # native app: a minimal titlebar so the window controls stay reachable
            # while the landing overlay is up (it covers the real titlebar)
            *([_landing_titlebar()] if desktop else []),
            Div(
                Div(NotStr(EGG_LOGO), cls="landing-logo"),
                Div("egg", cls="landing-title"),
                Div("egg aims to be an excellent grid generator", cls="landing-sub"),
                Div(
                    # shown by JS only when a cached session exists
                    Button(
                        "restore cached session", id="landing-restore",
                        cls="landing-opt", style="display:none",
                    ),
                    Button("recently opened", id="landing-recent", cls="landing-opt"),
                    # only in a source checkout, where the examples ship
                    *(
                        [Button(
                            "examples", id="landing-examples", cls="landing-opt",
                            data_dir=str(EXAMPLES_ROOT),
                        )]
                        if EXAMPLES_ROOT and EXAMPLES_ROOT.is_dir()
                        else []
                    ),
                    Button("new project", id="landing-new", cls="landing-opt"),
                    Button("open project", id="landing-open", cls="landing-opt"),
                    Button("open archive", id="landing-archive", cls="landing-opt"),
                    Button("documentation", id="landing-docs", cls="landing-opt"),
                    Button(
                        "configuration directory", id="landing-config",
                        cls="landing-opt",
                    ),
                    cls="landing-opts",
                ),
                cls="landing-card",
            ),
            id="landing",
            style="display:none",
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
    # No script argument -> start on the landing page (no default example
    # loaded); the editor stays empty until the user picks/creates one.
    p = _script_arg()
    return p.read_text() if p else ""


def _initial_path() -> str:
    p = _script_arg()
    return str(p) if p else ""


if __name__ == "__main__":
    serve(port=5001, reload=False)
