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
import json
import mimetypes
import os
import pickle
import re
import subprocess
import sys
import tempfile
import threading
import time
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
    render_sparkline,
    render_svg,
    scene_bounds,
    set_editable_blocking,
    set_guard_param,
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
_PCE_CSS = ("layout.css", "search.css", "autocomplete.css", "autocomplete-icons.css",
            "cursor.css")

app, rt = fast_app(
    pico=False,
    exts="ws",
    hdrs=(
        *(Link(rel="stylesheet", href=f"/vendor/{c}") for c in _PCE_CSS),
        Style(CSS),
        Style(CATPPUCCIN),
        Script(src=_SPLIT_SRC),
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
_last: dict = {"code": None, "harvest": None, "hist": None}

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
    runner = (
        [
            Button(
                "run",
                cls="primary",
                hx_post="/run",
                hx_include="[name='code'],[name='path']",
                hx_target="#viewbar",
                hx_swap="outerHTML",
                disabled=running or None,
                title="run (Ctrl+Enter)",
            ),
            Button(
                "stop",
                cls="danger",
                hx_post="/stop",
                hx_swap="none",
                disabled=(not running) or None,
                title="stop (Ctrl+Enter)",
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
        *chips,
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
    params = guard_params(code)
    if not params:
        return Div(id="params")
    fields = []
    for p in params:
        common = dict(cls="param-input", data_param=p.name)
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
                NotStr(json.dumps(r.edit_data)),
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
    msg = to_xml(view_bar(*chips, running=running, oob=True)) + to_xml(
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
        bounds = scene_bounds(h.scene)
        hist: dict[str, list[float]] = {"energy": [], "min det": []}
        prev_hist = _last["hist"]  # previous run, faded under the chart
        last_emit = 0.0
        last_q = 0.0  # quality stats debounce off the frame hot path
        last_phase = None
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
                case "done":
                    _last["code"], _last["harvest"] = code, h
                    if any(len(v) >= 2 for v in hist.values()):
                        _last["hist"] = {k: list(v) for k, v in hist.items()}
                    done = True
                case "step":
                    phase, info, X = msg[1], msg[2], msg[3]
                    # Mirror of pipeline._sync on the server-side grid.
                    h.grid.global_nodes = X
                    for bi, blk in enumerate(h.grid.blocks):
                        blk.nodes[...] = X[h.grid.block_dof_maps[bi]]
                    if (e := info.get("energy")) is not None:
                        hist["energy"].append(e)
                    if (m := info.get("min_det")) is not None:
                        hist["min det"].append(m)
                    stopped = _run["stop"]
                    chips = [Span(phase, cls="phase")]
                    chips += [
                        Span(_fmt_chip(k, v)) for k, v in info.items() if k != "min_det"
                    ]
                    if (c := _mindet_chip(info.get("min_det"))) is not None:
                        chips.append(c)
                    if stopped:
                        chips.append(Span("stopped", cls="warn"))
                    final = stopped or phase == "final"
                    now = time.perf_counter()
                    # Phase transitions always emit, so fast runs show each stage.
                    if (
                        final
                        or phase != last_phase
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
        if not done:
            # Channel closed without a terminal message: crash or hard kill.
            rc = proc.wait()
            if _run["stop"]:
                _fail_frame("stopped")
            else:
                _fail_frame(f"pipeline failed (worker exited {rc})")
    except Exception:
        _fail_frame("pipeline failed", detail=tb.format_exc())
    finally:
        for pipe in (proc.stdout, proc.stdin):
            try:
                pipe.close()
            except Exception:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        if _run["tmp"]:
            Path(_run["tmp"]).unlink(missing_ok=True)
        _run.update(
            proc=None,
            reader=None,
            stop=False,
            tmp=None,
            svg=None,
            quality=None,
            log=[],
        )


@rt("/run", methods=["post"])
def run_route(code: str, path: str = ""):
    if _running():
        return view_bar(Span("already running", cls="warn"), running=True)
    fd, tmp = tempfile.mkstemp(suffix=".py", prefix="egg-webui-run-")
    with os.fdopen(fd, "w") as f:
        f.write(code)
    proc = subprocess.Popen(
        [sys.executable, "-m", WORKER_MODULE, tmp, path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,  # the frame channel; stderr stays on the console
    )
    t = threading.Thread(target=_run_reader, args=(code, path, proc), daemon=True)
    _run.update(proc=proc, reader=t, stop=False, tmp=tmp, log=[])
    t.start()
    return view_bar(Span("starting…", cls="phase"), running=True)


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


@rt("/export/su2", methods=["post"])
async def export_su2_route(code: str, path: str = ""):
    """SU2 export of the current mesh — the last smoothed grid if the script
    hasn't changed since that run (exported straight from the resident grid,
    no exec), else a fresh TFI-initialized one built in the render worker."""
    if _last["harvest"] is not None and _last["code"] == code:
        h = _last["harvest"]
        if h.grid is None:
            return PlainTextResponse("no grid to export", status_code=400)
        try:
            text = grid_to_su2_text(h.grid)
        except Exception as exc:
            return PlainTextResponse(f"export failed: {exc}", status_code=400)
    else:
        # big grids TFI-init in the worker; give them room beyond the
        # editor-render timeout
        text, why = await asyncio.wrap_future(
            _render_worker.su2(code, path, timeout=120.0)
        )
        if text is None:
            return PlainTextResponse(why, status_code=400)
    return PlainTextResponse(
        text,
        headers={"Content-Disposition": 'attachment; filename="mesh.su2"'},
    )


# --- open/save: the server's filesystem IS the user's filesystem (local
# --- single-user tool — the script already execs with full privileges) ---


@rt("/api/files")
def api_files(dir: str = ""):
    """List one directory: subdirs + .py files (hidden/__pycache__ skipped)."""
    d = (Path(dir).expanduser() if dir else (_REPO_ROOT or Path.home())).resolve()
    if not d.is_dir():
        return JSONResponse({"error": f"not a directory: {d}"}, status_code=400)
    try:
        entries = sorted(d.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        return JSONResponse({"error": f"permission denied: {d}"}, status_code=400)
    skip = lambda n: n.startswith(".") or n == "__pycache__"  # noqa: E731
    return JSONResponse(
        {
            "dir": str(d),
            "parent": str(d.parent),
            "dirs": [p.name for p in entries if p.is_dir() and not skip(p.name)],
            "files": [p.name for p in entries if p.is_file() and p.suffix == ".py"],
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


@rt("/api/file/save", methods=["post"])
def api_file_save(path: str, code: str):
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code)
        return JSONResponse({"path": str(p.resolve())})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


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
    diags = await asyncio.wrap_future(_render_worker.validate(code, f, path))
    return JSONResponse({"diagnostics": diags})


@rt("/api/topo/commit", methods=["post"])
async def topo_commit(code: str, blocking: str = "{}", path: str = ""):
    """Write a blocking back into the ``editable({...})`` source span — but only
    when it has no errors. Validates in the worker; advisory ``warn_*``
    diagnostics don't block, real errors refuse with the reasons."""
    try:
        f = json.loads(blocking)
    except json.JSONDecodeError:
        return JSONResponse({"error": "bad blocking json"}, status_code=400)
    diags = await asyncio.wrap_future(_render_worker.validate(code, f, path))
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
    return view_fragment(await _render_scene(code, mode, path), code, mode=mode)


def _menu(label: str, *items):
    """Header dropdown: a button + absolutely-positioned item panel."""
    return Div(Button(label, cls="menu-btn"), Div(*items, cls="menu-items"), cls="menu")


@rt("/")
async def get(view: str = "grid"):
    mode = view if view in ("grid", "topo", "edit") else "grid"
    code, path = _initial_code(), _initial_path()
    scene_r = await _render_scene(code, mode, path)
    return (
        Title("egg webui"),
        Header(
            Span("egg", style="font-weight:600"),
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
                Div(cls="menu-sep"),
                Button("export svg", id="file-dl-svg"),
                Button("export su2", id="file-dl-su2"),
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
            ),
            Span(id="filechip"),
            Span("scroll or pinch to zoom, drag to pan, hover for names", cls="hint"),
        ),
        Div(
            Div(
                Textarea(
                    code,
                    name="code",
                    spellcheck="false",
                    data_persist="0" if _script_arg() is not None else "1",
                    data_file=str(_script_arg() or ""),
                    data_watch="1"
                    if os.environ.get("EGG_WEBUI_WATCH") and _script_arg()
                    else "0",
                    hx_post="/render",
                    hx_include="[name='view'],[name='path']",
                    hx_target="#view",
                    hx_trigger="input changed delay:500ms",
                    hx_sync="this:replace",
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
                Div(id="fs-dir", cls="fs-path"),
                Div(id="fs-list"),
                Div(
                    Input(id="fs-name", placeholder="filename.py", autocomplete="off"),
                    Button("save", id="fs-do-save", cls="primary"),
                    cls="fs-saverow",
                ),
                cls="fs-box",
            ),
            id="fsmodal",
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
                    Span("save edits"),
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
