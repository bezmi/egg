# egg webui — "code as CAD" prototype

A FastHTML + HTMX prototype of a 2D web UI: the left pane is a Python
script using the egg front-end (`Vector3`, `Line`, `Arc`, `Spline`, `Edge`,
`TopologyBuilder`, ...), the right pane is a live SVG render of whatever the
script defines. Rendering re-runs ~0.5 s after you stop typing; the
**run** button consumes the pipeline the script explicitly registered via
`egg_webui.run(grid, steps)` and streams the relaxing mesh back over a
websocket.

```bash
uv sync --group webui          # once
uv run --no-sync egg-webui                          # http://127.0.0.1:5001
uv run --no-sync egg-webui my_geometry.py           # open an existing script
uv run --no-sync egg-webui my_geometry.py --watch   # start in watch mode
uv run --no-sync egg-webui --host 0.0.0.0 --reload  # dev server: auto-restart
                                                    # on edits to webui/ or egg/
# (reload drops websocket connections and any in-flight run; open a
# script with the dev server via the positional arg or EGG_WEBUI_SCRIPT)
# equivalent without the console script: uv run --no-sync python webui/app.py
```

What gets drawn (harvested from the script's namespace, recursing a few
levels into lists/tuples/dicts):

- parametric curve entities (sampled via `eval_frac`), `Edge` wrappers,
- `Vector3` points (squares when `fixed=True`), placed `Node`s (rings),
- **construction points** (toggleable "control points" layer, off by
  default): the Vector3s each curve was built from, drawn CAD-style with
  dashed cage lines — spline through-point chords, Bézier control
  polygons, arc radius lines to the centre; `Polyline`s recurse per
  segment. Found via the gdtk-style retained attributes (`line.p0`,
  `arc.centre`, `spline.points`), so function-local construction shows
  too; clickable like any point;
- `SvgDomain` objects from `svg_import(...)`: every labeled Inkscape path
  draws under its label (`dom['egg']`, …);
- the structured grid, if the script builds one: an explicit
  `MultiBlockGrid` wins, else a `BlockTopology` (or a bare
  `TopologyBuilder`, best effort) is TFI-initialized for preview —
  including projection of associated boundary nodes onto their entities;
- cells whose corner-Jacobian det goes non-positive get a red overlay,
  and the status bar shows the preview min det (red when folded).

Grid view adds a **quality** strip (orthogonality / equiangular skewness /
aspect ratio, each as mean and worst cell; recomputed at most once
a second while a run streams, and always on the final frame). Clicking a
block reports its name and
`i × j = N` cell counts; clicking empty canvas reports the grid total.
The convergence chart draws the previous run's series faded under the
live ones (shared per-series scale) for before/after comparison.

**Theming**: [catppuccin](https://catppuccin.com), all four flavors —
**view → theme** picks between mocha / macchiato / frappé (dark) and
latte (light); the default follows the OS `prefers-color-scheme`, the
choice persists in localStorage, and the CodeMirror editor re-themes
live (its theme + syntax palette are built from the same CSS variables
via a `Compartment`). The raw palette lives in `--ctp-*` custom
properties on `:root` (flavors override via `[data-theme]`); every color
and linetype in the chrome *and* both views (grid lines, block outlines,
tangled overlay, topology lines, corner markers, curves, points,
highlight, sparkline) routes through them and the semantic `--egg-*`
layer, so a small CSS override still restyles everything — elements
carry semantic classes only, and derived colors (block-line shades,
hovers, fills) mix toward other palette entries, so every flavor stays
pure catppuccin. SVG downloads embed the stylesheet and
carry the active flavor on the root element, so exports stay
self-contained and themed. The Sphinx docs served at `/docs/` share the
localStorage key: opened from the UI they follow its flavor (live, via
storage events), and standalone they offer their own sidebar selector.
A hover/touch readout shows the pointer
position in geometry coordinates.

Header controls: a **file** menu (open… / examples… / save / save as… /
watch file / export svg / export su2 — examples… is just the open dialog
rooted at `examples/2D/`), a **view** menu (fit, per-layer visibility
toggles: grid / tangled / curves / points / control points, and the
theme picker — the view dropdown is disabled while a run streams, since
frames render the grid view), and a **help**
menu whose documentation link opens the locally served Sphinx site at
`/docs/` (`egg-webui` refreshes `docs/_build/html` at startup when the
docs group + doxygen are available; `--no-docs` skips). Scroll or pinch to
zoom, drag to pan (touch works), hover for variable names, view → fit
resets; the editor/viewer split is draggable (Split.js, pinned 1.6.5).
All CDN dependencies are version-pinned; every one degrades gracefully
if unreachable.

**Files**: open/save is a normal workflow against the *server's*
filesystem (which for this local single-user tool is the user's own
machine — the script already execs with full privileges, so this adds no
new exposure). file → open… browses directories (`.py` files only) and
loads a script into the editor; save (or Ctrl+S / Cmd+S) writes it back;
save as… picks a directory and filename. The filename chip in the header
shows the open file with a • when the buffer has unsaved changes; a
script passed on the CLI is pre-associated, and the association persists
across reloads.

**The `__egg_webui__` contract (no magic)**: scripts exec with
`__name__ = "__egg_webui__"`, so a script's `main()` guard is inert and
an `if __name__ == "__egg_webui__":` block — the mirror image of the
`__main__` guard — runs *only* in the UI. Every repo example's block
feeds its own `setup()` (the exact wiring `main()` uses) the CLI defaults —
declared with `egg_webui.params(...)`, an explicit marker (not "the first
dict in the guard") — and hands the result over explicitly:

```python
if __name__ == "__egg_webui__":
    import egg_webui

    # CLI defaults, mirroring driver.py — edit freely
    a = egg_webui.params(
        sweeps_per_delta=20, tmop_sweeps=60, chunk=10,
        smoother="jacobi", omega=0.8, device="cpu",
    )
    topo, ents, grid, cfg = setup(a)
    egg_webui.run(grid, generate_steps(grid, config=cfg, untangle_direct=False))
```

(`main()` calls the same `setup(vars(parse_args()))`, so CLI and UI share
one wiring; `egg_webui.params(...)` returns the plain dict at runtime and
marks it as the UI-side knob panel.) The **run parameters**
strip under the view bar renders those entries as a form: number fields keep
the source spelling (`5.0e-3` stays `5.0e-3`), `smoother`/`device` get
dropdowns, bools get checkboxes, and an edit rewrites *only that value's
source span* in the buffer (`/api/param`, AST-located) — the code stays
the single source of truth. Non-literal entries are skipped; watch mode
disables the panel (a watched file is never modified).

`egg_webui.print(...)` — or `egg.webui_print(...)` in code shared with
`main()`, which can't import `egg_webui` — is a `print` that reaches the
UI: its output streams to a log panel under the convergence chart during a
run, and folds into the render's *stdout* panel while editing. It is a
no-op headless (the CLI, or any run with no UI attached), so the identical
call is silent outside the UI. Both share one process-global sink the
worker (`worker.py`, streams it as `("print", text)` frames) and render
worker (`render_worker.py`, into captured stdout) install; the core lives
in `egg._webui_print`.

Any other literal joins the panel by wrapping it in
`egg_webui.editable(...)` — an identity function at runtime, a marker in
the AST — anywhere in the script, not just the dict:

```python
N_RINGS = egg_webui.editable(3, label="dipole rings")        # int box
a = egg_webui.params(
    metric=egg_webui.editable("shape_size",
                              choices=["shape", "shape_size"]),  # dropdown
    dipole=egg_webui.editable(True),                             # checkbox
)
```

The input type follows the literal (bool → checkbox, int/float → numeric
box, str → text box); `choices=[...]` renders a dropdown and rejects
values outside it; `label=` overrides the panel name, which otherwise
comes from the assignment target, dict key, or keyword argument the call
sits in. Edits splice the literal *inside* the call, so the marker
survives every round trip. Without an
`egg_webui.run(...)` registration the run button does nothing — the UI
never invents a pipeline behind the script's back. Opening a
library-style script without the block (a single parameterless `build*`
function, nothing drawn at top level) prompts with a ready-made block:
appended to the file *only on confirmation* — or, in watch mode, shown
in a copy-paste popup instead (a watched file is never modified).

**Watch mode** (file → watch file) — the expected day-to-day workflow
for anyone with an editor they already like: the editor pane disappears,
the UI polls the opened file (~1 s) and re-renders on every save from
your editor; run/stop, both views, and the exports keep working against
the synced buffer. Unsaved built-in-editor changes prompt before being
discarded; unticking restores the editor with the file's current
content; the watch state survives page reloads. A watched file is never
written to by the UI.

**Running**: the run button (or `Ctrl+Enter` / `Cmd+Enter`; the same key
stops a streaming run) hands the script to a **separate worker
process** (`worker.py`) that execs it and consumes the steps generator
the script registered — the pipeline and the SYCL runtime never run
inside the server process, so a hung or crashed solve can't take the UI
down. The worker streams node updates back over a pipe (script/pipeline
`print`s are rerouted to the server log); the server renders them and
pushes frames (throttled to ~16 fps) to every connected browser via the
websocket, with a live energy / min-det chart (axes + legend) under the
view. Pass `untangle_direct=False` so the untangle phase animates. One
run at a time; **stop** takes effect at the next chunk boundary — press
it again to hard-kill a worker that is stuck mid-chunk. A dev-server
reload closes the worker's stdin, which the worker treats as stop, so no
orphan keeps computing.

**Export**: the **su2** button downloads the current mesh in SU2 format
(markers from `tag_boundary`); if the script hasn't changed since the
last run, the *smoothed* grid is exported — what you see is what you
get. **svg** downloads the current view.

**Editor**: CodeMirror 6 (loaded as ES modules from esm.sh — if the CDN
is unreachable the plain textarea silently takes over) with Python syntax
highlighting and autocompletion. Completions are introspected from the
real egg API at `/api/completions`: geometry constructors,
`TopologyBuilder` methods with their signatures, and `PipelineConfig`
fields, layered on lang-python's builtin global/local completion
(Ctrl-Space to trigger explicitly).

Script errors show as a traceback under the view, with a clickable
"error at line N" chip that jumps the editor there; whatever was defined
before the error still renders. `print` output lands in a collapsible
"stdout" section. Editor content persists across page reloads
(localStorage) unless a script path was passed on the CLI.

Set `EGG_WEBUI_DEBUG=1` to debug a drawn topology: whenever an
`ExplicitTopology` produces a diagnostic (a warning or a hard error), its
diagnostics **and the full connectivity** (the `nodes`/`edges`/`res` blocking
dict) are dumped to the server console — the terminal running `egg-webui` — so
you can inspect the exact blocking that failed. Off by default (the render
loop would otherwise dump on every keystroke render).

**Rendering**: editor renders exec the script in a **persistent render
worker process** (`render_worker.py`, kept warm with egg imported), not
in the server — a script stuck in a loop is killed with its worker at
the 30 s render timeout and the UI stays live, and a native crash in a
script costs one render plus a respawn instead of the server. Keystroke
bursts coalesce: queued renders collapse to the newest code, so a slow
render followed by more typing never renders stale code. The same worker
serves the open-dialog's run-block probe and the SU2 export, so the only
in-server script exec left is the run pipeline's frame renderer (it
needs the live grid object the nodes stream into). Grid previews above
~200k nodes are skipped, and dense blocks draw at most 129 preview grid
lines per axis — the grid itself is untouched. Editing while a run
streams keeps the relaxing mesh in the canvas: the re-render updates the
chips / errors / stdout around it and never flashes back to a static
TFI view.

Deliberate prototype limitations:

- the script runs with **full interpreter privileges** (worker
  processes, plus the server itself for the run's frame renderer) —
  this is a local, single-user tool, not a deployable service;
- 2D only.
