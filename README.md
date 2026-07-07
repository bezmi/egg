# egg

Egg aims to be an **e**xcellent **g**rid **g**enerator.

Very early WIP.

A structured multi-block grid generator with a TMOP-style mesh smoother /
untangler. Written as a C++23 core and a python driver.
AdaptiveCpp SYCL compute for performance portable kernels
from one source (OMP, ROCm, CUDA).

NOTE: AI has been used heavily to get this to a proof-of-concept working state.

Everything else (topology, geometry, TFI init, projection, visualization,
pipeline orchestration) stays in Python.

## Setup

For now, see [`DEVELOPING.md`](DEVELOPING.md) to get started.

## Web UI

A browser front-end ("code as CAD"): a Python geometry script on the left,
a live SVG render of its topology/grid on the right, with the real
untangle + TMOP pipeline streaming into the view.

```bash
uv sync --group webui          # once (add --group docs for the help menu)
uv run --no-sync egg-webui     # → http://127.0.0.1:5001
```

`egg-webui my_geometry.py` opens a script; `--host 0.0.0.0` exposes it on
the network; `--reload` restarts on source edits; `--no-docs` skips the
Sphinx docs refresh (help → documentation serves them at `/docs/`).

The basics:

- **file → examples…** opens any script under `examples/2D/`; every
  example draws its topology and registers a run with the CLI-default
  settings. **file → open/save/save as** is a normal file workflow
  (Ctrl+S saves), and **file → watch file** — the expected day-to-day
  workflow — hides the built-in editor and follows the opened file on
  disk: edit in your own editor, the view updates on save, run/exports
  keep working.
- **run** executes the pipeline a script registered via
  `egg_webui.run(grid, steps)` in its `if __name__ == "__egg_webui__":`
  block (the mirror image of the `__main__` guard — see any example) and
  animates the relaxing mesh with live energy / min-det charts; **stop**
  halts at the next chunk boundary. No registration, no run: the UI never
  invents a pipeline. In the examples, the block's `a = dict(...)` is the
  knob panel (sweeps, smoother, device, …).
- The view switches between the smoothed **grid** and the block
  **topology** (tap elements for names + constraining geometry);
  **file → export su2 / svg** downloads the mesh (with `tag_boundary`
  markers) or the picture.

Details, theming, and the full script contract: [`webui/README.md`](webui/README.md).

## Running the demos

The circle demos accept `--device {auto,cpu,gpu}` to select the SYCL device:

```bash
# CPU
uv run examples/2D/circles/good-topo.py --tmop-sweeps 40 --device cpu
uv run examples/2D/circles/untangle.py --tmop-sweeps 1000 --device cpu

# GPU
uv run examples/2D/circles/good-topo.py --tmop-sweeps 40 --device gpu
uv run examples/2D/circles/untangle.py --tmop-sweeps 1000 --device gpu
```

Common flags: `--plot-live` (PyVista animation), `--plot-grid` (matplotlib
wireframe), `--chunk N`. Without a plot flag the demo runs headless and prints
per-chunk progress.

A `generic` AdaptiveCpp build exposes *all* visible SYCL devices (the OpenMP host
**and** the GPU). Use `ACPP_VISIBILITY_MASK` to restrict which backends are loaded:

```bash
# Force CPU only
ACPP_VISIBILITY_MASK=omp uv run ...

# Force GPU only (AMD)
ACPP_VISIBILITY_MASK=hip uv run ...
```

Set `OMP_NUM_THREADS=N` to control CPU parallelism.

## Tests

These should be taken with a grain of salt for now. They are mostly AI generated, but have proved useful in the development so far.

```bash
# Python tests (C++-dependent tests skip if cpp_core is not built)
ACPP_VISIBILITY_MASK=omp uv run pytest tests/

# C++ unit tests (Boost.UT)
ctest --test-dir build
```

Parity tests cross-check the C++ sweep / untangle against the NumPy sequential
solver (`local_relaxation_sweep`), on both CPU and GPU.
