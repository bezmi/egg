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

## Running the demos

The circle demos accept `--device {auto,cpu,gpu}` to select the SYCL device:

```bash
# CPU
uv run examples/circles/good-topo.py --tmop-sweeps 40 --device cpu
uv run examples/circles/untangle.py --tmop-sweeps 1000 --device cpu

# GPU
uv run examples/circles/good-topo.py --tmop-sweeps 40 --device gpu
uv run examples/circles/untangle.py --tmop-sweeps 1000 --device gpu
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
