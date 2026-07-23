# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`egg` is a structured multi-block grid generator with a TMOP-style mesh smoother/untangler: a header-only C++23 SYCL compute core (`src/`, compiled with AdaptiveCpp — one source targets OpenMP host, CUDA, and ROCm) driven by a Python layer (`egg/`) that owns everything non-numeric (topology, geometry front-end, TFI init, projection, pipeline orchestration, visualization). Licensed PolyForm Noncommercial; every source file carries a required-notice header — keep it on new files.

## Build, test, run

The project is a scikit-build-core editable install managed by `uv`. Python edits are live; C++ edits recompile on next import.

```bash
# Full clean rebuild (the reliable way to switch CMake defines — they are ONE-SHOT):
rm -rf build && uv sync --force-reinstall \
  -C cmake.define.EGG_REAL_IS_FP32=OFF \
  -C cmake.define.EGG_BUILD_TESTS=ON

# After building, run things WITHOUT re-syncing (avoids clobbering the -C defines):
uv run --no-sync pytest tests/                    # full Python suite
uv run --no-sync pytest tests/smoothing/test_fas.py -q          # one file
uv run --no-sync pytest "tests/test_cpp_composite.py::test_composite_device_relaxes_on_curve[lines]"  # one test

# C++ (Boost.UT) tests live under the wheel-tag subdir of build/:
ctest --test-dir build/cp313-cp313-linux_x86_64 --output-on-failure

# Examples select the SYCL device themselves:
uv run --no-sync python examples/3D/sphere3d/sphere_in_cube_nurbs.py --device gpu --smoother fas
uv run --no-sync python examples/2D/circles/good-topo.py --device cpu
```

Pitfalls that waste time if you don't know them:

- **Never delete `build/` and then plain `uv sync`** — the editable hook only re-runs `cmake --build`, not configure, so the next import dies on `missing CMakeCache.txt`. Use the `rm -rf build && uv sync --force-reinstall -C ...` pattern above (it takes seconds warm).
- **Precision is a build-time knob**: `EGG_REAL_IS_FP32=OFF` (double) is the correctness gate — run it with tight tolerances before calling anything done; `ON` (float) is for fast GPU iteration. Tests route tolerances through `tests/real_tol.py` / `tests/cpp/real_tol.hpp` (identity in fp64, floored in fp32).
- **Never assert exact energies** — reduction order makes console energies fluctuate in the 4th–5th digit between runs. Tests use monotonicity, relative bounds, and `real_tol` slack.
- A `generic` AdaptiveCpp build exposes the OpenMP host *and* the GPU; `--device {cpu,gpu,auto}` picks at runtime. `ACPP_VISIBILITY_MASK=omp` forces CPU-only, `OMP_NUM_THREADS` controls host parallelism.
- Format with `clang-format -i` (repo `.clang-format`) and `ruff format`. Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`, ...), single-line messages; iterative fixes get squashed/amended into the feature commit rather than stacked.
- Docs: `uv run sphinx-build -W -b html docs docs/_build/html` (Sphinx is a core dep; needs the `doxygen` binary; Doxygen runs automatically from `conf.py`).
- `FAS_plan.md` at the repo root is a deliberately untracked planning scratchpad — leave it untracked and do **not** add it to `.gitignore`.

## Architecture

### The Python/C++ split and the SoA wire

Python builds *descriptions*; C++ does *all* the numerics. `egg/topology/builder.py` (`TopologyBuilder`) declares corners/blocks/connections and produces a `MultiBlockGrid` (`egg/core/types.py`) initialized by TFI (`egg/init/tfi.py`). Geometry constraints (lines, arcs, Béziers, splines, composites, 3D surfaces — `egg/geometry/`) attach to DOFs via `grid.dof_constraints`. `egg/smoothing/solver.py::build_sweep_context` + `egg/smoothing/flat_context.py` flatten everything into typed SoA arrays (`egg/geometry/entity_soa.py` encodes each entity kind into a self-contained arena slice), which `src/bindings.cpp` uploads once; `src/entity_soa.hpp` reconstructs device-side entities from the same layout. Adding an entity type means touching both encoders in lockstep.

There are two C++ execution layouts:

- **Flat/unstructured**: `sweep.hpp` kernels over per-DOF patch tables — the general path, also the NumPy-parity reference target.
- **Structured (the fast path)**: `egg/smoothing/cpp_backend.py::build_block_structured_context` re-homes the grid onto a halo-padded per-block store (`src/structured*.hpp`); `CppStructuredSweepSession` keeps it resident on the device across calls (`run`, `run_fas`, `run_untangle`, `get_X`, `mg_levels`). Cross-block coupling is expressed purely as offset tables: ghost-fill entries plus owner→copy pairs for shared interface nodes, derived in Python by shared-global-DOF twin matching (non-conforming interfaces are rejected; singular fans get spare ghost slots allocated in `bindings.cpp`).

### The sweep engine (`src/sweep.hpp`)

Everything relaxes under a **frozen-halo block-Jacobi cadence**: double-buffered X, and a `before_sweep` hook (`fused_halo_broadcast`, `src/structured_halo.hpp`) that refreshes ghost shells and non-owner copies of shared nodes on the read buffer — every free DOF sees the same snapshot, so there is no intra-sweep dependency. The hook is a pure copy gather (linear in the buffer); code in `fas.hpp` relies on that linearity, so don't add non-copy behavior to it.

One sweep is three kernel families, split deliberately for GPU register pressure (measured; don't re-fuse or monomorphize the boundary kernel — both regressed):

1. `metric_kernel` writes per-node (grad, hess, e0); `interior_update_kernel` reads them back and does the Newton step + backtracking for free DOFs. Interior-eligible DOFs synthesize their patch from the block layout instead of reading stored tables.
2. Boundary (constrained) DOFs: a **cold** first sweep (`sweep_kernel`, robust world-space projection, seeds the warm-start cache) then **warm** sweeps (`sweep_boundary_paramls_kernel`, param-space line search — project once, step in parameter space).
3. `reduce_energy_mindet` — global energy + min-det reduction over the energy stencil.

Two invariants shared by every accept decision in the codebase (per-DOF line searches and the FAS safeguard alike): a trial is accepted iff `isfinite(e) && e <= e_before + tol::energy && objective.accept_mindet(mdet)` — energy is monotone and the shape barrier (`det > 0`) is never crossed. And **ω (SOR damping) scales the Newton step *before* the projected line search**, never as a post-hoc world-space blend — a post-blend pulls constrained DOFs off their geometry (this was a real regression, bisected and fixed; the composite curve-projection tests guard it).

### FAS nonlinear multigrid (`src/fas.hpp`, `src/mg_*.hpp`)

The TMOP shape phase can run FAS V-cycles instead of plain Jacobi (`tmop_smoother="fas"`; untangle is always Jacobi — its δ-continuation landscape is not elliptic-like). MG/OPT framing: the coarse correction is a search direction, applied through a backtracking line search on the *global* fine energy with the standard acceptance rule. Key pieces:

- `mg_level.hpp` builds the hierarchy: per-block, per-axis factor-2 coarsening when the interior node count is odd and ≥ 5; blocks that can't halve are *carried* unchanged (identity transfers); the ladder stops when nothing halves. Coarse free DOFs = interior candidates plus guarded interface face nodes.
- `mg_topology.hpp` derives each coarse level's halo/share tables from the fine tables (even-index + factor division; parity is guaranteed by odd counts; non-derivable entries drop to frozen rather than erroring), so interface DOFs relax on coarse levels too.
- `mg_transfer.hpp`: injection restriction (state), full-weighting restriction (gradients, for τ), multilinear prolongation, masked axpy.
- `fas.hpp`: recursive V-cycle with τ = ∇E_c(Rx) − R̂∇E_f(x) threaded through the `HasTau` kernel template; intermediate corrections are deliberately unsafeguarded (α=1) — only the fine-level line search gates what reaches the real grid. The safeguard evaluates the α ladder two trials per fused reduction (α=0 doubles as the pre-state energy → one host sync per typical cycle) and the per-cycle report is downloaded once after the loop. `nu_coarse` is a *cap*: the coarsest solve runs `min(nu_coarse, 2 × coarsest interior diameter)` sweeps.
- Performance profile worth knowing before optimizing: the cycle is GPU-launch-bound on small meshes (coarse-ladder launches), not host-sync-bound; FAS wins big (5–9× at equal energy, fp32 RTX-class) on ≥1M-node grids and is roughly break-even ~30k.

### Pipeline (`egg/pipeline.py`)

`run_pipeline` / `generate_steps` sequence the phases on one resident session: **untangle** (δ-continuation: relax `E` with a shifted barrier, shrink δ until `min det A > margin`) → **TMOP** shape smoothing (Jacobi sweeps or FAS cycles, `PipelineConfig.tmop_smoother`) → optional **pin** phase (freeze first boundary layers, re-smooth) → optional exact **respace** post-pass (`egg/smoothing/respace.py`). Examples under `examples/2D`/`examples/3D` are thin drivers over this; `examples/3D/sphere3d/sphere_in_cube_nurbs.py` is the standing performance testbed.

### Tests

- Python (`tests/`): parity gates against the sequential NumPy reference solver, plus behavior suites per subsystem. C++-dependent tests *skip* when `cpp_core` isn't built.
- C++ (`tests/cpp/`): a multi-TU `cpp_tests` runner for host-side logic, and **separate single-TU binaries for every test that launches kernels** — the AdaptiveCpp SSCP flow mis-resolves per-TU HCF object ids in multi-TU binaries ("Could not obtain hcf kernel info"). Keep new kernel tests in their own binary with their own `main`. Golden reference data is regenerated by `tests/cpp/gen_golden.py`.
- More ACPP runtime gotchas encoded in the tests: host-test binaries leak one long-lived queue on purpose (destroying the last queue tears down the runtime and deadlocks on the CUDA backend when no kernel ever ran); executor queues are in-order; transfer kernels are named functor types, not lambdas, for stable kernel names.
