# Development builds

There are three developer workflows:

| Workflow | AdaptiveCpp comes from | Use when |
|---|---|---|
| **(a) Devcontainer** | the image (built-in) | you want a known-good, reproducible toolchain |
| **(b) Local** | a system/user acpp you installed | you're comfortable managing dependencies on the host |
| **(c) Superbuild** | built from source as part of the egg build | you have no acpp and don't want/can't use the devcontainer |

---

## (a) Devcontainer workflow (recommended)

The devcontainer ([`.devcontainer/`](.devcontainer/)) builds/installs the pinned
AdaptiveCpp (see [`ACPP_VERSION`](ACPP_VERSION)) and deps.

Follow the instructions in [`.devcontainer/README.md`](.devcontainer/README.md).

Then use the [common editable loop](#the-editable-loop-all-workflows) below.

---

## (b) Local workflow

You provide AdaptiveCpp; `uv sync` builds against it. [Useful link for macOS users](https://github.com/AdaptiveCpp/AdaptiveCpp/blob/develop/doc/installing.md#using-a-2-stage-build-mac)

### Install AdaptiveCpp

AdaptiveCpp is a clang-based SYCL compiler, so it must be built against a real
LLVM (**not** Apple Clang on macOS).
We pin a specific commit ([`ACPP_VERSION`](ACPP_VERSION)), but for a local install you can
use any acpp/LLVM combination acpp supports (LLVM 15–21).

```bash
git clone https://github.com/AdaptiveCpp/AdaptiveCpp.git
cd AdaptiveCpp && mkdir build && cd build
cmake .. -GNinja \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build . && cmake --install
```

> **TIP:** acpp builds against whatever LLVM your `clang++`
> resolves to by default, and derives clang from that same LLVM. If you have
> several LLVMs, or the default isn't the one you want acpp tied to, point it
> explicitly:
>
> ```bash
> cmake .. -GNinja \
>   -DCMAKE_CXX_COMPILER=clang++ \
>   -DCMAKE_INSTALL_PREFIX=$HOME/.local \
>   -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm
> ```
>
> `LLVM_DIR` is the directory holding `LLVMConfig.cmake`. On distros it's under something like
> `/usr/lib/llvm-<ver>/lib/cmake/llvm`; with Homebrew LLVM,
> `$(brew --prefix llvm)/lib/cmake/llvm`.

### Point uv at it and sync

Make sure the location you installed

```bash
uv sync
```

Or pass it per-command (see [passing build args](#passing-build-args-to-uv-sync)):

```bash
uv sync -C cmake.define.AdaptiveCpp_DIR=$HOME/.local/lib/cmake/AdaptiveCpp
```

> If `uv sync` reports it can't find AdaptiveCpp, `CMAKE_PREFIX_PATH` isn't set
> (or doesn't point at the prefix that contains `lib/cmake/AdaptiveCpp`).

---

## (c) Editable AdaptiveCpp superbuild

If you have no acpp and don't want the devcontainer, the superbuild clones and
builds acpp (pinned to [`ACPP_VERSION`](ACPP_VERSION)) against an existing LLVM,
into the build tree (not system-wide). It is **heavy** (a clang-based SYCL
toolchain) and is a standalone CMake step — it is *not* run through `uv sync`
(passing `-DEGG_BUILD_ADAPTIVECPP=ON` to the wheel build fails by design).

The pattern is **two steps**: build acpp, then `uv sync` against it.

```bash
# 1. Bootstrap acpp from source (against your LLVM):
cmake -S . -B build_sb -DEGG_BUILD_ADAPTIVECPP=ON \
      -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm
cmake --build build_sb
#    -> installs acpp to build_sb/adaptivecpp/install

# 2. Editable install against that acpp:
uv sync -C cmake.define.AdaptiveCpp_DIR=$PWD/build_sb/adaptivecpp/install/lib/cmake/AdaptiveCpp
```

### Passing variables to the superbuild

The superbuild forwards backend toggles and locations to acpp's CMake:

| Variable | Purpose |
|---|---|
| `LLVM_DIR` | the LLVM acpp builds against (must match acpp); clang is derived from it |
| `EGG_ACPP_WITH_CUDA=ON` + `CUDA_TOOLKIT_ROOT_DIR` | CUDA (NVIDIA) backend |
| `EGG_ACPP_WITH_ROCM=ON` + `ROCM_PATH` | ROCm/HIP (AMD) backend |
| `EGG_ACPP_WITH_METAL=ON` + `METAL_INCLUDE_DIR` | Metal (Apple GPU) backend — ⚠️ highly experimental, **no `double` support**, so it **cannot run egg's double-precision kernels** (see below) |

(If acpp picks the wrong clang, also pass `CLANG_EXECUTABLE_PATH` /
`CLANG_INCLUDE_PATH`. These names are AdaptiveCpp's, per its `doc/install-*.md`.)

#### Example: Apple Silicon (CPU / OpenMP)

On macOS the supported path is the **CPU/OpenMP** backend. acpp needs a real
LLVM (Homebrew's — **not** Apple Clang) plus boost and libomp. Build acpp against
Homebrew LLVM and target `omp`:

```bash
brew install llvm boost ninja libomp     # acpp needs a real LLVM + boost; libomp for OpenMP

cmake -S . -B build_sb -DEGG_BUILD_ADAPTIVECPP=ON \
      -DLLVM_DIR="$(brew --prefix llvm)/lib/cmake/llvm"
cmake --build build_sb

uv sync -C cmake.define.AdaptiveCpp_DIR=$PWD/build_sb/adaptivecpp/install/lib/cmake/AdaptiveCpp \
        -C cmake.define.ACPP_TARGETS=omp
```

> **Why not the GPU (Metal) on macOS?** AdaptiveCpp's Metal backend exists
> (`-DEGG_ACPP_WITH_METAL=ON` + `-DMETAL_INCLUDE_DIR=…`), but it is **highly
> experimental** and — critically for this project — **does not support
> `double`**: Apple Silicon GPUs have no native fp64 and Metal Shading Language
> has no `double` type (soft-double emulation is only planned, and would be
> compatibility-only, not fast). This core is double-precision, so the Metal GPU
> path **cannot run egg kernels**. Stick to `ACPP_TARGETS=omp` on macOS.
>
> **Generic JIT caveat.** For a *fully working generic JIT* on Apple Silicon,
> AdaptiveCpp recommends a **2-stage LLVM build** (it builds LLVM with acpp
> bootstrapped in). The superbuild builds acpp against an *existing* LLVM (single
> stage) and does **not** do the 2-stage build — which is another reason to use
> the `omp` target here. If you do need generic JIT, follow AdaptiveCpp's
> `doc/installing.md` "2-stage build (Mac)" manually, then use workflow (b)
> against that install.

---

## The editable loop (all workflows)

After any of (a)/(b)/(c), you have an editable install:

- **Edit Python (`.py`)** → live, no rebuild (`editable.mode = "redirect"`).
- **Edit C++ (`src/*.cpp`, `*.hpp`)** → recompiles on next import
  (`editable.rebuild = true`). Trigger it without launching your app:

  ```bash
  uv run python -c "import egg._cpp.cpp_core"
  ```

- **Accidentally deleted `build/`?** The rebuild hook only runs `cmake --build`
  (no reconfigure), so the next import fails with `missing CMakeCache.txt` and a
  plain `uv sync` won't fix it (uv sees nothing changed). Force a reinstall to
  reconfigure: `uv sync --reinstall-package egg` (with
  `CMAKE_PREFIX_PATH`/`AdaptiveCpp_DIR` set so acpp is found).

- **Run things:**

  ```bash
  uv run examples/circles/phase5_good-topo_demo.py --device cpu
  uv run pytest tests/
  ```

### Selecting a device at runtime

A `generic` build exposes the CPU (OpenMP host) **and** the GPU. You select the
device through the application itself — the demos take `--device {auto,cpu,gpu}`,
and the C++ tests run on every visible device. `OMP_NUM_THREADS=N` controls CPU
parallelism.

> **`ACPP_VISIBILITY_MASK` is not normally needed.** It restricts which acpp
> backends load (e.g. `omp` for CPU-only, `hip` for AMD GPU). Reach for it only
> if acpp mis-selects a device or you want to force one backend for debugging
> e.g. `ACPP_VISIBILITY_MASK=omp uv run …`.

---

## Testing

Two independent suites — neither needs the in-tree (`EGG_INPLACE`) build.

### Python tests (pytest)

These import `egg._cpp.cpp_core` and exercise the C++ sweep/untangle against
the NumPy reference. They resolve the extension from the **editable install**, so
just `uv sync` first (any of workflows (a)/(b)/(c)), then:

```bash
uv run pytest tests/
```

The C++-dependent tests `skip` (not fail) if `cpp_core` isn't importable, so the
pure-Python suite runs even without a built extension. CPU and GPU paths are
covered by separate tests; the GPU ones skip when no GPU is visible.

### C++ tests (ctest)

The compute core has Boost.UT tests (metric/solve/geometry/patch/sweep, run on
every visible SYCL device). There are two ways to build and run them.

**(1) editable install.** The editable build

```bash
uv sync                                         # builds cpp_core + the C++ tests
uv run pytest tests/                            # Python suite
ctest --test-dir build/cp313-cp313-linux_x86_64 # C++ suite (the {wheel_tag} dir)
```

**(2) Standalone, no Python install required.**

```bash
cmake -S . -B build_cpp -DACPP_TARGETS=generic     # tests ON by default
cmake --build build_cpp
ctest --test-dir build_cpp --output-on-failure
```

> `ACPP_VISIBILITY_MASK=omp ctest --test-dir <build-dir>`.
---

## Install the library in-tree (not recommended)

```bash
cmake -S . -B build -DACPP_TARGETS=generic -DEGG_INPLACE=ON
cmake --build build --target cpp_core    # in-tree cpp_core*.so
```

> **This `.so` is static — it does NOT rebuild on import.** That recompile-on-
> import behavior belongs to the editable install only; here you re-run
> `cmake --build build --target cpp_core` yourself after every C++ change.
>
> **Don't mix `EGG_INPLACE` with the editable install in the same checkout.**
> If an editable install is active (`uv run` / the project venv), its redirect
> hook resolves `egg._cpp.cpp_core` to *its* copy and **shadows** the in-tree
> `egg/_cpp/*.so`, so your manual build is silently ignored. Pick one workflow:
> editable (`uv sync`, auto-rebuild) **or** `EGG_INPLACE` (manual, explicit).

---

## Passing build args to `uv sync`

`uv sync` accepts `-C/--config-setting` like `uv pip install`. Pass CMake cache
entries as `cmake.define.<VAR>=<VALUE>`:

```bash
uv sync \
  -C cmake.define.AdaptiveCpp_DIR=$HOME/.local/lib/cmake/AdaptiveCpp \
  -C cmake.define.ACPP_TARGETS=generic     # generic = CPU+GPU; omp = CPU only
```

To make a setting persistent (no flag each time), add it under
`[tool.scikit-build.cmake.define]` in `pyproject.toml` (where `ACPP_TARGETS`
already defaults to `generic`), or export the equivalent environment variable
(e.g. `CMAKE_PREFIX_PATH`).

---

## Clean state / resetting

Everything below is regenerated — none of it is source:

| Reset goal | Remove |
|---|---|
| Full from-zero (fresh-clone-like) | `.venv/ build/ dist/ egg/_cpp/*.so` + all `__pycache__/` |
| Just the C++ rebuild dir | `build/` **and** `uv sync --reinstall-package egg` |
| Cold dependency fetch | also `.cpm_cache/` (re-downloads mdspan/pybind11/etc.) |

```bash
# full reset
rm -rf .venv build dist egg/_cpp/*.so
find . -name __pycache__ -type d -prune -exec rm -rf {} +
```

> **Do not delete `build/` and then just `uv sync`.** `editable.rebuild` only
> re-runs `cmake --build` on the persistent build dir; it does **not**
> reconfigure. uv sees the project as unchanged and does nothing, so the next
> import fails with `missing CMakeCache.txt`. Recover by forcing a reinstall:
>
> ```bash
> uv sync --reinstall-package egg
> ```
>
> (and make sure `CMAKE_PREFIX_PATH` is set, or the reconfigure can't find acpp.)

`.cpm_cache/` (CPM's offline dependency cache) and `.cache/clangd/` (editor
index) are safe to keep; delete `.cpm_cache/` only to test a cold dependency
fetch. `uv.lock` is **not** disposable — it pins your dependency resolution.

See [DEPLOY.md](DEPLOY.md) for building distributable wheels.
