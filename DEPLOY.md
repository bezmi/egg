# Deploying egg

The C++ core links the **AdaptiveCpp runtime** (`libacpp-rt`, `libacpp-common`)
and, for the generic SSCP JIT, `libLLVM` + bitcode. A built wheel therefore has
to answer one question: *where does the AdaptiveCpp runtime come from on the
target machine?* Two distribution models:

| Model | Runtime source | Wheel | Portability |
|---|---|---|---|
| **System-library** (default) | acpp installed on the target | small | needs matching acpp present |
| **Standalone** | bundled inside the wheel | ~160 MB+ | self-contained, but platform/arch-pinned |

Both are produced from the same `pyproject.toml` build (scikit-build-core); the
difference is one CMake define. The extension is installed into the package at
`egg/_cpp/cpp_core*.so` with an rpath that self-locates its runtime — no
`LD_LIBRARY_PATH` needed in either model. The rpath token is platform-specific
(`$ORIGIN` on Linux, `@loader_path` on macOS) and resolves, in order:
`./` → `./acpp_runtime/` (bundled) → the AdaptiveCpp lib dir the wheel was built
against.

> Build-time prerequisite (both models): an AdaptiveCpp install on the build
> machine, located via `CMAKE_PREFIX_PATH` or `AdaptiveCpp_DIR`. See
> [DEVELOPING.md](DEVELOPING.md) for installing acpp.

---

## System-library distribution (default)

The lean wheel. It assumes the target has a *compatible* AdaptiveCpp install
(same major runtime — acpp has **no stable ABI**, so ideally the same acpp
version/build).

```bash
uv build --wheel \
  -C cmake.define.AdaptiveCpp_DIR=/opt/adaptivecpp/lib/cmake/AdaptiveCpp \
  -C cmake.define.ACPP_TARGETS=generic        # generic = CPU+GPU JIT; omp = CPU only
# -> dist/egg-<ver>-<pytag>-<platform>.whl
```

Install on the target (which must have acpp):

```bash
uv pip install egg-<ver>-<pytag>-<platform>.whl
# acpp's libs are found via the baked rpath if acpp is at the same prefix as on
# the build host; otherwise put acpp's lib dir on the loader path:
#   export LD_LIBRARY_PATH=/opt/adaptivecpp/lib:$LD_LIBRARY_PATH
```

**Use when:** you control the target environment (a cluster module, a base
image, the devcontainer) and can guarantee a matching acpp is installed. Smallest
artifact, and acpp can be upgraded independently of the wheel (within ABI).

---

## Standalone distribution (bundled runtime)

Vendors the AdaptiveCpp runtime *into* the wheel using acpp's own deployment
mechanism (`acpp --acpp-deploy`), so the wheel installs with **no** system acpp.

```bash
uv build --wheel \
  -C cmake.define.AdaptiveCpp_DIR=/opt/adaptivecpp/lib/cmake/AdaptiveCpp \
  -C cmake.define.ACPP_TARGETS=generic \
  -C cmake.define.EGG_BUNDLE_RUNTIME=ON \
  -C cmake.define.EGG_DEPLOY_COMPONENTS=core    # see components below
```

This populates `egg/_cpp/acpp_runtime/` (libacpp-rt/common, `libLLVM`, the
SSCP bitcode + JIT tools, plus the requested backend), resolved at runtime via
the `acpp_runtime` rpath entry. Install needs nothing extra:

```bash
uv pip install egg-<ver>-<pytag>-<platform>.whl   # done — runs standalone
```

### Deployment components

`EGG_DEPLOY_COMPONENTS` is passed to `acpp --acpp-deploy=<component>:<dir>`
(`;`-separated for several). Per AdaptiveCpp's `doc/deployment.md`:

| Component | Contents |
|---|---|
| `core` | **mandatory** — core infra, CPU backend, generic-JIT LLVM + bitcode |
| `cuda` | CUDA backend + deps |
| `hip`  | HIP/ROCm backend + deps |
| `ocl`  | OpenCL ICD loader (end user installs the actual OpenCL driver) |
| `all`  | all of the above |

```bash
# CPU-only standalone (smallest):
-C cmake.define.EGG_DEPLOY_COMPONENTS=core
# CPU + AMD GPU:
-C cmake.define.EGG_DEPLOY_COMPONENTS="core;hip"
```

Because the generic SSCP compiler decouples the app from hardware, **`core` is
enough to ship** — an end user can add a GPU component to the same install later
if their hardware changes. Only deploy a backend that acpp was *built* with.

### Caveats (read before shipping a standalone wheel)

- **Not a manylinux/PyPI wheel.** It embeds backend libraries built against
  *this* host's LLVM/ROCm/CUDA; it is platform- and arch-specific. Treat it as an
  internal/registry artifact, not a PyPI upload.
- **ABI pinned.** The acpp runtime has no stable ABI, so the wheel is locked to
  the acpp version it was built with. Rebuild the wheel to change acpp.
- **HIP has no forward compatibility** — a `hip` component only runs on GPUs
  supported by the ROCm it was built against. (CUDA/OpenCL forward-compat is
  fine; you can refresh just the component — see "Updating" below.)
- **Size.** Dominated by `libLLVM.so` (~150 MB); `all` with HIP can exceed
  300 MB. Shrink with [`upx`](https://upx.github.io/) (compresses `libLLVM`
  transparently, often ~40%).
- **macOS standalone is unverified.** acpp's deploy tool is Linux-centric
  (`LD_LIBRARY_PATH`-based). On macOS, prefer the system-library model.
- **Redistribution licensing.** `cuda`/vector-math components carry third-party
  license terms (CUDA EULA permits redistribution of the bundled CUDA libs; SVML
  / Arm PL / SLEEF have their own licenses). See `doc/deployment.md` "CUDA
  redistribution" / "Vector math library redistribution".

### Verifying a standalone wheel is self-contained

```bash
# 1. Runtime libs live inside the wheel:
python3 -c "import zipfile,glob; \
  n=zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist(); \
  print([x for x in n if 'acpp_runtime' in x and x.endswith('.so')])"

# 2. The extension resolves them (no system acpp on the path):
unzip -q dist/*.whl -d /tmp/whl && \
  env -u LD_LIBRARY_PATH ldd /tmp/whl/egg/_cpp/cpp_core*.so | grep -E "acpp|LLVM"
#   -> should point into .../egg/_cpp/acpp_runtime/, not /usr or ~/.local

# 3. End-to-end: install into a clean venv (no acpp), run a sweep.
#    (See the project tests; cpp_sweep JIT-compiles a SYCL kernel on import.)
```

### Updating a deployed package (without recompiling the app)

Because generic-SSCP binaries are hardware-decoupled, you can often refresh just
the runtime: rebuild the *same* acpp version against newer backend deps (e.g. a
newer ROCm), re-run the deploy for that component, and ship the updated
`acpp_runtime/`. You **cannot** change the acpp *version* without rebuilding the
wheel (ABI). After updating, tell users to clear the acpp JIT cache.

---

## Choosing a model

- **Internal cluster / CI / devcontainer-like targets** → system-library. Small,
  and acpp is already there (or is a managed module).
- **Hand a colleague a wheel that "just works" on the same OS/arch** → standalone
  `core` (+ a GPU component if they have that vendor's GPU).
- **PyPI / arbitrary machines** → neither is a clean fit today (SYCL/GPU + LLVM
  JIT don't fit manylinux). Ship source + a documented acpp prereq, or a
  devcontainer.

See [DEVELOPING.md](DEVELOPING.md) for the build/dev side, including
[building the documentation site](DEVELOPING.md#documentation).

---

## Release authenticity

The release tooling (`scripts/make_wheel_release.py`,
`scripts/make_source_release.py`) and the PolyForm Countdown notice are public,
so *anyone* can produce a look-alike artifact with a `LICENSE-COUNTDOWN.md`
attached. What makes a release **official** is not the presence of that file —
it is a signature under the maintainer's identity. (A Countdown notice attached
by a non-licensor is legally void anyway: only the copyright holders can grant
the conversion.)

Official artifacts are signed with **keyless Sigstore** (cosign) — a short-lived
certificate bound to the maintainer's OAuth identity, no long-lived key:

```bash
scripts/sign_release.py dist/wheel/egg-*.whl dist/egg-*-src.tar.gz
```

This writes `<artifact>.sig` + `<artifact>.pem` per artifact, published
alongside it. The `.sha256` files the build scripts emit are for **integrity**
(detect corruption); the signature is what proves **origin**.

### Verifying an official release

```bash
cosign verify-blob egg-<ver>-...whl \
  --signature egg-<ver>-...whl.sig --certificate egg-<ver>-...whl.pem \
  --certificate-identity s.imran@tuta.io \
  --certificate-oidc-issuer https://github.com/login/oauth
```

If it isn't signed by `s.imran@tuta.io` (verified against Sigstore's public
transparency log), it is **not** an official release, regardless of which
license files it carries.
