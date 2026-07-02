---
name: rocprof
description: >-
  Profile and interpret GPU kernel execution for this project with rocprofv3
  (ROCm) on the sphere_in_cube smoother test case. Use when asked to profile
  the GPU, check register/VGPR pressure or spills, find the hot kernel, measure
  occupancy / ALU- vs memory-boundedness / LDS bank conflicts, see where wall
  time goes (kernel vs memcpy vs API), or get
  ACPP (AdaptiveCpp) JIT/launch debug info. Triggers: "profile the gpu",
  "rocprof", "rocprofv3", "VGPR", "register spill", "occupancy", "why is the
  sweep slow", "kernel trace", "ACPP_DEBUG".
---

# Interpreting rocprofv3 results

You drive `rocprofv3` yourself and interpret the output. Pick the rocprofv3 args
that match the question being asked — do not blindly copy one recipe. Run
`rocprofv3 --help` and `rocprofv3 --list-avail` whenever you need a flag or a
counter you are unsure about.

`scripts/parse_trace.py` is an optional convenience for the kernel-trace +
register-pressure view only. You are free to ignore it and read any rocprofv3
output (CSV/JSON/summary/stderr) directly — especially when using rocprofv3
features it can't parse (counter collection, sys-trace, summaries, etc.).

## Working directory (read first)

**Run every command in this skill from the project root**
(`/home/simran/rep/egg-dod-3d`), e.g. start with `cd /home/simran/rep/egg-dod-3d`.
All paths here — `examples/sphere3d/sphere_in_cube_nurbs.py`,
`scripts/parse_trace.py` — are **relative to the project root, NOT to this skill
directory**. The app path inside the test command is hard-coded relative to the
project root, so launching rocprofv3 from anywhere else will fail to find it.
(The project root is the git repo root: `git rev-parse --show-toplevel`.)

## Environment (verify, don't assume)

- GPU at time of writing: **gfx1030 (RDNA2)**, rocprofv3 **1.1.0**. Re-check with
  `rocminfo | grep -m1 gfx` and `rocprofv3 --version` if results look wrong —
  counter names and slot limits are architecture-specific.
- `scripts/parse_trace.py` (i.e. `<project-root>/scripts/parse_trace.py`) needs
  **pandas**, which is NOT in the project venv. Run it from the project root as
  `uv run --with pandas python scripts/parse_trace.py <csv>`.
- **Write all trace output to a scratch dir, never into the repo.** Use
  `-d <scratch>/rocprof-<label>`. Your session scratchpad directory is ideal.
  `.rocprofv3/` is NOT gitignored — do not dump traces there and do not commit
  any `*.csv`/`*.dat` trace artifacts.

## The fixed test case (DO NOT MODIFY)

Everything after `--` is the application command. It is **fixed verbatim** — never
change, add, or drop any of its flags. To look at different things, change only
the **rocprofv3** flags that come *before* `--`.

```
uv run python examples/sphere3d/sphere_in_cube_nurbs.py \
  --sweeps 1000 --chunk 1000 --structured \
  --smoother block-jacobi --omega 0.8 --device gpu
```

The ONLY permitted change to the post-`--` portion is prepending
`env ACPP_DEBUG_LEVEL=N` (see Recipe D) — this does not alter the app's own flags.
Do not lower `--sweeps`, swap `--smoother`, change `--device`, etc., even to make
a profiling pass faster.

## Switching precision (double ↔ fp32)

The project supports building with `egg::real = float` via `-DEGG_REAL_IS_FLOAT=ON`.
By default (`uv sync`) the extension builds in **double** precision. Profile
changes in register pressure / occupancy between the two builds.

### Build the fp32 extension (with tests)

```
uv sync --force-reinstall \
  -C cmake.define.EGG_REAL_IS_FLOAT=ON
```

Or, if you only want the extension and don't need the device test suite:

```
uv sync --force-reinstall \
  -C cmake.define.EGG_REAL_IS_FLOAT=ON \
  -C cmake.define.EGG_BUILD_TESTS=OFF
```

If you do build with tests, they will pass under fp32 but the golden-table parity
check tolerances are floored at `real_tol` (5e-4 fp32 floor).

### Profile the fp32 build

After switching to fp32, **always** add `--no-sync` to `uv run` so it does not
silently rebuild the double extension:

```
rocprofv3 --kernel-trace --output-format csv -d "$OUT" -- \
  uv run --no-sync python examples/sphere3d/sphere_in_cube_nurbs.py \
    --sweeps 1000 --chunk 1000 --structured \
    --smoother block-jacobi --omega 0.8 --device gpu
```

Without `--no-sync` uv notices the changed cmake defines and re-syncs to the
default double build, overwriting the fp32 extension.

### Revert to double

```
uv sync --force-reinstall
```

### fp32 caveats

- The example script's final assertion (`assert sph_dev < 1e-9 and pl_dev < 1e-9`
  at line 569 of `sphere_in_cube_nurbs.py`) will **fail** under fp32 because the
  sphere/plane deviation floor is ~1e-7. The sweeps complete and converge; this
  is only a Python-side assertion at the very end. rocprofv3 still captures every
  kernel dispatch correctly.
- For the closed-form math tests under clang codegen, use
  `-DEGG_TEST_ACPP_CLANG=ON` (see the CMakeLists.txt `EGG_TEST_ACPP_CLANG`
  option).

### Expected fp32 vs double profile deltas (gfx1030 RDNA2)

| metric | double | fp32 | change |
|---|---|---|---|
| Interior kernel VGPRs | ~248 | ~144 | -42% |
| Boundary kernel VGPRs | 256 (capped) | ~184 | -28% |
| Boundary scratch spill | ~600 B | ~160 B | -73% |
| Total GPU time | ~1,950 ms | ~860 ms | ~2.3× |
| Reduction accumulator | `double` | `float` | ✓ |
| Sphere/plane deviation | ~1e-15 | ~1e-7 | — |

Occupancy stays at 1 wave/SIMD in both builds because the interior VGPRs (144)
are still above the 128 threshold for a second wave. The speedup comes from
halved scratch spills and halved memory-bus width per operation.

## Recipe A — Kernel overview: timing + register pressure (default)

Start here for "profile the gpu" / "VGPR" / "spills" / "what's the hot kernel".

```
rocprofv3 --kernel-trace --output-format csv -d "$OUT" -- \
  uv run python examples/sphere3d/sphere_in_cube_nurbs.py \
    --sweeps 1000 --chunk 1000 --structured \
    --smoother block-jacobi --omega 0.8 --device gpu
```

Then summarize (the script ranks by total time and flags scratch spills):

```
uv run --with pandas python scripts/parse_trace.py \
  "$(find "$OUT" -name '*_kernel_trace.csv' | head -1)"
```

Do NOT pass `-T/--truncate-kernels` for this recipe: `parse_trace.py` extracts
the C++ template type from the demangled `basic_parallel_for<...>` /
`ndrange_parallel_for<...>` name, which truncation destroys.

If you need columns the script doesn't surface (grid/workgroup size, queue, etc.)
read the `*_kernel_trace.csv` directly. Useful columns: `Kernel_Name`,
`VGPR_Count`, `SGPR_Count`, `Scratch_Size`, `LDS_Block_Size`,
`Workgroup_Size_{X,Y,Z}`, `Grid_Size_{X,Y,Z}`, `Start_Timestamp`,
`End_Timestamp` (ns).

### Interpreting VGPR / occupancy on gfx1030 (RDNA2)

- 256 VGPRs per SIMD lane, allocated in blocks. Occupancy (waves/SIMD) is
  roughly `floor(256 / VGPR_Count)`, capped at the SIMD max. High `VGPR_Count`
  (≳128) means ≤2 waves/SIMD → poor latency hiding.
- **`Scratch_Size > 0` means register spills** — the compiler ran out of VGPRs
  and spilled to scratch (global) memory. This is usually the first thing worth
  fixing; it shows up as the dedicated section in `parse_trace.py` output.
- `LDS_Block_Size` is per-workgroup LDS; large LDS also caps occupancy.
- Cross-check the project's own `VGPR_reduction_plan.md` framing if present.

## Recipe B — Where does wall time go? (kernel vs memcpy vs API)

For "why is it slow" / "is it launch-bound" — note this case is 1000 sweeps, so
per-launch overhead matters.

```
rocprofv3 --sys-trace --stats --summary-per-domain --summary-units usec \
  --output-format csv -d "$OUT" -- <app>
```

`--summary-per-domain` prints a per-domain breakdown (KERNEL_DISPATCH,
MEMORY_COPY, HIP_API, HSA_API, …) to stderr. If kernel time is small vs API
time, you're launch/dispatch-bound, not compute-bound. Use `--runtime-trace -S`
for a lighter single summary without the HSA layer.

## Recipe C — Hardware counters (occupancy, ALU vs memory bound, LDS conflicts)

Use `--pmc`. **Rules:** a run fails if its counters can't be collected in one
pass, so keep each group small (≈4–6), split across multiple runs, and verify
every name exists with `rocprofv3 --list-avail` first (availability is
arch-specific). Focus on the hot kernel (from Recipe A) to cut noise and ease
single-pass collection:

```
rocprofv3 --pmc VALUInsts SALUInsts MemUnitBusy WriteUnitStalled \
  --kernel-include-regex 'sweep|colour|jacobi' \
  --output-format csv -d "$OUT" -- <app>
```

Suggested gfx1030 groups (run separately, confirm names with `--list-avail`):

- **Pipeline mix / bound:** `VALUInsts SALUInsts MemUnitBusy WriteUnitStalled`
- **Occupancy:** `Wavefronts MeanOccupancyPerCU OccupancyPercent GPUBusy`
- **LDS pressure:** `LDSBankConflict ALUStalledByLDS SQ_INSTS_LDS`
- **Memory traffic:** `FETCH_SIZE GL2C_HIT GL2C_MISS MemUnitBusy`

Read the resulting `*_counter_collection.csv` directly (one row per dispatch).
Interpretation: high `MemUnitBusy`/`WriteUnitStalled` → memory bound; high
`VALUInsts` with low `MemUnitBusy` → compute bound; nonzero `LDSBankConflict` /
high `ALUStalledByLDS` → fix LDS access pattern; low occupancy with no spills →
likely VGPR- or LDS-limited (confirm against Recipe A).

## Recipe D — ACPP / AdaptiveCpp JIT & launch debug

When kernel names, JIT compilation, work-group sizing, or backend selection are
in question, prepend `env ACPP_DEBUG_LEVEL=3` to the app (goes to stderr —
capture it). This is independent of rocprofv3 and combines with any recipe:

```
rocprofv3 --kernel-trace --output-format csv -d "$OUT" -- \
  env ACPP_DEBUG_LEVEL=3 uv run python examples/sphere3d/sphere_in_cube_nurbs.py \
    --sweeps 1000 --chunk 1000 --structured \
    --smoother block-jacobi --omega 0.8 --device gpu \
  2> "$OUT/acpp.log"
```

Level 3 is verbose (JIT cache, kernel build, launch config). 4 may add more; if a
higher level yields nothing new, stay at 3. Grep the log for the specific kernel
/ launch / JIT events relevant to the task rather than dumping it wholesale.

## Workflow

1. `cd /home/simran/rep/egg-dod-3d` (the project root) — required, all paths and
   the hard-coded app path are relative to it.
2. Set `OUT` to a scratch dir (your session scratchpad), `mkdir -p` it.
3. Decide on the precision target (double is the default; see `## Switching
   precision` above to toggle to fp32).
4. Choose the recipe matching the question; adapt rocprofv3 flags as needed
   (`--help`, `--list-avail` are your reference).
5. Run, then parse/read the CSV. Report the finding (hot kernel, spill, bound
   type, occupancy limiter) with the concrete numbers, not just raw tables.
6. If profiling perturbs results (counter collection serializes kernels), say so
   and lean on Recipe A timing for wall-clock claims.
7. Clean up large trace files from scratch when done; never commit them.
