# FAS (nonlinear geometric multigrid) for the structured block-Jacobi smoother

> **DISCLAIMER — UNTRACKED WORKING DOCUMENT.**
> This file is a local planning scratchpad. It must remain **untracked**: do not
> `git add` it, do not commit it, and **do not add it to `.gitignore`** (it stays
> invisible to the repo by convention, not by ignore rules). Delete it when the
> work lands.

## 1. Goal

Accelerate convergence of the structured block-Jacobi smoother
(`run_block_jacobi`, `src/sweep.hpp:1439`) with a Full Approximation Scheme
(FAS) geometric multigrid, in the **MG/OPT safeguarded-minimization framing**:
the coarse-grid correction is treated as a search direction and accepted only
through the existing energy/min-det line-search machinery, so monotone energy
decrease is preserved and the `detA > 0` barrier of `ShapeObjectiveT`
(`src/metric.hpp:763`) is never crossed.

Scope decisions (settled in discussion):

- **Shape phase only.** `UntangleObjectiveT` (`src/metric.hpp:811`) keeps the
  plain Jacobi path; its δ-continuation landscape is not elliptic-like and the
  smooth-error argument does not apply.
- **Coarse levels relax interior free DOFs only.** Coarse images of
  boundary-constrained, block-interface, and singular nodes are frozen at their
  restricted positions (Dirichlet). Constrained and interface DOFs still relax
  every cycle in the fine-level pre/post-smooths. Coarse interface relaxation
  is milestone M4 (optional, only if M3 shows interface modes dominate).
- **Dual-precision testing.** Correctness tests are written against the
  **fp64 build with tight (double-tuned) tolerances** — that is the reference.
  The same tests must also pass in the **fp32 build**
  (`-DEGG_REAL_IS_FP32=ON`, `src/core.hpp:23`, `CMakeLists.txt:142-150`),
  where `real_tol` floors the thresholds (`tests/cpp/real_tol.hpp`,
  `tests/real_tol.py`) — exactly the existing cross-precision pattern. fp32 is
  the mode for quick GPU testing and profiling; fp64 is what gates
  correctness. **This machine has an NVIDIA GPU** — `rocprofv3` is not usable
  here; GPU profiling on this box goes through Nsight (`nsys`/`ncu`), or the
  rocprof checks wait for an AMD machine.

## 2. Why the codebase makes this cheap

- **Coarse levels need no metric tables.** Interior free DOFs evaluate their
  patch synthetically from the block layout with an identity target:
  `patch_eval_synth` (`src/structured_patch.hpp:176`) and
  `synth_trial_energy_mindet` (`src/structured_patch.hpp:221`) index nodes via
  `dev_node_index` (`src/structured_patch.hpp:160`) using only
  `block_off`/`nstride`. A coarse level is therefore *just* a coarsened
  `BlockLayout<D>` (`src/structured.hpp:46`) plus per-DOF index arrays — no
  `W_inv`, no `gc/gn/s` tables, no entity SoA.
- **The shape metric is scale-invariant** (`mu_shape2d`,
  `src/metric.hpp:38-43`: `A → cA` leaves `s/(2·detA)` unchanged in 2D; the
  3D condition-number metric likewise), so factor-2 coarsening needs no target
  rescaling — the identity `W_inv` is correct at every level.
- **The smoother is already the FAS smoother.** The per-DOF Newton +
  backtracking pair `metric_kernel` (`src/sweep.hpp:1039`) /
  `interior_update_kernel` (`src/sweep.hpp:1082`) runs unchanged on coarse
  levels over a coarse `GroupViewT<D>` whose every DOF is synth-eligible
  (`GroupViewT::interior`, `src/sweep.hpp:151-159`).
- **The fine gradient FAS needs for τ is already computed** per node into
  `grad_buf` by `metric_kernel` — no new fine-level work.
- **Global energy evaluation exists** for the safeguarded prolongation line
  search: `reduce_energy_mindet` (`src/sweep.hpp:1396`).

## 3. Algorithm (two-level first; recursion is a loop over the same pieces)

One V-cycle on level ℓ (fine = 0), state `x_ℓ` in the packed halo-padded
buffer of that level's `BlockLayout`:

1. **Pre-smooth**: ν₁ block-Jacobi sweeps (existing sweep body from
   `run_block_jacobi`, no per-sweep reduction).
2. **Restrict state**: injection `x_{ℓ+1}(I) = x_ℓ(2I)` per block (coarse node
   `I` ↔ fine node `2I` in logical space).
3. **Build τ (first-order coherence)**:
   `τ = ∇E_{ℓ+1}(R x_ℓ) − R̂ ∇E_ℓ(x_ℓ)`, where `∇E_ℓ` is the per-node gradient
   from `metric_kernel` on level ℓ, `R̂` is full weighting (3^D stencil,
   weights `2^{−D−|s|₁}`), and `∇E_{ℓ+1}` is one coarse `metric_kernel` pass at
   the restricted point. τ is stored per coarse free node (D reals).
4. **Coarse solve**: minimize `E_{ℓ+1}(x) − Σ_d τ_d · x_d` — recursively
   (V-cycle) or, on the coarsest level, with ~ν_c cheap Jacobi sweeps. The
   linear τ term shifts the per-DOF gradient and the line-search energies (see
   §5.2); frozen coarse nodes never move.
5. **Prolong the correction**: `c = P (x_{ℓ+1} − R x_ℓ)` by multilinear
   interpolation (copy at even fine nodes, midpoint averages at odd), then
   **zero `c` at every non-Free fine node** (constrained/boundary/singular
   nodes must not be pushed off their geometry).
6. **Safeguarded application**: backtracking on the *global* fine energy —
   trial `x_ℓ + α c` for `α ∈ {1, ½, …}` (≤ 10 halvings), accept iff
   `isfinite(E)`, `E ≤ E_pre + tol::energy`, and `objective.accept_mindet`
   holds (mirrors the acceptance in `interior_update_kernel`,
   `src/sweep.hpp:1154-1155`). All-fail ⇒ drop the correction (FAS still
   converges via the smooths).
7. **Post-smooth**: ν₂ sweeps.

Coarsenability per block per axis: interior node count `n_k` (i.e. `n_k − 1`
cells) coarsens iff `n_k` is odd and `n_k ≥ 5`; even axes are **semi-coarsened**
(left uncoarsened); a block with no coarsenable axis stops the hierarchy. Odd
`n_k` also guarantees interface conformity survives coarsening even under
reversed axis orientation (`i → n_k − 1 − i` preserves parity iff `n_k` odd) —
assert this at hierarchy-build time and fall back to semi-coarsening on any
axis where the matched face's parity disagrees.

## 4. New files

All header-only like the rest of `src/`, PolyForm license header first (see any
`src/*.hpp`; commit `9f95cc2`), clean C++23: `stdex::mdspan` views for all node
addressing (`InteriorView`/`HaloView`, `src/structured.hpp:161-166`, and the
`interior_view`/`halo_view` factories at `src/structured.hpp:194-210`),
`UsmBuffer<T>` (`src/device.hpp:31`) for every device allocation, **no raw
owning pointers, no `new`/`delete`**. Kernels capture trivially-copyable views
by value (pattern documented in `src/structured_smoother.hpp:14-17`).

### 4.1 `src/mg_transfer.hpp` — inter-grid transfer kernels

- `template <int D> sycl::event restrict_inject(sycl::queue&, InteriorView<D> fine, InteriorView<D> coarse, std::array<int, D> factor)`
  — one work-item per coarse node; `factor[k] ∈ {1, 2}` encodes
  semi-coarsening. Views built host-side from the two `BlockLayout`s via
  `interior_view` and passed by value.
- `template <int D> sycl::event restrict_full_weight(...)` — same signature
  over per-node gradient buffers (rank-(D+1) mdspans, coordinate axis
  innermost, matching the `X[D*i + k]` convention in
  `src/structured.hpp:21-24`). Clamped at block faces (frozen coarse DOFs
  ignore their τ anyway).
- `template <int D> sycl::event prolong_correction(...)` — multilinear; writes
  the fine *correction* buffer, does not touch `X`.
- `template <int D> sycl::event axpy_masked(sycl::queue&, real* x, const real* c, real alpha, const std::uint8_t* free_mask, std::size_t n_nodes)`
  — `x += α·c` at free nodes only; used by the safeguarded line search
  (§3.6). The mask is per fine node, built once (§4.2).

All loops over blocks launch one kernel per block per transfer (block counts
are small; fusing across blocks is a later optimization — note it, don't do
it).

### 4.2 `src/mg_level.hpp` — one level of the hierarchy

```
template <int D> struct MgLevel {
    BlockLayout<D>            layout;      // coarse packing geometry (host)
    std::array<...>           factors;     // per-block per-axis coarsen factor
    UsmBuffer<real>           x;           // packed coarse positions
    UsmBuffer<real>           tau;         // [num_nodes * D] linear shift
    UsmBuffer<real>           grad, hess, e0;   // smoother scratch (as in
                                                //   run_block_jacobi, sweep.hpp:1459-1462)
    UsmBuffer<real>           x_swap;      // double-buffer partner
    UsmBuffer<int>            dof_idx, interior_block, interior_logical;
    UsmBuffer<int>            block_off, nstride;  // node-unit layout (device)
    UsmBuffer<std::uint8_t>   free_mask;   // per coarse node
    std::size_t               n_free;
};
```

plus a host builder
`build_mg_hierarchy(const BlockLayout<D>& fine_layout, /* fine free info */)`
returning `std::vector<MgLevel<D>>`. Everything owned by `UsmBuffer`; the
`GroupViewT<D>` handed to the smoother kernels is assembled on demand from
these buffers (non-owning views, same split as
`DeviceGroup::view`, `src/sweep.hpp:458-491`): only `dof_idx`,
`interior_block`, `interior_logical`, `block_off`, `nstride` populated —
`partitions`/metric-table members stay null because **every coarse DOF is
synth-eligible by construction**, so `metric_kernel` and
`interior_update_kernel` never touch them.

Coarse free-DOF selection (host, at build time): coarse node `(b, I)` is free
iff its fine image `(b, 2I ⊙ factor)` is a Free-tag DOF **and** is
interior-eligible — both available from the host context:
`SweepGroupHostT::interior_block/interior_logical` (`src/sweep.hpp:80-82`) and
the Free partition's `dof_local` list (`SoAHostRecord`,
`src/sweep.hpp:44-56`). Everything else (physical boundary, block faces,
singular nodes from the `sing_*` tables) is frozen. No coarse
`BlockTopologyDevice` is needed in M1–M3: with faces frozen, no coarse DOF's
patch reads a ghost slot, so the coarse "halo hook" is a no-op (the empty-topology
behaviour already documented at `src/structured_sweep.hpp:41-42`).

The fine-level `free_mask` (for §3.5's zeroing and §4.1's `axpy_masked`) is
built once from the merged view's Free prefix: `dof_idx[0 .. n_free)` of
`merged_group_view()` (`src/sweep.hpp:399-406`).

### 4.3 `src/fas.hpp` — the V-cycle driver

```
struct FasOptions {
    int  n_cycles      = 10;
    int  nu_pre        = 2;
    int  nu_post       = 2;
    int  nu_coarsest   = 32;
    int  max_levels    = 2;      // M1–M3 ship with 2; recursion in M5
    real omega         = 1.0_r;
};

template <int D, class Obj>
std::pair<std::vector<real>, std::vector<real>>
run_fas(sycl::queue&, SweepDeviceContextT<D>&, std::vector<MgLevel<D>>&,
        Obj objective, const FasOptions&, /* fine halo hook */);
```

Returns per-cycle `(energy, mindet)` — same contract as `run_jacobi`
(`src/structured_sweep.hpp:62-79`). Rejects `UntangleObjectiveT` at the
`std::visit` dispatch layer (throw from the binding, §5.4, before reaching
device code).

Prerequisite refactor: extract the single-sweep body of `run_block_jacobi`
(`src/sweep.hpp:1473-1527`: halo hook → `metric_kernel` →
`interior_update_kernel` → boundary kernel → swap) into a reusable
`jacobi_sweep_once` so pre/post-smooths and the coarse smoother share it with
`run_block_jacobi` instead of duplicating the launch sequence. Coarse levels
call it with a null halo hook, an all-interior group view, and no boundary
launch.

## 5. Modifications to existing files

### 5.1 `src/sweep.hpp` — thread τ through the interior pair

`metric_kernel` and `interior_update_kernel` gain the linear shift as a
**compile-time branch** so the fine-level instantiation is bit-identical and
its VGPR budget untouched (the file's own comments record how carefully that
split was tuned, `src/sweep.hpp:1478-1481`):

```
template <int D, class Obj, bool HasTau = false>
inline void interior_update_kernel(..., const real* tau = nullptr, ...)
```

With `HasTau`:
- `metric_kernel`: `grad[k] -= tau[dof*D + k]` before storing (τ enters the
  Newton system exactly once).
- `interior_update_kernel`: line-search energies shift per DOF —
  `e0 -= dot(τ_d, pos)`, `e_new -= dot(τ_d, trial)` in both the synth
  (`synth_trial_energy_mindet` call site, `src/sweep.hpp:1131-1133`) and the
  stored-array branch. `accept_mindet` is untouched (the barrier is geometric,
  not shifted).

### 5.2 `src/structured_sweep.hpp` — executor surface

`StructuredExecutorT<D>` gains:

- a way to receive the fine `BlockLayout<D>` and the host free/interior info
  needed by `build_mg_hierarchy` (extend the ctor; the binding already has
  `rm.layout` in scope at construction, `bindings.cpp:757-765`);
- `run_fas(const FasOptions&, const ObjectiveKindT<D>&)`, mirroring
  `run_jacobi`'s `std::visit` dispatch (`src/structured_sweep.hpp:62-79`) but
  rejecting the untangle variant;
- lazy hierarchy construction on first `run_fas` call (zero cost for plain
  Jacobi users).

### 5.3 `src/structured_patch.hpp` — none expected

`patch_eval_synth`/`synth_trial_energy_mindet` are level-agnostic already
(they only see `block_off`/`nstride`). If τ handling turns out to be cleaner
inside `accumulate_sample`, resist it — keep τ in the kernels (§5.1) so the
patch math stays single-sourced.

### 5.4 `src/bindings.cpp` — Python surface

- `CppStructuredSweepSession` (`bindings.cpp:773-837`) gains
  `.run_fas(n_cycles, *, nu_pre=2, nu_post=2, nu_coarsest=32, omega=1.0, phase="barrier")`
  returning `(energies, mindets)` like `.run` (`bindings.cpp:798-810`).
  `phase="untangle"` raises `ValueError`.
- Expose the hierarchy shape for tests/introspection:
  `.mg_levels()` → list of per-level per-block interior shapes.

### 5.5 `egg/` Python — thin passthrough only

One forwarding method on the session wrapper in
`egg/smoothing/cpp_backend.py` (near `build_block_structured_context`,
line 166). No changes to the parametric layer (standalone constraint) and no
new Python-side logic.

### 5.6 `CMakeLists.txt` — tests

- Add `tests/cpp/test_multigrid.cpp` to `_cpp_tests_srcs`
  (`CMakeLists.txt:277-286`).
- Add `multigrid` to the device-test `foreach` (`CMakeLists.txt:301-308`) with
  `tests/cpp/test_multigrid_device.cpp`.

## 6. Testing — **fp64 correctness reference, fp32 for GPU quick-runs**

Tests are written once with **tight double-tuned tolerances** and gate
correctness on the **fp64 build**. Every tolerance goes through `real_tol`
(`tests/cpp/real_tol.hpp` / `tests/real_tol.py`), which is the identity in
fp64 and floors at ~`5e-4` relative in fp32 — so the same suite also runs in
the **fp32 build**, the mode used for quick GPU testing and profiling.
Remember the known nondeterminism: console energies fluctuate in the 4th–5th
digit between runs — **never assert exact energy values**; assert monotonicity
(with `tol::energy` slack, `src/core.hpp:35-36`) and convergence *ratios* with
generous margins.

Build + run (both precisions):

```sh
# fp64 — the correctness gate (tight tolerances active)
cmake -S . -B build_fp64 -GNinja -DEGG_BUILD_TESTS=ON \
      -DAdaptiveCpp_DIR=... -DACPP_TARGETS=...   # per DEVELOPING.md:38-53
cmake --build build_fp64
ctest --test-dir build_fp64 -R 'multigrid|cpp_tests'

# fp32 — GPU quick-testing/profiling build (real_tol floors kick in)
cmake -S . -B build_fp32 -GNinja -DEGG_REAL_IS_FP32=ON -DEGG_BUILD_TESTS=ON \
      -DAdaptiveCpp_DIR=... -DACPP_TARGETS=...
cmake --build build_fp32
ctest --test-dir build_fp32 -R 'multigrid|cpp_tests'
```

Python-level checks read `cpp_core.REAL_IS_FLOAT` (`bindings.cpp:858`) —
nothing to configure.

### 6.1 `tests/cpp/test_multigrid.cpp` (host + math, Boost.UT)

- **Hierarchy build**: odd/even/mixed interior shapes → expected coarse
  shapes, semi-coarsening flags, stop conditions; frozen-mask correctness on a
  hand-built 2-block layout (faces + physical boundary frozen, interior free).
- **Transfer exactness** (these are exact even in fp32):
  - injection: coarse values bitwise-equal to fine even-index values;
  - prolongation reproduces multilinear fields exactly (copy/midpoint of
    floats is exact);
  - `R ∘ P = identity` on coarse data.
- **τ coherence**: on a small perturbed single-block grid, the shifted coarse
  gradient at `R x_f` equals `R̂ ∇E_f(x_f)` per free coarse DOF to
  `real_tol(1e-10)` — tight (1e-10) in the fp64 gate, floored in fp32. This is
  the one identity that validates the whole FAS plumbing.

### 6.2 `tests/cpp/test_multigrid_device.cpp` (every visible device)

Follows the `test_*_device.cpp` pattern (`CMakeLists.txt:294-308`): run the
transfer kernels and one full V-cycle on the GPU and the OpenMP host device.

- **Safeguard**: construct a case where the raw prolonged correction inverts a
  fine cell (pinch two coarse nodes); assert the line search either damps α or
  rejects, and `mindet > 0` holds after the cycle.
- **Monotonicity**: energy non-increasing across every cycle of a 5-cycle run.
- **Convergence rate** (the acceptance test for the whole feature): on a
  single-block ~65² (2D) and a 2-block 3D case with a smooth low-frequency
  perturbation of the interior nodes, FAS(ν₁=ν₂=2) must reach the energy
  plateau in ≤ ⅓ of the *fine-sweep-equivalents* plain Jacobi needs (coarse
  sweeps counted at 2^{-D} weight). Margins wide enough for fp32 noise.
- **Parity**: FAS and plain Jacobi converge to the same energy within
  `real_tol`-floored tolerance (same minimizer, different path).

### 6.3 Python integration (`tests/smoothing/`)

One test driving `CppStructuredSweepSession.run_fas` on a small multiblock
grid built with `build_block_structured_context`
(`egg/smoothing/cpp_backend.py:166`): asserts monotone energies, positive
mindets, agreement of final `get_X()` with the plain-Jacobi result within
`real_tol.py` tolerances, and that `phase="untangle"` raises.

### 6.4 Performance validation (not CTest)

- `bench/bench_sweep.cpp`-style row comparing warm ms-to-plateau, and the
  `sphere_in_cube` example (`examples/3D/sphere3d/`) end-to-end, in the fp32
  build (the GPU quick-run mode).
- Kernel checks on the modified interior pair: confirm the `HasTau=false`
  fine instantiations kept their register budget (the interior split exists
  precisely to stay under the gfx1030 128-VGPR 2-wave tier,
  `src/sweep.hpp:1029-1030`) and that transfer kernels are noise in the
  trace. **This machine has an NVIDIA GPU**, so `rocprofv3` is unavailable
  here — use Nsight (`nsys profile` for the trace, `ncu` for per-kernel
  registers/occupancy) locally; the gfx1030 VGPR-tier verification itself
  must wait for an AMD box (or be read from the ACPP JIT stats there).

## 7. Milestones

> **Adaptive ν_c (2026-07-04), amended into the single feature commit (now
> `0e810b9`).** `FasOptions::nu_coarse` is now a CAP: the driver runs
> min(nu_coarse, 2 × the coarsest level's interior diameter) coarsest Newton
> sweeps per cycle (`detail::coarse_sweep_budget`, src/fas.hpp; host test in
> test_multigrid.cpp). Rationale: relaxation propagates ~one cell per sweep,
> so a 6-node-across coarsest block is converged after a handful of sweeps —
> the fixed 32 iterated on a converged state at ~0.15 ms of launch overhead
> each. Measured (fp32 RTX 5070, sphere): 32k steady-state cycle 9.5 → 6.7
> ms, trajectory unchanged; 2M cycle 117.8 → 114.5 ms, 50-cycle E 1.0812e6
> vs 1.0815e6 (noise). A big carried block or a shallow max_levels=2
> hierarchy keeps the full nu_coarse budget (diameter stays large).
> Equal-energy at 32k is now ~0.5–0.6× Jacobi (was ~0.3×): FAS wins to
> mid-accuracy targets (~1.5× at E≈1.6e4) but plateaus more slowly at the
> tight tail — the remaining small-mesh limits are the convergence rate
> (frozen coarse edges/corners) and fine-sweep cost, not the coarse ladder.
>
> **Safeguard-overhead follow-up (2026-07-04), amended into the same
> commit.** The per-cycle safeguard is now device-side:
> `detail::reduce_linesearch_pair` (src/fas.hpp) evaluates two α-ladder
> trials per stencil pass (four reductions, one launch) with α = 0 riding
> along as the pre-state energy; the prolonged correction is stripped to the
> Free set (`zero_unmasked`, src/mg_transfer.hpp) and halo-broadcast ONCE, so
> every trial x + α·corr is halo-consistent by linearity of the pure-copy
> hook — no per-trial `before_sweep`, no trial buffer (−nn·D reals of VRAM,
> ~25 MB at 2M nodes). The per-cycle report reduces into per-cycle device
> slots downloaded once after the loop (run_block_jacobi pattern). The common
> accept-at-α=1 cycle: 1 host sync + 1 readback of 4 scalars, down from
> ~3 waited reductions + 6 waited memcpys. Behavior-preserving: energies
> bit-identical old-vs-new at 32k and 2M (fp32); fp64 + fp32 gates green
> (7/7 ctest incl. new `zero_unmasked` + line-search-pair device tests, 322
> pytest, 4 known composite deselects).
>
> **Measured outcome — the "small meshes lose on safeguard syncs" premise
> was WRONG.** Wall time is unchanged: 32k sphere (n=21 m=6 mh=7) 9.33 →
> 9.31 ms/cycle; 2M (n=81 m=21 mh=21) 118.9 → 117.8 ms/cycle. The host syncs
> were microseconds against a 9 ms GPU-bound cycle. Breakdown at 32k
> (24-cycle warm averages): full cycle 9.5 ms; fine part alone
> (max_levels=1) 3.2 ms; nu_coarse=4 cuts to 5.2 ms — the coarsest level's
> ν_c = 32 Newton sweeps cost ~4.8 ms (~0.15 ms per tiny-grid sweep, pure
> launch overhead) — HALF the cycle. nu_coarse=4 converges essentially
> identically here (E 1.3272e4 vs 1.3233e4 after 24 cycles; nu_coarse=128
> buys nothing: 1.3230e4). So the real small-mesh lever is the coarse-ladder
> launch count: adaptive ν_c (scale with coarsest DOF count), fusing the
> per-block transfer launches (noted in mg_transfer.hpp's header), and/or
> batching coarse smooth launches. The safeguard change is kept for the VRAM
> save, the sync-count reduction (matters when the host is busy), and the
> simpler cycle tail.

> **STATUS 2026-07-03: all milestones (M1–M5, per-block coarsening, example
> integration) complete** on `feat-multigrid`, squashed into the single
> commit `c7ec336` on top of `eb629df`. fp64 + fp32 gates green (C++ 2D+3D
> device tests on the RTX 5070 + OMP host, 322 Python tests; the 4
> `test_cpp_composite.py::test_composite_device_relaxes_on_curve` fp64
> failures pre-date this branch — verified at `eb629df`). Milestone labels
> were removed from code comments; they remain here only. M5 recursion: the
> executor builds the full factor-2 ladder lazily (129² → 65 → 33 → 17 → 9 →
> 5 → 3), `FasOptions::max_levels` clips per call (Python kwarg exposed), and
> `detail::fas_coarse_vcycle` recurses the smooth/restrict/τ/correct leg with
> unsafeguarded α = 1 intermediate corrections (the fine-level line search
> still gates the only correction that reaches the real grid). Saturation is
> gone — fp32 RTX 5070 warm, 12 cycles, ω = 0.8: 129² full depth E = 0.0022 in
> 36 ms vs two-level nc=32 E = 27.2 / nc=128 E = 2.84 vs Jacobi 3072 E = 1.73
> in 66 ms; 257² full depth E = 0.0075 in 41 ms (plateau by cycle ~6, i.e.
> grid-size-independent rate) vs two-level nc=128 E = 138 vs Jacobi 3072
> E = 112 in 140 ms. ncu counters are admin-locked on this box; the gfx1030
> VGPR check waits for an AMD machine.
>
> **M4 complete (2026-07-03).** Coarse interface relaxation: per-level
> BlockTopologyDevice derived from the fine host tables in C++
> (`src/mg_topology.hpp` — even-index lookup + factor division, parity makes
> it exact across reversed/permuted and patchwork faces; drop-not-throw), the
> guarded owner face nodes become synth-path coarse DOFs (free, not
> free_interior), full weighting reads the ghost-inclusive fine view, and the
> FAS drivers run each level's halo hook on positions and gradients.
> `build_structured_context_from_block_maps` (the sphere path) now derives
> face-ghost tables by shared-DOF twin matching, so the flagship example gets
> both coalesced fine ghosts and coarse interface DOFs. Measured
> (sphere_in_cube fp32 RTX 5070): 2M nodes (n=81 m=21 mh=21) FAS 50 cyc
> 1.302M → **1.066M** ≈ Jacobi-3000's 1.022M at 7.5 s vs 61 s (pre-M4 it
> plateaued 27% above); FAS 12 cyc 2.43M → 1.66M in 1.8 s. Two-block 2D:
> FAS-12 lands 50× below Jacobi-1200. Constraint deviations unchanged
> (~1e-7 fp32).
>
> **Per-block coarsening (2026-07-03).** The all-or-nothing rule is gone: a
> block with no halvable axis is CARRIED unchanged (factor 1 everywhere,
> identity transfers, full DOF set) and the ladder stops only when no block
> halves — so the 3³ H-grid corners (216 nodes) no longer veto multigrid for
> the 889k-node mh=3 sphere. Measured: mh=3 went from zero levels to a
> 4-level ladder; FAS 100 cyc reaches Jacobi-3000's energy (4.762M) in 8.2 s
> vs 39.7 s (~5×); the deeper ladders also made the mh=11/21 runs ~20%
> FASTER at equal quality (the ν_c-sweep coarsest solve now runs on tiny
> grids). Cost caveat (documented in coarsen_shapes): a LARGE uncoarsenable
> block is carried at full size — prefer odd node counts. Remaining known
> limit: frozen block edges/corners + singular-edge neighbourhoods (M4
> guards keep them Dirichlet).

| # | Deliverable | Acceptance |
|---|-------------|------------|
| M1 | `jacobi_sweep_once` refactor of `run_block_jacobi`; behavior-preserving | existing `cpp_tests` + `tests/test_cpp_backend.py` green in fp64 (tight tolerances) and fp32; identical energies to pre-refactor within nondeterminism band |
| M2 | `mg_transfer.hpp` + `mg_level.hpp` + hierarchy builder | §6.1 host tests green in fp64 (tight) + fp32 |
| M3 | τ threading (§5.1) + `fas.hpp` two-level V-cycle + bindings | §6.2 + §6.3 green in fp64 (tight) + fp32; convergence-rate criterion met on sphere_in_cube |
| M4 | ~~*(optional)*~~ **done** — coarse halo/share tables derived per level from the fine `BlockTopologyDevice` inputs (`src/mg_topology.hpp`); guarded face-interior coarse DOFs unfrozen; per-level halo hook in the FAS drivers | interface-mode stall measured on multiblock (sphere_in_cube 2M nodes) and removed; parity/conformity asserted by exact division, non-derivable faces drop to the frozen behaviour |
| M5 | *(optional)* recursion to `max_levels > 2`, ν/ω tuning, per-block kernel fusion in transfers | rate improves over two-level on ≥129³-class blocks |

Ship after M3 if the rate win holds; M4/M5 are data-driven.

## 8. Risks / open questions

- **τ sign/scale bugs** are the classic FAS failure and look like "converges,
  but no faster". The §6.1 τ-coherence identity test is written *first*.
- **fp32 gradient restriction noise**: near convergence `∇E` is ~`tol::znorm`;
  τ built from differences of small gradients may be noise-dominated. Guard:
  skip the coarse correction when `‖R̂∇E_f‖∞` is below a floor scaled from
  `tol::znorm` (`src/core.hpp:35-36`).
- **Semi-coarsening bookkeeping** (per-block per-axis factors) is the fiddliest
  index math; it is confined to `mg_transfer.hpp`/`mg_level.hpp` and fully
  covered by exact host tests.
- **Frozen coarse faces** cap the achievable speedup on many-block meshes —
  measured, and escalated to M4 only with evidence.
- **VGPR regression** from τ threading if the compiler doesn't fold the
  `HasTau=false` branch — verified per §6.4 before merging (Nsight `ncu` on
  this NVIDIA box as a proxy; the authoritative gfx1030 VGPR check needs an
  AMD machine).
