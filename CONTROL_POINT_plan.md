# Control-point-form grid parameterization + TMOP-over-control-points

Untracked scratch plan (sibling of FAS_plan.md). Do not gitignore.

## Goal

Add a second smoother backend where the moving DOFs are a coarse **control net**
per block, not individual nodes. Node positions are a fixed *linear* function of
the control net (Eiseman control-point form + Boolean-sum boundary conformance).
The cell metric `W` and the TMOP energy are unchanged: `metric.hpp` / `patch.hpp`
still evaluate per-node grad/hess on the sampled node store. Only the DOF layer
and its transfer change.

This buys: a compact handle for algebraic refine / derefine / cluster while
holding block-interface continuity and grid quality; structurally smooth seams;
a path to non-conforming interfaces; cheap grid movement (moving inflow edges).
TMOP-per-node is NOT removed; the control-point smoother is an alternative
(and optional finishing node pass) selected by config.

## Invariants preserved (do not regress)

- Frozen-halo block-Jacobi cadence, double-buffered DOFs, pure-copy before_sweep
  hook (fas.hpp relies on that linearity).
- Accept rule everywhere: `isfinite(e) && e <= e0 + tol::energy && accept_mindet`.
- ω scales the Newton step BEFORE the projected line search, never a post-hoc
  world-space blend.
- fp64 build is the correctness gate; never assert exact energies.
- Same-layout SoA encode/decode in lockstep (Python encoder <-> device decoder).

---

## Part 0 — Math foundation (dimension-agnostic)

### 0.1 Control net and the map

Per block `b`, a control lattice of interior shape `Nc = (Nc_0, ..., Nc_{D-1})`,
each control point in R^D. Parameter domain `(r_0..r_{D-1}) in [0,1]^D`. Node
`i` in the block carries a fixed parameter vector `r(i)` (its clustered logical
position). The control-point form map is:

    x(i) = Q(r(i); C) = T(r(i); C)  ⊕  boundary(r(i))          (Boolean sum)

Tensor-product interior blend, per-axis first-difference (Eiseman) form:

    T(r; C) = sum_J ( prod_k phi_{j_k}(r_k) ) C_J

`phi` is a per-axis 1D blend policy (see 0.3). The Boolean sum is transfinite
interpolation of the boundary corrections over the `3^D - 1` boundary facets
(faces, edges, corners) with inclusion-exclusion — the same operator egg already
implements in `egg/init/tfi.py`, reused verbatim in D dims.

### 0.2 The one property that makes the backend tractable: linearity

For fixed sampling `r(i)` and fixed boundary paths, `Q` is **linear in the
control points and the boundary node positions**. Both `T` and the Boolean-sum
corrections are linear combinations. Therefore per block:

    X = M · C + b

- `M`  fixed sparse operator, node <- control-point weights (support = blend
        stencil; compact if `phi` is compact-support, see 0.3).
- `b`  fixed offset from truly-fixed geometry boundary paths (constant nodes not
        represented as control DOFs). Zero where all edges are control DOFs.

Consequences (all reuse FAS transfer patterns):
- prolong  `X = M C + b`           sparse matvec       (cp_prolong kernel)
- gather   `g_C = Mᵀ g_X`          transpose matvec    (cp_gather kernel)
- reduced Gauss-Newton system      `Mᵀ H_x M`          (cp_assemble kernel)

`M` and `b` are built ONCE per (re)sampling in Python and uploaded. Refine /
derefine / recluster = rebuild `M`,`b` at a new `r(i)` distribution and re-upload.

### 0.3 Blend policy (per-axis `phi`) — the sparsity AND continuity knob

Expose `phi` as a per-axis 1D basis table (values + derivatives up to the
continuity order at each node param). Two supported policies behind one
interface:
- **compact-support (default)**: B-spline-like / local Eiseman G. `M` sparse,
  control-net Hessian banded (bandwidth = support), thin ghost-control halo.
- **global Eiseman integration blend**: denser `M`; classic near-boundary
  control. Same backend, just a fuller weight table.

**Blend degree is the continuity knob.** A degree-`p` compact blend is `C^{p-1}`,
which sets both the intra-block smoothness and the achievable cross-interface
continuity (see 1.5). Default **cubic (p=3 → C²)** so the target C² along and
across interfaces is reachable; the table carries first AND second derivatives so
curvature-level (C²) interface conditions can be assembled. Degree is a per-axis
policy, so anisotropic choices are allowed.

Backend never sees the choice: it consumes `M` as a generic sparse operator. The
support width sets both the reduced-solve bandwidth and the control-halo width
`w = ceil(continuity_order)` (1.5): C¹ needs `w=1`, C² needs `w=2`.

### 0.4 DOF taxonomy on the control net (mirror of nodes today)

- **interior control point**: free, full D-dim step.
- **boundary control point on geometry**: constrained; tangent-reduced step +
  projection, reusing `geometry.hpp` / `entity_soa` / `project_with_seed`.
  (Boolean sum already forces sampled edge NODES onto the geometry path; we also
  constrain edge control points to that geometry for a consistent near-boundary
  blend.)
- **shared seam control point**: one owner across the interface, owner->copy
  broadcast; gradient gathered from BOTH incident blocks.

Reuse the `MgMasks`-style free/frozen mask, one level up on the control net.

---

## Part 1 — C++ core

Templated `template <int D>` throughout, `dim::` helpers, `PtN/VecN/MatN<D>`,
`real`, `_r`. New headers; existing metric/patch/geometry/structured reused.

### 1.1 New headers

- `src/control_point.hpp`
  - `ControlNetLayout<D>`: like `BlockLayout<D>` but with a **parameterized halo
    width `w`** (interior `Nc`, `w` ghost control layers per face, `w=2` for C2,
    plus the diagonal edge/corner ghosts). NOTE `BlockLayout<D>` hardcodes width-1
    padding (`padded = n+2`); generalize its padding to `n + 2w` (a `w` template/
    ctor param, default 1 keeps the node layout bit-identical) or fork a widened
    variant. For the seam
    exchange). Same offset/stride math, distinct instance from the node layout.
  - `ProlongOpView<D>`: device SoA view of `M`,`b`. CSR-like per node:
      - `row_off[num_nodes+1]`, `col[nnz]` (control global index),
        `wgt[nnz]` (blend weight, real), `boff` offset into `b` (D reals) or a
        `has_b` flag. Matches the entity_soa "self-contained arena slice" style.
  - `apply_prolong<D>(pv, C, X)`  and  `apply_gather<D>(pv, gX, gC)` device-side
    inline helpers (transpose shares the same table).

- `src/control_sweep.hpp`
  - `CpExecutorT<D>` mirroring `StructuredExecutorT<D>`: owns the in-order queue,
    a resident control-net store (double-buffered), the node store `X` as a
    derived scratch buffer, the `ProlongOpView<D>`, the control-level halo topo,
    and reduced-system scratch. Methods `run` (Jacobi over control net),
    `run_untangle` (node-level fallback — see 1.6), `get_X`, `get_C`,
    `set_sampling` (swap in a new `M`,`b` for refine/derefine/cluster).

- `src/control_solve.hpp`
  - per-block banded reduced-system assembly + solve (see 1.4).

Reused as-is: `metric.hpp`, `patch.hpp` (`metric_kernel`, `patch_eval`,
`accumulate_sample`), `geometry.hpp` + `entity_soa.hpp` (boundary control-point
projection), `structured_halo.hpp` pattern (control-level halo), `sweep.hpp`
`reduce_energy_mindet`, `fas.hpp` safeguard/line-search shape.

### 1.2 Resident state (CpExecutorT<D>)

- `C_`, `C_new_`  control net, double-buffered  `[cp_layout.total_reals()]`.
- `X_`           derived node store              `[node_layout.total_reals()]`.
- `prolong_`     `ProlongOpView<D>` (M,b) resident, swappable via set_sampling.
- `cp_topo_`     control-level `BlockTopologyDevice<D>` (ghost-control fill +
                 owner->copy shared-seam pairs), built by twin-matching on shared
                 control global ids (Python), same shape as node topology.
- `cp_free_`     control free/frozen/boundary mask + per-control entity tag/params
                 for constrained control points (entity_soa slice).
- metric scratch `grad_buf/hess_buf/e0_buf` over nodes (reuse SweepScratch).
- reduced-system scratch (per block banded system).

### 1.3 Per-sweep algorithm (control-point block-Jacobi)

One sweep = one merged sequence, double-buffered on `C`:

1. `before_sweep` hook on the READ control buffer: `fused_cp_halo_broadcast<D>`
   — refresh ghost-control columns + non-owner copies of shared seam control
   points. Pure copy gather (keep it linear, same rule as node halo).
2. `cp_prolong`: `X = M · C + b`  (fills the node store from the frozen net).
3. `metric_kernel<D>` on `X` -> per-node `grad`,`hess`,`e0`  (UNCHANGED).
4. `cp_gather` + `cp_assemble`: form each block's banded reduced system
   `(G_b, A_b)` over the block's FREE control points (interior + owned seam);
   ghost/frozen-seam control contributions move to the RHS. `A_b = Mᵀ H M`
   using the per-node block-diagonal `hess` (Gauss-Newton approx; safeguarded
   by the line search, so an approximate direction is fine). Boundary control
   points contribute tangent-reduced rows via `role_Jb`/entity tangent basis.
5. `cp_solve`: block-local banded solve -> control Newton step `δC_b`.
   (Blocks independent within the sweep: frozen halo makes each block's net one
   Jacobi subdomain — this is the concrete "each block is its own control-point
   patch".)
6. `cp_update`: ω-scale `δC` BEFORE line search; backtracking on the GLOBAL fine
   energy with the standard accept rule. Each trial re-prolongs `C_trial -> X`
   and re-evaluates energy+mindet; use the fas.hpp trick of 2 α-trials per fused
   reduction (α=0 doubles as e_before) to bound host syncs. Constrained control
   points project onto their entity each trial (`project_with_seed`, warm cache).
   Write accepted `C_new`.
7. `reduce_energy_mindet<D>` for the per-sweep/per-cycle report.
8. swap `C`,`C_new`.

Cost note: dominant term is step 3 (full metric pass), identical to node TMOP.
Steps 2/4/5/6 are the added overhead; step 5's system is tiny (`prod Nc` per
block, banded). Line-search re-prolong+metric per backtrack is the real cost;
cap backtracks and amortize with the 2-trials-per-reduction pattern.

### 1.4 Reduced system (cp_assemble / cp_solve)

- Reduced gradient at control `J`:  `G_J = sum_i W_J(r_i) · g_i` (scatter, weight).
- Reduced Hessian block `(J,K)`:     `A_JK = sum_{i in supp(J)∩supp(K)}
  W_J(r_i) W_K(r_i) · H_i`  (block-diagonal per-node `H_i`, D×D).
- Sparse because `supp(J)∩supp(K) ≠ ∅` only for control points within blend
  reach; the sparsity is the D-dim control-graph stencil (3D reaches diagonal
  neighbors, so it is a genuine sparse SPD system, not a fixed 2D bandwidth).
- Solve per block: sparse SPD (skyline/Cholesky or a few matrix-free PCG iters)
  on the reduced system; per constrained control point fall back to the
  tangent-reduced `newton_step_from_basis` (curve tdim=1, surface tdim=2).
- Small enough that the assembled `(G_b, A_b)` can be downloaded and solved host-
  side (one small sync/sweep, same shape as the FAS per-cycle report). Keep
  gather/prolong on device (they touch the full node field); never D2H the node
  gradient field.

### 1.5 Interface continuity (concrete, D-general)

Sharing in D dims is a hierarchy over facets: a shared facet of codimension `c`
is a `(D-c)`-dim control lattice shared by every incident block. D=2: shared
edges (curves, 2 blocks) + shared corners. D=3: shared faces (surfaces, 2
blocks) + shared edges (curves, ≥2 blocks) + shared corners. All are handled by
the SAME shared-DOF + owner->copy mechanism, parameterized by facet dimension;
no special 2D/3D code path.

Target: **C2 continuity both ALONG and ACROSS every interface**, and interface
orthogonality **as strong as feasible**. Continuity is structural, set by two
knobs: blend degree (`C^{p-1}`, 0.3) and control-halo width `w`. Orthogonality is
a separate geometric condition that does NOT fall out of the blend; it is imposed
as an aggressively-weighted near-hard term (below).

- **C0 conformal**: each shared facet is ONE shared `(D-c)`-dim control lattice.
  Its C0-level control points are shared DOFs with an owner + owner->copy
  broadcast (control-level twin matching on shared global ids, D-general, built
  in Python like node twin matching). Being degree >=3, the shared facet is
  ALSO C2 ALONG its own extent by construction — C2-along is free. Every incident
  block's Boolean-sum boundary correction references the same facet object, so
  conformal (C0) is exact. Conforming = same facet sampling across incident
  blocks; independent sampling = the non-conforming door (facet lattice shared,
  node counts differ), supported by resampling the facet onto each side's ghost.
- **C2 ACROSS the seam** (structural; needs degree >=3 AND width-`w=2` halo):
  model the cross-seam direction as ONE shared spline whose knot vector has the
  interface as an interior knot of multiplicity 1 — then the block boundary is
  just an interior knot and crossing continuity is `C^{p-1}` for free. Under
  block-Jacobi each block owns a piece, so the halo must carry enough of the
  neighbor's near-facet control layers to realize that single curve: **halo width
  `w` caps the cross-seam order** — `w=1` gives only C1 (slope), `w=2` gives C2
  (curvature) with a cubic blend. So the control halo is `w=2` deep (general:
  `w = ceil(target continuity order)`). One ghost layer is NOT enough for C2;
  this corrects the earlier C1 scoping.
- **Ghost neighborhood** (`3^D-1` directions, each `w` deep): pad the control net
  by `w` ghost control layers on every face PLUS the diagonal edge/corner ghosts
  (the D-dim blend stencil corners reach diagonal neighbors — same reason the 3D
  NODE halo fills edge/corner ghosts; each diagonal is `w`-deep too). Ghosts
  copied from the incident neighbor's near-facet control slab so both sides
  evaluate the same C2-consistent stencil across every facet. Direct lift of
  `structured_halo.hpp` (D-general at node level) one level up, widened to `w`;
  refreshed in step 1 under the frozen cadence.
- **Orthogonality — push it as strong as possible** (explicit goal, not soft-by-
  default). It does NOT fall out of the blend: it is a relation between two
  independent directions (the seam tangent(s) and the crossing tangent) that a
  smooth map does not force. Impose it and make it dominant:
  - Primary: `interface_ortho.py` weighted samples, generalized to the D-dim seam
    frame (3D: two tangential seam directions, crossing edge driven toward the
    seam normal), flow through `cp_gather` onto the near-seam control legs.
    Because the near-boundary crossing direction is governed by ONE adjacent
    control leg in the control-point form, a weighted orthogonality term is far
    more effective here than node-by-node: one leg steers a whole span of
    crossing edges, so a large weight moves the near-seam grid coherently instead
    of fighting node-local noise.
  - Weight continuation: ramp the orthogonality weight up across passes; the
    barrier-safeguarded accept still gates every step (`det>0`, energy monotone),
    so a large weight cannot invert a cell — it stalls instead. Drive it as high
    as the min-det safeguard tolerates to get near-orthogonal seams without a
    hard constraint.
  - Optional hard mode: additionally reduce the near-seam control leg's DOF onto
    the seam normal (exact perpendicular), with automatic fall-back to the
    weighted term where the hard constraint would collide with C2-across or drive
    `det<=0`. Hard-where-feasible, strong-soft-elsewhere.
  - Genuine tension to acknowledge: exact C2-across + exact orthogonality at every
    node + a skewed size target are jointly over-constrained near the seam. The
    design stance is "C2 structural, orthogonality maximized under the barrier,"
    not "both exact everywhere."
- **Non-conforming control resolution across a facet**: ghost fill becomes a
  resample (interpolate the neighbor's near-facet control slab onto this block's
  ghost) rather than a copy. The operator already supports it: ghost fill is an
  offset/weight table, and a resample is the same table with non-unit weights.
- **Singular fans (edges/corners with irregular incidence)**: D=2 corner fans and
  D=3 edge fans + corner fans are the same problem at different facet dims. A
  shared facet whose incident-block count differs from the regular valence gets
  spare ghost control slots and a frozen (copied, not derived) crossing, exactly
  as `bindings.cpp` already does for singular node fans, parameterized by facet
  dimension so 3D edge fans and corner fans reuse one code path.

### 1.6 Untangle

Control-point smoothing cannot put a kink at one inverted cell (smooth map).
So the untangle phase stays NODE-level Jacobi (same reasoning as FAS untangle).
`CpExecutorT::run_untangle` prolongs `C -> X` once, runs the existing node-level
untangle on `X`, then RE-FITS the control net to the untangled `X` (least-squares
`min_C ||M C + b - X||`, normal equations `MᵀM` — small, sparse SPD, reuse solve).
Pipeline: untangle (node) -> optional refit -> control-point shape smoothing ->
optional node-level TMOP finishing pass on `X`.

### 1.7 bindings.cpp

- `CppControlPointSweepSession` mirroring `CppStructuredSweepSession`:
  `run(n_sweeps, kind, omega, report_every)`, `run_untangle`, `get_X`, `get_C`,
  `set_sampling(prolong_soa)` (refine/derefine/cluster without rebuilding the
  session), introspection (`cp_shapes`). Uploads: node layout, control layout,
  `ProlongOpView` SoA (M,b), control free-mask + entity slice, control halo topo.
- Spare-ghost allocation for singular control fans (reuse existing pattern).

---

## Part 2 — Python frontend

Owns geometry, parameterization, and all `M`,`b` construction (numeric core stays
in C++). New numeric-free / numpy-only modules; parity-tested vs a NumPy prolong.

### 2.1 Control-net topology (`egg/topology/control_net.py`)

- Extend `TopologyBuilder` output: per block a `ControlNetSpec` (interior shape
  `Nc`, blend policy per axis, boundary-control geometry attachment).
- Shared seam nets: derive from the existing block-connection graph. A shared
  interface -> one seam control net; assign owner; produce shared-control global
  ids (twin matching, reuse the node twin-matcher generalized to the coarser
  lattice). Emit owner->copy pairs + ghost-control fill tables (the control-level
  analogue of `cpp_backend.build_block_structured_context` halo tables).
- Free/frozen/boundary mask over the control net; attach entity tag/params for
  boundary control points (same `dof_constraints` mechanism as nodes).

### 2.2 The map builder (`egg/geometry/control_point_form.py`)

Dimension-agnostic, numpy. Given a block's `Nc`, node sampling `r(i)`, blend
policy, boundary paths:
- per-axis blend tables `phi_{j}(r_k)` (+ derivatives) — B-spline / Eiseman-G.
- tensor-product interior weights `W_J(r_i) = prod_k phi_{j_k}(r_{k,i})`.
- Boolean-sum boundary corrections via the EXISTING D-dim TFI operator
  (`egg/init/tfi.py`) applied to boundary paths -> constant `b` (for fixed
  edges) or extra columns of `M` (for control-DOF edges).
- Assemble sparse `M` (CSR: row_off/col/wgt) + `b`. This is the single source of
  truth; both the NumPy reference and the C++ upload consume it.

Sampling `r(i)`:
- uniform, or a per-axis clustering/stretching distribution (reuse existing
  clustering functions). Clustering lives HERE, decoupled from control geometry.
- refine/derefine = new `r(i)` with more/fewer nodes -> rebuild `M`,`b` ->
  `session.set_sampling(...)`. Control net `C` and thus the geometry is invariant
  across resampling, so continuity + shape are preserved by construction.

### 2.3 SoA encoding (`egg/geometry/entity_soa.py` sibling)

- Encode `ProlongOpView` (M CSR + b) into a self-contained arena slice; add the
  matching decoder in `src/control_point.hpp` — encode/decode edited in lockstep
  (same rule as entity kinds).
- Control halo tables (ghost fill + owner->copy) encoded like the node halo in
  `cpp_backend.py`.

### 2.4 Solver / pipeline integration

- `egg/smoothing/cpp_backend.py`: `build_control_point_context(grid, ...)` next
  to `build_block_structured_context` — flattens control-net topology + `M`,`b`
  + masks into the upload payload; returns a `CppControlPointSweepSession`.
- `egg/smoothing/solver.py`: route `smoother="control_point"` to the CP session.
- `egg/pipeline.py`: `PipelineConfig.tmop_smoother` gains `"control_point"`;
  `generate_steps` sequences untangle (node) -> refit -> CP shape smoothing ->
  optional node-TMOP finish. Refine/derefine/cluster exposed as pipeline ops that
  call `set_sampling` between phases.
- Init: place the control net (regular lattice in parameter space) and set `C`
  either from the TFI/control-point-form of the boundary paths, or by fitting to
  an existing TFI node grid (`min_C ||M C + b - X0||`).

### 2.5 Reference solver (parity target)

Sequential NumPy: `X = M C + b`, `g_C = Mᵀ g_X`, block-diagonal reduced system,
banded solve, safeguarded line search. Same accept rule. This is the gate the
C++ CP backend is tested against (mirrors the existing flat NumPy reference).

---

## Part 3 — Dimension: 3D is first-class from the start

No `static_assert(D==2)` to-do sites. Every piece is written and tested for D=2
AND D=3 together. `real` fp64 gate runs both dims before anything is called done.
The design has no inherently 2D-only step; the pieces that LOOK 2D-specific are
recast as facet-dimension-parameterized loops so one code path serves both dims.

Facet algebra is the backbone. In D dims a block has `3^D - 1` boundary facets,
one per non-center element of `{-1,0,+1}^D`; a facet's codimension `c` = count of
nonzero components, so its intrinsic dimension is `D - c`. Enumerate facets by
iterating that ternary cube (D-general, `pow(3,D)-1` facets: D=2 -> 8, D=3 ->
26). Everything below indexes off this single enumeration:

- **Boolean sum / boundary conformance** is the D-dim transfinite projector sum
  `Q = 1 - prod_k (1 - P_k)`, expanded by inclusion-exclusion into tensor products
  of per-axis face interpolations. The expansion terms map one-to-one onto the
  ternary-cube facets with signs from the projector algebra. Implement the
  expansion as a loop over the facet enumeration, weights = products of the
  per-axis blend evaluated on the facet's constrained axes. Reuse / generalize
  `egg/init/tfi.py`; if it is not already `3^D-1`-general, generalize it there
  (single source, consumed by the reference and the upload).
- **Shared facets** (Part 1.5): a shared facet of codim `c` is a `(D-c)`-dim
  control lattice. Twin matching, owner assignment, owner->copy, and ghost
  resample all take the facet dimension as a parameter. D=3 exercises face
  (c=1, surface), edge (c=2, curve), and corner (c=3, point) sharing through the
  same code the D=2 edge+corner case uses.
- **Ghost neighborhood**: the control halo fills all `3^D-1` ghost directions
  (faces + diagonal edge/corner ghosts), each `w` layers deep (`w=2` for the C2
  interface target, 1.5). Reuse `structured_halo.hpp`'s node-level D-general fill,
  lifted to the control net and widened from 1 to `w`.
- **Blend / prolong / gather / reduced system**: naturally D-general (axis loops,
  tensor products, `BlockLayout<D>`, `geometry.hpp` D-general projection). The
  reduced system's sparsity is the D-dim control-graph stencil (3D -> wider
  band); solve as sparse SPD (skyline/Cholesky or a few matrix-free PCG iters),
  not a fixed-bandwidth 2D banded solver.
- **Singular fans**: D=3 edge fans and corner fans plus D=2 corner fans are one
  facet-dimension-parameterized special case (spare ghost slots, frozen crossing).

3D surface-geometry foundation is ALREADY complete and tested (audited): all five
3D entities (Plane/Sphere/Cylinder/Line3/BSplineSurface) have `TrimmedEntity`
project/tangent_basis (tdim=2), seeded/warm inversion where iterative
(BSplineSurface) and closed-form cold projection where analytic, device
`EntitySoA::load`, and Python encoders in `egg/geometry/entity_soa.py` (NOT the
stale 2D-only `entity_encoding.py`). `StructuredSession<3>` is live with dim==3
dispatch and 3D tests (`test_3d_sphere_in_cube_cad` drives surface-constrained
boundary nodes through the tdim=2 `newton_step_from_basis<3,2>` arm). So boundary
control points constrained to 3D surfaces (step 6) reuse this directly; no
surface prerequisite work. Drop the stale `patch.hpp:399` "only exercised once 3D
surface entities exist" comment when touched. The genuine 3D-new work is the
facet algebra above (shared face/edge/corner lattices, 3^D-1 ghost fill, edge and
corner singular fans), not geometry.

---

## Part 4 — Tests

- Python (`tests/smoothing/`):
  - `M`,`b` builder: partition-of-unity of interior weights; edge nodes land on
    boundary paths (Boolean sum) to tol; refine/derefine invariance of geometry.
  - **Interface continuity order**: on a two-block fixture, measure derivative
    matching across the seam — C1 (tangent) AND C2 (curvature/second-difference)
    to tol with a cubic blend and `w=2` halo; assert C2 FAILS with `w=1` (guards
    the width knob). C2 ALONG the seam: second difference of the shared facet
    control curve continuous. Both for D=2 and D=3 (3D: C2 across a shared face,
    C2 within the face surface).
  - **Orthogonality strength**: seam crossing-edge vs seam-tangent angle drives
    toward 90deg as the ortho weight ramps; report residual angle vs weight;
    assert it beats the node-based baseline at equal cell-validity (min-det).
  - NumPy CP reference vs flat reference on a single block: monotone energy,
    accept-rule parity, mindet > 0.
  - clustering: prescribed spacing recovered; TMOP maintains it via size target
    rather than fighting it (shape_size metric).
- C++ (`tests/cpp/`, SEPARATE single-TU binary per kernel test — HCF id bug):
  - cp_prolong / cp_gather transpose consistency (`<gX, M C> == <Mᵀ gX, C>`).
  - one CP sweep reduces energy; accept rule holds; mindet monotone (real_tol).
  - control-level halo broadcast correctness (ghost fill + owner->copy).
  - golden via `tests/cpp/gen_golden.py`.
- Parity gate: CppControlPointSweepSession vs NumPy CP reference (fp64, tight).
- **Every test above runs for D=2 AND D=3.** 3D is not a follow-on. Fixtures:
  a single 3D block (map, prolong/gather, sweep); a two-block 3D pair sharing a
  face (C0 face-lattice conformity, C2 across the face at `w=2`, orthogonality-
  strength on the face seam); a 3D fixture exercising a shared edge (c=2 curve
  lattice) and a shared corner; a 3D block with a curved surface boundary
  (surface-constrained control points, tdim=2 tangent step); a 3D edge/corner
  singular fan. Facet enumeration itself unit-tested: `3^D-1` facets, correct
  codim/intrinsic-dim, Boolean-sum inclusion-exclusion signs.

## Part 5 — Build / precision

- fp64 (`EGG_REAL_IS_FP32=OFF`) is the correctness gate for the reduced solve and
  the parity tests; fp32 for GPU iteration. Route tolerances through
  `real_tol`. Reduced banded solve: watch fp32 conditioning of `MᵀM` / `MᵀHM`
  (floor pivots via `tol::tiny`).
- New headers header-only; add CP kernel test binaries to the CMake single-TU
  list. `clang-format`/`ruff format`; single-line Conventional Commits.

## Part 6 — Implementation ordering (mechanics last; full scope above is in-scope)

Every step lands `template<int D>` for D in {2,3} and gates both dims before the
next step. 3D is never a trailing phase; a step is not done until its 3D fixture
passes.

1. Facet enumeration (`3^D-1`) + D-general Boolean-sum/TFI expansion, with the
   `M`,`b` builder + NumPy CP reference; Python tests for a single block in 2D
   and 3D.
2. `control_point.hpp` (`BlockLayout<D>` control layout, `ProlongOpView`,
   prolong/gather) + transpose test (2D and 3D).
3. `control_solve.hpp` reduced assembly + sparse SPD solve; single-block CP sweep;
   energy-descent + accept-rule C++ test; parity vs NumPy reference (2D and 3D).
4. `control_sweep.hpp` executor + double-buffer + reduce; bindings session.
5. Width-`w` control halo (generalize `BlockLayout` padding to `n+2w` + diagonal
   ghosts) + shared facets across dims: 2D edge+corner AND 3D face+edge+corner
   sharing (one facet-dim-parameterized path). C0 conformity + C1 tests at `w=1`,
   then C2-across tests at `w=2` (cubic blend); assert C2 fails at `w=1`. Both
   dims.
6. Boundary control-point projection on geometry: curve (tdim=1) and 3D surface
   (tdim=2) constrained control DOFs.
7. set_sampling refine/derefine/cluster; clustering-as-size-target integration.
8. Untangle path (node untangle -> refit) + pipeline wiring + finishing node pass.
9. Orthogonality maximization: interface_ortho through the CP gather onto near-
   seam control legs + weight continuation (ramp under the barrier) + optional
   hard-perpendicular leg reduction with barrier fall-back. Orthogonality-strength
   test vs node baseline at equal min-det (2D seam frame and 3D surface seam
   frame).
10. Singular fans (2D corner, 3D edge + corner) + non-conforming facet resample.
