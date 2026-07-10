# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""End-to-end grid-generation pipeline.

    rough topology -> TFI init -> boundary snap -> untangle (delta-continuation)
        -> TMOP quality opt (projection) -> validity check -> MultiBlockGrid

Runs on the C++ backend (barrier sweep + delta-untangle sweep via
``egg._cpp.cpp_core``). :func:`generate` attaches a :class:`PipelineReport`
(per-phase ``min det A`` / energy) to the returned grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal

import numpy as np

from .core.types import MultiBlockGrid

__all__ = [
    "generate",
    "generate_steps",
    "drain_steps",
    "run_pipeline",
    "PipelineConfig",
    "PipelineReport",
]


@dataclass
class PipelineConfig:
    """Pipeline options: per-phase caps, schedule params, and backend choice.

    Consumed by :func:`generate_steps` / :func:`run_pipeline` (field names
    match their keyword overrides) and by :func:`generate`.

    Attributes
    ----------
    target_fn : callable, optional
        Explicit TMOP target. When ``None`` (default) the pipeline builds
        one automatically from the topology (see ``cluster_boundary_layers``);
        an explicit target always wins and suppresses the auto-build.
    margin : float
        Validity gate: the grid counts as untangled when
        ``min det A > margin``.
    sweeps_per_delta : int
        Untangle sweeps per delta level.
    delta0_factor : float
        Initial delta = ``delta0_factor * |min det A|``.
    untangle_shrink : float
        Delta shrink per outer iteration. Gentle by default: block-Jacobi
        untangle smooths less per sweep than a sequential relaxation, so
        it needs more delta levels to clear a fold.
    max_outer : int
        Cap on delta levels.
    untangle_direct : bool
        True = the whole delta-continuation in one C++ call; False yields
        per delta so live views can animate the unfolding.
    tmop_sweeps, tmop_chunk : int
        Total TMOP sweeps and sweeps per yielded chunk. With
        ``tmop_smoother="fas"`` both count V-CYCLES instead (each cycle is
        ``nu_pre + nu_post = 4`` fine sweeps plus the coarse-grid work).
    tmop_smoother : str
        ``"jacobi"`` (default) or ``"fas"`` — FAS nonlinear geometric
        multigrid V-cycles for the TMOP phase (the untangle phase always
        relaxes plain block-Jacobi; FAS is barrier-phase-only by design).
        Blocks whose node counts admit no coarse level fall back to plain
        Jacobi with the equivalent fine-sweep budget automatically.
    tmop_metric : str
        ``"shape"`` (default) or ``"shape_size"`` — the TMOP objective.
        The shape metric is scale-invariant (angles/aspect only); shape+size
        adds ``(det T - 1)^2``, pushing every cell toward the target volume
        ``det W`` as well. Size-aware, so the target must carry physical
        scale: when no ``target`` is given the auto-build sizes the
        non-clustered far field to the grid's mean cell volume
        (:func:`egg.smoothing.targets.mean_size_target`) — global size
        uniformity. The untangle phase is metric-independent.
    cluster_boundary_layers : bool
        When ``True`` (default) and no explicit ``target`` is passed, the
        pipeline auto-builds the whole-domain TMOP target from the topology's
        ``set_boundary_layer`` specs
        (:func:`egg.smoothing.targets.build_topology_target`), passing
        ``metric=tmop_metric`` so the far field stays scale-consistent with
        the objective. With no specs this reduces to the metric's plain
        default (mean-size under ``shape_size``, identity under ``shape``).
        Set ``False`` to suppress clustering even when specs exist. Ignored
        when an explicit ``target`` is given.
    bl_blend_neighbours : bool
        Forwarded to :func:`~egg.smoothing.targets.build_topology_target` by
        the auto-build: continue a wall block's clustering profile into the
        block behind it. Default on.
    bl_interior_spacing : float, optional
        Forwarded to the auto-build: cap on the wall-normal growth
        (isotropic far field). ``None`` uses each block's natural face
        spacing.
    fas_nu_pre, fas_nu_post : int
        Pre-/post-smooth sweeps per level per FAS V-cycle.
    fas_nu_coarse : int
        Cap on Newton sweeps on the coarsest level per cycle (the driver
        runs ``min(fas_nu_coarse, 2 × coarsest interior diameter)``). The
        small-mesh tuning knob: the coarsest-level launches dominate cycle
        cost on small grids, so a lower cap trades convergence slack for
        wall time there.
    fas_max_levels : int
        V-cycle depth cap, fine level included (the hierarchy build stops
        on its own when blocks stop coarsening).
    omega : float
        Block-Jacobi SOR/damping weight for the TMOP phase.
    report_every : int
        Energy/min-det reduction throttle for the resident session: 0 =
        chunk-end only (lowest-launch default for the small-n GPU regime),
        1 = per-sweep (for live energy-curve plots), k > 1 = every k-th
        plus final.
    pin_sweeps : int
        TMOP sweeps after pinning the first boundary layers (0 = skip the
        pin phase; see :func:`generate_steps`).
    respace : bool
        Run the exact wall-respacing post-pass at the end.
    device : str
        ``"cpu"``, ``"gpu"``, or ``"auto"``.
    verbose : bool
    """

    target_fn: Callable[..., np.ndarray] | None = None

    # Validity gate / untangle schedule.
    margin: float = 1e-9
    sweeps_per_delta: int = 20
    delta0_factor: float = 2.0
    untangle_shrink: float = 0.8
    max_outer: int = 60
    untangle_direct: bool = True

    # TMOP quality phase.
    tmop_sweeps: int = 40
    tmop_chunk: int = 10
    tmop_smoother: Literal["jacobi", "fas"] = "jacobi"
    tmop_metric: Literal["shape", "shape_size"] = "shape"
    fas_nu_pre: int = 2
    fas_nu_post: int = 2
    fas_nu_coarse: int = 32
    fas_max_levels: int = 32
    omega: float = 0.8
    report_every: int = 0

    # Boundary-layer clustering target (see generate_steps / cluster_* below).
    cluster_boundary_layers: bool = True
    bl_blend_neighbours: bool = True
    bl_interior_spacing: float | None = None

    # Optional boundary-layer phases (see generate_steps).
    pin_sweeps: int = 0
    respace: bool = False

    # Optional composed block-interface term (2D). When set, the TMOP phase adds
    # weighted orthogonality/continuity shape samples at interface-adjacent cells
    # (see egg.smoothing.interface_ortho): a dict of kwargs forwarded to
    # interface_ortho_samples, e.g. {"mode": "normal", "weight": 0.3}. None (the
    # default) leaves the objective as the plain shape/shape_size metric.
    interface_ortho: dict | None = None

    # Optional block-interface C2 curvature term (2D). When set, the TMOP phase
    # adds the curvature-continuity term over grid-line windows (see
    # egg.smoothing.interface_c2): a dict of kwargs forwarded to
    # curvature_windows, e.g. {"weight": 0.0, "iface_boost": 20.0}. Concentrates
    # curvature continuity across block seams; None leaves it off.
    interface_c2: dict | None = None

    device: Literal["cpu", "gpu", "auto"] = "cpu"
    verbose: bool = False

    def validate(self) -> None:
        """Reject invalid enum knobs up front — a typo must fail before the
        untangle phase spends minutes, not after."""
        if self.tmop_smoother not in ("jacobi", "fas"):
            raise ValueError(
                f"PipelineConfig.tmop_smoother must be 'jacobi' or 'fas', got "
                f"{self.tmop_smoother!r}"
            )
        if self.tmop_metric not in ("shape", "shape_size"):
            raise ValueError(
                f"PipelineConfig.tmop_metric must be 'shape' or 'shape_size', "
                f"got {self.tmop_metric!r}"
            )
        if self.device not in ("cpu", "gpu", "auto"):
            raise ValueError(
                f"PipelineConfig.device must be 'cpu', 'gpu' or 'auto', got "
                f"{self.device!r}"
            )
        if self.interface_ortho is not None:
            mode = self.interface_ortho.get("mode", "normal")
            if mode not in ("normal", "continuous"):
                raise ValueError(
                    "PipelineConfig.interface_ortho['mode'] must be 'normal' or "
                    f"'continuous', got {mode!r}"
                )


@dataclass
class PipelineReport:
    """Per-phase diagnostics collected during a :func:`generate` run."""

    phases: list[dict[str, Any]] = field(default_factory=list)
    untangled: bool = False
    untangle_converged: bool = True
    final_min_det: float = float("nan")
    final_energy: float = float("nan")

    def add(self, name: str, **stats: Any) -> None:
        """Append a per-phase stats entry."""
        entry = {"phase": name, **stats}
        self.phases.append(entry)


def _min_det(grid, energy_stencil) -> float:
    from .smoothing.untangle import grid_min_det

    return grid_min_det(grid.global_nodes, energy_stencil)


def _energy(grid, energy_stencil, metric: str = "shape_2d") -> float:
    from .smoothing.objective import assemble_energy_vec

    es = energy_stencil
    return assemble_energy_vec(
        grid.global_nodes,
        es["gc"],
        es["gn0"],
        es["gn1"],
        es["s0"],
        es["s1"],
        es["W_inv"],
        metric=metric,
    )


def _cpp_reduce(grid, ctx, device: str, phase: str = "barrier") -> tuple[float, float]:
    """``(energy, min_det)`` from the C++ structured session without moving nodes.

    A single ``omega=0`` sweep runs the device reduction on the resident X, so
    this is dimension-general; the NumPy :func:`_min_det` / :func:`_energy` stay
    the 2D parity reference.
    """
    from .smoothing.cpp_backend import (
        CppStructuredSweepSession,
        build_block_structured_context,
    )

    session = CppStructuredSweepSession(
        ctx, build_block_structured_context(grid), grid.global_nodes, device=device
    )
    energies, mds = session.run(1, phase=phase, omega=0.0, report_every=0)
    return float(np.asarray(energies)[-1]), float(np.asarray(mds)[-1])


def _sync(grid, X) -> None:
    """Write a flat global-node array back into the grid + its per-block views."""
    grid.global_nodes = np.asarray(X)
    for bi, blk in enumerate(grid.blocks):
        blk.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]


def generate_steps(
    grid, target=None, config: PipelineConfig | None = None, **overrides
):
    """Step-wise pipeline on the C++ backend; mutates ``grid`` in place.

    Parameters
    ----------
    grid : MultiBlockGrid
        Initialised grid (``topology.initialize_grid()``); mutated in place.
    target : callable, optional
        Explicit TMOP target function. ``None`` (default) auto-builds one
        from the topology via
        :func:`egg.smoothing.targets.build_topology_target` when
        ``config.cluster_boundary_layers`` is set (else an isotropic quality
        target). Passing a target suppresses the auto-build.

        .. important::
           If you build the target yourself and pass it here, you MUST
           construct it with ``metric=config.tmop_metric``. That coupling is
           what keeps the non-clustered far-field ``det W`` consistent with
           the objective — under ``tmop_metric="shape_size"`` an
           ``IdentityTarget`` far field (``det W = 1``) would drive every
           non-wall cell toward area 1.0 instead of the grid's mean size. The
           auto-build handles this for you; the manual path cannot, so it is
           on the caller.

        Untangling always runs on the isotropic context regardless: geometric
        validity is target-independent, and a strongly anisotropic target can
        stall the continuation.
    config : PipelineConfig, optional
        Defaults are used when omitted. Keyword ``overrides`` update a copy
        of it, so one config can be reused across call sites; unknown
        override names raise.

    Yields
    ------
    (phase, info) : (str, dict)
        After each unit of work, so callers can animate the folded ->
        untangled -> smoothed transition or collect convergence history.
        Phase order: ``init`` (once) -> ``untangle`` (only if folded) ->
        ``tmop`` (per chunk) -> ``pin`` + ``tmop`` (with ``pin_sweeps``) ->
        ``respace`` (with ``respace``) -> ``final`` (once, after the
        closing boundary snap). :func:`generate` drains this for a batch
        run; the live demos drive it from a PyVista timer (see
        :func:`egg.io.visualize.animate_pipeline`).

    Notes
    -----
    Both the TMOP and untangle phases relax block-Jacobi over the
    halo-padded structured store (one merged double-buffered launch per
    sweep; ``omega`` is the TMOP SOR weight); X stays device-resident
    throughout. With ``tmop_smoother="fas"`` the TMOP phase runs FAS
    (nonlinear geometric multigrid) V-cycles instead — the untangle phase
    always stays plain Jacobi.

    The TMOP phase always steps per chunk (``tmop_chunk`` sweeps each); set
    ``tmop_chunk == tmop_sweeps`` for one direct call. The untangle phase
    by default runs *direct* — the whole delta-continuation in one
    ``cpp_untangle`` call (the analogue of chunk == sweeps), yielding a
    single ``untangle`` event; ``untangle_direct=False`` steps and yields
    per delta so a live view can animate the unfolding.

    With ``pin_sweeps > 0``, after the TMOP phase each boundary-layer
    spec's first ``n_fixed`` layers are set to their exact geometric
    heights and pinned (:func:`egg.smoothing.respace_first_layers`), the
    sweep context is rebuilt so the pinned DOFs leave the update set, and
    ``pin_sweeps`` more TMOP sweeps re-equilibrate the free grid. With
    ``respace`` the exact wall-respacing post-pass
    (:func:`egg.smoothing.enforce_boundary_layer_spacing`, nodes sliding
    along their smoothed columns) runs at the end instead.
    """
    from .smoothing.solver import build_sweep_context
    from .smoothing.targets import IdentityTarget
    from .smoothing.untangle import grid_min_det
    from .projection.project import project_nodes

    cfg = (
        replace(config, **overrides)
        if config is not None
        else PipelineConfig(**overrides)
    )
    cfg.validate()
    margin, device = cfg.margin, cfg.device
    tmop_chunk, omega, report_every = cfg.tmop_chunk, cfg.omega, cfg.report_every

    d = grid.topology.d
    iso = IdentityTarget(d)

    # initialize_grid step 3 already orthogonally projected every associated
    # boundary DOF onto its entity with the same entity.project this would call,
    # so the incoming grid needs no boundary snap here. project_nodes only
    # returns below after a phase that moves constrained DOFs (untangle, TMOP).
    ctx_iso = build_sweep_context(grid, iso)
    es = ctx_iso.energy_stencil

    # Min-det/energy monitoring: NumPy (the 2D parity reference) in 2D; the
    # d-general C++ device reduction in 3D, where the NumPy metric is 2D-only.
    def md_now() -> float:
        if d == 2:
            return grid_min_det(grid.global_nodes, es)
        return _cpp_reduce(grid, ctx_iso, device, phase="barrier")[1]

    def energy_now(score_ctx, metric: str) -> float:
        if d == 2:
            return _energy(grid, score_ctx.energy_stencil, metric=metric)
        phase = "shape_size" if metric == "shape_size" else "barrier"
        return _cpp_reduce(grid, score_ctx, device, phase=phase)[0]

    md = md_now()
    yield ("init", {"min_det": md})

    # --- Phase 1: δ-continuation untangle (only if folded) ---
    #
    # Two modes, mirroring the TMOP phase's chunking:
    #   stepped (default) — one resident-session run per δ, yielding each step so
    #     the live view animates the unfolding (X stays device-resident; only a
    #     host min-det check per δ, exactly like the TMOP per-chunk sync).
    #   direct (untangle_direct=True) — the whole δ-continuation in ONE C++ call
    #     (cpp_untangle), no per-δ host round-trips. This is the analogue of
    #     setting TMOP chunk == sweeps: production runs (generate) use it; only a
    #     single ``untangle`` event is yielded.
    if md <= margin:
        # Untangling relaxes a δ-barrier over the folded grid. Block-Jacobi is
        # damped (omega < 1): an undamped simultaneous update overshoots the
        # barrier and can deepen a fold instead of clearing it.
        untangle_omega = 0.5
        if cfg.untangle_direct:
            from .smoothing.cpp_backend import cpp_untangle

            X_out, md, outer_iters, delta_final = cpp_untangle(
                ctx_iso,
                grid,
                grid.global_nodes,
                sweeps_per_delta=cfg.sweeps_per_delta,
                delta0_factor=cfg.delta0_factor,
                shrink=cfg.untangle_shrink,
                max_outer=cfg.max_outer,
                device=device,
                margin=margin,
                omega=untangle_omega,
            )
            _sync(grid, X_out)
            yield (
                "untangle",
                {
                    "min_det": md,
                    "converged": md > margin,
                    "direct": True,
                    "outer_iters": outer_iters,
                    "delta": delta_final,
                },
            )
        else:
            from .smoothing.cpp_backend import (
                CppStructuredSweepSession,
                build_block_structured_context,
            )

            bsc = build_block_structured_context(grid)
            session = CppStructuredSweepSession(
                ctx_iso, bsc, grid.global_nodes, device=device
            )
            delta = cfg.delta0_factor * max(abs(md), 1e-12)
            for it in range(cfg.max_outer):
                _e, mds = session.run(
                    cfg.sweeps_per_delta,
                    phase="untangle",
                    delta=delta,
                    omega=untangle_omega,
                    report_every=report_every,
                )
                md = float(np.asarray(mds)[-1])
                _sync(grid, session.get_X())
                converged = md > margin
                yield (
                    "untangle",
                    {
                        "min_det": md,
                        "delta": delta,
                        "outer_iter": it + 1,
                        "converged": converged,
                    },
                )
                if converged:
                    break
                delta *= cfg.untangle_shrink
        # Re-snap boundary DOFs after untangling moved them.
        project_nodes(grid, grid.dof_constraints)

    # --- Phase 2: TMOP quality optimisation (resident session, per-chunk loop) ---
    # Auto-build the whole-domain target when the caller passed none. With
    # boundary-layer specs (and clustering on) this is the clustering target;
    # otherwise it is the metric's plain default — mean-size W under shape_size
    # (the size term needs physical scale, sized to the now-untangled grid's
    # mean cell volume), or the isotropic ctx_iso under shape. `metric` MUST
    # track cfg.tmop_metric so the non-wall far field stays scale-consistent
    # with the objective (see build_topology_target / the `target` docstring).
    tmop_phase = "shape_size" if cfg.tmop_metric == "shape_size" else "barrier"
    if target is None:
        specs = getattr(grid.topology, "boundary_layer_specs", {})
        if cfg.cluster_boundary_layers and specs:
            from .smoothing.targets import build_topology_target

            target = build_topology_target(
                grid.topology,
                grid,
                metric=cfg.tmop_metric,
                blend_neighbours=cfg.bl_blend_neighbours,
                interior_spacing=cfg.bl_interior_spacing,
            )
        elif cfg.tmop_metric == "shape_size":
            from .smoothing.targets import mean_size_target

            target = mean_size_target(grid)
        # else (shape metric, no clustering) — stay None and reuse ctx_iso.
    # The composed interface term (if any) lives in the sweep context, so an
    # interface_ortho request forces a fresh build even for the isotropic target
    # (ctx_iso was built without it).
    io = cfg.interface_ortho
    ic2 = cfg.interface_c2
    if target is None and io is None and ic2 is None:
        tmop_ctx = ctx_iso
    else:
        tmop_ctx = build_sweep_context(
            grid,
            iso if target is None else target,
            interface_ortho=io,
            interface_c2=ic2,
        )
    tmop_sweeps = max((cfg.tmop_sweeps // tmop_chunk) * tmop_chunk, tmop_chunk)
    # Block-Jacobi over the halo-padded structured store, built from the grid's
    # BlockTopology; one merged double-buffered launch per sweep, SOR weight omega.
    from .smoothing.cpp_backend import (
        CppStructuredSweepSession,
        build_block_structured_context,
    )

    bsc = build_block_structured_context(grid)
    session = CppStructuredSweepSession(tmop_ctx, bsc, grid.global_nodes, device=device)
    run_kwargs = {"omega": omega, "phase": tmop_phase}

    def run_tmop(sess, k):
        if cfg.tmop_smoother == "fas":
            return sess.run_fas(
                k,
                nu_pre=cfg.fas_nu_pre,
                nu_post=cfg.fas_nu_post,
                nu_coarse=cfg.fas_nu_coarse,
                max_levels=cfg.fas_max_levels,
                **run_kwargs,
            )
        return sess.run(k, report_every=report_every, **run_kwargs)

    done = 0
    while done < tmop_sweeps:
        k = min(tmop_chunk, tmop_sweeps - done)
        # The chunk-end min det comes from the device reduction (same as the
        # stepped untangle loop) — no host O(N) recompute per chunk.
        energies, mds = run_tmop(session, k)
        done += k
        _sync(grid, session.get_X())
        yield (
            "tmop",
            {
                "energy": float(np.asarray(energies)[-1]),
                "min_det": float(np.asarray(mds)[-1]),
                "sweeps": done,
            },
        )
    project_nodes(grid, grid.dof_constraints)
    score_ctx = tmop_ctx

    # --- Phase 3 (optional): pin the first n_fixed boundary layers exactly ---
    if cfg.pin_sweeps > 0:
        from .smoothing.respace import respace_first_layers

        pinned = respace_first_layers(grid, grid.topology)
        yield (
            "pin",
            {
                "n_dofs": int(pinned.size),
                "min_det": md_now(),
            },
        )
        if pinned.size:
            # Rebuild the context so the pinned DOFs are compiled out of the
            # update set, then re-equilibrate the free grid (warm start).
            pin_ctx = build_sweep_context(
                grid,
                iso if target is None else target,
                interface_ortho=io,
                interface_c2=ic2,
            )
            session = CppStructuredSweepSession(
                pin_ctx,
                build_block_structured_context(grid),
                grid.global_nodes,
                device=device,
            )
            total = max((cfg.pin_sweeps // tmop_chunk) * tmop_chunk, tmop_chunk)
            done = 0
            while done < total:
                k = min(tmop_chunk, total - done)
                energies, mds = run_tmop(session, k)
                done += k
                _sync(grid, session.get_X())
                yield (
                    "tmop",
                    {
                        "energy": float(np.asarray(energies)[-1]),
                        "min_det": float(np.asarray(mds)[-1]),
                        "sweeps": done,
                    },
                )
            project_nodes(grid, grid.dof_constraints)
            score_ctx = pin_ctx

    # --- Optional exact wall-respacing post-pass ---
    if cfg.respace:
        from .smoothing.respace import enforce_boundary_layer_spacing

        enforce_boundary_layer_spacing(grid, grid.topology, straighten_columns=False)
        yield ("respace", {"min_det": md_now()})

    # Terminal summary AFTER the closing boundary snap, so the reported/plotted
    # final numbers reflect the snapped mesh. Energy is scored on the SAME
    # context and metric TMOP optimised — for a BL/anisotropic target the
    # isotropic stencil reports a different (higher) objective on the same mesh.
    score_metric = "shape_size" if cfg.tmop_metric == "shape_size" else "shape_2d"
    yield (
        "final",
        {
            "min_det": md_now(),
            "energy": energy_now(score_ctx, score_metric),
        },
    )


def _fmt_info(info: dict) -> str:
    parts = [
        f"{k}={v:.4e}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items()
    ]
    return " ".join(parts)


def drain_steps(steps, *, mindet_history=None, energy_history=None, verbose=True):
    """Run a :func:`generate_steps` generator to completion (headless).

    Collects per-step ``min_det`` / ``energy`` into the supplied lists (for
    convergence plots) and, when ``verbose``, prints one line per step.
    """
    last_phase = None
    for phase, info in steps:
        if verbose:
            if phase != last_phase:
                print(f"\n[{phase}]")
                last_phase = phase
            print("  " + _fmt_info(info))
        if mindet_history is not None and "min_det" in info:
            mindet_history.append(info["min_det"])
        if energy_history is not None and "energy" in info:
            energy_history.append(info["energy"])


def run_pipeline(
    grid,
    target=None,
    config: PipelineConfig | None = None,
    *,
    mindet_history=None,
    energy_history=None,
    verbose=True,
    **overrides,
):
    """Batch convenience: :func:`drain_steps` over :func:`generate_steps`.

    Same options and in-place ``grid`` mutation as :func:`generate_steps`;
    the history lists collect per-step ``min_det`` / ``energy``. Use
    :func:`generate_steps` directly to drive the steps yourself (e.g.
    :func:`egg.io.visualize.animate_pipeline`).
    """
    drain_steps(
        generate_steps(grid, target, config, **overrides),
        mindet_history=mindet_history,
        energy_history=energy_history,
        verbose=verbose,
    )


def generate(topology, config: PipelineConfig | None = None) -> MultiBlockGrid:
    """Run the full pipeline and return the optimised :class:`MultiBlockGrid`.

    The returned grid carries a :class:`PipelineReport` at ``grid.pipeline_report``.
    """
    config = config or PipelineConfig()
    report = PipelineReport()

    # 1. TFI init.
    grid = topology.initialize_grid()

    # Batch-drive the step-wise generator: untangle runs DIRECT (one cpp_untangle
    # call — the analogue of TMOP chunk == sweeps), TMOP steps per chunk.
    # generate_steps mutates the grid and does the boundary snaps.
    tmop_sweeps_done = 0
    try:
        for phase, info in generate_steps(
            grid, config.target_fn, config, untangle_direct=True
        ):
            if phase == "init":
                report.add("init", min_det=info["min_det"])
            elif phase == "untangle":
                report.untangled = True
                report.untangle_converged = bool(info["converged"])
                report.add(
                    "untangle", min_det=info["min_det"], converged=info["converged"]
                )
            elif phase == "tmop":
                tmop_sweeps_done = info["sweeps"]
            elif phase == "final":
                report.final_min_det = info["min_det"]
                report.final_energy = info["energy"]
                report.add(
                    "tmop",
                    min_det=info["min_det"],
                    energy=info["energy"],
                    sweeps=tmop_sweeps_done,
                )
    except KeyboardInterrupt:
        # Grid is synced after each step; report what we have.
        print(
            f"\n[interrupted after {tmop_sweeps_done} TMOP sweeps; progress preserved]"
        )

    if config.verbose:
        print(
            f"[pipeline] final: min det A={report.final_min_det:.3e} "
            f"energy={report.final_energy:.4f}"
        )

    grid.pipeline_report = report
    return grid
