"""End-to-end grid-generation pipeline.

    rough topology -> TFI init -> boundary snap -> untangle (delta-continuation)
        -> TMOP quality opt (projection) -> validity check -> MultiBlockGrid

Runs on the C++ backend (barrier sweep + delta-untangle sweep via
``egg._cpp.cpp_core``). :func:`generate` attaches a :class:`PipelineReport`
(per-phase ``min det A`` / energy) to the returned grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

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
        TMOP target; defaults to ``IdentityTarget(d)``.
    metric : str
        TMOP quality metric.
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
        Total TMOP sweeps and sweeps per yielded chunk.
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
    metric: str = "shape_2d"

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
    omega: float = 0.8
    report_every: int = 0

    # Optional boundary-layer phases (see generate_steps).
    pin_sweeps: int = 0
    respace: bool = False

    device: str = "cpu"
    verbose: bool = False


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


def _energy(grid, energy_stencil) -> float:
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
    )


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
        TMOP target function. None uses an isotropic quality target.
        Untangling always runs on the isotropic context: geometric validity
        is target-independent, and a strongly anisotropic target can stall
        the continuation.
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
    throughout.

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
    margin, device = cfg.margin, cfg.device
    tmop_chunk, omega, report_every = cfg.tmop_chunk, cfg.omega, cfg.report_every

    d = grid.topology.d
    iso = IdentityTarget(d)

    # Snap boundary DOFs, then build the isotropic (untangle) context.
    project_nodes(grid, grid.dof_constraints)
    ctx_iso = build_sweep_context(grid, iso)
    es = ctx_iso.energy_stencil

    md = grid_min_det(grid.global_nodes, es)
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
    tmop_ctx = ctx_iso if target is None else build_sweep_context(grid, target)
    tmop_sweeps = max((cfg.tmop_sweeps // tmop_chunk) * tmop_chunk, tmop_chunk)
    # Block-Jacobi over the halo-padded structured store, built from the grid's
    # BlockTopology; one merged double-buffered launch per sweep, SOR weight omega.
    from .smoothing.cpp_backend import (
        CppStructuredSweepSession,
        build_block_structured_context,
    )

    bsc = build_block_structured_context(grid)
    session = CppStructuredSweepSession(tmop_ctx, bsc, grid.global_nodes, device=device)
    run_kwargs = {"omega": omega}
    done = 0
    while done < tmop_sweeps:
        k = min(tmop_chunk, tmop_sweeps - done)
        energies, _mds = session.run(k, report_every=report_every, **run_kwargs)
        done += k
        _sync(grid, session.get_X())
        yield (
            "tmop",
            {
                "energy": float(np.asarray(energies)[-1]),
                "min_det": grid_min_det(grid.global_nodes, es),
                "sweeps": done,
            },
        )
    project_nodes(grid, grid.dof_constraints)
    score_es = tmop_ctx.energy_stencil

    # --- Phase 3 (optional): pin the first n_fixed boundary layers exactly ---
    if cfg.pin_sweeps > 0:
        from .smoothing.respace import respace_first_layers

        pinned = respace_first_layers(grid, grid.topology)
        yield (
            "pin",
            {
                "n_dofs": int(pinned.size),
                "min_det": grid_min_det(grid.global_nodes, es),
            },
        )
        if pinned.size:
            # Rebuild the context so the pinned DOFs are compiled out of the
            # update set, then re-equilibrate the free grid (warm start).
            pin_ctx = build_sweep_context(grid, iso if target is None else target)
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
                energies, _mds = session.run(k, report_every=report_every, **run_kwargs)
                done += k
                _sync(grid, session.get_X())
                yield (
                    "tmop",
                    {
                        "energy": float(np.asarray(energies)[-1]),
                        "min_det": grid_min_det(grid.global_nodes, es),
                        "sweeps": done,
                    },
                )
            project_nodes(grid, grid.dof_constraints)
            score_es = pin_ctx.energy_stencil

    # --- Optional exact wall-respacing post-pass ---
    if cfg.respace:
        from .smoothing.respace import enforce_boundary_layer_spacing

        enforce_boundary_layer_spacing(grid, grid.topology, straighten_columns=False)
        yield ("respace", {"min_det": grid_min_det(grid.global_nodes, es)})

    # Terminal summary AFTER the closing boundary snap, so the reported/plotted
    # final numbers reflect the snapped mesh. Energy is scored on the SAME
    # context TMOP optimised — for a BL/anisotropic target the isotropic
    # stencil reports a different (higher) objective on the same mesh.
    yield (
        "final",
        {
            "min_det": grid_min_det(grid.global_nodes, es),
            "energy": _energy(grid, score_es),
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
