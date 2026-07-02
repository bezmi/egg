"""Orchestrates the full pipeline run.

``generate`` wires the end-to-end flow promised by the project:

    rough topology -> TFI init -> boundary snap -> untangle (delta-continuation)
        -> TMOP quality opt (projection) -> validity check -> MultiBlockGrid

It runs on the C++ backend (barrier sweep + delta-untangle sweep via
``egg._cpp.cpp_core``) and attaches a small
:class:`PipelineReport` (per-phase ``min det A`` / energy) to the returned grid
for tests and demos.
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
    """Pipeline options: per-phase caps, schedule params, and backend choices.

    Consumed by :func:`generate_steps` / :func:`run_pipeline` (field names
    match their keyword overrides) and by :func:`generate`.
    """

    target_fn: Callable[..., np.ndarray] | None = None  # default: IdentityTarget(d)
    metric: str = "shape_2d"

    # Validity gate / untangle schedule.
    margin: float = 1e-9
    sweeps_per_delta: int = 20
    delta0_factor: float = 2.0
    # Gentle δ shrink: block-Jacobi untangle smooths less per sweep than a
    # sequential relaxation, so it needs more δ levels to clear a fold.
    untangle_shrink: float = 0.8
    max_outer: int = 60
    # Direct = the whole δ-continuation in one C++ call; stepped (False)
    # yields per δ so live views can animate the unfolding.
    untangle_direct: bool = True

    # TMOP quality phase.
    tmop_sweeps: int = 40
    tmop_chunk: int = 10
    omega: float = 0.8  # block-Jacobi SOR/damping weight for the TMOP phase
    # report_every throttle for the resident session's energy/min-det reduction
    # (0 = chunk-end, 1 = per-sweep, k > 1 = every k-th plus final). 0 is the
    # lowest-launch default for the small-n GPU regime; set 1 for live plots
    # that animate per-sweep energy curves.
    report_every: int = 0

    # Optional boundary-layer phases (see :func:`generate_steps`).
    pin_sweeps: int = 0
    respace: bool = False

    # Backend.
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
        entry = {"phase": name, **stats}
        self.phases.append(entry)


def _min_det(grid, energy_stencil) -> float:
    from .smoothing.untangle import grid_min_det
    return grid_min_det(grid.global_nodes, energy_stencil)


def _energy(grid, energy_stencil) -> float:
    from .smoothing.objective import assemble_energy_vec
    es = energy_stencil
    return assemble_energy_vec(
        grid.global_nodes, es["gc"], es["gn0"], es["gn1"],
        es["s0"], es["s1"], es["W_inv"])


def _sync(grid, X) -> None:
    """Write a flat global-node array back into the grid + its per-block views."""
    grid.global_nodes = np.asarray(X)
    for bi, blk in enumerate(grid.blocks):
        blk.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]


def generate_steps(grid, target=None, config: PipelineConfig | None = None,
                   **overrides):
    """Step-wise pipeline on the C++ backend; mutates ``grid`` in place.

    All options come from ``config`` (a :class:`PipelineConfig`; defaults are
    used when omitted); keyword ``overrides`` update a copy of it, so
    ``generate_steps(grid, target, cfg, untangle_direct=False)`` reuses one
    config across call sites. Unknown override names raise.

    Yields ``(phase, info)`` after each unit of work so callers can animate the
    folded → untangled → smoothed transition or collect per-step convergence
    history. :func:`generate` drains this to do a batch run; the live demos
    drive it from a PyVista timer (see :func:`egg.io.visualize.animate_pipeline`).

    The TMOP phase always steps per chunk (``tmop_chunk`` sweeps each); set
    ``tmop_chunk == tmop_sweeps`` to run it in one direct call. Both the TMOP
    and untangle phases relax block-Jacobi over the halo-padded structured store
    (one merged double-buffered launch per sweep; ``omega`` is the TMOP SOR
    weight). The untangle
    phase by default runs **direct** — the whole δ-continuation
    in one ``cpp_untangle`` call (the analogue of TMOP chunk == sweeps), yielding
    a single ``untangle`` event. Pass ``untangle_direct=False`` to step per δ
    instead, yielding each δ so a live view can animate the unfolding. Either way
    X stays device-resident. ``target=None`` uses an isotropic quality target;
    untangling always runs on the isotropic context (geometric validity is
    target-independent and a strongly anisotropic target can stall the
    continuation).

    With ``pin_sweeps > 0``, after the TMOP phase each boundary-layer spec's
    first ``n_fixed`` layers are set to their exact geometric heights and
    pinned (:func:`egg.smoothing.respace_first_layers`), the sweep context is
    rebuilt so the pinned DOFs leave the update set, and ``pin_sweeps`` more
    TMOP sweeps re-equilibrate the free grid. With ``respace`` the exact wall
    respacing post-pass (:func:`egg.smoothing.enforce_boundary_layer_spacing`,
    nodes sliding along their smoothed columns) runs at the end instead.

    Phases: ``init`` (once) → ``untangle`` (per δ, only if folded) → ``tmop``
    (per chunk) → ``pin`` + ``tmop`` (with ``pin_sweeps``) → ``respace``
    (with ``respace``) → ``final`` (once, after the closing boundary snap).
    """
    from .smoothing.solver import build_sweep_context
    from .smoothing.targets import IdentityTarget
    from .smoothing.untangle import grid_min_det
    from .projection.project import project_nodes

    cfg = replace(config, **overrides) if config is not None \
        else PipelineConfig(**overrides)
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
                ctx_iso, grid, grid.global_nodes,
                sweeps_per_delta=cfg.sweeps_per_delta,
                delta0_factor=cfg.delta0_factor,
                shrink=cfg.untangle_shrink,
                max_outer=cfg.max_outer,
                device=device,
                margin=margin,
                omega=untangle_omega)
            _sync(grid, X_out)
            yield ("untangle", {"min_det": md, "converged": md > margin,
                                "direct": True, "outer_iters": outer_iters,
                                "delta": delta_final})
        else:
            from .smoothing.cpp_backend import (
                CppStructuredSweepSession, build_block_structured_context)
            bsc = build_block_structured_context(grid)
            session = CppStructuredSweepSession(
                ctx_iso, bsc, grid.global_nodes, device=device)
            delta = cfg.delta0_factor * max(abs(md), 1e-12)
            for it in range(cfg.max_outer):
                _e, mds = session.run(
                    cfg.sweeps_per_delta, phase="untangle", delta=delta,
                    omega=untangle_omega,
                    report_every=report_every,
                )
                md = float(np.asarray(mds)[-1])
                _sync(grid, session.get_X())
                converged = md > margin
                yield ("untangle", {"min_det": md, "delta": delta,
                                    "outer_iter": it + 1, "converged": converged})
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
        CppStructuredSweepSession, build_block_structured_context)
    bsc = build_block_structured_context(grid)
    session = CppStructuredSweepSession(
        tmop_ctx, bsc, grid.global_nodes, device=device)
    run_kwargs = {"omega": omega}
    done = 0
    while done < tmop_sweeps:
        k = min(tmop_chunk, tmop_sweeps - done)
        energies, _mds = session.run(k, report_every=report_every, **run_kwargs)
        done += k
        _sync(grid, session.get_X())
        yield ("tmop", {"energy": float(np.asarray(energies)[-1]),
                        "min_det": grid_min_det(grid.global_nodes, es),
                        "sweeps": done})
    project_nodes(grid, grid.dof_constraints)
    score_es = tmop_ctx.energy_stencil

    # --- Phase 3 (optional): pin the first n_fixed boundary layers exactly ---
    if cfg.pin_sweeps > 0:
        from .smoothing.respace import respace_first_layers

        pinned = respace_first_layers(grid, grid.topology)
        yield ("pin", {"n_dofs": int(pinned.size),
                       "min_det": grid_min_det(grid.global_nodes, es)})
        if pinned.size:
            # Rebuild the context so the pinned DOFs are compiled out of the
            # update set, then re-equilibrate the free grid (warm start).
            pin_ctx = build_sweep_context(
                grid, iso if target is None else target)
            session = CppStructuredSweepSession(
                pin_ctx, build_block_structured_context(grid),
                grid.global_nodes, device=device)
            total = max((cfg.pin_sweeps // tmop_chunk) * tmop_chunk, tmop_chunk)
            done = 0
            while done < total:
                k = min(tmop_chunk, total - done)
                energies, _mds = session.run(
                    k, report_every=report_every, **run_kwargs)
                done += k
                _sync(grid, session.get_X())
                yield ("tmop", {"energy": float(np.asarray(energies)[-1]),
                                "min_det": grid_min_det(grid.global_nodes, es),
                                "sweeps": done})
            project_nodes(grid, grid.dof_constraints)
            score_es = pin_ctx.energy_stencil

    # --- Optional exact wall-respacing post-pass ---
    if cfg.respace:
        from .smoothing.respace import enforce_boundary_layer_spacing

        enforce_boundary_layer_spacing(grid, grid.topology,
                                       straighten_columns=False)
        yield ("respace", {"min_det": grid_min_det(grid.global_nodes, es)})

    # Terminal summary AFTER the closing boundary snap, so the reported/plotted
    # final numbers reflect the snapped mesh. Energy is scored on the SAME
    # context TMOP optimised — for a BL/anisotropic target the isotropic
    # stencil reports a different (higher) objective on the same mesh.
    yield ("final", {"min_det": grid_min_det(grid.global_nodes, es),
                     "energy": _energy(grid, score_es)})


def _fmt_info(info: dict) -> str:
    parts = [f"{k}={v:.4e}" if isinstance(v, float) else f"{k}={v}"
             for k, v in info.items()]
    return " ".join(parts)


def drain_steps(steps, *, mindet_history=None, energy_history=None, verbose=True):
    """Run a :func:`generate_steps` generator to completion (headless).

    Optionally collects per-step ``min_det`` / ``energy`` into the supplied lists
    (for convergence plots) and prints one line per step.
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


def run_pipeline(grid, target=None, config: PipelineConfig | None = None, *,
                 mindet_history=None, energy_history=None, verbose=True,
                 **overrides):
    """Batch convenience: :func:`drain_steps` over :func:`generate_steps`.

    Options come from ``config`` (a :class:`PipelineConfig`) with keyword
    ``overrides`` applied on top, exactly as in :func:`generate_steps`
    (which mutates ``grid`` in place); the history lists collect per-step
    ``min_det`` / ``energy``. Use :func:`generate_steps` directly when you
    need to drive the steps yourself
    (e.g. :func:`egg.io.visualize.animate_pipeline`).
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
                grid, config.target_fn, config, untangle_direct=True):
            if phase == "init":
                report.add("init", min_det=info["min_det"])
            elif phase == "untangle":
                report.untangled = True
                report.untangle_converged = bool(info["converged"])
                report.add("untangle", min_det=info["min_det"],
                           converged=info["converged"])
            elif phase == "tmop":
                tmop_sweeps_done = info["sweeps"]
            elif phase == "final":
                report.final_min_det = info["min_det"]
                report.final_energy = info["energy"]
                report.add("tmop", min_det=info["min_det"],
                           energy=info["energy"], sweeps=tmop_sweeps_done)
    except KeyboardInterrupt:
        # Grid is synced after each step; report what we have.
        print(f"\n[interrupted after {tmop_sweeps_done} TMOP sweeps; "
              "progress preserved]")

    if config.verbose:
        print(f"[pipeline] final: min det A={report.final_min_det:.3e} "
              f"energy={report.final_energy:.4f}")

    grid.pipeline_report = report
    return grid
