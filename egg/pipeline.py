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

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .core.types import MultiBlockGrid

__all__ = [
    "generate",
    "generate_steps",
    "drain",
    "PipelineConfig",
    "PipelineReport",
]


@dataclass
class PipelineConfig:
    """Per-phase caps, schedule params, and backend options for :func:`generate`."""

    target_fn: Callable[..., np.ndarray] | None = None  # default: IdentityTarget(d)
    metric: str = "shape_2d"

    # Validity gate / untangle schedule.
    margin: float = 1e-9
    untangle_sweeps_per_delta: int = 20
    untangle_delta0_factor: float = 2.0
    untangle_shrink: float = 0.5
    untangle_max_outer: int = 60

    # TMOP quality phase.
    tmop_sweeps: int = 40
    tmop_chunk: int = 10
    # TMOP smoother: "colored-gs" (global in-place chain) or "block-jacobi" (the
    # halo-padded structured store, one merged double-buffered launch per sweep).
    smoother: str = "colored-gs"
    omega: float = 1.0  # block-Jacobi SOR/damping weight (ignored by colored-gs)
    # report_every throttle for the resident session's energy/min-det reduction
    # (0 = chunk-end, 1 = per-sweep, k > 1 = every k-th plus final). 0 is the
    # lowest-launch default for the small-n GPU regime; set 1 for live plots
    # that animate per-sweep energy curves.
    report_every: int = 0

    # Backend.
    use_cpp: bool = True
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


def generate_steps(
    grid,
    target=None,
    *,
    margin: float = 1e-9,
    sweeps_per_delta: int = 20,
    delta0_factor: float = 2.0,
    untangle_shrink: float = 0.5,
    max_outer: int = 60,
    untangle_direct: bool = True,
    tmop_sweeps: int = 40,
    tmop_chunk: int = 10,
    smoother: str = "colored-gs",
    omega: float = 1.0,
    report_every: int = 0,
    device: str = "cpu",
):
    """Step-wise pipeline on the C++ backend; mutates ``grid`` in place.

    Yields ``(phase, info)`` after each unit of work so callers can animate the
    folded → untangled → smoothed transition or collect per-step convergence
    history. :func:`generate` drains this to do a batch run; the Phase-5 demos
    drive it from a PyVista timer (see :func:`egg.io.visualize.animate_pipeline`).

    The TMOP phase always steps per chunk (``tmop_chunk`` sweeps each); set
    ``tmop_chunk == tmop_sweeps`` to run it in one direct call. ``smoother``
    selects the TMOP relaxation: ``"colored-gs"`` (default, the global in-place
    colour chain) or ``"block-jacobi"`` (the halo-padded structured store, one
    merged double-buffered launch per sweep, SOR weight ``omega``) — block-Jacobi
    converges to the same minimiser but needs more sweeps, so raise
    ``tmop_sweeps``. Only the TMOP phase honours ``smoother``; the untangle phase
    always runs the global colored-GS / ``cpp_untangle`` path. The untangle
    phase mirrors this: by default it runs **direct** — the whole δ-continuation
    in one ``cpp_untangle`` call (the analogue of TMOP chunk == sweeps), yielding
    a single ``untangle`` event. Pass ``untangle_direct=False`` to step per δ
    instead, yielding each δ so a live view can animate the unfolding. Either way
    X stays device-resident. ``target=None`` uses an isotropic quality target;
    untangling always runs on the isotropic context (geometric validity is
    target-independent and a strongly anisotropic target can stall the
    continuation).

    Phases: ``init`` (once) → ``untangle`` (per δ, only if folded) → ``tmop``
    (per chunk) → ``final`` (once, after the closing boundary snap).
    """
    from .smoothing.solver import build_sweep_context
    from .smoothing.targets import IdentityTarget
    from .smoothing.cpp_backend import CppSweepSession
    from .smoothing.untangle import grid_min_det
    from .projection.project import project_nodes

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
        if untangle_direct:
            from .smoothing.cpp_backend import cpp_untangle
            X_out, md, outer_iters, delta_final = cpp_untangle(
                ctx_iso, grid.global_nodes,
                sweeps_per_delta=sweeps_per_delta,
                delta0_factor=delta0_factor,
                shrink=untangle_shrink,
                max_outer=max_outer,
                device=device,
                margin=margin)
            _sync(grid, X_out)
            yield ("untangle", {"min_det": md, "converged": md > margin,
                                "direct": True, "outer_iters": outer_iters,
                                "delta": delta_final})
        else:
            session = CppSweepSession(ctx_iso, grid.global_nodes, device=device)
            delta = delta0_factor * max(abs(md), 1e-12)
            for it in range(max_outer):
                _e, mds = session.run(
                    sweeps_per_delta, phase="untangle", delta=delta,
                    report_every=report_every,
                )
                md = float(np.asarray(mds)[-1])
                _sync(grid, session.get_X())
                converged = md > margin
                yield ("untangle", {"min_det": md, "delta": delta,
                                    "outer_iter": it + 1, "converged": converged})
                if converged:
                    break
                delta *= untangle_shrink
        # Re-snap boundary DOFs after untangling moved them.
        project_nodes(grid, grid.dof_constraints)

    # --- Phase 2: TMOP quality optimisation (resident session, per-chunk loop) ---
    tmop_ctx = ctx_iso if target is None else build_sweep_context(grid, target)
    tmop_sweeps = max((tmop_sweeps // tmop_chunk) * tmop_chunk, tmop_chunk)
    # The default colored-GS keeps the global in-place session (unchanged). The
    # block-Jacobi smoother needs the halo-padded structured store, built from the
    # grid's BlockTopology; it runs one merged double-buffered launch per sweep and
    # converges to the same minimiser (more sweeps — bump tmop_sweeps accordingly).
    if smoother == "block-jacobi":
        from .smoothing.cpp_backend import (
            CppStructuredSweepSession, build_block_structured_context)
        bsc = build_block_structured_context(grid)
        session = CppStructuredSweepSession(
            tmop_ctx, bsc, grid.global_nodes, device=device)
        run_kwargs = {"smoother": "block-jacobi", "omega": omega}
    elif smoother == "colored-gs":
        session = CppSweepSession(tmop_ctx, grid.global_nodes, device=device)
        run_kwargs = {}
    else:
        raise ValueError(
            f"smoother must be 'colored-gs' or 'block-jacobi', got {smoother!r}")
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

    # Terminal summary AFTER the closing boundary snap, so the reported/plotted
    # final numbers reflect the snapped mesh. Energy is scored on the SAME
    # context TMOP optimised (tmop_ctx) — for a BL/anisotropic target the
    # isotropic stencil reports a different (higher) objective on the same mesh.
    yield ("final", {"min_det": grid_min_det(grid.global_nodes, es),
                     "energy": _energy(grid, tmop_ctx.energy_stencil)})


def _fmt_info(info: dict) -> str:
    parts = [f"{k}={v:.4e}" if isinstance(v, float) else f"{k}={v}"
             for k, v in info.items()]
    return " ".join(parts)


def drain(steps, *, mindet_history=None, energy_history=None, verbose=True):
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


def generate(topology, config: PipelineConfig | None = None) -> MultiBlockGrid:
    """Run the full pipeline and return the optimised :class:`MultiBlockGrid`.

    The returned grid carries a :class:`PipelineReport` at ``grid.pipeline_report``.
    """
    config = config or PipelineConfig()
    report = PipelineReport()

    # 1. TFI init.
    grid = topology.initialize_grid()

    if config.use_cpp:
        # Batch-drive the step-wise generator: untangle runs DIRECT (one
        # cpp_untangle call — the analogue of TMOP chunk == sweeps), TMOP steps
        # per chunk. generate_steps mutates the grid and does the boundary snaps.
        tmop_sweeps_done = 0
        try:
            for phase, info in generate_steps(
                    grid, config.target_fn,
                    margin=config.margin,
                    sweeps_per_delta=config.untangle_sweeps_per_delta,
                    delta0_factor=config.untangle_delta0_factor,
                    untangle_shrink=config.untangle_shrink,
                    max_outer=config.untangle_max_outer,
                    untangle_direct=True,
                    tmop_sweeps=config.tmop_sweeps,
                    tmop_chunk=config.tmop_chunk,
                    smoother=config.smoother,
                    omega=config.omega,
                    report_every=config.report_every,
                    device=config.device):
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
    else:
        # NumPy reference TMOP (untangle still runs on the C++ backend). Kept as a
        # separate path since generate_steps is C++-only.
        from .smoothing.solver import build_sweep_context, local_relaxation_sweep
        from .smoothing.targets import IdentityTarget
        from .smoothing.untangle import untangle
        from .projection.project import project_nodes

        d = topology.d
        target = config.target_fn or IdentityTarget(d)
        project_nodes(grid, grid.dof_constraints)
        ctx = build_sweep_context(grid, target)
        es = ctx.energy_stencil
        report.add("init", min_det=_min_det(grid, es), energy=_energy(grid, es))

        if _min_det(grid, es) <= config.margin:
            untangle_ctx = (ctx if config.target_fn is None
                            else build_sweep_context(grid, IdentityTarget(d)))
            res = untangle(
                grid, untangle_ctx, shrink=config.untangle_shrink,
                margin=config.margin,
                sweeps_per_delta=config.untangle_sweeps_per_delta,
                delta0_factor=config.untangle_delta0_factor,
                max_outer=config.untangle_max_outer, verbose=config.verbose)
            report.untangled = True
            report.untangle_converged = res.converged
            report.add("untangle", min_det=res.min_det, converged=res.converged,
                       delta=res.delta, outer_iters=res.outer_iters)
            project_nodes(grid, grid.dof_constraints)

        for _ in range(config.tmop_sweeps):
            local_relaxation_sweep(grid, target, config.metric, ctx)
        project_nodes(grid, grid.dof_constraints)
        report.final_min_det = _min_det(grid, es)
        report.final_energy = _energy(grid, es)
        report.add("tmop", min_det=report.final_min_det,
                   energy=report.final_energy, sweeps=config.tmop_sweeps)

    if config.verbose:
        print(f"[pipeline] final: min det A={report.final_min_det:.3e} "
              f"energy={report.final_energy:.4f}")

    grid.pipeline_report = report
    return grid
