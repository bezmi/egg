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
    tmop_smoother: Literal["jacobi", "fas", "control_point"] = "jacobi"
    tmop_metric: Literal["shape", "shape_size"] = "shape"
    fas_nu_pre: int = 2
    fas_nu_post: int = 2
    fas_nu_coarse: int = 32
    fas_max_levels: int = 32
    omega: float = 0.8
    report_every: int = 0

    # Control-point smoother knobs (tmop_smoother="control_point"): the shape
    # phase moves a coarse per-block B-spline control net instead of nodes
    # (egg.smoothing.control_topology / control_backend). The proven nodal
    # untangle still owns validity; control_presmooth nodal sweeps give the
    # net fit a smooth valid start. control_ratio is the cells-per-control
    # coarsening (finer nets track curved sliding walls; 2 is safe, 4 is
    # cheaper on smooth interiors). Seam C2/orthogonality dials mirror
    # run_control_topo.
    control_presmooth: int = 100
    # Smoother for the pre-fit nodal smooth: plain sweeps, FAS V-cycles at an
    # equivalent sweep budget, or "auto" — FAS once the grid is large enough
    # for the hierarchy to pay for itself (the fit only needs a smooth valid
    # start, and at scale a handful of V-cycles reaches a far better one in a
    # fraction of the sweeps' wall time).
    control_presmooth_smoother: str = "auto"
    control_ratio: int = 2
    control_max_outer: int = 30
    # Outer GN iterations per yielded "control" chunk (live views animate
    # per chunk; 0 = the whole phase in one call).
    control_chunk: int = 5
    control_c2_weight: float = 0.0
    control_ortho: str = "off"
    control_ortho_weight: float = 1.0
    # Extra knots inserted at every net axis end landing on a singular fan
    # (chain-wide so seams stay conforming): the fan window is C1-penalty-only
    # and a coarse net underfits the tight turning there. 0 disables.
    control_fan_refine: int = 2
    # Weight of the interior line-straightness rows (normal component of the
    # control polygon's second divided difference, frame frozen at the fit).
    # Grid lines can bow at near-zero shape-energy cost and a converged GN
    # step parks in whichever bowed state the initial fit seeded; a small
    # weight makes the straight member of that flat family the optimum.
    # Zero for collinear controls at any spacing, so clustering is untouched.
    control_smooth_weight: float = 10.0

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
        if self.tmop_smoother not in ("jacobi", "fas", "control_point"):
            raise ValueError(
                f"PipelineConfig.tmop_smoother must be 'jacobi', 'fas' or "
                f"'control_point', got {self.tmop_smoother!r}"
            )
        if self.tmop_smoother == "control_point":
            if self.interface_ortho is not None or self.interface_c2 is not None:
                raise ValueError(
                    "interface_ortho / interface_c2 are node-mode terms; the "
                    "control-point smoother expresses seam orthogonality and "
                    "curvature continuity directly on control legs — use "
                    "control_ortho / control_c2_weight instead"
                )
            if self.control_ortho not in ("off", "penalty", "hard"):
                raise ValueError(
                    "PipelineConfig.control_ortho must be 'off', 'penalty' or "
                    f"'hard', got {self.control_ortho!r}"
                )
            if self.control_presmooth_smoother not in ("auto", "jacobi", "fas"):
                raise ValueError(
                    "PipelineConfig.control_presmooth_smoother must be 'auto', "
                    f"'jacobi' or 'fas', got {self.control_presmooth_smoother!r}"
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

    if cfg.tmop_smoother == "control_point":
        # The shape phase moves the coarse control net instead of nodes.
        # Sequence (see egg/smoothing/control_topology.py): nodal pre-smooth
        # (the LSQ fit needs a valid, smooth start — fitting a rough grid can
        # fold the evaluated net), fit + reduced Gauss-Newton on the device
        # session (sliding CAD walls, exact C1 seams, fan fallback), then the
        # net is kept on the grid — refine/regrid becomes algebraic
        # re-evaluation.
        # Boundary-layer clustering: the control solve runs on the SAME
        # clustered target as node mode (chord-parameter fits keep the
        # clustered grid representable by the net), so the smoother shapes
        # the layer profile — and, crucially, the block interfaces — during
        # the solve. Exact first-layer heights stay the opt-in respace/pin
        # phases, same as node mode; `resample_block` remains available for
        # re-gridding the stored net after the fact.
        if cfg.control_presmooth > 0:
            n_nodes = int(np.asarray(grid.global_nodes).shape[0])
            use_fas = cfg.control_presmooth_smoother == "fas" or (
                cfg.control_presmooth_smoother == "auto" and n_nodes >= 50_000
            )
            if use_fas:
                # Equivalent-work budget: a V-cycle costs roughly
                # 4 x (nu_pre + nu_post) fine-sweep equivalents including the
                # coarse ladder, so the nominal presmooth budget converts to
                # a handful of cycles (floored at 2, capped at 10).
                cycles = max(
                    2,
                    min(
                        10,
                        cfg.control_presmooth
                        // (4 * (cfg.fas_nu_pre + cfg.fas_nu_post)),
                    ),
                )
                energies, mds = session.run_fas(
                    cycles,
                    nu_pre=cfg.fas_nu_pre,
                    nu_post=cfg.fas_nu_post,
                    nu_coarse=cfg.fas_nu_coarse,
                    max_levels=cfg.fas_max_levels,
                    **run_kwargs,
                )
                budget = {"cycles": cycles}
            else:
                energies, mds = session.run(
                    cfg.control_presmooth, report_every=report_every, **run_kwargs
                )
                budget = {"sweeps": cfg.control_presmooth}
            _sync(grid, session.get_X())
            yield (
                "tmop",
                {
                    "energy": float(np.asarray(energies)[-1]),
                    "min_det": float(np.asarray(mds)[-1]),
                    **budget,
                },
            )
        from .smoothing.control_backend import run_control_topo
        from .smoothing.control_topology import (
            build_control_topology,
            default_ctrl_shapes,
        )

        # fit_spacing="chord": the fit samples the grid's own (seam-
        # harmonized) parameter distribution, so a boundary-layer-clustered
        # initial grid stays representable by the net instead of pushing the
        # whole clustering into b (where the sliding-wall re-extension would
        # perturb hair-thin cells at their own scale).
        ctrl_topo = build_control_topology(
            grid,
            default_ctrl_shapes(grid, r=cfg.control_ratio),
            walls=True,
            fit_spacing="chord",
            fan_refine=cfg.control_fan_refine,
        )
        # On the grid from the fitted initial state onward: the chunk loop
        # mutates this same topology, so live views can stream the net as it
        # moves (the final state is the solved net).
        grid.control_net = ctrl_topo
        # Chunked so live views animate the control iterations; the session
        # stays resident across chunks (report["session"]).
        ctrl_kw = dict(
            topo=ctrl_topo,
            device=device,
            phase=tmop_phase,
            c2_weight=cfg.control_c2_weight,
            ortho=cfg.control_ortho,
            ortho_weight=cfg.control_ortho_weight,
            smooth_weight=cfg.control_smooth_weight,
        )
        ctrl_target = iso if target is None else target
        chunk_n = cfg.control_chunk if cfg.control_chunk > 0 else cfg.control_max_outer
        sess_c = None
        done_outer = 0
        ctrl_rep = None
        # The sliding-wall frame rebuilds keep the energy monotone within a
        # frame but not across rebuilds; on hard configs the loop can ratchet
        # (b re-extension injecting energy each frame). Track the best valid
        # chunk state (q, b, X) and restore it if the loop drifted past it.
        # Stall exit: two consecutive chunks without a meaningful energy
        # improvement end the phase — this covers both the ratchet (worse
        # chunks) and a flat optimum, where the GN gradient never reaches the
        # tolerance but chunks stop buying anything (the reduction-order
        # energy noise sits in the 4th-5th digit, hence the relative margin).
        best = None
        stalled_chunks = 0
        while done_outer < cfg.control_max_outer:
            k = min(chunk_n, cfg.control_max_outer - done_outer)
            ctrl_topo, ctrl_rep = run_control_topo(
                grid, ctrl_target, max_outer=k, session=sess_c, **ctrl_kw
            )
            sess_c = ctrl_rep["session"]
            done_outer += max(int(ctrl_rep["iters"]), 1)
            e_c = float(ctrl_rep["final_fine_energy"])
            md_c = float(ctrl_rep["final_mindet"])
            yield (
                "control",
                {
                    "energy": e_c,
                    "min_det": md_c,
                    "iters": done_outer,
                    "converged": bool(ctrl_rep["converged"]),
                    "frame_jumps": list(ctrl_rep.get("frame_jumps", [])),
                },
            )
            improved = md_c > 0.0 and (best is None or e_c < best[0] * (1.0 - 1e-5))
            if md_c > 0.0 and (best is None or e_c <= best[0]):
                best = (
                    e_c,
                    md_c,
                    ctrl_topo.q.copy(),
                    [np.array(b) for b in ctrl_topo.b_fields],
                    np.array(grid.global_nodes),
                )
            if improved:
                stalled_chunks = 0
            else:
                stalled_chunks += 1
                if stalled_chunks >= 2:
                    break
            if ctrl_rep["converged"] or ctrl_rep["iters"] == 0:
                break
        if best is not None and ctrl_rep is not None:
            e_last = float(ctrl_rep["final_fine_energy"])
            md_last = float(ctrl_rep["final_mindet"])
            if md_last <= 0.0 or e_last > best[0] * (1.0 + 1e-9):
                e_b, md_b, q_b, b_b, X_b = best
                ctrl_topo.q = q_b
                ctrl_topo.b_fields = b_b
                grid.global_nodes[:] = X_b
                for bi, block in enumerate(grid.blocks):
                    block.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]
                yield (
                    "control",
                    {
                        "energy": e_b,
                        "min_det": md_b,
                        "iters": done_outer,
                        "converged": False,
                        "restored": True,
                    },
                )
    else:
        done = 0
        while done < tmop_sweeps:
            k = min(tmop_chunk, tmop_sweeps - done)
            # The chunk-end min det comes from the device reduction (same as
            # the stepped untangle loop) — no host O(N) recompute per chunk.
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
    if cfg.tmop_smoother == "control_point" and md_now() <= 0.0:
        # The boundary snap can fold hair-thin near-wall cells when the
        # control state ended slightly off its frame (a reverted closing
        # rebuild). Recover with the proven nodal untangler (plain sweeps
        # cannot clear a fold) plus a short re-smooth — constrained nodes
        # stay on their entities, so the result is valid AND on-CAD (and
        # slightly off the stored net, which keeps the smooth shape).
        from .smoothing.cpp_backend import cpp_untangle

        X_out, _md_u, _oi, _df = cpp_untangle(
            ctx_iso,
            grid,
            grid.global_nodes,
            sweeps_per_delta=cfg.sweeps_per_delta,
            delta0_factor=cfg.delta0_factor,
            shrink=cfg.untangle_shrink,
            max_outer=cfg.max_outer,
            device=device,
            margin=margin,
            omega=0.5,
        )
        _sync(grid, X_out)
        project_nodes(grid, grid.dof_constraints)
        rec_sess = CppStructuredSweepSession(
            build_sweep_context(grid, iso if target is None else target),
            build_block_structured_context(grid),
            grid.global_nodes,
            device=device,
        )
        energies, mds = rec_sess.run(200, phase=tmop_phase, omega=omega)
        _sync(grid, rec_sess.get_X())
        project_nodes(grid, grid.dof_constraints)
        yield (
            "recover",
            {
                "energy": float(np.asarray(energies)[-1]),
                "min_det": md_now(),
            },
        )
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
