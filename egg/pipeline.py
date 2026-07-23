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

The pipeline is a list of stages run in order over one shared state
(:class:`MeshState`). Each stage owns only its own parameters:
:class:`Untangle`, one smoother (:class:`JacobiSmoother`, :class:`FasSmoother`
or :class:`ControlPointSmoother`), an optional :class:`Presmooth` prep pass
before the control-point smoother, and the optional :class:`Pin` /
:class:`Respace` clean-up stages (both of which take the node smoother to use).
Compose them and pass the list to :func:`run_pipeline` / :func:`generate_steps`::

    for ev in generate_steps(grid, stages=[Untangle(), JacobiSmoother()]):
        ...

:class:`PipelineConfig` is the older single-object form. It still works but is
deprecated: it translates to the equivalent stage list and warns. Runs on the
C++ backend (barrier sweep + delta-untangle sweep via ``egg._cpp.cpp_core``).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, TypedDict

import numpy as np

from collections.abc import Mapping

from .core.types import MultiBlockGrid
from .enums import (
    ControlOrtho,
    Device,
    PresmoothSmoother,
    TmopMetric,
    TmopSmoother,
    coerce_enum,
    enum_errors,
)
from .errors import EggValidationError
from .smoothing.config_types import (
    Directional,
    FasParams,
    InterfaceC2,
    InterfaceOrtho,
    config_errors,
)

if TYPE_CHECKING:
    from .smoothing.control_topology import ControlTopology
    from .smoothing.solver import SweepContext

__all__ = [
    "generate",
    "generate_steps",
    "drain_steps",
    "run_pipeline",
    "PipelineConfig",
    "PipelineReport",
    "MeshState",
    "Stage",
    "Untangle",
    "JacobiSmoother",
    "FasSmoother",
    "Presmooth",
    "ControlPointSmoother",
    "Pin",
    "Respace",
    "Refit",
    "Resample",
    "Save",
    "validate",
]


class _SharedSmootherKwargs(TypedDict):
    """The smoother constructor arguments the deprecated config path shares
    across every smoother it builds; splatted into each so the type checker
    validates them against the constructor signatures."""

    metric: TmopMetric
    cluster_boundary_layers: bool
    bl_blend_neighbours: bool
    bl_interior_spacing: float | None
    directional: Directional | Mapping[str, object] | bool | None


class _ControlRunKwargs(TypedDict):
    """Per-chunk arguments the control-point smoother forwards to
    ``run_control_topo``; typed so the splat is checked at the call site."""

    topo: "ControlTopology"
    device: Device
    phase: str
    c2_weight: float
    ortho: ControlOrtho
    ortho_weight: float
    smooth_weight: float
    directional: dict[str, object] | None


@dataclass
class PipelineConfig:
    """Deprecated single-object pipeline configuration.

    Kept for backward compatibility. Construct a stage list instead
    (:class:`Untangle`, a smoother, :class:`Pin`, :class:`Respace`) and pass it
    as ``stages=`` to :func:`run_pipeline` / :func:`generate_steps`. Using this
    class (or the matching keyword overrides) still runs, but it builds the
    equivalent stage list and emits a ``DeprecationWarning``. Field meanings
    are unchanged from before the stage split.

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
    tmop_smoother: TmopSmoother = TmopSmoother.JACOBI
    tmop_metric: TmopMetric = TmopMetric.SHAPE
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
    control_presmooth_smoother: PresmoothSmoother = PresmoothSmoother.AUTO
    control_ratio: int = 2
    control_max_outer: int = 30
    # Outer GN iterations per yielded "control" chunk (live views animate
    # per chunk; 0 = the whole phase in one call).
    control_chunk: int = 5
    control_c2_weight: float = 0.0
    control_ortho: ControlOrtho = ControlOrtho.OFF
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
    interface_ortho: InterfaceOrtho | Mapping[str, object] | None = None

    # Optional block-interface C2 curvature term (2D). When set, the TMOP phase
    # adds the curvature-continuity term over grid-line windows (see
    # egg.smoothing.interface_c2): a dict of kwargs forwarded to
    # curvature_windows, e.g. {"weight": 0.0, "iface_boost": 20.0}. Concentrates
    # curvature continuity across block seams; None leaves it off.
    interface_c2: InterfaceC2 | Mapping[str, object] | None = None

    # Directional soft-energy terms over declared parallel chains and fan
    # frames (see egg.smoothing.directional). None (the default) enables the
    # terms automatically whenever the topology declares them, with unit
    # weights; a dict of kwargs is forwarded to build_directional_samples
    # (lambda_parallel/lambda_line/lambda_stem/energy_scale — 0 disables that
    # term); False forces the terms off. References re-freeze from the current
    # node positions at every phase start.
    directional: Directional | Mapping[str, object] | bool | None = None

    device: Device = Device.CPU
    verbose: bool = False

    def validate(self) -> None:
        """Reject invalid knobs up front — a typo must fail before the untangle
        phase spends minutes, not after. Normalizes string knobs to their enum."""
        # coerce_enum validates and returns the enum member; store it so the rest
        # of the pipeline sees a normalized value.
        self.tmop_smoother = coerce_enum(
            TmopSmoother, self.tmop_smoother, "PipelineConfig.tmop_smoother"
        )
        self.tmop_metric = coerce_enum(
            TmopMetric, self.tmop_metric, "PipelineConfig.tmop_metric"
        )
        self.control_ortho = coerce_enum(
            ControlOrtho, self.control_ortho, "PipelineConfig.control_ortho"
        )
        self.control_presmooth_smoother = coerce_enum(
            PresmoothSmoother,
            self.control_presmooth_smoother,
            "PipelineConfig.control_presmooth_smoother",
        )
        self.device = coerce_enum(Device, self.device, "PipelineConfig.device")
        if self.tmop_smoother == TmopSmoother.CONTROL_POINT and (
            self.interface_ortho is not None or self.interface_c2 is not None
        ):
            raise EggValidationError(
                "interface_ortho / interface_c2 are node-mode terms; the "
                "control-point smoother expresses seam orthogonality and "
                "curvature continuity directly on control legs — use "
                "control_ortho / control_c2_weight instead"
            )
        # Coercion validates (ranges, unknown keys); discard the result here.
        InterfaceOrtho.coerce(self.interface_ortho)
        InterfaceC2.coerce(self.interface_c2)
        if not isinstance(self.directional, bool):
            Directional.coerce(self.directional)

    def to_stages(self) -> list["Stage"]:
        """Build the equivalent stage list and warn.

        The translation is exact: the same untangle schedule, smoother, and
        optional pin/respace phases the old flat config produced. New code
        should compose stages directly instead.
        """
        self.validate()
        warnings.warn(
            "The PipelineConfig / keyword pipeline API is deprecated; compose "
            "the pipeline from stages (Untangle, a smoother, Pin, Respace) and "
            "pass stages=[...] to run_pipeline / generate_steps.",
            DeprecationWarning,
            stacklevel=3,
        )
        untangle = Untangle(
            margin=self.margin,
            sweeps_per_delta=self.sweeps_per_delta,
            delta0_factor=self.delta0_factor,
            shrink=self.untangle_shrink,
            max_outer=self.max_outer,
            direct=self.untangle_direct,
            report_every=self.report_every,
        )
        fas_params = FasParams(
            nu_pre=self.fas_nu_pre,
            nu_post=self.fas_nu_post,
            nu_coarse=self.fas_nu_coarse,
            max_levels=self.fas_max_levels,
        )
        obj: _SharedSmootherKwargs = {
            "metric": self.tmop_metric,
            "cluster_boundary_layers": self.cluster_boundary_layers,
            "bl_blend_neighbours": self.bl_blend_neighbours,
            "bl_interior_spacing": self.bl_interior_spacing,
            "directional": self.directional,
        }
        stages: list[Stage] = [untangle]
        if self.tmop_smoother == "control_point":
            # The old control smoother pre-smoothed internally; that is now an
            # explicit Presmooth stage in front of it. "auto" maps to a plain
            # Jacobi pre-pass (the run-time FAS-at-scale choice is not available
            # without the grid, so new code should compose Presmooth directly).
            if self.control_presmooth > 0:
                if self.control_presmooth_smoother == "fas":
                    nu = self.fas_nu_pre + self.fas_nu_post
                    pre = FasSmoother(
                        sweeps=max(2, min(10, self.control_presmooth // (4 * nu))),
                        chunk=self.tmop_chunk,
                        omega=self.omega,
                        report_every=self.report_every,
                        fas_params=fas_params,
                        **obj,
                    )
                else:
                    pre = JacobiSmoother(
                        sweeps=self.control_presmooth,
                        chunk=self.control_presmooth,
                        omega=self.omega,
                        report_every=self.report_every,
                        **obj,
                    )
                stages.append(Presmooth(pre))
            stages.append(
                ControlPointSmoother(
                    omega=self.omega,
                    report_every=self.report_every,
                    chunk=self.tmop_chunk,
                    fas_params=fas_params,
                    ratio=self.control_ratio,
                    max_outer=self.control_max_outer,
                    control_chunk=self.control_chunk,
                    c2_weight=self.control_c2_weight,
                    ortho=self.control_ortho,
                    ortho_weight=self.control_ortho_weight,
                    fan_refine=self.control_fan_refine,
                    smooth_weight=self.control_smooth_weight,
                    **obj,
                )
            )
        elif self.tmop_smoother == "fas":
            stages.append(
                FasSmoother(
                    sweeps=self.tmop_sweeps,
                    chunk=self.tmop_chunk,
                    omega=self.omega,
                    report_every=self.report_every,
                    fas_params=fas_params,
                    interface_ortho=self.interface_ortho,
                    interface_c2=self.interface_c2,
                    **obj,
                )
            )
        else:
            stages.append(
                JacobiSmoother(
                    sweeps=self.tmop_sweeps,
                    chunk=self.tmop_chunk,
                    omega=self.omega,
                    report_every=self.report_every,
                    interface_ortho=self.interface_ortho,
                    interface_c2=self.interface_c2,
                    **obj,
                )
            )
        if self.pin_sweeps > 0:
            # The pin re-smooth uses the same nodal kind as the main phase
            # (jacobi for control mode, matching the old behaviour).
            if self.tmop_smoother == "fas":
                pin_sm = FasSmoother(
                    sweeps=self.pin_sweeps,
                    chunk=self.tmop_chunk,
                    omega=self.omega,
                    report_every=self.report_every,
                    fas_params=fas_params,
                    **obj,
                )
            else:
                pin_sm = JacobiSmoother(
                    sweeps=self.pin_sweeps,
                    chunk=self.tmop_chunk,
                    omega=self.omega,
                    report_every=self.report_every,
                    **obj,
                )
            stages.append(Pin(pin_sm))
        if self.respace:
            stages.append(Respace())
        return stages


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


def _refit_net(grid, existing, ratio: int):
    """Fit a control net to the grid's CURRENT coordinates (read only on nodes).

    When ``existing`` is a control net, the refit first tries to keep its
    control counts and knots, so the stored net keeps the same
    resolution-independent structure and only its state is re-fitted to the
    pinned/respaced grid. If that fit folds the new grid (the pinned near-wall
    spacing can be sharper than the pre-pin net was built for), it falls back
    to a fresh fit whose knots follow the new grid. Returns the fitted
    topology, or ``None`` when no fold-free fit exists (the grid stays valid
    and carries no net).
    """
    from .smoothing.control_fit import ControlFitError, _net_min_det, fit_control_net

    try:
        if existing is not None:
            from .smoothing.control_topology import build_control_topology

            knots = [
                [np.asarray(u, dtype=float) for u in cm.knots] for cm in existing.cmaps
            ]
            topo = build_control_topology(
                grid,
                existing.ctrl_shapes,
                walls=bool(existing.wall_faces),
                fit_spacing=existing.axis_params,
                knots=knots,
            )
            if _net_min_det(topo) > 0.0:
                return topo
        return fit_control_net(grid, ratio=ratio)
    except ControlFitError:
        return None


class MeshState:
    """Shared, mutable state threaded through the pipeline stages.

    A stage reads the grid (and whatever objective the previous stage set up),
    moves nodes or the control net, and records what it produced for the stages
    after it. This is a plain shared object run through the stage list in
    order, not a streaming buffer.
    """

    def __init__(
        self,
        grid: MultiBlockGrid,
        *,
        device: Device | str = Device.CPU,
        verbose: bool = False,
        target: Callable[..., np.ndarray] | None = None,
    ) -> None:
        from .smoothing.solver import build_sweep_context
        from .smoothing.targets import IdentityTarget

        self.grid = grid
        self.device = coerce_enum(Device, device, "device")
        self.verbose = verbose
        self.d = grid.topology.d
        self.iso = IdentityTarget(self.d)
        # initialize_grid step 3 already projected every associated boundary
        # DOF onto its entity, so the incoming grid needs no boundary snap
        # here; the isotropic context is the untangle / min-det reference.
        self.ctx_iso = build_sweep_context(grid, self.iso)
        self.es = self.ctx_iso.energy_stencil

        # Explicit user target (wins over any auto-build); None lets a smoother
        # build its own.
        self.target: Callable[..., np.ndarray] | None = target

        # The objective the last smoother established. pin / respace reuse it
        # so they optimise and score the same energy, and the final summary is
        # scored on it.
        self.active_target: Callable[..., np.ndarray] | None = None
        self.interface_ortho: InterfaceOrtho | None = None
        self.interface_c2: InterfaceC2 | None = None
        self.directional: dict[str, object] | None = None
        self.score_ctx: SweepContext = self.ctx_iso
        self.score_metric: str = "shape_2d"

        # Control-net state, kept faithful to the output by a later refit.
        self.control_net: ControlTopology | None = None
        # Set by a Resample stage: the grid starts from a loaded net (a solved
        # shape), so a following smoother polishes it instead of solving cold.
        self.warm: bool = False
        self.net_stale: bool = False

    def md_now(self) -> float:
        from .smoothing.untangle import grid_min_det

        if self.d == 2:
            return grid_min_det(self.grid.global_nodes, self.es)
        return _cpp_reduce(self.grid, self.ctx_iso, self.device, phase="barrier")[1]

    def energy_now(self, score_ctx, metric: str) -> float:
        if self.d == 2:
            return _energy(self.grid, score_ctx.energy_stencil, metric=metric)
        phase = "shape_size" if metric == "shape_size" else "barrier"
        return _cpp_reduce(self.grid, score_ctx, self.device, phase=phase)[0]


def _directional_config(directional) -> dict | None:
    """The directional-samples config dict (pipeline defaults merged with the
    caller's keys), or ``None`` when disabled (``False``) or absent (``None``).

    The pipeline's default scale: the directional penalties compete with the
    shape energy's own preference (a singular fan's 72-degree sectors, a rail's
    resistance to transverse cell shear), so unit weights barely move anything.
    200 straightens a framed five-fan to ~1 degree and pulls a marked wedge
    rail to a few degrees while the accept rule keeps the term soft
    (energy-monotone, barrier intact). The chain fairness term rides at the
    same lambda: orientation-only, it rounds block-corner kinks on a marked
    rail and damps the short-wavelength ripples a pure parallelism pull can
    leave, without hurting alignment. User keys override; the per-declaration
    ``"weight"`` tunes a single chain.
    """
    if directional is False:
        return None
    if directional is None or directional is True:
        return asdict(Directional())
    cfg = Directional.coerce(directional)
    assert cfg is not None  # directional is neither None nor a bool here
    return asdict(cfg)


def _establish_objective(
    s: MeshState,
    *,
    metric,
    cluster,
    blend,
    interior,
    io,
    ic2,
    directional=None,
):
    """Fix the TMOP target + sweep context a smoother (and pin/respace) uses.

    Mirrors the old auto-build: an explicit ``s.target`` wins; otherwise a
    boundary-layer clustering target is built from the topology specs, or a
    mean-size target under the size metric, or the plain isotropic context.
    Records the objective on ``s`` so later phases score the same energy.

    ``directional``: None enables the directional soft-energy terms whenever
    the topology declares parallel chains or fan frames (unit weights), a dict
    is kwargs for ``build_directional_samples``, False forces them off. The
    frozen references are rebuilt from the grid's CURRENT node positions here,
    at every phase start — the same staleness contract as ``interface_ortho``.
    """
    from .smoothing.solver import build_sweep_context

    target = s.target
    if target is None:
        specs = getattr(s.grid.topology, "boundary_layer_specs", {})
        if cluster and specs:
            from .smoothing.targets import build_topology_target

            target = build_topology_target(
                s.grid.topology,
                s.grid,
                metric=metric,
                blend_neighbours=blend,
                interior_spacing=interior,
            )
        elif metric == "shape_size":
            from .smoothing.targets import mean_size_target

            target = mean_size_target(s.grid)
        # else (shape metric, no clustering) stays None and reuses ctx_iso.
    dir_cfg = _directional_config(directional)
    dir_samples = None
    if dir_cfg is not None:
        from .smoothing.directional import build_directional_samples

        dir_samples = build_directional_samples(s.grid, **dir_cfg)
    if target is None and io is None and ic2 is None and dir_samples is None:
        ctx = s.ctx_iso
    else:
        ctx = build_sweep_context(
            s.grid,
            s.iso if target is None else target,
            interface_ortho=io,
            interface_c2=ic2,
            directional=dir_samples,
        )
    s.active_target = target
    s.interface_ortho = io
    s.interface_c2 = ic2
    s.directional = dir_cfg if dir_samples is not None else None
    s.score_ctx = ctx
    s.score_metric = "shape_size" if metric == "shape_size" else "shape_2d"
    return target, ctx


def _unencodable_entity_errors(grid: MultiBlockGrid) -> list[str]:
    """Report any constraint geometry the device cannot encode, up front.

    Every distinct entity a DOF slides on must encode to the device SoA layout
    (and stay within the compiled degree caps). Encoding fails late otherwise,
    at wire-build time inside the first stage; check it here instead.
    """
    from egg.geometry.entity_soa import encode_entity_soa

    d = grid.topology.d
    errs: list[str] = []
    seen: set[int] = set()
    for entity in grid.dof_constraints.values():
        if entity is None or id(entity) in seen:
            continue
        seen.add(id(entity))
        try:
            encode_entity_soa(entity, d=d)
        except NotImplementedError as e:
            errs.append(f"constraint geometry cannot be used: {e}")
    return errs


def _scalar_errors(who: str, sweeps: object, chunk: object, omega: object) -> list[str]:
    """Range checks shared by the smoother stages."""
    errs: list[str] = []
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 0:
        errs.append(f"{who} sweeps must be a non-negative int, got {sweeps!r}")
    if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
        errs.append(f"{who} chunk must be an int >= 1, got {chunk!r}")
    if isinstance(omega, bool) or not isinstance(omega, (int, float)) or omega <= 0:
        errs.append(f"{who} omega must be a positive number, got {omega!r}")
    return errs


# Capability tokens for validate(). "grid" = a built grid; "smoothed" = a
# quality-optimised grid; "net" = a control net faithful to the grid. A stage
# declares what must be present before it (``requires``) and what it leaves
# behind (``produces``); the capability chain is the first compatibility check.
class Stage:
    """One pipeline step. ``run`` yields the same ``(phase, info)`` events the
    old inline phases did."""

    requires: frozenset = frozenset({"grid"})
    produces: frozenset = frozenset()
    # True for the stages that do the TMOP quality smoothing.
    is_smoother: bool = False

    def validate(self, pipeline: "list[Stage]", index: int) -> None:
        """Reject this stage given the others and their order.

        Called by :func:`validate` for each stage, so a stage can check what it
        sits next to and refuse an order it cannot run in (beyond the
        ``requires`` / ``produces`` capability chain). ``pipeline`` is the whole
        stage list and ``index`` is this stage's position in it. The default
        adds no rule of its own.
        """

    def check(self) -> list[str]:
        """Return this stage's own config problems as messages (empty if valid).

        Called for every stage at the start of a run so the pipeline reports all
        bad values at once and aborts before any solving. Building a stage never
        validates, so a live-reloaded script stays quiet while a value is typed.
        """
        return []

    def check_grid(self, grid: MultiBlockGrid) -> list[str]:
        """Config problems that depend on the grid (empty if valid).

        Runs alongside :meth:`check` in the mandatory validation stage, for
        rules that need the grid the run will use (e.g. a term that only works
        in 2D). The default has none.
        """
        return []

    def normalize(self) -> None:
        """Replace this stage's raw inputs with typed config objects.

        Called once, after :meth:`check` passes, so the rest of the run works
        with enums and config dataclasses instead of the strings/dicts the user
        may have passed. Coercion cannot fail here because ``check`` already
        accepted the values.
        """

    def run(self, s: MeshState) -> Iterator[tuple[str, dict]]:
        raise NotImplementedError
        yield  # pragma: no cover


class Untangle(Stage):
    """Clear folds with delta-continuation (only runs if the grid is folded).

    Relaxes a shifted barrier energy and shrinks the shift until every cell is
    valid. ``direct`` runs the whole continuation in one C++ call (one event);
    otherwise it steps per delta level so a live view can animate the
    unfolding.
    """

    requires = frozenset({"grid"})
    produces = frozenset({"grid"})

    def validate(self, pipeline, index):
        # Untangling clears folds so the smoother has a valid start; running it
        # after a smoother is meaningless (and the smoother cannot smooth a
        # folded grid). So no smoother may come before Untangle.
        if any(st.is_smoother for st in pipeline[:index]):
            raise ValueError(
                "Untangle must come before the smoother — it prepares a valid "
                "grid for it; move Untangle to the front of the pipeline"
            )

    def __init__(
        self,
        *,
        margin: float = 1e-9,
        sweeps_per_delta: int = 20,
        delta0_factor: float = 2.0,
        shrink: float = 0.8,
        max_outer: int = 60,
        direct: bool = True,
        report_every: int = 0,
    ) -> None:
        self.margin = margin
        self.sweeps_per_delta = sweeps_per_delta
        self.delta0_factor = delta0_factor
        self.shrink = shrink
        self.max_outer = max_outer
        self.direct = direct
        self.report_every = report_every

    def check(self) -> list[str]:
        errs: list[str] = []
        if (
            isinstance(self.sweeps_per_delta, bool)
            or not isinstance(self.sweeps_per_delta, int)
            or self.sweeps_per_delta < 1
        ):
            errs.append(
                f"Untangle sweeps_per_delta must be an int >= 1, "
                f"got {self.sweeps_per_delta!r}"
            )
        if (
            isinstance(self.max_outer, bool)
            or not isinstance(self.max_outer, int)
            or self.max_outer < 1
        ):
            errs.append(f"Untangle max_outer must be an int >= 1, got {self.max_outer!r}")
        if not isinstance(self.shrink, (int, float)) or not 0.0 < self.shrink < 1.0:
            errs.append(f"Untangle shrink must be in (0, 1), got {self.shrink!r}")
        if not isinstance(self.delta0_factor, (int, float)) or self.delta0_factor <= 0:
            errs.append(
                f"Untangle delta0_factor must be a positive number, "
                f"got {self.delta0_factor!r}"
            )
        return errs

    def run(self, s: MeshState):
        from .projection.project import project_nodes

        md = s.md_now()
        if md > self.margin:
            return  # valid start: nothing to do, no event (self-skip)
        grid, device = s.grid, s.device
        # Undamped simultaneous updates overshoot the barrier; damp block-Jacobi.
        untangle_omega = 0.5
        if self.direct:
            from .smoothing.cpp_backend import cpp_untangle

            X_out, md, outer_iters, delta_final = cpp_untangle(
                s.ctx_iso,
                grid,
                grid.global_nodes,
                sweeps_per_delta=self.sweeps_per_delta,
                delta0_factor=self.delta0_factor,
                shrink=self.shrink,
                max_outer=self.max_outer,
                device=device,
                margin=self.margin,
                omega=untangle_omega,
            )
            _sync(grid, X_out)
            yield (
                "untangle",
                {
                    "min_det": md,
                    "converged": md > self.margin,
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
                s.ctx_iso, bsc, grid.global_nodes, device=device
            )
            delta = self.delta0_factor * max(abs(md), 1e-12)
            for it in range(self.max_outer):
                _e, mds = session.run(
                    self.sweeps_per_delta,
                    phase="untangle",
                    delta=delta,
                    omega=untangle_omega,
                    report_every=self.report_every,
                )
                md = float(np.asarray(mds)[-1])
                _sync(grid, session.get_X())
                converged = md > self.margin
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
                delta *= self.shrink
        # Re-snap boundary DOFs after untangling moved them.
        project_nodes(grid, grid.dof_constraints)


class _NodalSmoother(Stage):
    """Shared body of the node-moving TMOP smoothers (Jacobi and FAS).

    Builds the objective (target + sweep context), runs the sweeps per chunk,
    and re-snaps the boundary. Subclasses set ``nodal_kind``.
    """

    requires = frozenset({"grid"})
    produces = frozenset({"grid", "smoothed"})
    is_smoother = True
    nodal_kind = "jacobi"

    def __init__(
        self,
        *,
        sweeps: int = 40,
        chunk: int = 10,
        omega: float = 0.8,
        report_every: int = 0,
        metric: TmopMetric = TmopMetric.SHAPE,
        cluster_boundary_layers: bool = True,
        bl_blend_neighbours: bool = True,
        bl_interior_spacing: float | None = None,
        fas_params: FasParams | Mapping[str, object] | None = None,
        interface_ortho: InterfaceOrtho | Mapping[str, object] | None = None,
        interface_c2: InterfaceC2 | Mapping[str, object] | None = None,
        directional: Directional | Mapping[str, object] | bool | None = None,
    ) -> None:
        # Store inputs as given; validation runs together at pipeline start (see
        # Stage.check / _run_stages), so building a stage never raises. This keeps
        # a live-reloaded script quiet while a value is being typed.
        self.sweeps = sweeps
        self.chunk = chunk
        self.omega = omega
        self.report_every = report_every
        self.metric = metric
        self.cluster_boundary_layers = cluster_boundary_layers
        self.bl_blend_neighbours = bl_blend_neighbours
        self.bl_interior_spacing = bl_interior_spacing
        self.fas_params = fas_params
        self.interface_ortho = interface_ortho
        self.interface_c2 = interface_c2
        self.directional = directional

    def check(self) -> list[str]:
        """Config problems, gathered so the pipeline reports them all at once."""
        who = type(self).__name__
        errs = _scalar_errors(who, self.sweeps, self.chunk, self.omega)
        errs += enum_errors(TmopMetric, self.metric, f"{who} metric")
        errs += config_errors(FasParams, self.fas_params, f"{who} fas_params")
        errs += config_errors(InterfaceOrtho, self.interface_ortho, f"{who} interface_ortho")
        errs += config_errors(InterfaceC2, self.interface_c2, f"{who} interface_c2")
        if not isinstance(self.directional, bool):
            errs += config_errors(Directional, self.directional, f"{who} directional")
        return errs

    def check_grid(self, grid: MultiBlockGrid) -> list[str]:
        who = type(self).__name__
        errs: list[str] = []
        if grid.topology.d != 2:
            if self.interface_ortho is not None:
                errs.append(f"{who} interface_ortho is a 2D-only term (grid is 3D)")
            if self.interface_c2 is not None:
                errs.append(f"{who} interface_c2 is a 2D-only term (grid is 3D)")
        return errs

    def normalize(self) -> None:
        self.metric = coerce_enum(TmopMetric, self.metric, "metric")
        self.fas_params = FasParams.coerce(self.fas_params) or FasParams()
        self.interface_ortho = InterfaceOrtho.coerce(self.interface_ortho)
        self.interface_c2 = InterfaceC2.coerce(self.interface_c2)
        if not isinstance(self.directional, bool):
            self.directional = Directional.coerce(self.directional)

    def run_chunk(self, session, k, phase):
        """Run ``k`` sweeps (Jacobi) or V-cycles (FAS) on ``session``.

        The unit of ``k`` depends on ``nodal_kind``. Shared by this smoother's
        own loop and by :class:`Pin` (which reuses the smoother it was given).
        """
        kw = {"omega": self.omega, "phase": phase}
        if self.nodal_kind == "fas":
            fp = FasParams.coerce(self.fas_params) or FasParams()
            return session.run_fas(k, **asdict(fp), **kw)
        return session.run(k, report_every=self.report_every, **kw)

    def run(self, s: MeshState):
        from .projection.project import project_nodes
        from .smoothing.cpp_backend import (
            CppStructuredSweepSession,
            build_block_structured_context,
        )

        _target, tmop_ctx = _establish_objective(
            s,
            metric=self.metric,
            cluster=self.cluster_boundary_layers,
            blend=self.bl_blend_neighbours,
            interior=self.bl_interior_spacing,
            io=self.interface_ortho,
            ic2=self.interface_c2,
            directional=self.directional,
        )
        phase = "shape_size" if self.metric == "shape_size" else "barrier"
        total = max((self.sweeps // self.chunk) * self.chunk, self.chunk)
        session = CppStructuredSweepSession(
            tmop_ctx,
            build_block_structured_context(s.grid),
            s.grid.global_nodes,
            device=s.device,
        )
        done = 0
        prev_e = None
        while done < total:
            k = min(self.chunk, total - done)
            energies, mds = self.run_chunk(session, k, phase)
            done += k
            _sync(s.grid, session.get_X())
            e = float(np.asarray(energies)[-1])
            yield (
                "tmop",
                {
                    "energy": e,
                    "min_det": float(np.asarray(mds)[-1]),
                    "sweeps": done,
                },
            )
            # A warm start (a Resample stage seeded the grid from a saved net)
            # begins near a good grid, so the polish stops once a chunk stops
            # lowering the energy by more than a small fraction, instead of
            # burning the whole sweep budget. A cold run keeps its full budget
            # (this exit never fires cold).
            if s.warm and prev_e is not None and e > prev_e * (1.0 - 1e-3):
                break
            prev_e = e
        project_nodes(s.grid, s.grid.dof_constraints)


class JacobiSmoother(_NodalSmoother):
    """TMOP shape smoothing by damped block-Jacobi node updates."""

    nodal_kind = "jacobi"


class FasSmoother(_NodalSmoother):
    """TMOP shape smoothing by FAS (nonlinear geometric multigrid) V-cycles.

    ``sweeps`` / ``chunk`` count V-cycles here. Blocks that admit no coarse
    level fall back to plain Jacobi at the equivalent fine-sweep budget.
    """

    nodal_kind = "fas"


class Presmooth(Stage):
    """A nodal pre-pass that prepares the grid for the control-point smoother.

    The control-net fit needs a smooth, valid grid to start from, so a short
    node smoothing runs first. Pass the smoother to use for it (a
    :class:`JacobiSmoother` or :class:`FasSmoother`). This is a preparation
    step, not the quality smoother, so it must sit before a
    :class:`ControlPointSmoother`. Putting it before a nodal smoother is an
    error: that smoother already smooths from scratch, so the pre-pass would
    just be a second, redundant node smooth.
    """

    requires = frozenset({"grid"})
    produces = frozenset({"grid"})

    def __init__(self, smoother):
        if not isinstance(smoother, _NodalSmoother):
            raise TypeError(
                "Presmooth takes a JacobiSmoother or FasSmoother, got "
                f"{type(smoother).__name__}"
            )
        self.smoother = smoother

    def validate(self, pipeline, index):
        after = [st for st in pipeline[index + 1 :] if st.is_smoother]
        if not after:
            raise ValueError(
                "Presmooth must come before a control-point smoother (it "
                "prepares the grid for the net fit)"
            )
        if not isinstance(after[0], ControlPointSmoother):
            raise ValueError(
                "Presmooth before a jacobi/fas smoother is redundant — that "
                "smoother already smooths the grid from scratch; a pre-pass "
                "only helps the control-point smoother"
            )

    def check(self) -> list[str]:
        return self.smoother.check()

    def check_grid(self, grid: MultiBlockGrid) -> list[str]:
        return self.smoother.check_grid(grid)

    def normalize(self) -> None:
        self.smoother.normalize()

    def run(self, s: MeshState):
        # A warm start (a Resample stage seeded the grid from a saved net) already
        # holds a smooth grid, so there is nothing to prepare.
        if s.warm and getattr(s, "control_net", None) is not None:
            return
        # The nodal smoother does the whole pre-pass (build objective, sweep,
        # re-snap). The control-point smoother re-establishes its own objective
        # afterwards, so composing matching metric/clustering keeps them aligned.
        yield from self.smoother.run(s)


class ControlPointSmoother(Stage):
    """TMOP shape smoothing that moves a coarse per-block B-spline control net.

    Fit + reduced Gauss-Newton on the device session (sliding CAD walls, exact
    C1 seams, fan fallback), then the net is kept on the grid so refine / regrid
    is algebraic re-evaluation. The fit needs a smooth, valid start, so put a
    :class:`Presmooth` before this stage. Rejects the node-mode interface terms;
    expresses seam orthogonality / curvature on control legs instead.
    """

    requires = frozenset({"grid"})
    produces = frozenset({"grid", "smoothed", "net"})
    is_smoother = True

    def __init__(
        self,
        *,
        omega: float = 0.8,
        report_every: int = 0,
        chunk: int = 10,
        fas_params: FasParams | Mapping[str, object] | None = None,
        metric: TmopMetric = TmopMetric.SHAPE,
        cluster_boundary_layers: bool = True,
        bl_blend_neighbours: bool = True,
        bl_interior_spacing: float | None = None,
        ratio: int = 2,
        max_outer: int = 30,
        control_chunk: int = 5,
        c2_weight: float = 0.0,
        ortho: ControlOrtho = ControlOrtho.OFF,
        ortho_weight: float = 1.0,
        fan_refine: int = 2,
        smooth_weight: float = 10.0,
        directional: Directional | Mapping[str, object] | bool | None = None,
    ) -> None:
        # Store inputs as given; validated together at pipeline start (check).
        self.omega = omega
        self.report_every = report_every
        self.chunk = chunk
        self.fas_params = fas_params
        self.metric = metric
        self.cluster_boundary_layers = cluster_boundary_layers
        self.bl_blend_neighbours = bl_blend_neighbours
        self.bl_interior_spacing = bl_interior_spacing
        self.ratio = ratio
        self.max_outer = max_outer
        self.control_chunk = control_chunk
        self.c2_weight = c2_weight
        self.ortho = ortho
        self.ortho_weight = ortho_weight
        self.fan_refine = fan_refine
        self.smooth_weight = smooth_weight
        self.directional = directional

    def check(self) -> list[str]:
        who = "ControlPointSmoother"
        errs = enum_errors(TmopMetric, self.metric, f"{who} metric")
        errs += enum_errors(ControlOrtho, self.ortho, f"{who} ortho")
        errs += config_errors(FasParams, self.fas_params, f"{who} fas_params")
        if not isinstance(self.directional, bool):
            errs += config_errors(Directional, self.directional, f"{who} directional")
        if isinstance(self.chunk, bool) or not isinstance(self.chunk, int) or self.chunk < 1:
            errs.append(f"{who} chunk must be an int >= 1, got {self.chunk!r}")
        return errs

    def normalize(self) -> None:
        self.metric = coerce_enum(TmopMetric, self.metric, "metric")
        self.ortho = coerce_enum(ControlOrtho, self.ortho, "ortho")
        self.fas_params = FasParams.coerce(self.fas_params) or FasParams()
        if not isinstance(self.directional, bool):
            self.directional = Directional.coerce(self.directional)

    def run(self, s: MeshState):
        from .projection.project import project_nodes
        from .smoothing.cpp_backend import (
            CppStructuredSweepSession,
            build_block_structured_context,
        )

        grid, device = s.grid, s.device
        target, _ctx = _establish_objective(
            s,
            metric=self.metric,
            cluster=self.cluster_boundary_layers,
            blend=self.bl_blend_neighbours,
            interior=self.bl_interior_spacing,
            io=None,
            ic2=None,
            directional=self.directional,
        )
        tmop_phase = "shape_size" if self.metric == "shape_size" else "barrier"
        # A warm start (a Resample stage seeded the grid from a saved net) already
        # holds a solved, smooth grid, so the fresh fit is skipped and the
        # loaded net is polished directly. A cold run expects a Presmooth before
        # it to have prepared a smooth, valid grid for the fit.
        #
        # Boundary-layer clustering: the control solve runs on the SAME
        # clustered target as node mode (chord-parameter fits keep the
        # clustered grid representable by the net), so the smoother shapes the
        # layer profile (and the block interfaces) during the solve. Exact
        # first-layer heights stay the opt-in pin/respace phases.
        warm = s.warm and s.control_net is not None

        from .smoothing.control_backend import run_control_topo

        if warm:
            # Polish the loaded net in place (the chunk loop self-limits from a
            # near-optimal start).
            assert s.control_net is not None  # implied by warm
            ctrl_topo = s.control_net
        else:
            from .smoothing.control_topology import (
                build_control_topology,
                default_ctrl_shapes,
            )

            # fit_spacing="chord": the fit samples the grid's own (seam-
            # harmonized) parameter distribution, so a clustered initial grid
            # stays representable by the net instead of pushing the clustering
            # into b.
            ctrl_topo = build_control_topology(
                grid,
                default_ctrl_shapes(grid, r=self.ratio),
                walls=True,
                fit_spacing="chord",
                fan_refine=self.fan_refine,
            )
        # The chunk loop mutates this same topology, so live views can stream
        # the net as it moves (the final state is the solved net).
        grid.control_net = ctrl_topo
        s.control_net = ctrl_topo
        ctrl_kw: _ControlRunKwargs = {
            "topo": ctrl_topo,
            "device": device,
            "phase": tmop_phase,
            "c2_weight": self.c2_weight,
            "ortho": self.ortho,
            "ortho_weight": self.ortho_weight,
            "smooth_weight": self.smooth_weight,
            "directional": s.directional,
        }
        ctrl_target = s.iso if target is None else target
        chunk_n = self.control_chunk if self.control_chunk > 0 else self.max_outer
        sess_c = None
        done_outer = 0
        ctrl_rep = None
        # The sliding-wall frame rebuilds keep the energy monotone within a
        # frame but not across rebuilds; on hard configs the loop can ratchet
        # (b re-extension injecting energy each frame). Track the best valid
        # chunk state (q, b, X) and restore it if the loop drifted past it.
        # Stall exit: two consecutive chunks without a meaningful energy
        # improvement end the phase (covers the ratchet and a flat optimum,
        # where the GN gradient never reaches tolerance but chunks stop buying
        # anything; the reduction-order energy noise sits in the 4th-5th digit,
        # hence the relative margin).
        best = None
        stalled_chunks = 0
        while done_outer < self.max_outer:
            k = min(chunk_n, self.max_outer - done_outer)
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
        project_nodes(grid, grid.dof_constraints)
        if s.md_now() <= 0.0:
            # The boundary snap can fold hair-thin near-wall cells when the
            # control state ended slightly off its frame (a reverted closing
            # rebuild). Recover with the proven nodal untangler (plain sweeps
            # cannot clear a fold) plus a short re-smooth: constrained nodes
            # stay on their entities, so the result is valid AND on-CAD (and
            # slightly off the stored net, which keeps the smooth shape).
            from .smoothing.cpp_backend import cpp_untangle
            from .smoothing.solver import build_sweep_context

            X_out, _md_u, _oi, _df = cpp_untangle(
                s.ctx_iso,
                grid,
                grid.global_nodes,
                sweeps_per_delta=20,
                delta0_factor=2.0,
                shrink=0.8,
                max_outer=60,
                device=device,
                margin=1e-9,
                omega=0.5,
            )
            _sync(grid, X_out)
            project_nodes(grid, grid.dof_constraints)
            rec_sess = CppStructuredSweepSession(
                build_sweep_context(grid, s.iso if target is None else target),
                build_block_structured_context(grid),
                grid.global_nodes,
                device=device,
            )
            energies, mds = rec_sess.run(200, phase=tmop_phase, omega=self.omega)
            _sync(grid, rec_sess.get_X())
            project_nodes(grid, grid.dof_constraints)
            s.net_stale = True
            yield (
                "recover",
                {
                    "energy": float(np.asarray(energies)[-1]),
                    "min_det": s.md_now(),
                },
            )


class Pin(Stage):
    """Pin the first boundary layers to exact heights, then re-equilibrate.

    Sets each boundary-layer spec's first ``n_fixed`` layers to their exact
    geometric heights and freezes them, rebuilds the sweep context so the
    pinned DOFs leave the update set, and re-smooths the free grid. ``smoother``
    is the node smoother (a :class:`JacobiSmoother` or :class:`FasSmoother`) to
    run for that re-smooth, so the pin phase is not tied to the main smoother's
    kind. It runs under the objective the main smoother established.

    ``Pin()`` (no smoother) stamps and freezes the layers WITHOUT any
    re-smooth: the layers are placed at their exact per-column heights and
    left, with nothing after them re-disturbing the grid. The stamp itself is
    per-column exact (``respace_first_layers`` slides each column's rows along
    its own polyline), so this is the reliable way to land absolute layer
    heights; the far cells above the band keep whatever the smoother left.
    """

    requires = frozenset({"smoothed"})
    produces = frozenset({"smoothed"})

    def __init__(self, smoother=None):
        if smoother is not None and not isinstance(smoother, _NodalSmoother):
            raise TypeError(
                "Pin takes a JacobiSmoother or FasSmoother for the re-smooth "
                f"(or None for a stamp-only pin), got {type(smoother).__name__}"
            )
        self.smoother = smoother

    def check(self) -> list[str]:
        return self.smoother.check() if self.smoother is not None else []

    def check_grid(self, grid: MultiBlockGrid) -> list[str]:
        return self.smoother.check_grid(grid) if self.smoother is not None else []

    def normalize(self) -> None:
        if self.smoother is not None:
            self.smoother.normalize()

    def run(self, s: MeshState):
        from .projection.project import project_nodes
        from .smoothing.cpp_backend import (
            CppStructuredSweepSession,
            build_block_structured_context,
        )
        from .smoothing.respace import respace_first_layers
        from .smoothing.solver import build_sweep_context

        pinned = respace_first_layers(s.grid, s.grid.topology)
        yield ("pin", {"n_dofs": int(pinned.size), "min_det": s.md_now()})
        if not pinned.size:
            return
        if self.smoother is None:
            # Stamp-only: the layers are at their exact heights and frozen;
            # leave the rest of the grid exactly as the main smoother left it.
            project_nodes(s.grid, s.grid.dof_constraints)
            s.net_stale = True
            return
        sm = self.smoother
        # The pin re-smooth uses ITS OWN smoother's composed terms, not the
        # main smoother's: whatever interface_ortho / interface_c2 / directional
        # the given smoother carries is applied; a term left unset (None) on it
        # is simply off (no fall-back to what the main phase used). The target
        # stays the main smoother's so the re-smooth scores the same energy
        # above the frozen band, and directional references re-freeze from the
        # pinned grid's current node positions.
        pin_dir = (
            None if sm.directional is None else _directional_config(sm.directional)
        )
        pin_ctx = build_sweep_context(
            s.grid,
            s.iso if s.active_target is None else s.active_target,
            interface_ortho=sm.interface_ortho,
            interface_c2=sm.interface_c2,
            directional=pin_dir,
            frozen_interface=pinned,
        )
        session = CppStructuredSweepSession(
            pin_ctx,
            build_block_structured_context(s.grid),
            s.grid.global_nodes,
            device=s.device,
        )
        phase = "shape_size" if s.score_metric == "shape_size" else "barrier"
        total = max((sm.sweeps // sm.chunk) * sm.chunk, sm.chunk)
        done = 0
        while done < total:
            k = min(sm.chunk, total - done)
            energies, mds = sm.run_chunk(session, k, phase)
            done += k
            _sync(s.grid, session.get_X())
            yield (
                "tmop",
                {
                    "energy": float(np.asarray(energies)[-1]),
                    "min_det": float(np.asarray(mds)[-1]),
                    "sweeps": done,
                },
            )
        project_nodes(s.grid, s.grid.dof_constraints)
        s.score_ctx = pin_ctx
        s.net_stale = True


class Respace(Stage):
    """Exact wall-respacing post-pass (nodes slide along their columns)."""

    requires = frozenset({"smoothed"})
    produces = frozenset({"smoothed"})

    def run(self, s: MeshState):
        from .smoothing.respace import enforce_boundary_layer_spacing

        enforce_boundary_layer_spacing(
            s.grid, s.grid.topology, straighten_columns=False
        )
        s.net_stale = True
        yield ("respace", {"min_det": s.md_now()})


class Refit(Stage):
    """Fit the control net to the FINAL grid so it represents the output.

    After pin/respace move nodes off the control net, ``grid.control_net`` no
    longer matches the grid. This stage re-fits it (read only on the nodes: it
    stores the net alongside the grid, it does NOT write the net back, which
    would undo the pin cleanup). A control solve with no later node motion is
    already faithful and is kept as is. A fit that folds declines to ``None``
    (the grid stays valid and simply carries no net).
    """

    requires = frozenset({"smoothed"})
    produces = frozenset({"net"})

    def __init__(self, *, ratio=4):
        self.ratio = ratio

    def run(self, s: MeshState):
        existing = getattr(s.grid, "control_net", None)
        if existing is not None and not s.net_stale:
            # The live net already equals the grid (control solve, no pin /
            # respace / recover after it); keep it.
            yield (
                "refit",
                {
                    "min_det": s.md_now(),
                    "declined": False,
                    "kept": True,
                    "fit_residual": float(existing.fit_residual),
                },
            )
            return
        fitted = _refit_net(s.grid, existing, self.ratio)
        s.grid.control_net = fitted
        s.control_net = fitted
        s.net_stale = False
        yield (
            "refit",
            {
                "min_det": s.md_now(),
                "declined": fitted is None,
                "kept": False,
                "fit_residual": (
                    float(fitted.fit_residual) if fitted is not None else float("nan")
                ),
            },
        )


class Resample(Stage):
    """Sample a saved (or in-memory) control net onto the grid — no re-solve.

    The net is a resolution-independent map; this stage evaluates it at the
    grid's node parameters, so the mesh is the net re-sampled, not a fresh
    solve. Where the net comes from:

    - ``path`` given (the usual regrid): load the saved net (``.npz``). If the
      grid matches the saved sampling it is an exact restore; if the grid was
      built at a DIFFERENT resolution (same topology) the net re-tabulates onto
      it (:func:`retabulate_control_net`).
    - ``path=None``: re-sample the net already on the grid
      (``grid.control_net``, e.g. straight after a control solve).

    The sampling distribution — clustering is just a *mode* of resampling, not a
    separate step, since both cases evaluate the same map:

    - default: uniform in parameter (the resolution regrid);
    - ``cluster=True``: the boundary-layer clustering from the topology's
      ``set_boundary_layer`` specs (or the ``specs`` override), so a grid
      rebuilt with a new ``first_height`` clusters to it — the clustering
      regrid;
    - ``spacing=...``: a fully explicit per-block list of per-axis parameter
      arrays.

    So ``Resample(net, cluster=True)`` does the resolution regrid (onto whatever
    grid you built) AND the clustering regrid in one shot. The re-evaluated net
    stays faithful to the grid (``b`` and the stored parameters are updated), so
    a following :class:`Save` / :class:`Refit` serialises it. The run is marked a
    warm start so a following smoother polishes rather than solving cold.

    When the net carries an exact-layer residual (saved with
    ``Save(..., exact=True)``), it is re-interpolated to the final sampling and
    added on top as a deflection map (validity-gated), so the actual saved grid
    — including any pin / respace / nodal post-processing the net alone cannot
    represent — is reproduced (exactly at the original sampling, an
    interpolation at a new one). A missing file or a net whose topology does not
    fit the grid is a loud error.
    """

    produces = frozenset({"grid", "smoothed", "net"})

    def __init__(self, path=None, *, cluster=False, spacing=None, specs=None):
        self.path = path
        self.cluster = cluster
        self.spacing = spacing
        self.specs = specs
        # Loading from disk only needs a grid; re-sampling an in-memory net
        # needs that net to already be present.
        self.requires = frozenset({"grid"}) if path is not None else frozenset({"net"})

    def run(self, s: MeshState):
        from .io import retabulate_control_net, try_load_control_net
        from .projection.project import project_nodes

        grid = s.grid
        exact_restore = None
        if self.path is not None:
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"Resample: no saved net at {self.path!r}")
            # Same sampling: exact restore. Otherwise re-tabulate onto this
            # grid's resolution (the regrid path).
            topo = try_load_control_net(grid, self.path)
            exact_restore = topo is not None
            if topo is None:
                topo = retabulate_control_net(grid, self.path)
            if topo is None:
                raise ValueError(
                    f"Resample: saved net {self.path!r} does not fit this grid's "
                    "topology (different blocks / seams / fans, or the "
                    "re-tabulated net folds) — it can only be loaded onto the "
                    "same topology it was solved on"
                )
            grid.control_net = topo
            s.control_net = topo
        else:
            topo = getattr(grid, "control_net", None)
            if topo is None:
                raise ValueError(
                    "Resample without a path needs a control net on the grid — "
                    "load one with Resample(path=...) or run a control smoother "
                    "first"
                )

        clustered = self.cluster or self.spacing is not None or self.specs is not None
        if clustered:
            self._resample_clustered(s, topo)
        else:
            topo.write_to_grid()
        # Add the exact-layer residual (the post-processing the net cannot
        # represent) at the final sampling, when present and still valid.
        residual = topo.apply_residual()
        project_nodes(grid, grid.dof_constraints)
        s.warm = True
        yield (
            "resample",
            {
                "min_det": s.md_now(),
                "path": self.path,
                "exact_restore": exact_restore,
                "cluster": bool(clustered),
                "residual": residual,
                "fit_residual": float(topo.fit_residual),
            },
        )

    def _resample_clustered(self, s: MeshState, topo):
        """Re-evaluate every block at a clustered / explicit node distribution.

        The node COUNTS stay the same; only the per-axis parameter distribution
        moves (clustered axes toward their wall). Keeps ``b`` and the stored
        parameters faithful to the reclustered grid.
        """
        from .smoothing.control_topology import (
            cluster_spacings,
            resample_block,
            wall_b_field,
        )

        grid, d = s.grid, s.d
        if self.spacing is not None:
            spacings = dict(self.spacing)
        else:
            specs = self.specs
            if specs is None:
                specs = getattr(grid.topology, "boundary_layer_specs", {})
            spacings = cluster_spacings(topo, specs) if specs else {}
        for bi in range(len(grid.blocks)):
            node_shape = tuple(grid.blocks[bi].logical_shape)
            sp = spacings.get(bi, [None] * d)
            # Axes with no clustering keep the net's own stored parameters, so
            # only the clustered (wall-normal) axes actually move.
            params = [
                np.asarray(sp[k], dtype=float)
                if sp[k] is not None
                else np.asarray(topo.axis_params[bi][k], dtype=float)
                for k in range(d)
            ]
            Xb = resample_block(topo, bi, node_shape=node_shape, spacing=params)
            grid.blocks[bi].nodes[...] = Xb
            grid.global_nodes[grid.block_dof_maps[bi]] = Xb
            topo.b_fields[bi] = wall_b_field(topo, bi, Xb)
            topo.axis_params[bi] = params


class Save(Stage):
    """Serialise the grid's control net to ``path`` (``.npz``) — the write side.

    The mirror of :class:`Resample`. Writes the faithful ``grid.control_net`` (put
    there by the control-point smoother or a :class:`Refit`), so a later run can
    :class:`Resample` it and regrid. With ``exact=True`` it also stores the residual
    layer (per-node ``grid − net_eval``), making a same-resolution reload
    bit-for-bit exact. Declines silently when there is no net to save (a fit
    that folded), leaving the valid grid untouched.
    """

    requires = frozenset({"net"})
    produces = frozenset({"net"})

    def __init__(self, path, *, exact=False):
        self.path = path
        self.exact = exact

    def run(self, s: MeshState):
        net = getattr(s.grid, "control_net", None)
        if net is None:
            yield ("save", {"path": self.path, "saved": False})
            return
        from .io import save_control_net

        save_control_net(
            net, self.path, residual=s.grid.global_nodes if self.exact else None
        )
        yield ("save", {"path": self.path, "saved": True})


def validate(stages) -> None:
    """Check a stage list is well-ordered before it runs.

    Two compatibility checks per stage:

    - the capability chain: walking the list tracking which capabilities are
      present (starting from the built ``grid``), a stage whose ``requires`` are
      not yet available raises (pinning before smoothing, saving with no net);
    - each stage's own :meth:`Stage.validate`, which sees the whole list and its
      position, so it can refuse an order or a neighbour it cannot run with.

    Either way a bad composition fails up front instead of part way through a
    long run.
    """
    have = {"grid"}
    for i, st in enumerate(stages):
        st.validate(stages, i)
        missing = set(st.requires) - have
        if missing:
            raise ValueError(
                f"{type(st).__name__} needs {sorted(missing)} but the pipeline "
                f"has only {sorted(have)} at that point; reorder the stages"
            )
        have |= set(st.produces)


class ConfigValidationStage(Stage):
    """Mandatory first stage: validate the whole run before anything solves.

    Runs ahead of every real stage (not when a stage is built), so a
    live-reloaded script never raises mid-edit. It checks stage order, gathers
    every stage's config problems and the grid's validity into one message, and
    aborts if anything is wrong. It never touches the grid and yields no events.
    """

    def __init__(self, stages: "list[Stage]") -> None:
        self.stages = stages

    def run(self, s: MeshState) -> Iterator[tuple[str, dict]]:
        validate(self.stages)  # stage order / capability chain
        errs: list[str] = []
        for st in self.stages:
            errs += st.check()
            errs += st.check_grid(s.grid)
        if not np.all(np.isfinite(s.grid.global_nodes)):
            errs.append(
                "grid has non-finite node coordinates; check the topology, "
                "geometry constraints, and target before smoothing"
            )
        errs += _unencodable_entity_errors(s.grid)
        if errs:
            raise EggValidationError(
                "invalid pipeline configuration:\n  " + "\n  ".join(errs)
            )
        # Inputs are valid: normalize every stage so the run works with typed
        # config objects (enums, dataclasses) instead of raw strings/dicts.
        for st in self.stages:
            st.normalize()
        return
        yield  # pragma: no cover  (never yields)


def _run_stages(state: MeshState, stages):
    """Emit ``init``, run every stage, then the ``final`` summary."""
    stages = list(stages)
    # The mandatory validation stage runs first and aborts on any bad input.
    yield from ConfigValidationStage(stages).run(state)
    yield ("init", {"min_det": state.md_now()})
    for stage in stages:
        yield from stage.run(state)
    # Final summary AFTER the closing boundary snap, scored on the SAME context
    # and metric the smoother optimised (a BL/anisotropic target reports a
    # different objective than the isotropic stencil on the same mesh).
    yield (
        "final",
        {
            "min_det": state.md_now(),
            "energy": state.energy_now(state.score_ctx, state.score_metric),
        },
    )


def generate_steps(
    grid,
    target=None,
    config: PipelineConfig | None = None,
    *,
    stages=None,
    device: Device | None = None,
    verbose: bool = False,
    **overrides,
):
    """Step-wise pipeline on the C++ backend; mutates ``grid`` in place.

    Two ways to drive it:

    - **stages** (preferred): ``generate_steps(grid, stages=[Untangle(),
      JacobiSmoother(), ...])``. Each stage owns its own parameters. Persist a
      net with a :class:`Save` stage and reload / regrid it with
      :class:`Resample` (``cluster=True`` to re-cluster in the same step).
    - **config / keywords** (deprecated): ``generate_steps(grid,
      config=PipelineConfig(...))`` or ``generate_steps(grid,
      tmop_sweeps=20)``. This translates to the equivalent stage list and
      warns.

    Parameters
    ----------
    grid : MultiBlockGrid
        Initialised grid (``topology.initialize_grid()``); mutated in place.
    target : callable, optional
        Explicit TMOP target function. ``None`` (default) lets the smoother
        auto-build one from the topology; passing a target suppresses the
        auto-build. If you build the target yourself you MUST construct it with
        the smoother's metric so the non-clustered far field stays
        scale-consistent (the auto-build handles this for you). Untangling
        always runs on the isotropic context regardless.
    stages : list of Stage, optional
        The stage list to run. Mutually exclusive with ``config`` / keyword
        overrides.
    device : str, optional
        ``"cpu"`` / ``"gpu"`` / ``"auto"`` for the ``stages=`` path (the
        config path takes it from the config).

    Yields
    ------
    (phase, info) : (str, dict)
        After each unit of work, so callers can animate the folded ->
        untangled -> smoothed transition or collect convergence history.
        Phase order: ``init`` (once) -> ``untangle`` (only if folded) ->
        ``tmop`` / ``control`` (per chunk) -> ``pin`` + ``tmop`` (with a Pin
        stage) -> ``respace`` (with a Respace stage) -> ``final`` (once, after
        the closing boundary snap). :func:`generate` drains this for a batch
        run; the live demos drive it from a PyVista timer.
    """
    if stages is not None:
        if config is not None or overrides:
            raise TypeError(
                "pass either stages=... or config=/keyword overrides, not both"
            )
        state = MeshState(
            grid,
            device=device if device is not None else Device.CPU,
            verbose=verbose,
            target=target,
        )
        yield from _run_stages(state, list(stages))
        return

    cfg = (
        replace(config, **overrides)
        if config is not None
        else PipelineConfig(**overrides)
    )
    built_stages = cfg.to_stages()
    state = MeshState(grid, device=cfg.device, verbose=cfg.verbose, target=target)
    yield from _run_stages(state, built_stages)


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
    stages=None,
    device: Device | None = None,
    mindet_history=None,
    energy_history=None,
    verbose=True,
    **overrides,
):
    """Batch convenience: :func:`drain_steps` over :func:`generate_steps`.

    Same options and in-place ``grid`` mutation as :func:`generate_steps`;
    the history lists collect per-step ``min_det`` / ``energy``. Use
    :func:`generate_steps` directly to drive the steps yourself.
    """
    drain_steps(
        generate_steps(
            grid,
            target,
            config,
            stages=stages,
            device=device,
            verbose=verbose,
            **overrides,
        ),
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
    # call), TMOP steps per chunk. generate_steps mutates the grid.
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
