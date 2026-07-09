# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Argument parsing for the phoebus capsule example (see ``phoebus.py``)."""

import argparse


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Phoebus capsule grid")
    p.add_argument(
        "--plot-grid", action="store_true", help="matplotlib final wireframe grid"
    )
    p.add_argument(
        "--plot-topology",
        action="store_true",
        help="Plot the declared topology only — no pipeline run",
    )
    p.add_argument(
        "--grid-level",
        type=int,
        default=1,
        help="grid refinement factor (as in the Lua example)",
    )
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--tmop-sweeps", type=int, default=5000)
    p.add_argument(
        "--smoother",
        choices=["jacobi", "fas"],
        default="jacobi",
        help="TMOP-phase smoother: plain block-Jacobi sweeps or FAS (nonlinear "
        "geometric multigrid) V-cycles; with fas, --tmop-sweeps/--chunk count "
        "V-cycles (4 fine sweeps each plus coarse-grid work)",
    )
    p.add_argument(
        "--metric",
        choices=["shape", "shape_size"],
        default="shape_size",
        help="TMOP objective: shape+size (default; adds (det T - 1)^2 "
        "against a mean-cell-size target — pushes cell areas toward "
        "uniform) or the classic scale-invariant shape metric",
    )
    p.add_argument("--chunk", type=int, default=1000)
    p.add_argument(
        "--omega",
        type=float,
        default=0.8,
        help="block-Jacobi SOR/damping weight for the TMOP phase",
    )
    p.add_argument(
        "--pin-layers",
        type=int,
        default=2,
        help="0: off (respace post-pass instead). TMOP with the "
        "aspect-ratio target, fix this many near-wall layers at their "
        "exact geometric heights (n_fixed in set_boundary_layer), then "
        "re-run TMOP with them pinned",
    )
    p.add_argument(
        "--pin-sweeps",
        type=int,
        default=5000,
        help="TMOP sweeps for the pinned re-run",
    )
    p.add_argument(
        "--c2-weight",
        type=float,
        default=0.0,
        help="block-interface C2 curvature-continuity weight (0 = off); de-kinks "
        "grid lines crossing block seams (interface-only, leaving the clustered "
        "near-wall interior alone)",
    )
    p.add_argument(
        "--ortho-weight",
        type=float,
        default=0.0,
        help="block-interface orthogonality weight (0 = off); pulls the cross-seam "
        "edge perpendicular to the seam (mode='normal'), straightening the crossing",
    )
    p.add_argument(
        "--ortho-layers",
        type=int,
        default=3,
        help="near-seam band depth the orthogonality term acts on (1 = seam only)",
    )
    p.add_argument(
        "--ortho-relax",
        type=float,
        default=1.0,
        help="[0,1] relax orthogonality where near-seam cells are clustered slivers "
        "(follows the wall-normal instead of forcing perpendicular-to-seam)",
    )
    a = p.parse_args(argv)
    print("=" * 56)
    print("Phoebus capsule → TMOP smooth")
    print("=" * 56)
    for k, v in vars(a).items():
        print(f"  {k} = {v}")
    print("=" * 56)
    return a


def finish(grid, topo, ents, steps, a, *, title, mindet_title="min det A"):
    """Act on the CLI flags: drive the steps (live or batch), report, plot.

    ``steps`` is the (lazy, unconsumed) :func:`egg.pipeline.generate_steps`
    generator — nothing has run yet, so ``--plot-topology`` can still short-
    circuit, and live mode animates every phase.
    """
    if a.plot_topology:
        from egg.io.visualize import plot_topology

        plot_topology(topo, highlight_singularities=True, show=True)
        return

    mindet_history, energy_history = [], []
    if getattr(a, "plot_live", False):
        from egg.io.visualize import animate_pipeline

        animate_pipeline(
            grid,
            list(ents.values()),
            steps,
            show_edge_verts=a.colour_edge_verts,
            title=title,
        )
    else:
        from egg.pipeline import drain_steps

        drain_steps(steps, mindet_history=mindet_history, energy_history=energy_history)

    if getattr(topo, "boundary_layer_specs", None):
        from egg.smoothing import first_layer_heights

        # Height / its own spec target per wall column (1.0 = exact).
        r = first_layer_heights(grid, topo, relative=True)
        print(f"First layer heights / target: min={r.min():.6f} max={r.max():.6f}")

    if not getattr(a, "plot_live", False):
        print(f"\nFinal min det A: {mindet_history[-1]:.4e}")

        if a.plot_grid:
            from egg.io.visualize import plot_grid

            plot_grid(grid)
        if getattr(a, "plot_energy", False):
            from egg.io.visualize import plot_convergence

            plot_convergence(energy_history, mindet_history, mindet_title=mindet_title)

    if getattr(a, "export", None):
        from egg.io import export_su2

        export_su2(grid, a.export)
        print(f"Exported SU2 mesh to {a.export}")

    print("Done.")
