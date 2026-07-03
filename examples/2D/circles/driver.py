"""Command-line surface shared by the circle examples (see the demo scripts):
argument parsing in, presentation of the finished run out."""

import argparse


def _add_common_args(p, *, tmop_sweeps, sweeps_per_delta, chunk, pin_sweeps):
    p.add_argument(
        "--plot-live", action="store_true", help="PyVista animated relaxation"
    )
    p.add_argument(
        "--plot-energy",
        action="store_true",
        help="matplotlib energy + min-det convergence curves",
    )
    p.add_argument(
        "--plot-grid", action="store_true", help="matplotlib final wireframe grid"
    )
    p.add_argument(
        "--plot-topology",
        action="store_true",
        help="Plot the declared topology only — no pipeline run",
    )
    p.add_argument(
        "--colour-edge-verts",
        action="store_true",
        help="Toggle blue/red edge-vertex spheres in live plot",
    )
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--tmop-sweeps", type=int, default=tmop_sweeps)
    p.add_argument("--sweeps-per-delta", type=int, default=sweeps_per_delta)
    p.add_argument("--chunk", type=int, default=chunk)
    p.add_argument(
        "--omega",
        type=float,
        default=0.8,
        help="block-Jacobi SOR/damping weight (1.0 = undamped)",
    )
    p.add_argument(
        "--pin-layers",
        type=int,
        default=0,
        help="fix this many near-wall layers at their exact geometric "
        "heights after TMOP and re-run TMOP with them pinned (0: off)",
    )
    p.add_argument(
        "--pin-sweeps",
        type=int,
        default=pin_sweeps,
        help="TMOP sweeps for the pinned re-run",
    )


def parse_single_args(description):
    """Flags for the single-circle demos (good-topo.py / untangle.py)."""
    p = argparse.ArgumentParser(description=description)
    _add_common_args(
        p, tmop_sweeps=1000, sweeps_per_delta=200, chunk=100, pin_sweeps=500
    )
    # Boundary-layer clustering on the circle (off by default): just a first
    # layer height and a growth ratio — the clustering grows until it reaches
    # the grid's own ambient spacing.
    p.add_argument(
        "--bl-first-height",
        type=float,
        default=0.0,
        help="first off-wall layer height on the circle (<=0: off)",
    )
    p.add_argument(
        "--bl-growth",
        type=float,
        default=1.3,
        help="near-wall geometric growth ratio",
    )
    a = p.parse_args()
    print("=" * 56)
    print(description)
    print("=" * 56)
    for k, v in vars(a).items():
        print(f"  {k} = {v}")
    print("=" * 56)
    return a


def parse_twin_args(description):
    """Flags for the twin-circle demos (*_side-by-side.py)."""
    p = argparse.ArgumentParser(description=description)
    _add_common_args(
        p, tmop_sweeps=5000, sweeps_per_delta=200, chunk=500, pin_sweeps=5000
    )
    p.add_argument(
        "--bl-first-height",
        type=float,
        default=0.05,
        help="first bl cell height for left circle",
    )
    p.add_argument(
        "--bl-growth", type=float, default=1.5, help="bl growth for left circle"
    )
    p.add_argument(
        "--bl-first-height2",
        type=float,
        default=0.08,
        help="first bl cell height for right circle",
    )
    p.add_argument(
        "--bl-growth2", type=float, default=1.3, help="bl growth for right circle"
    )
    p.add_argument(
        "--resolution", type=int, default=2, help="Resolution scaling factor"
    )
    a = p.parse_args()
    print("=" * 56)
    print(description)
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
