"""Argument parsing for the spline-blob example (see ``blob.py``)."""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="spline blob → TMOP smoothed.")
    p.add_argument("--plot-live", action="store_true",
                   help="PyVista animated relaxation")
    p.add_argument("--plot-energy", action="store_true",
                   help="matplotlib energy + min-det convergence curves")
    p.add_argument("--plot-grid", action="store_true",
                   help="final wireframe grid")
    p.add_argument("--plot-topology", action="store_true",
                   help="Plot the declared topology only — no pipeline run")
    p.add_argument("--colour-edge-verts", action="store_true",
                   help="Toggle blue/red edge-vertex spheres in live plot")
    p.add_argument("--export", metavar="FILE",
                   help="write the final grid as an SU2 mesh (markers: blob, "
                        "bottom, right, top, left)")
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--tmop-sweeps", type=int, default=40)
    p.add_argument("--sweeps-per-delta", type=int, default=20)
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--omega", type=float, default=0.8,
                   help="block-Jacobi SOR/damping weight (1.0 = undamped)")
    a = p.parse_args()
    print("=" * 56)
    print("spline blob in rectangle → TMOP smooth")
    print("=" * 56)
    for k, v in vars(a).items():
        print(f"  {k} = {v}")
    print("=" * 56)
    return a


def finish(grid, topo, ents, steps, a, *, title,
           mindet_title="min det A"):
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

        drain_steps(steps, mindet_history=mindet_history,
                    energy_history=energy_history)

    if getattr(a, "pin_layers", 0) > 0 and             getattr(topo, "boundary_layer_specs", None):
        from egg.smoothing import first_layer_heights

        # Height / its own spec target per wall column (1.0 = exact).
        r = first_layer_heights(grid, topo, relative=True)
        print(f"First layer heights / target: min={r.min():.6f} "
              f"max={r.max():.6f}")

    if not getattr(a, "plot_live", False):
        print(f"\nFinal min det A: {mindet_history[-1]:.4e}")

        if a.plot_grid:
            from egg.io.visualize import plot_grid

            plot_grid(grid)
        if getattr(a, "plot_energy", False):
            from egg.io.visualize import plot_convergence

            plot_convergence(energy_history, mindet_history,
                             mindet_title=mindet_title)

    if getattr(a, "export", None):
        from egg.io import export_su2

        export_su2(grid, a.export)
        print(f"Exported SU2 mesh to {a.export}")

    print("Done.")
