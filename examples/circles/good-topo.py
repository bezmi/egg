"""good single-circle O-grid → TMOP smooth (pipeline).

Starts from a *proper* circle-in-rectangle topology (inner O-ring corners placed
so that ``min det A > 0``) and runs the pipeline:

    TFI init → boundary snap → TMOP quality opt → valid

Since the initial grid is already valid the δ-continuation untangling phase is
skipped automatically; only the fused TMOP quality optimisation runs.

Usage::

    uv run good-topo.py [--plot-live] [--plot-energy]
        [--plot-grid] [--plot-topology] [--colour-verts-by-graph]
         [--colour-edge-verts] [--device cpu|gpu|auto] [--tmop-sweeps N]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from topologies import build_circle_in_rectangle
from egg.pipeline import generate_steps, drain
from egg.io.visualize import animate_pipeline
from egg.smoothing.targets import IdentityTarget
from egg.smoothing.solver import build_sweep_context


def main():
    p = argparse.ArgumentParser(description="good topology → TMOP smoothed.")
    p.add_argument(
        "--plot-live",
        action="store_true",
        help="PyVista animated relaxation (TMOP only)",
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
        "--colour-verts-by-graph",
        action="store_true",
        help="Colour each node by its graph-colouring colour in live plot",
    )
    p.add_argument(
        "--colour-edge-verts",
        action="store_true",
        help="Toggle blue/red edge-vertex spheres in live plot",
    )
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--tmop-sweeps", type=int, default=40)
    p.add_argument("--sweeps-per-delta", type=int, default=20)
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument(
        "--solver",
        choices=["gauss-seidel", "jacobi"],
        default="gauss-seidel",
        help="TMOP solver: gauss-seidel (faster per-sweep) or jacobi (higher CPU utilisation)",
    )
    # Boundary-layer clustering on the (single) circle. <=0 disables it; the
    # ``*2`` flags exist only for CLI parity with the side-by-side demo (which
    # clusters a second circle) and are ignored here.
    p.add_argument(
        "--bl-first-height",
        type=float,
        default=0.0,
        help="first off-wall layer height on the circle (<=0: off)",
    )
    p.add_argument(
        "--bl-growth", type=float, default=1.3, help="near-wall geometric growth ratio"
    )
    p.add_argument(
        "--bl-first-height2",
        type=float,
        default=0.0,
        help="(ignored; single circle) parity with side-by-side demo",
    )
    p.add_argument(
        "--bl-growth2",
        type=float,
        default=1.25,
        help="(ignored; single circle) parity with side-by-side demo",
    )
    a = p.parse_args()

    print("=" * 56)
    print("good circle-in-rectangle → TMOP smooth")
    print("=" * 56)

    topo, ents = build_circle_in_rectangle(rough=False)

    if a.plot_topology:
        from egg.io.visualize import plot_topology

        plot_topology(topo, highlight_singularities=True, show=True)
        return

    # Optional boundary-layer clustering on the circle (off by default).
    use_bl = a.bl_first_height > 0.0
    target = None
    if use_bl:
        from egg.smoothing.targets import build_boundary_layer_target

        topo.boundary_layer_specs = {
            id(ents["circle"]): dict(
                first_height=a.bl_first_height,
                growth=a.bl_growth,
                n_layers=4,
                max_height=None,
                tangential_spacing=None,
            )
        }
        target = build_boundary_layer_target(topo, interior_spacing=0.2)
        print(
            f"Boundary layer: first_height={a.bl_first_height} "
            f"growth={a.bl_growth} on circle"
        )

    grid = topo.initialize_grid()

    steps = generate_steps(
        grid,
        target,
        sweeps_per_delta=a.sweeps_per_delta,
        tmop_sweeps=a.tmop_sweeps,
        tmop_chunk=a.chunk,
        device=a.device,
        # Step the untangle per δ only when animating; otherwise run it direct.
        untangle_direct=not a.plot_live,
    )

    if a.plot_live:
        graph_colours = None
        if a.colour_verts_by_graph:
            ctx = build_sweep_context(grid, IdentityTarget(d=topo.d))
            graph_colours = ctx.dof_colours
        animate_pipeline(
            grid,
            list(ents.values()),
            steps,
            graph_colours=graph_colours,
            show_edge_verts=a.colour_edge_verts,
            title="good single-circle",
        )
    else:
        mindet_history, energy_history = [], []
        drain(steps, mindet_history=mindet_history, energy_history=energy_history)
        print(f"\nFinal min det A: {mindet_history[-1]:.4e}")

        if a.plot_grid:
            from egg.io.visualize import plot_grid

            plot_grid(grid)
        if a.plot_energy:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            ax[0].plot(energy_history, "-o", ms=3)
            ax[0].set(title="TMOP energy", xlabel="chunk", ylabel="F")
            ax[1].axhline(0, color="r", lw=0.8)
            ax[1].plot(mindet_history, "-o", ms=2)
            ax[1].set(title="min det A (TMOP only)", xlabel="chunk")
            plt.tight_layout()
            plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
