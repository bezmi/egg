"""side-by-side demo: good twin-circle TMOP smooth + per-wall boundary layers.

Two circles in a 7×4 channel sharing a bridge block, started from a *proper*
(not folded) configuration and run through the pipeline. Each circle gets its
**own** boundary-layer clustering (``--bl-first-height`` / ``--bl-growth``).

Since the initial grid is already valid the δ-continuation untangling phase is
skipped automatically; only the fused TMOP quality optimisation runs.

Usage::

    uv run good-topo_side-by-side.py [--plot-live] [--plot-grid]
        [--plot-energy] [--plot-topology]
        [--colour-edge-verts] [--bl-first-height H] [--bl-growth G]
        [--device cpu|gpu|auto] [--tmop-sweeps N]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from egg.pipeline import generate_steps, drain
from egg.io.visualize import animate_pipeline

from topologies import build_twin_circle
from egg.smoothing.targets import build_boundary_layer_target


def main():
    p = argparse.ArgumentParser(description="twin-circle good-topo + BL.")
    p.add_argument
    p.add_argument(
        "--plot-live",
        action="store_true",
        help="PyVista animated relaxation (untangle then TMOP)",
    )
    p.add_argument("--plot-grid", action="store_true")
    p.add_argument("--plot-energy", action="store_true")
    p.add_argument(
        "--plot-topology",
        action="store_true",
        help="Plot the declared (rough) topology only — no pipeline run",
    )
    p.add_argument(
        "--colour-edge-verts",
        action="store_true",
        help="Toggle blue/red edge-vertex spheres in live plot",
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
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--tmop-sweeps", type=int, default=5000, help="total TMOP sweeps")
    p.add_argument("--sweeps-per-delta", type=int, default=200)
    p.add_argument("--chunk", type=int, default=500, help="TMOP sweeps per chunk")
    p.add_argument("--omega", type=float, default=1.0,
                   help="block-Jacobi SOR/damping weight (1.0 = undamped)")
    p.add_argument(
        "--resolution", type=int, default=2, help="Resolution scaling factor"
    )
    a = p.parse_args()

    print("=" * 56)
    print("side-by-side: good twin-circle TMOP smooth + per-wall BL")
    print("=" * 56)

    bl = {
        "circle": dict(first_height=a.bl_first_height, growth=a.bl_growth, n_layers=4),
        "circle2": dict(
            first_height=a.bl_first_height2, growth=a.bl_growth2, n_layers=4
        ),
    }
    topo, ents = build_twin_circle(rough=False, bl=bl, R=a.resolution)

    if a.plot_topology:
        from egg.io.visualize import plot_topology

        plot_topology(topo, highlight_singularities=True, show=True)
        return

    tgt = build_boundary_layer_target(topo, interior_spacing=0.2)
    print(
        f"Boundary layers: circle (h={a.bl_first_height}, g={a.bl_growth}), "
        f"circle2 (h={a.bl_first_height2}, g={a.bl_growth2})"
    )

    grid = topo.initialize_grid()

    steps = generate_steps(
        grid,
        tgt,
        sweeps_per_delta=a.sweeps_per_delta,
        tmop_sweeps=a.tmop_sweeps,
        tmop_chunk=a.chunk,
        omega=a.omega,
        device=a.device,
        # Step the untangle per δ only when animating; otherwise run it direct.
        untangle_direct=not a.plot_live,
    )

    if a.plot_live:
        animate_pipeline(
            grid,
            list(ents.values()),
            steps,
            show_edge_verts=a.colour_edge_verts,
            title="good twin-circle",
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
