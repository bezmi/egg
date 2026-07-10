# MIT License
#
# Copyright (c) 2026 Shahzeb Imran and the Egg contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Argument parsing for the SVG-imported egg example (see ``egg_svg.py``)."""

import argparse


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="egg in rectangle, geometry from Inkscape SVG → TMOP smoothed."
    )
    p.add_argument(
        "--plot-live", action="store_true", help="PyVista animated relaxation"
    )
    p.add_argument(
        "--plot-energy",
        action="store_true",
        help="matplotlib energy + min-det convergence curves",
    )
    p.add_argument("--plot-grid", action="store_true", help="final wireframe grid")
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
    p.add_argument(
        "--export",
        metavar="FILE",
        help="write the final grid as an SU2 mesh (markers: egg, "
        "inflow, outflow, wall)",
    )
    p.add_argument(
        "--bl-first-height",
        type=float,
        default=5.0e-3,
        help="first wall-normal cell height on the egg (0 disables clustering)",
    )
    p.add_argument(
        "--bl-growth",
        type=float,
        default=1.5,
        help="boundary-layer geometric growth ratio",
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
        default=300,
        help="TMOP sweeps for the pinned re-run",
    )
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--tmop-sweeps", type=int, default=600)
    p.add_argument(
        "--smoother",
        choices=["jacobi", "fas"],
        default="jacobi",
        help="TMOP-phase smoother: plain block-Jacobi sweeps or FAS (nonlinear "
        "geometric multigrid) V-cycles; with fas, --tmop-sweeps/--chunk count "
        "V-cycles (4 fine sweeps each plus coarse-grid work)",
    )
    p.add_argument("--sweeps-per-delta", type=int, default=20)
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument(
        "--omega",
        type=float,
        default=0.8,
        help="block-Jacobi SOR/damping weight (1.0 = undamped)",
    )
    a = p.parse_args(argv)
    print("=" * 56)
    print("egg in rectangle (from Inkscape SVG) → TMOP smooth")
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
