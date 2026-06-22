"""Smooth blob in a rectangle, with geometry defined via the egg 2D front-end.

Same O-grid-in-rectangle topology as ``examples/circles/good-topo.py``, but the
geometry is built with :mod:`egg.geometry.frontend2d` (a gdtk-style construction
API that returns egg entities directly) instead of the analytic primitives: the
inner body is a closed :class:`~egg.geometry.frontend2d.Spline` through points
sampled from ``r(theta) = 0.8 + 0.12 sin(3 theta)`` (a wavy blob), produced as a
``CompositePath`` of cubic Bézier segments; the outer walls are
:class:`~egg.geometry.frontend2d.Line` segments.

Pipeline: TFI init → boundary snap → TMOP quality optimisation.

Usage::

    uv run blob.py [--plot-live] [--plot-energy] [--plot-grid]
        [--plot-topology] [--device cpu|gpu|auto] [--tmop-sweeps N] [--chunk N]
"""

import argparse

import numpy as np

from egg.geometry.frontend2d import Line, Spline, Vector3
from egg.pipeline import generate_steps, drain
from egg.topology.builder import TopologyBuilder


def build_blob_in_rectangle():
    """Blob-in-rectangle topology; geometry authored via the egg 2D front-end."""
    # Inner body: closed cubic spline through a wavy-radius point ring.
    theta = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
    r = 0.8 + 0.12 * np.sin(3.0 * theta)
    ring = [Vector3(2.0 + ri * np.cos(th), 2.0 + ri * np.sin(th))
            for th, ri in zip(theta, r)]
    blob = Spline(ring, closed=True)

    # Outer walls.
    bottom = Line(Vector3(0, 0), Vector3(4, 0))
    right = Line(Vector3(4, 0), Vector3(4, 4))
    top = Line(Vector3(4, 4), Vector3(0, 4))
    left = Line(Vector3(0, 0), Vector3(0, 4))

    b = TopologyBuilder(d=2)
    for n, p in [("sw", (0, 0)), ("se", (4, 0)), ("ne", (4, 4)), ("nw", (0, 4))]:
        b.add_corner(n, p, fixed=True)
    for n, p in [("msw", (1, 1)), ("mse", (3, 1)), ("mne", (3, 3)), ("mnw", (1, 3))]:
        b.add_corner(n, p, fixed=False)
    for n, p in [
        ("isw", (1.3, 1.3)),
        ("ise", (2.7, 1.3)),
        ("ine", (2.7, 2.7)),
        ("inw", (1.3, 2.7)),
    ]:
        b.add_corner(n, p, fixed=False)
    for n, p in [
        ("bsw", (1, 0)),
        ("bse", (3, 0)),
        ("rse", (4, 1)),
        ("rne", (4, 3)),
        ("tne", (3, 4)),
        ("tnw", (1, 4)),
        ("lnw", (0, 3)),
        ("lsw", (0, 1)),
    ]:
        b.add_corner(n, p, fixed=False)

    for nm, sw, nw, se, ne in [
        ("o_s", "msw", "isw", "mse", "ise"),
        ("o_e", "mse", "ise", "mne", "ine"),
        ("o_n", "mne", "ine", "mnw", "inw"),
        ("o_w", "mnw", "inw", "msw", "isw"),
    ]:
        b.add_block(nm, (sw, nw, se, ne), (10, 4))
    for nm, sw, nw, se, ne in [
        ("e_s", "bsw", "msw", "bse", "mse"),
        ("e_e", "rse", "mse", "rne", "mne"),
        ("e_n", "tne", "mne", "tnw", "mnw"),
        ("e_w", "lnw", "mnw", "lsw", "msw"),
    ]:
        b.add_block(nm, (sw, nw, se, ne), (10, 5))
    for nm, sw, nw, se, ne in [
        ("c_sw", "sw", "lsw", "bsw", "msw"),
        ("c_se", "se", "bse", "rse", "mse"),
        ("c_ne", "ne", "rne", "tne", "mne"),
        ("c_nw", "nw", "tnw", "lnw", "mnw"),
    ]:
        b.add_block(nm, (sw, nw, se, ne), (5, 5))

    for a, b_ in [("o_s", "o_e"), ("o_e", "o_n"), ("o_n", "o_w"), ("o_w", "o_s")]:
        b.connect(a, 0, 1, b_, 0, 0)
    for e, o in [("e_s", "o_s"), ("e_e", "o_e"), ("e_n", "o_n"), ("e_w", "o_w")]:
        b.connect(e, 1, 1, o, 1, 0)
    for cb, ca, cs, eb, ea, es in [
        ("c_sw", 0, 1, "e_s", 0, 0),
        ("c_sw", 1, 1, "e_w", 0, 1),
        ("c_se", 0, 1, "e_e", 0, 0),
        ("c_se", 1, 1, "e_s", 0, 1),
        ("c_ne", 0, 1, "e_n", 0, 0),
        ("c_ne", 1, 1, "e_e", 0, 1),
        ("c_nw", 0, 1, "e_w", 0, 0),
        ("c_nw", 1, 1, "e_n", 0, 1),
    ]:
        b.connect(cb, ca, cs, eb, ea, es)

    for blk in ("o_s", "o_e", "o_n", "o_w"):
        b.associate(blk, 1, 1, blob)
    for blk, ent in [("e_s", bottom), ("e_e", right), ("e_n", top), ("e_w", left)]:
        b.associate(blk, 1, 0, ent)
    for blk, a0, a1 in [
        ("c_sw", left, bottom),
        ("c_se", bottom, right),
        ("c_ne", right, top),
        ("c_nw", top, left),
    ]:
        b.associate(blk, 0, 0, a0)
        b.associate(blk, 1, 0, a1)

    topology = b.build()
    entities = {
        "blob": blob,
        "bottom": bottom,
        "right": right,
        "top": top,
        "left": left,
    }
    return topology, entities


def main():
    p = argparse.ArgumentParser(description="spline blob → TMOP smoothed.")
    p.add_argument("--plot-live", action="store_true",
                   help="PyVista animated relaxation")
    p.add_argument("--plot-energy", action="store_true",
                   help="matplotlib energy + min-det convergence curves")
    p.add_argument("--plot-grid", action="store_true",
                   help="final wireframe grid")
    p.add_argument("--plot-topology", action="store_true",
                   help="Plot the declared topology only — no pipeline run")
    p.add_argument("--colour-verts-by-graph", action="store_true",
                   help="Colour each node by its graph-colouring colour in live plot")
    p.add_argument("--colour-edge-verts", action="store_true",
                   help="Toggle blue/red edge-vertex spheres in live plot")
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--tmop-sweeps", type=int, default=40)
    p.add_argument("--sweeps-per-delta", type=int, default=20)
    p.add_argument("--chunk", type=int, default=10)
    a = p.parse_args()

    print("=" * 56)
    print("spline blob in rectangle → TMOP smooth")
    print("=" * 56)

    topo, ents = build_blob_in_rectangle()

    if a.plot_topology:
        from egg.io.visualize import plot_topology

        plot_topology(topo, highlight_singularities=True, show=True)
        return

    grid = topo.initialize_grid()

    steps = generate_steps(
        grid,
        None,
        sweeps_per_delta=a.sweeps_per_delta,
        tmop_sweeps=a.tmop_sweeps,
        tmop_chunk=a.chunk,
        device=a.device,
        # Step the untangle per δ only when animating; otherwise run it direct.
        untangle_direct=not a.plot_live,
    )

    if a.plot_live:
        from egg.io.visualize import animate_pipeline

        graph_colours = None
        if a.colour_verts_by_graph:
            from egg.smoothing.solver import build_sweep_context
            from egg.smoothing.targets import IdentityTarget

            ctx = build_sweep_context(grid, IdentityTarget(d=topo.d))
            graph_colours = ctx.dof_colours
        animate_pipeline(
            grid,
            list(ents.values()),
            steps,
            graph_colours=graph_colours,
            show_edge_verts=a.colour_edge_verts,
            title="spline blob",
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
