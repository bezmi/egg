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

from egg.geometry import Edge, Line, Spline, Vector3
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

    # Outer walls, wrapped as parametric grid edges.
    sw, se = Vector3(x=0, y=0, fixed=True), Vector3(x=4, y=0, fixed=True)
    ne, nw = Vector3(x=4, y=4, fixed=True), Vector3(x=0, y=4, fixed=True)
    bottom = Edge(Line(p0=sw, p1=se))
    right = Edge(Line(p0=se, p1=ne))
    top = Edge(Line(p0=ne, p1=nw))
    left = Edge(Line(p0=sw, p1=nw))

    # O-ring corners: mid square and inner square around the blob.
    msw, mse, mne, mnw = (Vector3(*p) for p in [(1, 1), (3, 1), (3, 3), (1, 3)])
    isw, ise, ine, inw = (
        Vector3(*p) for p in [(1.3, 1.3), (2.7, 1.3), (2.7, 2.7), (1.3, 2.7)]
    )
    # Wall corners placed along the wall edges in parametric space.
    bsw, bse = bottom.place_node(0.25), bottom.place_node(0.75)
    rse, rne = right.place_node(0.25), right.place_node(0.75)
    tne, tnw = top.place_node(0.25), top.place_node(0.75)
    lnw, lsw = left.place_node(0.75), left.place_node(0.25)

    b = TopologyBuilder(d=2)
    for nm, c_sw, c_nw, c_se, c_ne in [
        ("o_s", msw, isw, mse, ise),
        ("o_e", mse, ise, mne, ine),
        ("o_n", mne, ine, mnw, inw),
        ("o_w", mnw, inw, msw, isw),
    ]:
        b.add_block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=(10, 4))
    for nm, c_sw, c_nw, c_se, c_ne in [
        ("e_s", bsw, msw, bse, mse),
        ("e_e", rse, mse, rne, mne),
        ("e_n", tne, mne, tnw, mnw),
        ("e_w", lnw, mnw, lsw, msw),
    ]:
        b.add_block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=(10, 5))
    for nm, c_sw, c_nw, c_se, c_ne in [
        ("c_sw", sw, lsw, bsw, msw),
        ("c_se", se, bse, rse, mse),
        ("c_ne", ne, rne, tne, mne),
        ("c_nw", nw, tnw, lnw, mnw),
    ]:
        b.add_block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=(5, 5))

    # Connectivity and wall associations are inferred; the O-ring faces on
    # the blob (rough corners) and the corner-block faces (corners shared by
    # two walls) stay explicit.
    for blk in ("o_s", "o_e", "o_n", "o_w"):
        b.associate(blk, 1, 1, blob)
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
        "bottom": bottom.entity,
        "right": right.entity,
        "top": top.entity,
        "left": left.entity,
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
    p.add_argument("--colour-edge-verts", action="store_true",
                   help="Toggle blue/red edge-vertex spheres in live plot")
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--tmop-sweeps", type=int, default=40)
    p.add_argument("--sweeps-per-delta", type=int, default=20)
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--omega", type=float, default=1.0,
                   help="block-Jacobi SOR/damping weight (1.0 = undamped)")
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
        omega=a.omega,
        device=a.device,
        # Step the untangle per δ only when animating; otherwise run it direct.
        untangle_direct=not a.plot_live,
    )

    if a.plot_live:
        from egg.io.visualize import animate_pipeline

        animate_pipeline(
            grid,
            list(ents.values()),
            steps,
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
