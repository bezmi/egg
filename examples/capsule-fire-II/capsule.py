"""FIRE II capsule forebody, ported from gdtk's lmr 2D capsule-fire-II case.

The gdtk original (``examples/lmr/2D/capsule-fire-II/grid.lua``) builds one
structured grid over a ``ControlPointPatch`` with a 4 x 13 net of control
points placed by a channel-e2w guide patch. Here those control-point
locations (exported by the Lua case to ``capsule_ctrl_pts.vts``, including
its three hand-tuned shoulder points) become the *corners* of a 3 x 12
multiblock topology instead, and each cell of the net becomes an egg block.

Boundary faces are associated with the gdtk paths (outer arc = inflow,
capsule body = wall, stagnation line = symmetry, exit line = outflow) and
tagged with those names for SU2 export. One deviation from the gdtk case:
the outflow is vertical rather than slanted, meeting the (horizontal)
post-shoulder wall at a right angle so the boundary-layer cells stay
orthogonal into that corner.

Pipeline: TFI init → boundary snap → TMOP quality optimisation with a
boundary-layer target → exact wall respacing post-pass
(:func:`egg.smoothing.enforce_boundary_layer_spacing`).

Usage::

    uv run capsule.py [--plot-live] [--plot-energy] [--plot-grid]
        [--plot-topology] [--export mesh.su2] [--device cpu|gpu|auto]
        [--tmop-sweeps N] [--chunk N]
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np

from egg.geometry import Arc, Line, Polyline, Vector3
from egg.pipeline import drain, generate_steps
from egg.topology.builder import TopologyBuilder

# Number of control points across the shock layer (i) and along the body (j).
NCPI = 4
NCPJ = 13


def build_paths():
    """The four boundary paths of the FIRE II forebody domain (grid.lua)."""
    Ri = 0.9347  # nose radius
    ri = 0.0102  # shoulder radius
    A = 0.3358  # capsule frontal radius
    thetai = math.asin((A - ri) / (Ri - ri))
    L = 0.05  # length of conical section after the shoulder
    diffo = 0.07  # shock-layer standoff of the outer boundary
    Ro = Ri + diffo

    oi = Vector3(Ri, 0.0)
    ai = oi + Ri * Vector3(-1.0, 0.0)
    bi = oi + Ri * Vector3(-math.cos(thetai), math.sin(thetai))
    pi_ = oi + (Ri - ri) * Vector3(-math.cos(thetai), math.sin(thetai))
    ci = pi_ + ri * Vector3(0.0, 1.0)
    di = ci + L * Vector3(1.0, 0.0)

    body = Polyline([Arc(ai, bi, oi), Arc(bi, ci, pi_), Line(ci, di)])

    # The wall is horizontal after the shoulder, so a vertical outflow meets
    # it at a right angle (the gdtk original slants it to 1.5*thetai, which
    # forces sheared cells where it meets the wall). The outer arc ends
    # directly above the wall's end point.
    ao = oi + Ro * Vector3(-1.0, 0.0)
    thetao = math.acos((Ri - di.x) / Ro)
    do = oi + Ro * Vector3(-math.cos(thetao), math.sin(thetao))

    outer = Arc(ao, do, oi)
    south = Line(ao, ai)
    north = Line(do, di)
    return outer, body, south, north


def load_control_points(vts_path):
    """Read the (NCPI, NCPJ, 2) control-point net from the Lua-exported VTS.

    gdtk's ``writeCtrlPtsAsVtkXml`` writes the points i-fastest, j-slowest.
    """
    txt = Path(vts_path).read_text()
    vals = [
        float(x)
        for x in re.findall(r"[-+0-9.]+e[-+][0-9]+", txt.split("<DataArray", 1)[1])
    ]
    C = np.array(vals).reshape(NCPJ, NCPI, 3)
    return C[:, :, :2].transpose(1, 0, 2)


def build_capsule(res_i=20, res_j=20, bl_first_height=0.0, bl_growth=1.3):
    """3 x 12 block topology with the control points as block corners."""
    outer, body, south, north = build_paths()
    C = load_control_points(Path(__file__).parent / "capsule_ctrl_pts.vts")

    # The VTS control net was generated for the original slanted outflow;
    # move its last column onto the vertical outflow line (north: do -> di).
    do, di = north.p0, north.p1
    for i, r in enumerate((0.0, 0.25, 0.75, 1.0)):
        C[i, -1] = (di.x, do.y + (di.y - do.y) * r)

    inflow = outer
    wall = body
    symmetry = south
    outflow = north

    # Control points as shared corner objects — block-to-block connectivity
    # is inferred from the shared identities at build().
    corner = {
        (i, j): Vector3(
            *C[i, j],
            fixed=(i in (0, NCPI - 1)) and (j in (0, NCPJ - 1)),
        )
        for i in range(NCPI)
        for j in range(NCPJ)
    }

    b = TopologyBuilder(d=2)
    for i in range(NCPI - 1):
        for j in range(NCPJ - 1):
            b.add_block(
                f"b{i}_{j}",
                sw=corner[i, j],
                se=corner[i + 1, j],
                nw=corner[i, j + 1],
                ne=corner[i + 1, j + 1],
                res=(res_i, res_j),
            )

    # The corners are free control points, not nodes on the paths, so the
    # boundary associations stay explicit (faces are snapped at init).
    for j in range(NCPJ - 1):
        b.associate(f"b0_{j}", 0, 0, inflow)
        b.tag_boundary("inflow", f"b0_{j}", 0, 0)
        b.associate(f"b{NCPI - 2}_{j}", 0, 1, wall)
        b.tag_boundary("wall", f"b{NCPI - 2}_{j}", 0, 1)
    for i in range(NCPI - 1):
        b.associate(f"b{i}_0", 1, 0, symmetry)
        b.tag_boundary("symmetry", f"b{i}_0", 1, 0)
        b.associate(f"b{i}_{NCPJ - 2}", 1, 1, outflow)
        b.tag_boundary("outflow", f"b{i}_{NCPJ - 2}", 1, 1)

    if bl_first_height > 0.0:
        b.set_boundary_layer(
            wall, first_height=bl_first_height, growth=bl_growth, n_layers=6
        )

    topology = b.build()
    entities = {
        "inflow": inflow,
        "wall": wall,
        "symmetry": symmetry,
        "outflow": outflow,
    }
    return topology, entities


def main():
    p = argparse.ArgumentParser(description="FIRE II capsule forebody → TMOP smoothed.")
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
        help="write the final grid as an SU2 mesh (markers: inflow, "
        "wall, symmetry, outflow)",
    )
    p.add_argument(
        "--res-i", type=int, default=10, help="cells per block across the shock layer"
    )
    p.add_argument(
        "--res-j", type=int, default=10, help="cells per block along the body"
    )
    p.add_argument(
        "--bl-first-height",
        type=float,
        default=4.0e-4,
        help="first wall-normal cell height on the capsule body "
        "(0 disables clustering)",
    )
    p.add_argument(
        "--bl-growth",
        type=float,
        default=1.3,
        help="boundary-layer geometric growth ratio",
    )
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--tmop-sweeps", type=int, default=40)
    p.add_argument("--sweeps-per-delta", type=int, default=20)
    p.add_argument("--chunk", type=int, default=10)
    a = p.parse_args()

    print("=" * 56)
    print("FIRE II capsule forebody → TMOP smooth")
    print("=" * 56)

    topo, ents = build_capsule(
        res_i=a.res_i,
        res_j=a.res_j,
        bl_first_height=a.bl_first_height,
        bl_growth=a.bl_growth,
    )

    if a.plot_topology:
        from egg.io.visualize import plot_topology

        plot_topology(topo, highlight_singularities=True, show=True)
        return

    target = None
    if a.bl_first_height > 0.0:
        from egg.smoothing import build_boundary_layer_target

        # Per-block tangential spacings and the cap on the wall-normal growth
        # are derived from the topology; neighbour blocks continue the profile.
        target = build_boundary_layer_target(topo)
        print(
            f"Boundary layer: first_height={a.bl_first_height} "
            f"growth={a.bl_growth} on wall "
            f"({len(target.per_block)} blocks incl. blended neighbours)"
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

    if a.bl_first_height > 0.0:
        # Chain the exact respacing onto the pipeline so the live view's last
        # frame (and any later plotting/export) reflects it.
        from egg.smoothing import enforce_boundary_layer_spacing

        def _with_respace(inner):
            yield from inner
            enforce_boundary_layer_spacing(grid, topo)
            yield ("respace", {"first_height": a.bl_first_height,
                               "growth": a.bl_growth})

        steps = _with_respace(steps)

    if a.plot_live:
        from egg.io.visualize import animate_pipeline

        animate_pipeline(
            grid,
            list(ents.values()),
            steps,
            show_edge_verts=a.colour_edge_verts,
            title="FIRE II capsule",
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

    if a.export:
        from egg.io import export_su2

        export_su2(grid, a.export)
        print(f"Exported SU2 mesh to {a.export}")

    print("Done.")


if __name__ == "__main__":
    main()
