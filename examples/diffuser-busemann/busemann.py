"""Busemann diffuser grid, ported from the Eilmer example (diffuser-busemann).

Two blocks matching the Lua patches 1:1: the diffuser section between the
theory-derived wall contour and the symmetry axis, and the straight throat
section downstream. The contour (``bd-contour.dat``, M1=5.77 -> M3=2.48
design from Moelder 2003) is a ``Spline`` wrapped arc-length parameterized —
the egg equivalent of the Lua ``ArcLengthParameterizedPath``. Where the Lua
example shapes the diffuser interior with a ``ControlPointPatch``, the TMOP
smoothing pass does that job.

The contour corners are placed with ``bodyside.place_node``, so the diffuser
wall association and the block-to-block connection (shared corner objects)
are inferred; the remaining faces are associated explicitly.

Usage::

    uv run busemann.py [--plot-grid] [--plot-topology]
        [--tmop-sweeps N] [--device cpu|gpu|auto]
"""

import argparse
import math
import os

import numpy as np

from egg.geometry import Edge, Line, Spline, Vector3
from egg.pipeline import drain, generate_steps
from egg.topology.builder import TopologyBuilder


def build_busemann():
    thrt_length = 0.2

    # -- diffuser contour: spline through theory-derived points ---------------
    pts = np.loadtxt(os.path.join(os.path.dirname(__file__),
                                  "bd-contour.dat"), skiprows=1)
    bodyside = Edge(Spline(pts), arc_length=True)
    lip = bodyside.place_node(0.0, fixed=True)    # entrance lip
    thrt_start = bodyside.place_node(1.0)          # contour end / throat start
    thrt_end = Vector3(x=thrt_start.x + thrt_length, y=thrt_start.y,
                       fixed=True)

    # -- axis and boundary lines ----------------------------------------------
    s0a = Vector3(x=lip.x, y=0.0, fixed=True)
    s0b = Vector3(x=thrt_start.x, y=0.0)
    s1b = Vector3(x=thrt_end.x, y=0.0, fixed=True)
    symmetry0 = Edge(Line(p0=s0a, p1=s0b))
    symmetry1 = Edge(Line(p0=s0b, p1=s1b))
    entrance = Edge(Line(p0=s0a, p1=lip))
    thrt = Edge(Line(p0=thrt_start, p1=thrt_end))
    exit_ = Edge(Line(p0=s1b, p1=thrt_end))

    # -- blocks: 1:1 with the Lua patches -------------------------------------
    # Squarish cells at the entrance, squashed towards the throat (as in Lua).
    nycells = 20
    len_entr = lip.y
    len_sym0 = s0b.x - s0a.x
    len_sym1 = thrt_length
    nxcells0 = math.ceil(nycells * len_sym0 / len_entr)
    nxcells1 = math.ceil(nxcells0 * len_sym1 / len_sym0)

    bld = TopologyBuilder(d=2)
    bld.add_block("diffuser", sw=s0a, se=s0b, nw=lip, ne=thrt_start,
                  res=(nxcells0, nycells))
    bld.add_block("throat", sw=s0b, se=s1b, nw=thrt_start, ne=thrt_end,
                  res=(nxcells1, nycells))
    # The diffuser-throat connection (shared s0b/thrt_start objects) and the
    # bodyside association (lip/thrt_start node provenance) are inferred.
    bld.associate("diffuser", 0, 0, entrance)   # west
    bld.associate("diffuser", 1, 0, symmetry0)  # south
    bld.associate("throat", 0, 1, exit_)        # east
    bld.associate("throat", 1, 0, symmetry1)    # south
    bld.associate("throat", 1, 1, thrt)         # north

    topology = bld.build()
    entities = {
        "bodyside": bodyside.entity,
        "entrance": entrance.entity,
        "thrt": thrt.entity,
        "exit": exit_.entity,
        "symmetry0": symmetry0.entity,
        "symmetry1": symmetry1.entity,
    }
    return topology, entities


def main():
    p = argparse.ArgumentParser(description="Busemann diffuser grid")
    p.add_argument("--plot-grid", action="store_true",
                   help="matplotlib final wireframe grid")
    p.add_argument("--plot-topology", action="store_true",
                   help="Plot the declared topology only — no pipeline run")
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--tmop-sweeps", type=int, default=40)
    a = p.parse_args()

    print("=" * 56)
    print("Busemann diffuser → TMOP smooth")
    print("=" * 56)

    topo, ents = build_busemann()

    if a.plot_topology:
        from egg.io.visualize import plot_topology

        plot_topology(topo, highlight_singularities=True, show=True)
        return

    grid = topo.initialize_grid()

    steps = generate_steps(
        grid,
        None,
        tmop_sweeps=a.tmop_sweeps,
        device=a.device,
        untangle_direct=True,
    )
    mindet_history, energy_history = [], []
    drain(steps, mindet_history=mindet_history, energy_history=energy_history)
    print(f"\nFinal min det A: {mindet_history[-1]:.4e}")

    if a.plot_grid:
        from egg.io.visualize import plot_grid

        plot_grid(grid)

    print("Done.")


if __name__ == "__main__":
    main()
