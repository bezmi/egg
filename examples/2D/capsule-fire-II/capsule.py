"""FIRE II capsule forebody, ported from gdtk's lmr 2D capsule-fire-II case.

The domain is bounded by the four gdtk paths (outer arc = inflow, capsule
body = wall, stagnation line = symmetry, exit line = outflow) and filled
with a 3 x 12 block array: sub-block corners are placed parametrically on
the bounding paths (TFI for the interior ones) and every face is tagged for
SU2 export. Where the gdtk original shapes the interior with a hand-tuned
``ControlPointPatch`` net, the TMOP smoothing pass does that job — nothing
is read from an external file.

One deviation from the gdtk case: the outflow is vertical rather than
slanted, meeting the (horizontal) post-shoulder wall at a right angle so
the boundary-layer cells stay orthogonal into that corner.

The command-line surface lives in ``driver.py``; run
``uv run capsule.py --help`` for options.
"""

import math

from egg.geometry import Arc, Edge, Line, Polyline, Vector3
from egg.pipeline import PipelineConfig, generate_steps
from egg.topology.builder import TopologyBuilder


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


def build_capsule(res_i=20, res_j=20, bl_first_height=0.0, bl_growth=1.3, n_fixed=0):
    """3 x 12 block array between the capsule body and the outer arc."""
    outer, body, south, north = build_paths()

    # Grid edges (axis 0 = inflow -> wall, axis 1 = along the body).
    inflow = Edge(outer)  # west, symmetry -> outflow
    wall = Edge(body, arc_length=True)  # east, symmetry -> outflow
    symmetry = Edge(south)  # south, inflow -> wall
    outflow = Edge(north)  # north, inflow -> wall

    nib, njb = 3, 12
    b = TopologyBuilder(d=2)
    # Sub-block corners on the bounding paths (TFI inside), blocks from the
    # shared corner objects, boundary faces associated with their edges.
    _corner, names = b.add_block_array(
        south=symmetry,
        north=outflow,
        west=inflow,
        east=wall,
        nib=nib,
        njb=njb,
        res=(nib * res_i, njb * res_j),
    )
    for j in range(njb):
        b.tag_boundary("inflow", names[0][j], 0, 0)
        b.tag_boundary("wall", names[nib - 1][j], 0, 1)
    for i in range(nib):
        b.tag_boundary("symmetry", names[i][0], 1, 0)
        b.tag_boundary("outflow", names[i][njb - 1], 1, 1)

    if bl_first_height > 0.0:
        # relax_orthogonality is a no-op while the outflow meets the wall at
        # a right angle; declared for consistency with the phoebus example
        # (and so a slanted outflow, as in the gdtk original, keeps its
        # layer heights).
        b.set_boundary_layer(
            wall,
            first_height=bl_first_height,
            growth=bl_growth,
            n_fixed=n_fixed,
            relax_orthogonality=(outflow,),
        )

    topology = b.build()
    entities = {
        "inflow": inflow.entity,
        "wall": wall.entity,
        "symmetry": symmetry.entity,
        "outflow": outflow.entity,
    }
    return topology, entities


def main():
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from driver import finish, parse_args

    from egg.smoothing import build_boundary_layer_target

    a = parse_args()

    topo, ents = build_capsule(
        res_i=a.res_i,
        res_j=a.res_j,
        bl_first_height=a.bl_first_height,
        bl_growth=a.bl_growth,
        n_fixed=a.pin_layers,
    )

    # --pin-layers > 0 smooths against the aspect-ratio target and pins the
    # first n_fixed layers exactly; the default path runs the pure shape
    # metric and restores the wall spacing with the respace post-pass.
    pin = a.bl_first_height > 0.0 and a.pin_layers > 0
    target = build_boundary_layer_target(topo) if pin else None
    grid = topo.initialize_grid()
    cfg = PipelineConfig(
        sweeps_per_delta=a.sweeps_per_delta,
        tmop_sweeps=a.tmop_sweeps,
        tmop_chunk=a.chunk,
        omega=a.omega,
        device=a.device,
        pin_sweeps=a.pin_sweeps if pin else 0,
        respace=a.bl_first_height > 0.0 and not pin,
    )
    steps = generate_steps(grid, target, cfg, untangle_direct=not a.plot_live)

    finish(
        grid,
        topo,
        ents,
        steps,
        a,
        title="FIRE II capsule",
        mindet_title="min det A (TMOP only)",
    )


if __name__ == "__main__":
    main()
