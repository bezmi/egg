"""Phoebus capsule grid, ported from the Eilmer example (capsule-phoebus).

A 2x4 block array between the capsule wall and the inflow boundary,
matching the Lua ``ControlPointPatch`` + ``registerFluidGridArray{nib=2,
njb=4}``: the wall is an arc-line-arc-line ``Polyline`` (nose arc, cone
flank, shoulder arc, aft wall) and the inflow boundary is a cubic
``Bezier``, both arc-length parameterized. Sub-block corners are placed
parametrically along the bounding edges (TFI for the interior ones);
block-to-block connectivity is inferred from the shared corner objects.

The Lua wall clustering (GeometricFunction, h_wall=1e-4, r=1.2) is realised
with egg's boundary-layer machinery: ``set_boundary_layer`` records the spec,
the TMOP pass smooths against :func:`egg.smoothing.build_boundary_layer_target`,
and :func:`egg.smoothing.enforce_boundary_layer_spacing` enforces the exact
wall spacing at initialization and as a post-pass. Where the Lua example
massages the interior with manual control points, the TMOP smoothing pass
does that job.

Reference: D. Bianchi et al., Int. J. Heat Mass Transfer 177 (2021) 121430.

The command-line surface lives in ``driver.py``; run
``uv run phoebus.py --help`` for options.
"""

import math

from egg.pipeline import PipelineConfig, generate_steps
from egg.smoothing import build_boundary_layer_target

from egg.geometry import Arc, Bezier, Edge, Line, Polyline, Vector3
from egg.topology.builder import TopologyBuilder

H_WALL = 1.0e-4  # first cell height at the wall [m] (h_wall in the Lua)
BL_GROWTH = 1.2  # near-wall geometric growth ratio (r in the Lua)


def build_phoebus(grid_level: int = 1, n_fixed: int = 1):
    # -- geometric parameters (verbatim from grid.lua) -----------------------
    Rn = 20.0e-3  # nose radius [m]
    beta = math.radians(45.0)  # cone angle [rad]
    Rs = 1.57e-3  # shoulder radius [m]
    Rb = 20.0e-3  # base radius [m]
    L_aft = Rs  # artificial wall length downstream of shoulder [m]
    Ri = Rn * 1.2  # initial radius of inflow boundary [m]
    Rf = Rn * 1.5  # final radius of inflow boundary [m]

    # -- points ---------------------------------------------------------------
    A = Vector3(x=0.0, y=0.0)
    B = Vector3(x=Rn * (1.0 - math.cos(beta)), y=Rn * math.sin(beta))
    Cy = Rb + Rs * (math.cos(beta) - 1.0)
    C = Vector3(x=B.x + (Cy - B.y) / math.tan(beta), y=Cy)
    D = Vector3(x=C.x + Rs * math.sin(beta), y=Rb)
    E = Vector3(x=D.x + L_aft, y=D.y, fixed=True)
    theta = math.atan(E.y / (Rn - E.x))  # angle of outflow boundary
    F = Vector3(x=Rn - Rf * math.cos(theta), y=Rf * math.sin(theta), fixed=True)
    G = Vector3(x=Rn - Ri, y=0.0, fixed=True)
    centre_n = Vector3(x=Rn, y=0.0)
    centre_s = Vector3(x=D.x, y=D.y - Rs)

    # -- paths -> grid edges (axis0 = inflow->wall, axis1 = along the wall) ---
    wall = Edge(
        Polyline(
            [
                Arc(p0=A, p1=B, centre=centre_n),
                Line(p0=B, p1=C),
                Arc(p0=C, p1=D, centre=centre_s),
                Line(p0=D, p1=E),
            ]
        ),
        arc_length=True,
    )
    inflow = Edge(
        Bezier(points=[G, Vector3(x=G.x, y=B.y), Vector3(x=B.x, y=0.8 * F.y), F]),
        arc_length=True,
    )
    symm = Edge(Line(p0=G, p1=A))  # south: symmetry axis
    outflow = Edge(Line(p0=F, p1=E))  # north: outflow boundary

    # -- 2x4 block array (nib x njb of registerFluidGridArray) ----------------
    n_refine = 2 ** (grid_level / 2)
    n_wall_normal = math.ceil(20 * n_refine)  # cells inflow->wall
    n_along_wall = math.ceil(100 * n_refine)  # cells along the wall

    bld = TopologyBuilder(d=2)
    # Sub-block corners on the bounding edges (TFI inside), blocks from the
    # shared corner objects, boundary faces associated with their edges.
    bld.add_block_array(
        south=symm, north=outflow, west=inflow, east=wall,
        nib=2, njb=4, res=(n_wall_normal, n_along_wall),
    )

    # The outflow meets the wall obliquely; relaxing orthogonality towards it
    # lets the clustering target follow it with sheared cells instead of
    # trading away the near-wall layer heights.
    bld.set_boundary_layer(
        wall, first_height=H_WALL, growth=BL_GROWTH, n_fixed=n_fixed,
        relax_orthogonality=(outflow,))

    topology = bld.build()
    entities = {
        "wall": wall.entity,
        "inflow": inflow.entity,
        "symm": symm.entity,
        "outflow": outflow.entity,
    }
    return topology, entities


def main():
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from driver import finish, parse_args

    a = parse_args()

    topo, ents = build_phoebus(grid_level=a.grid_level, n_fixed=a.pin_layers)

    # --pin-layers > 0 smooths against the aspect-ratio target and pins the
    # first n_fixed layers exactly; the default path runs the pure shape
    # metric and restores the wall spacing with the respace post-pass.
    target = build_boundary_layer_target(topo) if a.pin_layers > 0 else None
    grid = topo.initialize_grid()
    cfg = PipelineConfig(
        tmop_sweeps=a.tmop_sweeps,
        tmop_chunk=a.chunk,
        omega=a.omega,
        device=a.device,
        pin_sweeps=a.pin_sweeps if a.pin_layers > 0 else 0,
        respace=a.pin_layers == 0,
    )
    steps = generate_steps(grid, target, cfg, untangle_direct=True)

    finish(grid, topo, ents, steps, a, title="Phoebus capsule")


if __name__ == "__main__":
    main()
