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

"""FIRE II capsule forebody, the simplified read-along version of capsule.py.

This example runs in the egg web UI. The flow is a single straight line, split
into three readable pieces:

    build_geometry()   -> the four boundary paths as named curves
    build_topology()   -> a plain 3 x 12 block array over them (the base)
    the web UI block   -> wrap the base in an ExplicitTopology whose blocking is
                          editable(), so you can draw extra blocks over the array
                          live; flatten it and smooth (untangle -> TMOP -> pin)

Everything that would branch the pipeline in capsule.py is fixed here: the corner
dipole is off (plain block array), the smoother is plain block-Jacobi, and the
boundary-layer clustering + pin pass always run. The knobs that don't change the
pipeline shape -- resolution, TMOP sweeps, and the TMOP metric -- stay editable
in the run panel. See capsule.py for the full configurable version.
"""

import math

from egg.geometry import Arc, Edge, Line, Polyline, Vector3
from egg.enums import TmopMetric
from egg.pipeline import (
    JacobiSmoother,
    Pin,
    Untangle,
    generate_steps,
)
from egg.topology import ExplicitTopology
from egg.topology.builder import TopologyBuilder

# Fixed pipeline knobs, hard-coded from capsule.py's defaults so the pipeline
# below stays one straight line (capsule.py exposes these on its command line).
BL_FIRST_HEIGHT = 4.0e-4  # first wall-normal cell height on the capsule body
BL_GROWTH = 1.3  # boundary-layer geometric growth ratio away from the wall
PIN_LAYERS = 2  # near-wall layers pinned at their exact geometric heights
PIN_SWEEPS = 40  # TMOP sweeps for the pinned re-run
SWEEPS_PER_DELTA = 20  # untangle sweeps per delta-continuation level
OMEGA = 0.8  # block-Jacobi SOR / damping weight for the TMOP phase
CHUNK = 10  # TMOP sweeps per yielded step (animation granularity)


def build_geometry():
    """The four boundary paths of the FIRE II forebody, as named curves keyed by
    the label the blocking binds to (``{label: entity}``)."""
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

    body = Polyline([Arc(ai, bi, oi), Arc(bi, ci, pi_), Line(ci, di)]).named("wall")

    # The wall is horizontal after the shoulder, so a vertical outflow meets it
    # at a right angle (the gdtk original slants it, which shears the cells where
    # it meets the wall). The outer arc ends directly above the wall's end point.
    ao = oi + Ro * Vector3(-1.0, 0.0)
    thetao = math.acos((Ri - di.x) / Ro)
    do = oi + Ro * Vector3(-math.cos(thetao), math.sin(thetao))

    outer = Arc(ao, do, oi).named("inflow")
    south = Line(ao, ai).named("symmetry")
    north = Line(do, di).named("outflow")
    return {"inflow": outer, "wall": body, "symmetry": south, "outflow": north}


def build_topology(geometry, res_i, res_j):
    """A plain 3 x 12 block array over the geometry, with boundary-layer
    clustering on the wall. Returns the base topology."""
    # Axis 0 = inflow -> wall, axis 1 = along the body. Each Edge inherits its
    # curve's name, so the outer faces the array associates auto-derive the
    # inflow / wall / symmetry / outflow SU2 markers.
    wall = Edge(geometry["wall"], arc_length=True)

    b = TopologyBuilder(d=2)
    b.add_block_array(
        south=Edge(geometry["symmetry"]),
        north=Edge(geometry["outflow"]),
        west=Edge(geometry["inflow"]),
        east=wall,
        nib=3,
        njb=12,
        res=(3 * res_i, 12 * res_j),
    )

    # Cluster cells toward the capsule wall: a fixed first cell height growing
    # geometrically away from the body (the pin pass fixes the first PIN_LAYERS).
    wall.clustered(first_height=BL_FIRST_HEIGHT, growth=BL_GROWTH, n_fixed=PIN_LAYERS)

    return b.build()


if __name__ == "__egg_webui__":  # this example runs in the egg web UI
    import egg.webui as egg_webui

    from egg import editable

    # Run-panel knobs (edit from the run-parameters strip). editable() surfaces a
    # value with a typed input; choices=[...] renders a dropdown. These are the
    # pipeline-neutral knobs.
    a = egg_webui.params(
        res_i=egg_webui.editable(10, label="cells per block (i)"),
        res_j=egg_webui.editable(10, label="cells per block (j)"),
        tmop_sweeps=egg_webui.editable(3000, label="TMOP sweeps"),
        metric=egg_webui.editable("shape_size", choices=["shape", "shape_size"]),
        device="cpu",
    )

    geometry = build_geometry()
    topo = build_topology(geometry, a["res_i"], a["res_j"])

    # Draw extra blocking on top of the 3 x 12 base array in the topology edit
    # view: snap corners onto the geometry, bind faces to the named curves, set
    # per-edge resolution; `save edits` writes the connectivity back into this
    # literal. Empty to start, so it begins as the plain array.
    capsule_topo = ExplicitTopology(
        base=topo,
        geometry=geometry,
        connectivity=editable({"nodes": {}, "edges": []}),
    )

    # Flatten the (possibly edited) blocking to a topology, then smooth it. The
    # base is always valid, so the run button is live from the start; edits that
    # don't yet flatten leave flat_topo None until they're fixed.
    flat_topo, _diagnostics = capsule_topo.flatten()
    if flat_topo is not None:
        grid = flat_topo.initialize_grid()
        # direct=False yields the untangle per delta so the web UI can animate it.
        stages = [
            Untangle(
                sweeps_per_delta=SWEEPS_PER_DELTA, direct=False, name="untangle folds"
            ),
            JacobiSmoother(
                sweeps=a["tmop_sweeps"],
                chunk=CHUNK,
                omega=OMEGA,
                metric=TmopMetric(a["metric"]),
                cluster_boundary_layers=True,
            ),
            Pin(
                JacobiSmoother(sweeps=PIN_SWEEPS, chunk=CHUNK, omega=OMEGA),
                name="pin boundary layers",
            ),
        ]
        egg_webui.run(grid, generate_steps(grid, stages=stages, device=a["device"]))
