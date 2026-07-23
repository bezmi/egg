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

"""An egg in a rectangle, with the geometry imported from an Inkscape SVG.

The same domain and topology as the ``egg`` example, but no curve is
constructed in Python: :func:`egg.geometry.svg_import` reads ``egg.svg``
(alongside this script) and every boundary is looked up by the label it
carries in Inkscape's Objects panel — ``inflow`` (left), ``outflow``
(right), ``wall_top`` / ``wall_bottom``, and the closed ``egg`` curve.
Domain corners come from the imported wall endpoints; topology nodes are
placed on the imported curves with ``Edge.place_node`` exactly as if the
geometry had been built in code.

Reshape the drawing in Inkscape (drag the egg's spline handles, move a
wall) and rerun: the topology follows the geometry. Two things must
survive an edit: the labels, and each path's direction (the node-placement
fractions and wall-to-wall corner associations assume bottom west→east,
right south→north, top east→west, left south→north in model space — SVG
y-down is flipped on import, so that is top-left→bottom-left etc. as drawn).

Boundary markers carry the same physical names as the ``egg`` example:
``inflow`` / ``outflow`` / ``wall`` / ``egg``. The egg surface gets the
same wall-normal boundary-layer clustering and pinned-layer options too
(``--bl-first-height`` / ``--bl-growth`` / ``--pin-layers`` /
``--pin-sweeps``) — the clustering attaches to the *imported* curve.

The command-line surface lives in ``driver.py``; run
``uv run egg_svg.py --help`` for options.
"""

from pathlib import Path

from egg.pipeline import (
    ControlPointSmoother,
    FasSmoother,
    JacobiSmoother,
    Presmooth,
    Pin,
    Respace,
    Untangle,
    generate_steps,
)

from egg.geometry import svg_import, svg_topology

SVG_FILE = Path(__file__).resolve().parent / "egg.svg"


def build_egg_from_svg(bl_first_height=0.0, bl_growth=1.5, n_fixed=2, res=12):
    """Egg-in-rectangle topology, geometry *and blocking* drawn in ``egg.svg``.

    Nothing about the block layout is written in Python: the ``topology``
    layer of the SVG holds a straight-line wireframe whose endpoints are the
    block corners, and :func:`egg.geometry.svg_topology` turns its planar
    faces into the twelve O-grid blocks. Corners, geometry associations and
    boundary tags are all inferred geometrically (a corner on two walls is
    pinned; an edge whose both ends lie on a labelled curve follows it), and
    the four 5-valent singularities where the O-ring meets the mid square
    fall out for free — they are just the nodes where five wireframe edges
    cross.

    Redraw either layer in Inkscape and rerun: reshape the egg, or drag a
    blocking node to move a singularity, and the mesh follows. ``res`` is a
    single uniform cells-per-axis count (so every shared edge stays
    conforming); per-block clustering is a refinement left to code.
    """
    dom = svg_import(SVG_FILE)
    b, entities = svg_topology(dom, res=res)

    if bl_first_height > 0.0:
        # Wall-normal clustering on the imported egg curve; the egg is a
        # closed interior wall, so no domain boundary meets it obliquely
        # and relax_orthogonality stays empty.
        b.set_boundary_layer(
            dom.edge("egg"),
            first_height=bl_first_height,
            growth=bl_growth,
            n_fixed=n_fixed,
        )

    return b.build(), entities


def _smoother(a, *, metric, cluster):
    """The smoothing-phase stage(s) chosen by ``--smoother``.

    The clustering target is built by the pipeline from the topology's bl
    specs, so the smoother carries only ``metric`` and whether clustering is
    on. Control mode is preceded by a matching nodal pre-pass so the net fit
    starts from a smooth grid.
    """
    common = dict(
        chunk=a["chunk"],
        omega=a["omega"],
        metric=metric,
        cluster_boundary_layers=cluster,
        bl_blend_neighbours=False,
    )
    s = a["smoother"]
    if s == "fas":
        return [FasSmoother(sweeps=a["tmop_sweeps"], **common)]
    if s == "control_point":
        return [
            Presmooth(JacobiSmoother(sweeps=100, **dict(common, chunk=100))),
            ControlPointSmoother(**common),
        ]
    return [JacobiSmoother(sweeps=a["tmop_sweeps"], **common)]


def setup(a, *, direct=True):
    """Topology, grid, and stage list from parsed args — shared by the CLI
    ``main()`` and the web UI (which passes the parser defaults).

    NB: ``--pin-layers`` (→ ``n_fixed``) only takes effect when
    ``--bl-first-height`` > 0 — without a first height there is no
    boundary layer to pin, and ``--pin-sweeps`` pins those same layers,
    so it too is a no-op without one.
    """
    topo, ents = build_egg_from_svg(
        bl_first_height=a["bl_first_height"],
        bl_growth=a["bl_growth"],
        n_fixed=a["pin_layers"],
        res=a.get("res", 12),
    )

    # --pin-layers > 0 smooths against the boundary-layer clustering target
    # and pins the first n_fixed layers exactly; the default path runs the
    # pure shape metric (no clustering) and restores the egg spacing with the
    # respace post-pass. cluster_boundary_layers picks between them.
    #
    # bl_blend_neighbours=False: continuing the clustering profile into the
    # blocks behind the O-ring would drag them toward the mid square — they
    # are far too coarse to carry it. Confine the profile to the ring; its
    # outer rows reach the neighbour spacing on their own.
    pin = a["bl_first_height"] > 0.0 and a["pin_layers"] > 0
    grid = topo.initialize_grid()
    metric = a.get("metric", "shape")
    stages = [
        Untangle(sweeps_per_delta=a["sweeps_per_delta"], direct=direct),
        *_smoother(a, metric=metric, cluster=pin),
    ]
    # pin: fix the first layers exactly and re-smooth; else, with a boundary
    # layer requested, restore the spacing with the respace post-pass.
    if pin:
        stages.append(
            Pin(
                JacobiSmoother(
                    sweeps=a["pin_sweeps"], chunk=a["chunk"], omega=a["omega"]
                )
            )
        )
    elif a["bl_first_height"] > 0.0:
        stages.append(Respace())
    return topo, ents, grid, stages


def main():
    from driver import finish, parse_args

    a = parse_args()
    topo, ents, grid, stages = setup(vars(a), direct=not a.plot_live)
    steps = generate_steps(grid, stages=stages, device=a.device)

    finish(
        grid,
        topo,
        ents,
        steps,
        a,
        title="egg in rectangle (from Inkscape SVG)",
        mindet_title="min det A (TMOP only)",
    )


if __name__ == "__egg_webui__":  # running inside the egg web UI
    import egg.webui as egg_webui

    # CLI defaults, mirroring driver.py — edit freely
    a = egg_webui.params(
        # Uniform cells per block axis, inferred wholly from the SVG blocking
        # (editable() surfaces it as a numeric box in the run panel).
        res=egg_webui.editable(12, label="resolution"),
        # editable() surfaces a value in the run panel with a typed input;
        # choices=[...] renders a dropdown. metric="shape" is the classic
        # scale-invariant metric; "shape_size" also equalises cell areas.
        metric=egg_webui.editable("shape_size", choices=["shape", "shape_size"]),
        bl_first_height=5.0e-3,
        bl_growth=1.5,
        pin_layers=2,
        pin_sweeps=300,
        sweeps_per_delta=20,
        tmop_sweeps=600,
        chunk=50,
        smoother="jacobi",
        omega=0.8,
        device="cpu",
    )
    topo, ents, grid, stages = setup(a, direct=False)
    egg_webui.run(grid, generate_steps(grid, stages=stages, device=a["device"]))

if __name__ == "__main__":
    main()
