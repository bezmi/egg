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

The acute (~36°) corner where the inflow arc meets the outflow gets the
best treatment found across an extensive comparison (nested 3-valent
dipole + the size-aware ``shape_size`` metric, both on by default):

- ``dipole=True`` telescopes n nested 3-valent splits into the corner
  block, refining the corner cell geometrically from topology alone.
- ``metric="shape_size"`` adds ``(det T - 1)^2`` to the objective, so the
  smoother actively equalises cell areas (the pure shape metric is
  scale-invariant and cannot).

Measured at res 10x10 (cell-area CV / max-min ratio): stock + shape
1.78 / 568; dipole + shape 0.74 / 400; stock + shape_size 0.65 / 47;
**dipole + shape_size 0.47 / 79** — and with the BL target active the
combination still wins (0.53 / 41 vs 1.00 / 419). Set ``dipole=False`` /
``metric="shape"`` to reproduce the standard single-block-corner approach.
The other corner experiments (boundary fan, apex split, sliding corner,
reversed 5→3 polarity) live in the git history of this file.

The command-line surface lives in ``driver.py``; run
``uv run capsule.py --help`` for options.
"""

import math

from egg.geometry import Arc, Edge, Line, Polyline, Vector3
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


def build_capsule(
    res_i=20, res_j=20, bl_first_height=0.0, bl_growth=1.3, n_fixed=2, dipole=True
):
    """3 x 12 block array between the capsule body and the outer arc.

    With ``dipole`` (default) the inflow/outflow corner block is rebuilt
    around n nested 3-valent splits telescoping into the acute corner —
    every level's diagonal runs from the previous singularity to the new
    one and its lines land on the inflow and outflow faces, wrapping the
    shrinking corner block. Adding a level regularises the previous
    singularity to 4-valent, so ANY n carries exactly one 3-valent (the
    innermost s_n) and one 5-valent (c1_11, the classic 3-5 pair) — a
    geometric corner refinement built purely from topology, every node
    free. ``dipole=False`` keeps the plain tensor block array.
    """
    outer, body, south, north = build_paths()

    # Grid edges (axis 0 = inflow -> wall, axis 1 = along the body). Naming each
    # edge auto-derives its SU2 marker on every associated face (the block array
    # associates the outer faces below), so no tag_boundary calls are needed.
    inflow = Edge(outer, name="inflow")  # west, symmetry -> outflow
    wall = Edge(body, arc_length=True, name="wall")  # east, symmetry -> outflow
    symmetry = Edge(south, name="symmetry")  # south, inflow -> wall
    outflow = Edge(north, name="outflow")  # north, inflow -> wall

    nib, njb = 3, 12
    j1 = njb - 1
    b = TopologyBuilder(d=2)

    # Sub-block corners on the bounding paths (TFI inside), blocks from the
    # shared corner objects, boundary faces associated with their edges.
    # With the dipole, the corner block (0, j1) is skipped and rebuilt by
    # hand below.
    skip = {(0, j1)} if dipole else set()
    corner, names = b.add_block_array(
        south=symmetry,
        north=outflow,
        west=inflow,
        east=wall,
        nib=nib,
        njb=njb,
        res=(nib * res_i, njb * res_j),
        skip=skip,
    )
    # The array associated the outer faces with their named edges, so inflow /
    # wall / symmetry / outflow markers auto-derive.

    boundary = []
    if dipole:
        from egg.geometry.frontend2d import tfi_point

        n = 2  # nesting depth (1 = the plain 3-block dipole)
        f = 0.5  # per-level extent shrink toward the corner
        # Radial cells per ring, outermost -> innermost. This is each
        # ring's one free count (tangential counts are inherited from the
        # array: res_i along the arc, res_j along the outflow), so the
        # cell size blends into the corner at a controllable rate: with
        # extents shrinking by f, equal counts already halve the radial
        # size per ring.
        kr = [max(2, res_j // 2)] * n
        u1, v1 = 1 / nib, 1 - 1 / njb  # b0_11's outflow/inflow face spans
        s_prev, an_prev, aw_prev = corner[1, j1], corner[1, njb], corner[0, j1]
        for m in range(1, n + 1):
            um, vm = u1 * f**m, 1 - (1 - v1) * f**m
            b.add_corner(f"an{m}", outflow.place_node(um), fixed=False)
            b.add_corner(f"aw{m}", inflow.place_node(vm), fixed=False)
            b.add_corner(
                f"s{m}",
                tfi_point(um, vm, symmetry, outflow, inflow, wall),
                fixed=False,
            )
            b.add_block(
                f"b0_11b{m}",
                sw=f"s{m}",
                se=s_prev,
                ne=an_prev,
                nw=f"an{m}",
                res=(kr[m - 1], res_j),
            )
            b.add_block(
                f"b0_11c{m}",
                sw=aw_prev,
                se=s_prev,
                ne=f"s{m}",
                nw=f"aw{m}",
                res=(res_i, kr[m - 1]),
            )
            boundary.append((f"b0_11c{m}", 0, 0, inflow))
            boundary.append((f"b0_11b{m}", 1, 1, outflow))
            s_prev, an_prev, aw_prev = f"s{m}", f"an{m}", f"aw{m}"
        b.add_block(
            "b0_11a",
            sw=aw_prev,
            se=s_prev,
            ne=an_prev,
            nw=corner[0, njb],
            res=(res_i, res_j),
        )
        boundary.append(("b0_11a", 0, 0, inflow))
        boundary.append(("b0_11a", 1, 1, outflow))
    for name, axis, side, edge in boundary:
        b.associate(name, axis, side, edge)  # marker auto-derives from edge name

    if bl_first_height > 0.0:
        # relax_orthogonality is a no-op while the outflow meets the wall at
        # a right angle; declared for consistency with the phoebus example
        # (and so a slanted outflow, as in the gdtk original, keeps its
        # layer heights).
        b.relax_orthogonality(outflow)
        b.set_boundary_layer(
            wall,
            first_height=bl_first_height,
            growth=bl_growth,
            n_fixed=n_fixed,
        )

    topology = b.build()
    return topology, topology.entities


def _smoother(a, *, metric, cluster, c2, ortho):
    """The TMOP-phase smoother stage chosen by ``--smoother``.

    The interface C2 / orthogonality terms are node-mode, so they attach to the
    nodal smoothers; the control-point smoother has its own seam dials.
    """
    common = dict(
        chunk=a["chunk"],
        omega=a["omega"],
        metric=metric,
        cluster_boundary_layers=cluster,
    )
    s = a["smoother"]
    if s == "control_point":
        return [
            Presmooth(JacobiSmoother(sweeps=100, **dict(common, chunk=100))),
            ControlPointSmoother(**common),
        ]
    node = dict(
        sweeps=a["tmop_sweeps"], interface_c2=c2, interface_ortho=ortho, **common
    )
    return [FasSmoother(**node) if s == "fas" else JacobiSmoother(**node)]


def setup(a, *, direct=True):
    """Topology, grid, and stage list from parsed args — shared by the CLI
    ``main()`` and the web UI (which passes the parser defaults).

    NB: ``--pin-layers`` (→ ``n_fixed``) only takes effect when
    ``--bl-first-height`` > 0 — without a first height there is no
    boundary layer to pin, and ``--pin-sweeps`` pins those same layers,
    so it too is a no-op without one.
    """
    topo, ents = build_capsule(
        res_i=a["res_i"],
        res_j=a["res_j"],
        bl_first_height=a["bl_first_height"],
        bl_growth=a["bl_growth"],
        n_fixed=a["pin_layers"],
        dipole=a.get("dipole", True),
    )

    # --pin-layers > 0 smooths against the boundary-layer clustering target
    # and pins the first n_fixed layers exactly; the default path runs on the
    # plain metric (no clustering) and restores the wall spacing with the
    # respace post-pass. cluster_boundary_layers picks between them: the
    # pipeline builds the clustering target from the set_boundary_layer specs
    # itself, sizing the shape_size far field correctly.
    pin = a["bl_first_height"] > 0.0 and a["pin_layers"] > 0
    grid = topo.initialize_grid()
    metric = a.get("metric", "shape")
    # Optional block-interface C2 curvature term (interface_only: de-kink the
    # seams between the O-grid and the wake/outer blocks, without touching the
    # legitimately curved clustered near-wall cells).
    c2w, c2s = a.get("c2_weight", 0.0), a.get("c2_singularity", 0.0)
    c2 = (
        {"weight": c2w, "interface_only": True, "singularity_weight": c2s}
        if (c2w > 0.0 or c2s > 0.0)
        else None
    )
    # Optional block-interface orthogonality term (pulls the cross-seam edge
    # perpendicular to the seam); composes with the C2 term.
    ortho = (
        {
            "mode": "normal",
            "weight": a["ortho_weight"],
            "n_layers": a.get("ortho_layers", 3),
            "cluster_relax": a.get("ortho_relax", 1.0),
        }
        if a.get("ortho_weight", 0.0) > 0.0
        else None
    )
    stages = [
        Untangle(sweeps_per_delta=a["sweeps_per_delta"], direct=direct,
                 name="untangle folds"),
        *_smoother(a, metric=metric, cluster=pin, c2=c2, ortho=ortho),
    ]
    if pin:
        stages.append(
            Pin(
                JacobiSmoother(
                    sweeps=a["pin_sweeps"], chunk=a["chunk"], omega=a["omega"]
                ),
                name="pin boundary layers",
            )
        )
    elif a["bl_first_height"] > 0.0:
        stages.append(Respace(name="respace walls"))
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
        title="FIRE II capsule",
        mindet_title="min det A (TMOP only)",
    )


if __name__ == "__egg_webui__":  # running inside the egg web UI
    import egg.webui as egg_webui

    # CLI defaults, mirroring driver.py — edit freely
    a = egg_webui.params(
        res_i=10,
        res_j=10,
        bl_first_height=4.0e-4,
        bl_growth=1.3,
        pin_layers=2,
        pin_sweeps=40,
        sweeps_per_delta=20,
        tmop_sweeps=5000,
        chunk=10,
        smoother="jacobi",
        # editable() surfaces a value in the run-parameters panel with a
        # typed input; choices=[...] renders a dropdown. The classic
        # combination for comparison: metric="shape", dipole=False.
        metric=egg_webui.editable("shape_size", choices=["shape", "shape_size"]),
        dipole=egg_webui.editable(True, label="corner dipole"),
        omega=0.8,
        # block-interface C2 curvature-continuity weight (0 = off); de-kinks the
        # grid lines crossing the block seams. interface-only, so it leaves the
        # clustered near-wall cells alone.
        c2_weight=egg_webui.editable(
            0.0, label="interface C2 weight", show_if={"smoother": ["jacobi", "fas"]}
        ),
        c2_singularity=egg_webui.editable(
            0.0,
            label="singularity ring C2 weight",
            show_if={"smoother": ["jacobi", "fas"]},
        ),
        ortho_weight=egg_webui.editable(
            0.0,
            label="interface orthogonality weight",
            show_if={"smoother": ["jacobi", "fas"]},
        ),
        ortho_layers=egg_webui.editable(
            3,
            label="orthogonality band layers",
            show_if={"smoother": ["jacobi", "fas"]},
        ),
        ortho_relax=egg_webui.editable(
            1.0,
            label="orthogonality clustering relax",
            show_if={"smoother": ["jacobi", "fas"]},
        ),
        device="cpu",
    )
    topo, ents, grid, stages = setup(a, direct=False)
    egg_webui.run(grid, generate_steps(grid, stages=stages, device=a["device"]))

if __name__ == "__main__":
    main()
