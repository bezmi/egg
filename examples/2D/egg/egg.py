# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""An egg in a rectangle — O-grid ring, edge blocks, corner blocks.

The namesake example. The egg boundary is a closed spline through an egg
curve (fat end down); around it sits the same O-grid-in-rectangle topology
as the spline-blob example, built leaning on the builder's inference:
corners made with ``Edge.place_node`` slide on their host curve, shared
corner objects become block connections, and faces whose corners are all
nodes on one edge become geometry associations. Only the corner blocks
need explicit ``associate()`` calls — their wall faces span two walls each.

Boundary markers carry physical names for a flow solve: the left wall is
``inflow``, the right wall ``outflow``, the top and bottom walls share the
``wall`` marker, and the egg surface is ``egg``.

The egg surface gets wall-normal boundary-layer clustering
(``--bl-first-height`` / ``--bl-growth``); ``--pin-layers`` smooths
against the aspect-ratio target and pins the first layers at their exact
geometric heights, ``--pin-layers 0`` restores the spacing with the
respace post-pass instead.

The command-line surface lives in ``driver.py``; run
``uv run egg.py --help`` for options.
"""

import numpy as np

from egg.pipeline import PipelineConfig, generate_steps

from egg.geometry import Edge, Line, Spline, Vector3
from egg.topology.builder import TopologyBuilder


def build_egg_in_rectangle(bl_first_height=0.0, bl_growth=1.5, n_fixed=2):
    """Egg-in-rectangle topology; markers: inflow / outflow / wall / egg."""
    # Egg boundary: closed spline through an egg curve, fat end down.
    theta = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
    ring = [
        Vector3(2.0 + (0.66 - 0.15 * np.sin(t)) * np.cos(t), 2.0 + 0.85 * np.sin(t))
        for t in theta
    ]
    egg = Edge(Spline(ring, closed=True), name="egg")

    # Domain walls and the sliding nodes that split them. Naming each curve
    # auto-derives its SU2 marker on every associated face. The top and bottom
    # walls are named distinctly (so both draw) but carry tag="wall", so they
    # export under one shared 'wall' marker — no per-face tag_boundary needed.
    sw, se = Vector3(0, 0, fixed=True), Vector3(4, 0, fixed=True)
    ne, nw = Vector3(4, 4, fixed=True), Vector3(0, 4, fixed=True)
    bottom = Edge(Line(p0=sw, p1=se), name="wall_bottom", tag="wall")
    right = Edge(Line(p0=se, p1=ne), name="outflow")
    top = Edge(Line(p0=ne, p1=nw), name="wall_top", tag="wall")
    left = Edge(Line(p0=sw, p1=nw), name="inflow")
    bsw, bse = bottom.place_node(0.25), bottom.place_node(0.75)
    rse, rne = right.place_node(0.25), right.place_node(0.75)
    tne, tnw = top.place_node(0.25), top.place_node(0.75)
    lnw, lsw = left.place_node(0.75), left.place_node(0.25)

    # O-ring inner corners slide on the egg itself; mid square ties the
    # ring to the outer blocks.
    ine, inw, isw, ise = (egg.place_node(f) for f in (0.125, 0.375, 0.625, 0.875))
    msw, mse, mne, mnw = (Vector3(*p) for p in [(1, 1), (3, 1), (3, 3), (1, 3)])

    b = TopologyBuilder(d=2)
    # O-ring around the egg; the inner face's association with the egg is
    # inferred, and its 'egg' marker auto-derives from the egg entity's name.
    for nm, c_sw, c_nw, c_se, c_ne in [
        ("o_s", msw, isw, mse, ise),
        ("o_e", mse, ise, mne, ine),
        ("o_n", mne, ine, mnw, inw),
        ("o_w", mnw, inw, msw, isw),
    ]:
        # Clustering needs the ring deep enough to carry the layer profile:
        # a shallow ring forces the steep part of the target across the
        # mid-square interface, which folds cells there (measured).
        o_res_j = 16 if bl_first_height > 0.0 else 8
        b.add_block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=(20, o_res_j))
    # Edge blocks between walls and mid square; wall associations inferred, so
    # every marker (inflow / outflow / wall) auto-derives from the named edges.
    for nm, c_sw, c_nw, c_se, c_ne in [
        ("e_s", bsw, msw, bse, mse),
        ("e_e", rse, mse, rne, mne),
        ("e_n", tne, mne, tnw, mnw),
        ("e_w", lnw, mnw, lsw, msw),
    ]:
        b.add_block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=(20, 10))
    # Corner blocks: two walls meet at each, so associations stay explicit;
    # markers auto-derive from the edges' tags (wall_top/wall_bottom -> 'wall').
    for nm, w0, w1, c_sw, c_nw, c_se, c_ne in [
        ("c_sw", left, bottom, sw, lsw, bsw, msw),
        ("c_se", bottom, right, se, bse, rse, mse),
        ("c_ne", right, top, ne, rne, tne, mne),
        ("c_nw", top, left, nw, tnw, lnw, mnw),
    ]:
        b.add_block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=(10, 10))
        b.associate(nm, 0, 0, w0)
        b.associate(nm, 1, 0, w1)

    if bl_first_height > 0.0:
        # Wall-normal clustering on the egg surface; the egg is a closed
        # interior wall, so no domain boundary meets it obliquely and
        # relax_orthogonality stays empty.
        b.set_boundary_layer(
            egg, first_height=bl_first_height, growth=bl_growth, n_fixed=n_fixed, n_layers=2
        )

    topology = b.build()
    return topology, topology.entities


def setup(a):
    """Topology, grid, and config from parsed args — shared by the CLI
    ``main()`` and the web UI (which passes the parser defaults).

    NB: ``--pin-layers`` (→ ``n_fixed``) only takes effect when
    ``--bl-first-height`` > 0 — without a first height there is no
    boundary layer to pin, and ``--pin-sweeps`` pins those same layers,
    so it too is a no-op without one.
    """
    topo, ents = build_egg_in_rectangle(
        bl_first_height=a["bl_first_height"],
        bl_growth=a["bl_growth"],
        n_fixed=a["pin_layers"],
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
    # Optional block-interface C2 curvature-continuity term: de-kinks grid lines
    # crossing the O-grid/H-grid seams (and the 5-way singularities between them).
    # interface_only: act on windows that cross a seam, not the (legitimately
    # curved) clustered interior near the egg wall.
    c2w, c2s = a.get("c2_weight", 0.0), a.get("c2_singularity", 0.0)
    c2 = (
        {"weight": c2w, "interface_only": True, "singularity_weight": c2s}
        if (c2w > 0.0 or c2s > 0.0)
        else None
    )
    # Optional block-interface orthogonality term: pulls the cross-seam edge
    # perpendicular to the seam (mode="normal"), which also straightens the
    # crossing (continuity). Composes with the C2 term above.
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
    cfg = PipelineConfig(
        sweeps_per_delta=a["sweeps_per_delta"],
        tmop_sweeps=a["tmop_sweeps"],
        tmop_chunk=a["chunk"],
        tmop_smoother=a["smoother"],
        tmop_metric="shape_size",
        cluster_boundary_layers=pin,
        bl_blend_neighbours=False,
        omega=a["omega"],
        interface_c2=c2,
        interface_ortho=ortho,
        device=a["device"],
        pin_sweeps=a["pin_sweeps"] if pin else 0,
        respace=a["bl_first_height"] > 0.0 and not pin,
    )
    return topo, ents, grid, cfg


def main():
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from driver import finish, parse_args

    a = parse_args()
    topo, ents, grid, cfg = setup(vars(a))
    steps = generate_steps(grid, config=cfg, untangle_direct=not a.plot_live)

    finish(
        grid,
        topo,
        ents,
        steps,
        a,
        title="egg in rectangle",
        mindet_title="min det A (TMOP only)",
    )


if __name__ == "__egg_webui__":  # running inside the egg web UI
    import egg_webui

    from egg import editable
    from egg.topology import ExplicitTopology

    # CLI defaults, mirroring driver.py — edit freely
    a = egg_webui.params(
        bl_first_height=5.0e-3,
        bl_growth=1.5,
        pin_layers=1,
        pin_sweeps=5000,
        sweeps_per_delta=20,
        tmop_sweeps=5000,
        chunk=50,
        smoother="jacobi",
        omega=0.8,
        # block-interface terms (0 = off), editable in the run panel: C2 de-kinks
        # the grid lines crossing block seams; orthogonality pulls the cross-seam
        # edge perpendicular to the seam.
        c2_weight=egg_webui.editable(10, label="interface C2 weight"),
        c2_singularity=egg_webui.editable(1, label="singularity ring C2 weight"),
        ortho_weight=egg_webui.editable(0, label="interface orthogonality weight"),
        ortho_layers=egg_webui.editable(3, label="orthogonality band layers"),
        ortho_relax=egg_webui.editable(1.0, label="orthogonality clustering relax"),
        device="cpu",
    )
    topo, ents, grid, cfg = setup(a)

    # The egg O-grid is the frozen base; the editable({}) blocking is what the
    # topology edit view (the "edit" view) draws into — snap to a base corner,
    # bifurcate a block edge, bind a face to a curve by name — and `save edits`
    # writes it back into this literal. The grid the smoother relaxes is rebuilt
    # from the edited topology, so the blocking is actually meshed, not just drawn.
    egg_topo = ExplicitTopology(
        base=topo,
        geometry=ents,
        connectivity=editable({
            "nodes": {
            },
            "edges": [
                {"a": "_c6", "b": "_c7", "res": 10},
            ],
            "res": 10,
        }),
    )
    topo = egg_topo.build()
    grid = topo.initialize_grid()
    # cfg already carries cluster_boundary_layers / bl_blend_neighbours, so the
    # pipeline rebuilds the clustering target from the edited topology itself.
    egg_webui.run(grid, generate_steps(grid, config=cfg, untangle_direct=False))

if __name__ == "__main__":
    main()
