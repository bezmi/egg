# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Smooth blob in a rectangle, with geometry defined via the egg 2D front-end.

Same O-grid-in-rectangle topology as ``examples/circles/good-topo.py``, but the
geometry is built with :mod:`egg.geometry.frontend2d` (a gdtk-style construction
API that returns egg entities directly) instead of the analytic primitives: the
inner body is a closed :class:`~egg.geometry.frontend2d.Spline` through points
sampled from ``r(theta) = 0.8 + 0.12 sin(3 theta)`` (a wavy blob), produced as a
``CompositePath`` of cubic Bézier segments; the outer walls are
:class:`~egg.geometry.frontend2d.Line` segments.

Pipeline: TFI init → boundary snap → TMOP quality optimisation.

The command-line surface lives in ``driver.py``; run
``uv run blob.py --help`` for options.
"""

import numpy as np

from egg.pipeline import PipelineConfig, generate_steps

from egg.geometry import Edge, Line, Spline, Vector3
from egg.topology.builder import TopologyBuilder


def build_blob_in_rectangle():
    """Blob-in-rectangle topology; geometry authored via the egg 2D front-end."""
    # Inner body: closed cubic spline through a wavy-radius point ring.
    theta = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
    r = 0.8 + 0.12 * np.sin(3.0 * theta)
    ring = [
        Vector3(2.0 + ri * np.cos(th), 2.0 + ri * np.sin(th))
        for th, ri in zip(theta, r)
    ]
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
        b.tag_boundary("blob", blk, 1, 1)
    # Wall associations of the e_* blocks are inferred (place_node corners);
    # markers for SU2 export stay explicit.
    wall_names = {
        id(bottom): "bottom",
        id(right): "right",
        id(top): "top",
        id(left): "left",
    }
    for blk, ent in [("e_s", bottom), ("e_e", right), ("e_n", top), ("e_w", left)]:
        b.tag_boundary(wall_names[id(ent)], blk, 1, 0)
    for blk, a0, a1 in [
        ("c_sw", left, bottom),
        ("c_se", bottom, right),
        ("c_ne", right, top),
        ("c_nw", top, left),
    ]:
        b.associate(blk, 0, 0, a0)
        b.associate(blk, 1, 0, a1)
        b.tag_boundary(wall_names[id(a0)], blk, 0, 0)
        b.tag_boundary(wall_names[id(a1)], blk, 1, 0)

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
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from driver import finish, parse_args

    a = parse_args()

    topo, ents = build_blob_in_rectangle()

    grid = topo.initialize_grid()
    cfg = PipelineConfig(
        sweeps_per_delta=a.sweeps_per_delta,
        tmop_sweeps=a.tmop_sweeps,
        tmop_chunk=a.chunk,
        omega=a.omega,
        device=a.device,
    )
    steps = generate_steps(grid, None, cfg, untangle_direct=not a.plot_live)

    finish(
        grid,
        topo,
        ents,
        steps,
        a,
        title="spline blob",
        mindet_title="min det A (TMOP only)",
    )


if __name__ == "__main__":
    main()
