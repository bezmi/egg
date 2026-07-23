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

from egg.pipeline import (
    ControlPointSmoother,
    FasSmoother,
    JacobiSmoother,
    Presmooth,
    Untangle,
    generate_steps,
)

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
    blob = Spline(ring, closed=True).named("blob")

    # Outer walls, wrapped as parametric grid edges. Naming each curve is the
    # single source of truth: build() auto-tags every associated face with the
    # name (an SU2 marker), and the topology exposes them as topo.entities.
    sw, se = Vector3(x=0, y=0, fixed=True), Vector3(x=4, y=0, fixed=True)
    ne, nw = Vector3(x=4, y=4, fixed=True), Vector3(x=0, y=4, fixed=True)
    bottom = Edge(Line(p0=sw, p1=se), name="bottom")
    right = Edge(Line(p0=se, p1=ne), name="right")
    top = Edge(Line(p0=ne, p1=nw), name="top")
    left = Edge(Line(p0=sw, p1=nw), name="left")

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
    # two walls) stay explicit. Every associated face inherits its entity's
    # name as its boundary marker — no separate tag_boundary calls.
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
    return topology, topology.entities


def _smoother(a):
    """The smoothing-phase stage(s) chosen by ``--smoother``.

    Control mode is preceded by a nodal pre-pass so the net fit starts smooth.
    """
    s = a["smoother"]
    if s == "fas":
        return [
            FasSmoother(sweeps=a["tmop_sweeps"], chunk=a["chunk"], omega=a["omega"])
        ]
    if s == "control_point":
        return [
            Presmooth(JacobiSmoother(sweeps=100, chunk=100, omega=a["omega"])),
            ControlPointSmoother(chunk=a["chunk"], omega=a["omega"]),
        ]
    return [JacobiSmoother(sweeps=a["tmop_sweeps"], chunk=a["chunk"], omega=a["omega"])]


def setup(a, *, direct=True):
    """Topology, grid, and stage list from parsed args — shared by the CLI
    ``main()`` and the web UI."""
    topo, ents = build_blob_in_rectangle()
    grid = topo.initialize_grid()
    stages = [
        Untangle(sweeps_per_delta=a["sweeps_per_delta"], direct=direct),
        *_smoother(a),
    ]
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
        title="spline blob",
        mindet_title="min det A (TMOP only)",
    )


if __name__ == "__egg_webui__":  # running inside the egg web UI
    import egg.webui as egg_webui

    # CLI defaults, mirroring driver.py — edit freely
    a = egg_webui.params(
        sweeps_per_delta=20,
        tmop_sweeps=40,
        chunk=10,
        smoother="jacobi",
        omega=0.8,
        device="cpu",
    )
    topo, ents, grid, stages = setup(a, direct=False)
    egg_webui.run(grid, generate_steps(grid, stages=stages, device=a["device"]))

if __name__ == "__main__":
    main()
