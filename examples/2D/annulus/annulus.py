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

"""Quarter annulus — the smallest complete egg example.

One block, two arc associations, named boundary markers (``inner`` /
``outer``), and a web-UI guard. A good first file to open in the web UI
and edit.

The command-line surface lives in ``driver.py``; run
``uv run annulus.py --help`` for options.
"""

from egg.pipeline import (
    ControlPointSmoother,
    FasSmoother,
    JacobiSmoother,
    Presmooth,
    Untangle,
    generate_steps,
)

from egg.geometry import Arc, Vector3
from egg.topology.builder import TopologyBuilder


def build_annulus():
    """Quarter-annulus topology; markers: inner / outer."""
    centre = Vector3(0, 0)
    i0, i1 = Vector3(1, 0), Vector3(0, 1)
    o0 = Vector3(3, 0, fixed=True)
    o1 = Vector3(0, 3, fixed=True)

    # Naming each arc auto-derives its 'inner'/'outer' marker on the associated
    # face and exposes it as an entity (topo.entities).
    inner = Arc(i0, i1, centre).named("inner")
    outer = Arc(o0, o1, centre).named("outer")

    b = TopologyBuilder(d=2)
    b.add_block("ring", sw=i0, se=o0, nw=i1, ne=o1, res=(10, 24))
    b.associate("ring", 0, 0, inner)
    b.associate("ring", 0, 1, outer)

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
    ``main()`` and the web UI (which passes the parser defaults)."""
    topo, ents = build_annulus()
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

    finish(grid, topo, ents, steps, a, title="quarter annulus")


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
