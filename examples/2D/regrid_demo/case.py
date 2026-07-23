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

"""Regrid demo: solve once, save the net, regrid cheaply.

A quarter annulus with two curved (arc) walls. The point of this example is
the workflow, not the shape:

1. run it once. It solves and saves the control net beside the script
   (``net.npz``). Pack the folder into ``demo.eggy`` to share it.
2. run it again at a different resolution (``--n``). The saved net does not
   depend on resolution, so the run LOADS it — re-tabulating it onto the new
   grid — and polishes, instead of solving from scratch. ``regrid.py`` does
   this from an ``.eggy`` archive.

The command line exposes only the grid-shape choice (``--n``): the resolution
you are allowed to change on a regrid. The solve effort (sweep / iteration
counts) is fixed in the script, because a warm regrid never re-uses those
anyway.
"""

from __future__ import annotations

import argparse
import os

from egg.geometry import Arc, Vector3
from egg.io import eggy
from egg.pipeline import (
    ControlPointSmoother,
    JacobiSmoother,
    Resample,
    Presmooth,
    Refit,
    Save,
    Untangle,
    generate_steps,
)
from egg.topology.builder import TopologyBuilder

# Solve effort: author-fixed constants. A cold solve uses them; a warm regrid
# ignores them (it polishes to convergence from the loaded net).
_TMOP_SWEEPS = 60
_CONTROL_MAX_OUTER = 30

# The net lives beside the script, so packing the folder into an .eggy carries
# it and a later run finds it by this relative path.
_NET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "net.npz")


def build_case(n: int = 1):
    """Quarter-annulus topology; ``n`` scales the resolution. Walls: inner / outer."""
    centre = Vector3(0, 0)
    i0, i1 = Vector3(1, 0), Vector3(0, 1)
    o0, o1 = Vector3(3, 0, fixed=True), Vector3(0, 3, fixed=True)
    inner = Arc(i0, i1, centre).named("inner")
    outer = Arc(o0, o1, centre).named("outer")

    b = TopologyBuilder(d=2)
    b.add_block("ring", sw=i0, se=o0, nw=i1, ne=o1, res=(8 * n, 16 * n))
    b.associate("ring", 0, 0, inner)
    b.associate("ring", 0, 1, outer)
    return b.build()


def _cold_stages(smoother: str, net_path: str):
    """Solve from scratch, then fit and save the net (exact layer included)."""
    if smoother == "control_point":
        # A nodal pre-pass gives the net fit a smooth, valid start.
        solve = [
            Presmooth(JacobiSmoother(sweeps=_TMOP_SWEEPS, chunk=_TMOP_SWEEPS)),
            ControlPointSmoother(max_outer=_CONTROL_MAX_OUTER),
        ]
    else:
        solve = [JacobiSmoother(sweeps=_TMOP_SWEEPS, chunk=10)]
    return [Untangle(), *solve, Refit(), Save(net_path, exact=True)]


def _warm_stages(net_path: str, smoother: str):
    """Resample the saved net onto the (possibly finer) grid and polish it."""
    if smoother == "control_point":
        return [Resample(net_path), ControlPointSmoother(max_outer=_CONTROL_MAX_OUTER)]
    return [Resample(net_path), JacobiSmoother(sweeps=_TMOP_SWEEPS, chunk=10)]


def run_case(
    n: int = 1,
    smoother: str = "control_point",
    cache_path: str | None = None,
    verbose: bool = True,
):
    """Build, solve (or warm-regrid from the saved net), and return the result.

    Loads the saved net when one exists beside the script (re-tabulating it to
    this resolution) and polishes; otherwise solves cold and saves. Returns
    ``(grid, final, warm)``: the final event dict and whether the run
    warm-started from the saved net (a ``"resample"`` phase in the stream).
    """
    net_path = cache_path or _NET
    grid = build_case(n).initialize_grid()
    if os.path.exists(net_path):
        stages = _warm_stages(net_path, smoother)
    else:
        stages = _cold_stages(smoother, net_path)
    final = {}
    warm = False
    for phase, ev in generate_steps(grid, stages=stages):
        if phase == "resample":
            warm = True
        if verbose:
            print(f"[{phase}] " + " ".join(f"{k}={v}" for k, v in ev.items()))
        final = ev
    if verbose:
        print(f"{'warm regrid' if warm else 'cold solve'}: {final}")
    return grid, final, warm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Grid-shape only: what a regrid may change.
    p.add_argument("--n", type=int, default=2, help="resolution scale")
    p.add_argument(
        "--smoother", default="control_point", choices=["control_point", "jacobi"]
    )
    p.add_argument(
        "--export-eggy",
        default=None,
        metavar="PATH",
        help="after solving, pack this folder into a .eggy archive",
    )
    a = p.parse_args()
    run_case(n=a.n, smoother=a.smoother)
    if a.export_eggy:
        eggy.pack(a.export_eggy, os.path.dirname(os.path.abspath(__file__)))
        print(f"exported {a.export_eggy}")


if __name__ == "__main__":
    main()
