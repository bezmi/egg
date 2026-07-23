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

"""An egg in a rectangle — the simplest complete egg example.

A closed egg-shaped spline sits inside a 4x4 square, meshed with an O-grid: a
ring of four blocks hugs the egg, four edge blocks bridge the ring out to the
domain walls, and four corner blocks fill the diagonals (twelve blocks). The
egg surface carries wall-normal boundary-layer clustering.

The pipeline is fixed (no knobs): clear the folded start, smooth the nodes with
block-Jacobi against the clustering target, pin the first wall layers to their
exact heights and re-smooth, then fit a control net to the result and save it
to ``./net.npz``. The saved net does not depend on the grid resolution, so it
can be reloaded and re-evaluated at a new resolution and clustering without
solving again — see the commented-out regrid block at the bottom of ``main``.

Runs on the command line (``uv run egg.py``) and in the egg web UI.
"""

import os

import numpy as np

from egg.geometry import Edge, Line, Spline, Vector3
from egg.enums import TmopMetric
from egg.pipeline import (
    JacobiSmoother,
    Pin,
    Refit,
    Save,
    Untangle,
    generate_steps,
    run_pipeline,
)
from egg.topology.builder import TopologyBuilder

# The net is saved beside the script so the regrid block in main() finds it by
# this path regardless of the working directory.
NET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "net.npz")


def build_egg_in_rectangle(
    bl_first_height=5.0e-3, bl_growth=1.5, n_fixed=2, res_scale=1
):
    """Egg-in-rectangle O-grid topology; markers: inflow / outflow / wall / egg.

    ``res_scale`` multiplies every block's cell count (1 = the base resolution),
    so the regrid block in ``main`` can rebuild the SAME topology at a finer
    grid and load the base-resolution net onto it. The topology (blocks, seams,
    fans) is identical across scales — only the node counts change — which is
    exactly what the saved net needs to re-tabulate.
    """

    def _res(*ns):
        # Scale cell counts (node count n -> (n-1)*scale + 1) so shared faces
        # stay conforming across the uniform scale.
        return tuple((n - 1) * res_scale + 1 for n in ns)

    # Egg boundary: closed spline through an egg curve, fat end down.
    theta = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
    ring = [
        Vector3(2.0 + (0.66 - 0.15 * np.sin(t)) * np.cos(t), 2.0 + 0.85 * np.sin(t))
        for t in theta
    ]
    egg = Edge(Spline(ring, closed=True)).named("egg")

    # Domain walls and the sliding nodes that split them. Naming each curve
    # auto-derives its SU2 marker on every associated face. The top and bottom
    # walls are named distinctly (so both draw) but carry tag="wall", so they
    # export under one shared 'wall' marker — no per-face tag_boundary needed.
    sw, se = Vector3(0, 0, fixed=True), Vector3(4, 0, fixed=True)
    ne, nw = Vector3(4, 4, fixed=True), Vector3(0, 4, fixed=True)
    bottom = Edge(Line(p0=sw, p1=se)).named("wall_bottom", tag="wall")
    right = Edge(Line(p0=se, p1=ne)).named("outflow")
    top = Edge(Line(p0=ne, p1=nw)).named("wall_top", tag="wall")
    left = Edge(Line(p0=sw, p1=nw)).named("inflow")
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
        b.add_block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=_res(20, o_res_j))
    # Edge blocks between walls and mid square; wall associations inferred, so
    # every marker (inflow / outflow / wall) auto-derives from the named edges.
    for nm, c_sw, c_nw, c_se, c_ne in [
        ("e_s", bsw, msw, bse, mse),
        ("e_e", rse, mse, rne, mne),
        ("e_n", tne, mne, tnw, mnw),
        ("e_w", lnw, mnw, lsw, msw),
    ]:
        b.add_block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=_res(20, 10))
    # Corner blocks: two walls meet at each, so associations stay explicit;
    # markers auto-derive from the edges' tags (wall_top/wall_bottom -> 'wall').
    for nm, w0, w1, c_sw, c_nw, c_se, c_ne in [
        ("c_sw", left, bottom, sw, lsw, bsw, msw),
        ("c_se", bottom, right, se, bse, rse, mse),
        ("c_ne", right, top, ne, rne, tne, mne),
        ("c_nw", top, left, nw, tnw, lnw, mnw),
    ]:
        blk = b.block(nm, sw=c_sw, nw=c_nw, se=c_se, ne=c_ne, res=_res(10, 10))
        blk.west.on(w0)
        blk.south.on(w1)

    if bl_first_height > 0.0:
        # Wall-normal clustering on the egg surface; the egg is a closed
        # interior wall, so no domain boundary meets it obliquely and
        # relax_orthogonality stays empty.
        egg.clustered(
            first_height=bl_first_height,
            growth=bl_growth,
            n_fixed=n_fixed,
            n_layers=2,
        )

    topology = b.build()
    return topology, topology.entities


def pipeline():
    """The fixed pipeline: untangle -> Jacobi smooth -> pin -> refit -> save.

    - :class:`~egg.pipeline.Untangle` clears any folded cells in the TFI start
      (this egg starts valid, so it self-skips).
    - :class:`~egg.pipeline.JacobiSmoother` relaxes the nodes with block-Jacobi
      against the boundary-layer clustering target the pipeline builds from the
      egg's spec (``shape_size`` equalises cell size too; the profile stays in
      the O-ring, ``bl_blend_neighbours=False``).
    - :class:`~egg.pipeline.Pin` sets the first wall layers to their exact
      heights, freezes them, and re-smooths with the given node smoother.
    - :class:`~egg.pipeline.Refit` fits a control net to the final grid.
    - :class:`~egg.pipeline.Save` writes that net to ``./net.npz`` so the
      regrid block in ``main`` can load it. ``exact=True`` also stores the
      residual (the pinned grid minus the net evaluation), so the regrid
      reproduces the actual pinned mesh, not just the net's approximation.
    """
    return [
        Untangle(),
        JacobiSmoother(
            sweeps=200, chunk=50, metric=TmopMetric.SHAPE_SIZE, bl_blend_neighbours=False
        ),
        Pin(JacobiSmoother(sweeps=100, chunk=50)),
        Refit(),
        Save(NET_PATH, exact=True),
    ]


def main():
    topo, _ents = build_egg_in_rectangle()
    grid = topo.initialize_grid()
    # Solve and save the fitted net to ./net.npz.
    run_pipeline(grid, stages=pipeline())

    # --- Regrid without solving again -----------------------------------------
    # The saved net is resolution-independent, so you can regrid this mesh to a
    # NEW resolution and clustering by re-evaluating the net instead of solving.
    # This block ADDS a regrid after the solve; to regrid INSTEAD of solving,
    # comment out the `run_pipeline(grid, stages=pipeline())` solve call above
    # (a prior solve run must have left ./net.npz), then uncomment this block:
    # it rebuilds the SAME topology at 2x resolution with a tighter boundary
    # layer, then Resamples the saved net onto it with clustering, and runs a
    # short pin polish for neat exact layers.
    #
    # from egg.pipeline import Resample
    #
    # topo, _ents = build_egg_in_rectangle(res_scale=2, bl_first_height=2.5e-3)
    # grid = topo.initialize_grid()
    # run_pipeline(
    #     grid,
    #     stages=[
    #         # load the saved net onto the finer grid AND re-cluster in one step
    #         Resample(NET_PATH, cluster=True),
    #         Untangle(),
    #         JacobiSmoother(
    #             sweeps=100, chunk=50, metric="shape_size", bl_blend_neighbours=False
    #         ),
    #         Pin(JacobiSmoother(sweeps=60, chunk=30)),   # exact first layers
    #         Refit(),
    #     ],
    # )


if __name__ == "__egg_webui__":  # running inside the egg web UI
    import egg.webui as egg_webui

    # Bind entities to a non-underscore name so the web UI harvests the egg /
    # wall curves for the geometry layer.
    topo, ents = build_egg_in_rectangle()
    grid = topo.initialize_grid()
    egg_webui.run(grid, generate_steps(grid, stages=pipeline()))

    # --- Regrid in the web UI ------------------------------------------------
    # To regrid instead of solving: run this script once as-is (the Save stage
    # writes ./net.npz beside it), then comment out the egg_webui.run solve call
    # above, uncomment the block below, and run again. It rebuilds the SAME
    # topology at 2x resolution with a tighter boundary layer, then Resamples the
    # saved net onto it with clustering, and runs a short pin polish.
    #
    # from egg.pipeline import Resample
    #
    # topo, ents = build_egg_in_rectangle(res_scale=2, bl_first_height=2.5e-3)
    # grid = topo.initialize_grid()
    # stages = [
    #     # load the saved net onto the finer grid AND re-cluster in one step
    #     Resample(NET_PATH, cluster=True),
    #     Untangle(),
    #     JacobiSmoother(
    #         sweeps=100, chunk=50, metric="shape_size", bl_blend_neighbours=False
    #     ),
    #     Pin(JacobiSmoother(sweeps=60, chunk=30)),   # exact first layers
    #     Refit(),
    # ]
    # egg_webui.run(grid, generate_steps(grid, stages=stages))

if __name__ == "__main__":
    main()
