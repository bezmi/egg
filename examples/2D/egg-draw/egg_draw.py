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

"""An egg in a rectangle — geometry only; you draw the topology in the web UI.

Unlike :mod:`egg` (which builds the whole O-grid in code) and :mod:`egg_svg`
(which reads the blocking from an SVG wireframe), this script defines **no
topology at all**: just the named egg curve and the four rectangle walls. It
runs *only* in the egg web UI — there is no ``driver.py`` / command-line path.

Open it in the web UI and use the **topology edit view** to draw the
**blocking** onto this geometry — snap corners, bind faces to the named curves,
set per-edge resolution. ``save edits`` writes the blocking back into the empty
``connectivity`` literal below. Because the geometry is *named* (``.named(...)``),
binding a face to a curve both constrains its nodes and — via the curve's
:attr:`~egg.geometry.base.GeometryEntity.tag` — auto-derives its SU2 boundary
marker; the two walls share ``tag="wall"`` while keeping distinct names so both
draw. The whole topology is the blocking you draw (``base=None``).

Boundary-layer clustering rides on the geometry too: the egg carries a
wall-normal clustering request via
:meth:`~egg.geometry.base.GeometryEntity.clustered`, which ``build()`` collects
for every face the blocking binds to it (just like the marker from ``tag``) — so
this drawn, base-less topology clusters with no ``TopologyBuilder``. The
pipeline's ``cluster_boundary_layers`` then builds the target.
"""

import numpy as np

from egg.geometry import Line, Spline, Vector3
from egg.enums import TmopMetric
from egg.pipeline import (
    ControlPointSmoother,
    FasSmoother,
    JacobiSmoother,
    Presmooth,
    Pin,
    Untangle,
    generate_steps,
)
from egg.topology import ExplicitTopology


def build_geometry(bl_first_height=0.0, bl_growth=1.5, n_fixed=1):
    """The named geometry the web UI draws and the drawn blocking binds to.

    Returns a ``{name: entity}`` dict — the egg curve plus the four walls,
    each :meth:`~egg.geometry.base.GeometryEntity.named` so the edit view can
    bind faces to it by name and so its marker auto-derives on export. With
    ``bl_first_height > 0`` the egg also carries a
    :meth:`~egg.geometry.base.GeometryEntity.clustered` request, so whatever
    O-ring the blocking wraps around it gets wall-normal clustering.
    """
    # Egg boundary: closed spline through an egg curve, fat end down.
    theta = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
    ring = [
        Vector3(2.0 + (0.66 - 0.15 * np.sin(t)) * np.cos(t), 2.0 + 0.85 * np.sin(t))
        for t in theta
    ]
    egg = Spline(ring, closed=True).named("egg")
    if bl_first_height > 0.0:
        egg.clustered(first_height=bl_first_height, growth=bl_growth, n_fixed=n_fixed)

    # Rectangle walls. Left/right carry the flow markers; top and bottom keep
    # distinct names (so both draw) but tag="wall", so they export as one marker.
    sw, se = Vector3(0, 0), Vector3(4, 0)
    ne, nw = Vector3(4, 4), Vector3(0, 4)
    bottom = Line(p0=sw, p1=se).named("wall_bottom", tag="wall")
    right = Line(p0=se, p1=ne).named("outflow")
    top = Line(p0=ne, p1=nw).named("wall_top", tag="wall")
    left = Line(p0=sw, p1=nw).named("inflow")

    return {
        "egg": egg,
        "inflow": left,
        "outflow": right,
        "wall_bottom": bottom,
        "wall_top": top,
    }


if __name__ == "__egg_webui__":  # this example runs ONLY in the egg web UI
    import egg.webui as egg_webui

    from egg import editable

    geometry = build_geometry(
        bl_first_height=5.0e-3,  # egg-surface first-layer height (0 disables clustering)
        bl_growth=1.5,
        n_fixed=2,  # near-wall layers pinned exactly in the pin phase
    )

    # Empty to start — draw the blocking in the topology edit view: snap corners
    # onto the geometry, bind faces to the named curves, set per-edge resolution,
    # then `save edits` writes the connectivity back into this literal.
    egg_topo = ExplicitTopology(
        base=None,
        geometry=geometry,
        connectivity=editable(
            {
                "nodes": {
                    "u16": {"xy": [-0.2546, 4.1576], "on": ["wall_top"]},
                    "u17": {"xy": [4.2442, 4.1517], "on": ["outflow"]},
                    "u18": {"xy": [4.2297, -0.2027], "on": ["wall_bottom"]},
                    "u19": {"xy": [-0.2774, -0.2232], "on": ["wall_bottom"]},
                    "u20": {"xy": [1.226, 2.9356], "on": ["egg"]},
                    "u21": {"xy": [1.2747, 1.0794], "on": ["egg"]},
                    "u22": {"xy": [2.8207, 1.0278], "on": ["egg"]},
                    "u23": {"xy": [2.7081, 2.8992], "on": ["egg"]},
                    "u24": {"xy": [0.9201, 3.2541]},
                    "u25": {"xy": [3.1005, 3.3728]},
                    "u26": {"xy": [3.1005, 0.7061]},
                    "u27": {"xy": [0.9181, 0.7322]},
                    "u19_u16": {"xy": [-0.2312, 3.3123], "on": ["inflow"]},
                    "u16_u17": {"xy": [0.9807, 4.1742], "on": ["wall_top"]},
                    "u16_u17_u17": {"xy": [3.1091, 4.1291], "on": ["wall_top"]},
                    "u17_u18": {"xy": [4.2199, 3.3362], "on": ["outflow"]},
                    "u17_u18_u18": {"xy": [4.2196, 0.6692], "on": ["outflow"]},
                    "u18_u19": {"xy": [3.1023, -0.1838], "on": ["wall_bottom"]},
                    "u18_u19_u19": {"xy": [0.9199, -0.1842], "on": ["wall_bottom"]},
                    "u19_u19_u16": {"xy": [-0.2286, 0.8749], "on": ["inflow"]},
                    "u24_u16_u17": {"xy": [0.9543, 3.7102]},
                    "u28": {"xy": [0.3536, 3.7099]},
                    "u24_u19_u16": {"xy": [0.3476, 3.2479]},
                    "u27_u19_u19_u16": {"xy": [0.3655, 0.7738]},
                    "u29": {"xy": [0.378, 0.3738]},
                    "u27_u18_u19_u19": {"xy": [0.9224, 0.3436]},
                    "u26_u18_u19": {"xy": [3.1049, 0.3006]},
                    "u30": {"xy": [3.6163, 0.2883]},
                    "u26_u17_u18_u18": {"xy": [3.6164, 0.7096]},
                    "u25_u17_u18": {"xy": [3.5805, 3.3629]},
                    "u31": {"xy": [3.5675, 3.6732]},
                    "u25_u16_u17_u17": {"xy": [3.0921, 3.6477]},
                },
                "edges": [
                    {"a": "u20", "b": "u21", "bind": "egg"},
                    {"a": "u21", "b": "u22", "bind": "egg"},
                    {"a": "u22", "b": "u23", "bind": "egg"},
                    {"a": "u23", "b": "u20", "bind": "egg"},
                    {"a": "u20", "b": "u24"},
                    {"a": "u24", "b": "u25"},
                    {"a": "u23", "b": "u25"},
                    {"a": "u25", "b": "u26"},
                    {"a": "u26", "b": "u22"},
                    {"a": "u26", "b": "u27"},
                    {"a": "u21", "b": "u27"},
                    {"a": "u27", "b": "u24"},
                    {"a": "u19_u16", "b": "u16", "bind": "inflow"},
                    {"a": "u16", "b": "u16_u17", "bind": "wall_top"},
                    {"a": "u16_u17", "b": "u16_u17_u17", "bind": "wall_top"},
                    {"a": "u16_u17_u17", "b": "u17", "bind": "wall_top"},
                    {"a": "u17", "b": "u17_u18", "bind": "outflow"},
                    {"a": "u17_u18", "b": "u17_u18_u18", "bind": "outflow"},
                    {"a": "u17_u18_u18", "b": "u18", "bind": "outflow"},
                    {"a": "u18", "b": "u18_u19", "bind": "wall_bottom"},
                    {"a": "u18_u19", "b": "u18_u19_u19", "bind": "wall_bottom"},
                    {"a": "u18_u19_u19", "b": "u19", "bind": "wall_bottom"},
                    {"a": "u19", "b": "u19_u19_u16", "bind": "inflow"},
                    {"a": "u19_u19_u16", "b": "u19_u16", "bind": "inflow"},
                    {"a": "u24", "b": "u24_u16_u17"},
                    {"a": "u24_u16_u17", "b": "u16_u17"},
                    {"a": "u24_u16_u17", "b": "u28"},
                    {"a": "u24", "b": "u24_u19_u16"},
                    {"a": "u24_u19_u16", "b": "u19_u16", "res": 5},
                    {"a": "u28", "b": "u24_u19_u16"},
                    {"a": "u27", "b": "u27_u19_u19_u16", "res": 8},
                    {"a": "u27_u19_u19_u16", "b": "u19_u19_u16"},
                    {"a": "u24_u19_u16", "b": "u27_u19_u19_u16"},
                    {"a": "u27_u19_u19_u16", "b": "u29"},
                    {"a": "u27", "b": "u27_u18_u19_u19"},
                    {"a": "u27_u18_u19_u19", "b": "u18_u19_u19"},
                    {"a": "u29", "b": "u27_u18_u19_u19"},
                    {"a": "u26", "b": "u26_u18_u19"},
                    {"a": "u26_u18_u19", "b": "u18_u19"},
                    {"a": "u27_u18_u19_u19", "b": "u26_u18_u19"},
                    {"a": "u26_u18_u19", "b": "u30"},
                    {"a": "u26", "b": "u26_u17_u18_u18"},
                    {"a": "u26_u17_u18_u18", "b": "u17_u18_u18"},
                    {"a": "u30", "b": "u26_u17_u18_u18"},
                    {"a": "u25", "b": "u25_u17_u18"},
                    {"a": "u25_u17_u18", "b": "u17_u18"},
                    {"a": "u26_u17_u18_u18", "b": "u25_u17_u18"},
                    {"a": "u25_u17_u18", "b": "u31"},
                    {"a": "u25", "b": "u25_u16_u17_u17"},
                    {"a": "u25_u16_u17_u17", "b": "u16_u17_u17"},
                    {"a": "u31", "b": "u25_u16_u17_u17"},
                    {"a": "u25_u16_u17_u17", "b": "u24_u16_u17"},
                    {"a": "u28", "b": "u16"},
                    {"a": "u31", "b": "u17"},
                    {"a": "u18", "b": "u30"},
                    {"a": "u29", "b": "u19"},
                ],
                "res": 10,
            }
        ),
    )

    # Run-panel knobs (edit freely from the run-parameters strip). Any bare
    # literal here is surfaced: bool -> checkbox, int/float -> number box, str ->
    # text box; editable(..., choices=[...]) renders a dropdown.
    a = egg_webui.params(
        metric=editable("shape_size", choices=["shape", "shape_size"]),
        cluster_boundary_layers=True,  # checkbox — untangle without clustering by unticking
        bl_blend_neighbours=True,  # checkbox — let the clustering extend into neighbour blocks
        pin_sweeps=300,  # number box (0 disables the pin phase)
        sweeps_per_delta=20,
        tmop_sweeps=5000,
        chunk=500,
        smoother="jacobi",
        omega=0.8,
        device="cpu",
    )

    # Wire up the pipeline only once the drawn blocking flattens to a valid
    # topology, so the empty starting point just draws the geometry without
    # erroring; the run button comes alive as soon as the blocking is valid.
    topo, _diagnostics = egg_topo.flatten()
    if topo is not None:
        grid = topo.initialize_grid()
        # The egg's carried .clustered() request populates the topology's
        # boundary_layer_specs, so cluster_boundary_layers builds the target and
        # the pin phase freezes the near-wall layers.
        metric = TmopMetric(a["metric"])
        if a["smoother"] == "fas":
            smoothing = [
                FasSmoother(
                    sweeps=a["tmop_sweeps"],
                    chunk=a["chunk"],
                    omega=a["omega"],
                    metric=metric,
                    cluster_boundary_layers=a["cluster_boundary_layers"],
                    bl_blend_neighbours=a["bl_blend_neighbours"],
                )
            ]
        elif a["smoother"] == "control_point":
            smoothing = [
                Presmooth(
                    JacobiSmoother(
                        sweeps=100,
                        chunk=100,
                        omega=a["omega"],
                        metric=metric,
                        cluster_boundary_layers=a["cluster_boundary_layers"],
                        bl_blend_neighbours=a["bl_blend_neighbours"],
                    )
                ),
                ControlPointSmoother(
                    chunk=a["chunk"],
                    omega=a["omega"],
                    metric=metric,
                    cluster_boundary_layers=a["cluster_boundary_layers"],
                    bl_blend_neighbours=a["bl_blend_neighbours"],
                ),
            ]
        else:
            smoothing = [
                JacobiSmoother(
                    sweeps=a["tmop_sweeps"],
                    chunk=a["chunk"],
                    omega=a["omega"],
                    metric=metric,
                    cluster_boundary_layers=a["cluster_boundary_layers"],
                    bl_blend_neighbours=a["bl_blend_neighbours"],
                )
            ]
        stages = [Untangle(sweeps_per_delta=a["sweeps_per_delta"]), *smoothing]
        if a["pin_sweeps"] > 0:
            stages.append(
                Pin(
                    JacobiSmoother(
                        sweeps=a["pin_sweeps"], chunk=a["chunk"], omega=a["omega"]
                    )
                )
            )
        egg_webui.run(grid, generate_steps(grid, stages=stages, device=a["device"]))
