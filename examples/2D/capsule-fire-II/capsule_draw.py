# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

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
from egg.pipeline import PipelineConfig, generate_steps
from egg.topology.builder import TopologyBuilder
from egg.topology import ExplicitTopology


def build_geometry(bl_first_height=0.0, bl_growth=1.3, n_fixed=2):
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

    body = Polyline([Arc(ai, bi, oi), Arc(bi, ci, pi_), Line(ci, di)]).named("wall")
    if bl_first_height > 0.0:
        body.clustered(first_height=bl_first_height, growth=bl_growth, n_fixed=n_fixed)

    ao = oi + Ro * Vector3(-1.0, 0.0)
    #thetao = math.acos((Ri - di.x) / Ro)
    thetao = 1.5*thetai
    do = oi + Ro * Vector3(-math.cos(thetao), math.sin(thetao))

    outer = Arc(ao, do, oi).named("inflow")
    south = Line(ao, ai).named("symmetry")
    north = Line(do, di).named("outflow")
    return {"capsule": body, "inflow": outer, "outflow": north, "symmetry": south}


if __name__ == "__egg_webui__":  # this example runs ONLY in the egg web UI
    import egg_webui

    from egg import editable

    # Run-panel knobs (edit freely from the run-parameters strip). Any bare
    # literal here is surfaced: bool -> checkbox, int/float -> number box, str ->
    # text box; editable(..., choices=[...]) renders a dropdown.

    a = egg_webui.params(
        bl_first_height=4.0e-4,
        bl_growth=1.3,
        pin_layers=1,
        pin_sweeps=5000,
        sweeps_per_delta=200,
        tmop_sweeps=5000,
        chunk=10,
        smoother='jacobi',
        # editable() surfaces a value in the run-parameters panel with a
        # typed input; choices=[...] renders a dropdown. The classic
        # combination for comparison: metric="shape", dipole=False.
        metric=egg_webui.editable('shape_size', choices=["shape", "shape_size"]),
        dipole=egg_webui.editable(False, label="corner dipole"),
        omega=0.8,
        # block-interface C2 curvature-continuity weight (0 = off); de-kinks the
        # grid lines crossing the block seams. interface-only, so it leaves the
        # clustered near-wall cells alone.
        c2_weight=egg_webui.editable(10, label="interface C2 weight"),
        c2_singularity=egg_webui.editable(0.0, label="singularity ring C2 weight"),
        ortho_weight=egg_webui.editable(0, label="interface orthogonality weight"),
        ortho_layers=egg_webui.editable(5, label="orthogonality band layers"),
        ortho_relax=egg_webui.editable(1.0, label="orthogonality clustering relax"),
        device="cpu",
    )

    geometry = build_geometry(
        bl_first_height=a["bl_first_height"],  # egg-surface first-layer height (0 disables clustering)
        bl_growth=a["bl_growth"],
        n_fixed=a["pin_layers"],  # near-wall layers pinned exactly in the pin phase
    )
    print(geometry)

    # Empty to start — draw the blocking in the topology edit view: snap corners
    # onto the geometry, bind faces to the named curves, set per-edge resolution,
    # then `save edits` writes the connectivity back into this literal.
    egg_topo = ExplicitTopology(
        base=None,
        geometry=geometry,
        connectivity=editable(
            {
                "nodes": {
                    "u0": {"xy": [0.0735, 0.524], "on": ["inflow"]},
                    "u1": {"xy": [0.1211, 0.3369], "on": ["outflow"]},
                    "u2": {"xy": [0.0499, 0.3086], "on": ["capsule"]},
                    "u3": {"xy": [0.0453, 0.4679], "on": ["inflow"]},
                    "u4": {"xy": [-0.0678, 0.0025], "on": ["symmetry"]},
                    "u5": {"xy": [-0.0077, 0.0033], "on": ["symmetry"]},
                },
                "edges": [
                    {"a": "u0", "b": "u1", "bind": "outflow", "res": 15},
                    {"a": "u2", "b": "u3", "res": 20},
                    {"a": "u3", "b": "u0", "bind": "inflow", "res": 10},
                    {"a": "u3", "b": "u4", "bind": "inflow", "res": 45},
                    {"a": "u4", "b": "u5", "bind": "symmetry"},
                    {"a": "u5", "b": "u2", "bind": "capsule"},
                    {"a": "u1", "b": "u2", "res": 15},
                ],
                "res": 10,
            }
        ),
    )

    # Wire up the pipeline only once the drawn blocking flattens to a valid
    # topology, so the empty starting point just draws the geometry without
    # erroring; the run button comes alive as soon as the blocking is valid.
    topo, _diagnostics = egg_topo.flatten()
    if topo is not None:
      # --pin-layers > 0 smooths against the boundary-layer clustering target
      # and pins the first n_fixed layers exactly; the default path runs on the
      # plain metric (no clustering) and restores the wall spacing with the
      # respace post-pass. cluster_boundary_layers picks between them: the
      # pipeline builds the clustering target from the set_boundary_layer specs
      # itself, sizing the shape_size far field correctly (see PipelineConfig).
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
      cfg = PipelineConfig(
          sweeps_per_delta=a["sweeps_per_delta"],
          tmop_sweeps=a["tmop_sweeps"],
          tmop_chunk=a["chunk"],
          tmop_smoother=a["smoother"],
          tmop_metric=metric,
          cluster_boundary_layers=pin,
          omega=a["omega"],
          interface_c2=c2,
          interface_ortho=ortho,
          device=a["device"],
          pin_sweeps=a["pin_sweeps"] if pin else 0,
          respace=a["bl_first_height"] > 0.0 and not pin,
      )
      

      egg_webui.run(grid, generate_steps(grid, config=cfg, untangle_direct=True))
