"""good single-circle O-grid → TMOP smooth (pipeline).

Starts from a *proper* circle-in-rectangle topology (inner O-ring corners
placed so that ``min det A > 0``, see
``topologies.build_circle_in_rectangle``) and runs the pipeline; since the
initial grid is already valid the δ-continuation untangling phase is
skipped automatically.

The command-line surface lives in ``driver.py``; run ``--help`` for options.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from driver import finish, parse_single_args
from egg.pipeline import PipelineConfig, generate_steps
from egg.smoothing import build_boundary_layer_target
from topologies import build_circle_in_rectangle


def main():
    a = parse_single_args("good circle-in-rectangle → TMOP smooth")

    # Geometry + topology; optional boundary-layer spec on the circle.
    bl = (dict(first_height=a.bl_first_height, growth=a.bl_growth,
               n_fixed=a.pin_layers)
          if a.bl_first_height > 0.0 else None)
    topo, ents = build_circle_in_rectangle(rough=False, bl=bl)

    # Quality target (aspect-ratio clustering where specs exist), grid, and
    # the step-wise pipeline: TFI init -> boundary snap -> untangle -> TMOP
    # (-> pinned layers with --pin-layers).
    target = build_boundary_layer_target(topo)
    grid = topo.initialize_grid()
    cfg = PipelineConfig(
        sweeps_per_delta=a.sweeps_per_delta,
        tmop_sweeps=a.tmop_sweeps,
        tmop_chunk=a.chunk,
        omega=a.omega,
        device=a.device,
        pin_sweeps=a.pin_sweeps if bl and a.pin_layers > 0 else 0,
    )
    steps = generate_steps(grid, target, cfg, untangle_direct=not a.plot_live)

    finish(grid, topo, ents, steps, a, title="good single-circle",
           mindet_title="min det A (TMOP only)")


if __name__ == "__main__":
    main()
