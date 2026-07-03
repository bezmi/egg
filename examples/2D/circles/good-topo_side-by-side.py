"""side-by-side demo: good twin-circle TMOP smooth + per-wall boundary layers.

Two circles in a 7×4 channel sharing a bridge block (see
``topologies.build_twin_circle``), started from a *proper* (not folded)
configuration. Each circle gets its **own** boundary-layer clustering
(``--bl-first-height`` / ``--bl-growth``).

The command-line surface lives in ``driver.py``; run ``--help`` for options.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from driver import finish, parse_twin_args
from egg.pipeline import PipelineConfig, generate_steps
from egg.smoothing import build_boundary_layer_target
from topologies import build_twin_circle


def main():
    a = parse_twin_args("side-by-side: good twin-circle TMOP smooth + per-wall BL")

    # Geometry + topology; each circle carries its own boundary-layer spec.
    bl = {
        "circle": dict(
            first_height=a.bl_first_height, growth=a.bl_growth, n_fixed=a.pin_layers
        ),
        "circle2": dict(
            first_height=a.bl_first_height2, growth=a.bl_growth2, n_fixed=a.pin_layers
        ),
    }
    topo, ents = build_twin_circle(rough=False, bl=bl, R=a.resolution)

    # Quality target (aspect-ratio clustering per wall), grid, and the
    # step-wise pipeline: TFI init -> boundary snap -> untangle -> TMOP
    # (-> pinned layers with --pin-layers).
    target = build_boundary_layer_target(topo)
    grid = topo.initialize_grid()
    cfg = PipelineConfig(
        sweeps_per_delta=a.sweeps_per_delta,
        tmop_sweeps=a.tmop_sweeps,
        tmop_chunk=a.chunk,
        omega=a.omega,
        device=a.device,
        pin_sweeps=a.pin_sweeps if a.pin_layers > 0 else 0,
    )
    steps = generate_steps(grid, target, cfg, untangle_direct=not a.plot_live)

    finish(
        grid,
        topo,
        ents,
        steps,
        a,
        title="good twin-circle",
        mindet_title="min det A (TMOP only)",
    )


if __name__ == "__main__":
    main()
