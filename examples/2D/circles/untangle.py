# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""folded single-circle O-grid → untangle → smooth (pipeline).

Starts from a deliberately *folded* circle-in-rectangle topology (rough
inner O-ring corners give ``min det A < 0``, see
``topologies.build_circle_in_rectangle``) and runs the full pipeline,
δ-continuation untangle included.

The command-line surface lives in ``driver.py``; run ``--help`` for options.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from driver import finish, parse_single_args
from egg.pipeline import generate_steps
from topologies import setup_single


def main():
    a = parse_single_args("folded circle-in-rectangle → untangle → smooth")
    topo, ents, grid, cfg = setup_single(vars(a), rough=True)
    steps = generate_steps(grid, config=cfg, untangle_direct=not a.plot_live)

    finish(
        grid,
        topo,
        ents,
        steps,
        a,
        title="single-circle",
        mindet_title="min det A (untangle+TMOP)",
    )


if __name__ == "__egg_webui__":  # running inside the egg web UI
    import egg_webui

    # CLI defaults, mirroring driver.py — edit freely
    a = egg_webui.params(
        bl_first_height=0.0,
        bl_growth=1.3,
        pin_layers=0,
        pin_sweeps=500,
        sweeps_per_delta=200,
        tmop_sweeps=1000,
        chunk=100,
        smoother="jacobi",
        omega=0.8,
        device="cpu",
    )
    topo, ents, grid, cfg = setup_single(a, rough=True)
    egg_webui.run(grid, generate_steps(grid, config=cfg, untangle_direct=False))

if __name__ == "__main__":
    main()
