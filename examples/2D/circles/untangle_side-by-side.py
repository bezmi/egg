# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""side-by-side demo: folded twin-circle untangle + per-wall boundary layers.

Two circles in a 7×4 channel sharing a bridge block (see
``topologies.build_twin_circle``), started from a deliberately *folded*
configuration and run through the full pipeline, δ-continuation untangle
included. Each circle gets its **own** boundary-layer clustering.

The command-line surface lives in ``driver.py``; run ``--help`` for options.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from driver import finish, parse_twin_args
from egg.pipeline import generate_steps
from topologies import setup_twin


def main():
    a = parse_twin_args("side-by-side: twin-circle untangle + per-wall BL")
    topo, ents, grid, cfg = setup_twin(vars(a), rough=True)
    steps = generate_steps(grid, config=cfg, untangle_direct=not a.plot_live)

    finish(
        grid,
        topo,
        ents,
        steps,
        a,
        title="twin-circle",
        mindet_title="min det A (untangle+TMOP)",
    )


if __name__ == "__egg_webui__":  # running inside the egg web UI
    import egg_webui

    # CLI defaults, mirroring driver.py — edit freely
    a = egg_webui.params(
        bl_first_height=0.05,
        bl_growth=1.5,
        bl_first_height2=0.08,
        bl_growth2=1.3,
        resolution=2,
        pin_layers=0,
        pin_sweeps=5000,
        sweeps_per_delta=200,
        tmop_sweeps=5000,
        chunk=500,
        smoother="jacobi",
        omega=0.8,
        device="cpu",
    )
    topo, ents, grid, cfg = setup_twin(a, rough=True)
    egg_webui.run(grid, generate_steps(grid, config=cfg, untangle_direct=False))

if __name__ == "__main__":
    main()
