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
