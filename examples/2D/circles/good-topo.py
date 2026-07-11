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

"""good single-circle O-grid → TMOP smooth (pipeline).

Starts from a *proper* circle-in-rectangle topology (inner O-ring corners
placed so that ``min det A > 0``, see
``topologies.build_circle_in_rectangle``) and runs the pipeline; since the
initial grid is already valid the δ-continuation untangling phase is
skipped automatically.

The command-line surface lives in ``driver.py``; run ``--help`` for options.
"""

from driver import finish, parse_single_args
from egg.pipeline import generate_steps
from topologies import setup_single


def main():
    a = parse_single_args("good circle-in-rectangle → TMOP smooth")
    topo, ents, grid, cfg = setup_single(vars(a))
    steps = generate_steps(grid, config=cfg, untangle_direct=not a.plot_live)

    finish(
        grid,
        topo,
        ents,
        steps,
        a,
        title="good single-circle",
        mindet_title="min det A (TMOP only)",
    )


if __name__ == "__egg_webui__":  # running inside the egg web UI
    import egg.webui as egg_webui

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
    topo, ents, grid, cfg = setup_single(a)
    egg_webui.run(grid, generate_steps(grid, config=cfg, untangle_direct=False))

if __name__ == "__main__":
    main()
