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

"""Imperative 3D multiblock built with ``TopologyBuilder(d=3)``.

Two hex blocks tile the slab ``[0, 2] x [0, 1] x [0, 1]`` and share the x=1
plane. The shared-face node correspondence is resolved by the general
``_interface_axis_map`` (every rotation/reflection of the quad), the grid is
TFI-initialised, and it smooths through the standard pipeline. In 3D the
pipeline's min-det/energy monitoring routes through the C++ device reduction
(the NumPy metric is 2D-only); the C++ core smooths at dim=3.

Run: ``uv run --no-sync python two_block.py --device cpu``.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from egg.pipeline import JacobiSmoother, Refit, Save, Untangle, generate_steps
from egg.topology.builder import TopologyBuilder

CORNERS = {
    "a00": (0, 0, 0),
    "a01": (0, 0, 1),
    "a10": (0, 1, 0),
    "a11": (0, 1, 1),
    "s00": (1, 0, 0),
    "s01": (1, 0, 1),
    "s10": (1, 1, 0),
    "s11": (1, 1, 1),
    "b00": (2, 0, 0),
    "b01": (2, 0, 1),
    "b10": (2, 1, 0),
    "b11": (2, 1, 1),
}


def build(n: int) -> TopologyBuilder:
    tb = TopologyBuilder(d=3)
    for name, pos in CORNERS.items():
        tb.add_corner(name, pos)
    tb.add_block(
        "west",
        corners=("a00", "a01", "a10", "a11", "s00", "s01", "s10", "s11"),
        resolutions=(n, n, n),
    )
    tb.add_block(
        "east",
        corners=("s00", "s01", "s10", "s11", "b00", "b01", "b10", "b11"),
        resolutions=(n, n, n),
    )
    return tb


def main(argv=None):
    p = argparse.ArgumentParser(description="imperative 3D two-block slab")
    p.add_argument("--n", type=int, default=4, help="cells per axis per block")
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--sweeps", type=int, default=200)
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument(
        "--export-eggy",
        metavar="PATH",
        default=None,
        help="after the run, pack this example folder into a .eggy archive",
    )
    a = p.parse_args(argv)

    topo = build(a.n).build()
    grid = topo.initialize_grid()
    print(
        f"blocks={len(topo.block_specs)} nodes={grid.global_node_count} "
        f"singularities={len(topo.singularities)}"
    )

    # Refit + Save leave a control net beside the script so the exported .eggy
    # carries one (a regrid can resample from it); the smoother itself is nodal.
    net_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "net.npz")
    stages = [
        Untangle(),
        JacobiSmoother(sweeps=a.sweeps, chunk=a.chunk),
        Refit(),
        Save(net_path),
    ]
    for phase, info in generate_steps(grid, stages=stages, device=a.device):
        if phase in ("init", "final") or phase == "tmop":
            bits = " ".join(
                f"{k}={v:.4e}" if isinstance(v, float) else f"{k}={v}"
                for k, v in info.items()
            )
            print(f"  {phase}: {bits}")

    if a.export_eggy:
        from egg.io import eggy

        eggy.pack(a.export_eggy, os.path.dirname(os.path.abspath(__file__)))
        print(f"Exported .eggy archive to {a.export_eggy}")

    assert not np.any(np.isnan(grid.global_nodes))
    print("Done.")


if __name__ == "__main__":
    main()
