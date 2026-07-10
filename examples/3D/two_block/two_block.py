# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

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

import numpy as np

from egg.pipeline import PipelineConfig, generate_steps
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
    a = p.parse_args(argv)

    topo = build(a.n).build()
    grid = topo.initialize_grid()
    print(
        f"blocks={len(topo.block_specs)} nodes={grid.global_node_count} "
        f"singularities={len(topo.singularities)}"
    )

    cfg = PipelineConfig(device=a.device, tmop_sweeps=a.sweeps, tmop_chunk=a.chunk)
    for phase, info in generate_steps(grid, config=cfg, untangle_direct=True):
        if phase in ("init", "final") or phase == "tmop":
            bits = " ".join(
                f"{k}={v:.4e}" if isinstance(v, float) else f"{k}={v}"
                for k, v in info.items()
            )
            print(f"  {phase}: {bits}")

    assert not np.any(np.isnan(grid.global_nodes))
    print("Done.")


if __name__ == "__main__":
    main()
