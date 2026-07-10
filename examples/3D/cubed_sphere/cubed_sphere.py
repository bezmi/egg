# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Sphere-in-cube O-shell, built with ``TopologyBuilder(d=3)``.

The 6-block cubed-sphere O-shell of the ``sphere3d`` example, expressed
declaratively: 8 sphere corners and 8 cube corners, six radial blocks (one per
cube face) sharing corners so their interfaces merge, each inner face bound to
the ``Sphere`` and outer face to a cube ``Plane``. The eight octant corners are
3-block fans (singular), which the general orientation map and singularity
detection handle. It initialises (face nodes project onto the sphere) and smooths
through the standard pipeline; in 3D the pipeline's min-det/energy monitoring
routes through the C++ device reduction.

Run: ``uv run --no-sync python cubed_sphere.py --device cpu``.
"""

from __future__ import annotations

import argparse

import numpy as np

from egg.geometry.analytic3d import Plane, Sphere
from egg.pipeline import PipelineConfig, generate_steps
from egg.topology.builder import TopologyBuilder

# sign of (e_i x e_j) . e_k for the two axes other than k, in ascending order.
_SIGN_IJ = {0: 1, 1: -1, 2: 1}


def cubed_sphere(n_rad: int, n_tan: int, r0: float = 0.5, cw: float = 1.0):
    """The 6-block O-shell topology (radial x tangential x tangential per block)."""
    signs = (-1, 1)
    tb = TopologyBuilder(d=3)

    def sph(d):
        return "S_%+d%+d%+d" % tuple(d)

    def cub(d):
        return "C_%+d%+d%+d" % tuple(d)

    for sx in signs:
        for sy in signs:
            for sz in signs:
                d = np.array([sx, sy, sz], float)
                tb.add_corner(sph([sx, sy, sz]), r0 * d / np.linalg.norm(d))
                tb.add_corner(cub([sx, sy, sz]), cw * d, fixed=True)

    for k in (0, 1, 2):
        i, j = (a for a in (0, 1, 2) if a != k)
        for s in signs:
            # order the two tangential axes so the block frame is right-handed
            a1, a2 = (i, j) if s == _SIGN_IJ[k] else (j, i)

            def corner(radial, t1, t2, _k=k, _s=s, _a1=a1, _a2=a2):
                d = [0, 0, 0]
                d[_k], d[_a1], d[_a2] = _s, t1, t2
                return (sph if radial == 0 else cub)(d)

            corners = tuple(
                corner(i0, signs[i1], signs[i2])
                for i0 in (0, 1)
                for i1 in (0, 1)
                for i2 in (0, 1)
            )
            name = "blk_%d%+d" % (k, s)
            tb.add_block(name, corners=corners, resolutions=(n_rad, n_tan, n_tan))
            tb.associate(name, 0, 0, Sphere((0, 0, 0), r0, (1, 0, 0), (0, 1, 0)))
            o = [0.0, 0.0, 0.0]
            o[k] = s * cw
            axa = [1.0 if m == a1 else 0.0 for m in (0, 1, 2)]
            axb = [1.0 if m == a2 else 0.0 for m in (0, 1, 2)]
            tb.associate(name, 0, 1, Plane(o, axa, axb))
    return tb


def main(argv=None):
    p = argparse.ArgumentParser(description="sphere-in-cube O-shell (builder)")
    p.add_argument("--n", type=int, default=4, help="radial cells per block")
    p.add_argument("--nt", type=int, default=6, help="tangential cells per block")
    p.add_argument("--r0", type=float, default=0.5, help="sphere radius")
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--sweeps", type=int, default=300)
    p.add_argument("--chunk", type=int, default=100)
    a = p.parse_args(argv)

    topo = cubed_sphere(a.n, a.nt, r0=a.r0).build()
    grid = topo.initialize_grid()
    print(
        f"blocks={len(topo.block_specs)} nodes={grid.global_node_count} "
        f"singular fans={len(topo.singularities)}"
    )

    cfg = PipelineConfig(device=a.device, tmop_sweeps=a.sweeps, tmop_chunk=a.chunk)
    for phase, info in generate_steps(grid, config=cfg, untangle_direct=True):
        if phase in ("init", "untangle", "final"):
            bits = " ".join(
                f"{k}={v:.4e}" if isinstance(v, float) else f"{k}={v}"
                for k, v in info.items()
            )
            print(f"  {phase}: {bits}")
    print("Done.")


if __name__ == "__main__":
    main()
