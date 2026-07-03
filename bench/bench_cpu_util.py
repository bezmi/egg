# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Focused CPU-utilisation benchmark for the C++ coloured Gauss-Seidel sweep.

Builds a large circle-in-rectangle O-grid, then runs many barrier sweeps on a
persistent device-resident ``CppSweepSession`` (``device="cpu"``). After the
one-time setup + warm-up, ~all wall time is the sweep hot loop, so
``/usr/bin/time -v`` "Percent of CPU this job got" reflects sweep-thread
utilisation. With the #1 per-colour launch merge it should approach
``100 % × OMP_NUM_THREADS``.

  OMP_NUM_THREADS=8 /usr/bin/time -v \
      uv run bench/bench_cpu_util.py --R 80 --sweeps 4000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=80, help="O-grid resolution")
    ap.add_argument("--sweeps", type=int, default=4000, help="timed barrier sweeps")
    ap.add_argument("--warmup", type=int, default=50, help="untimed JIT/warm sweeps")
    args = ap.parse_args()

    from topologies import build_circle_in_rectangle

    from egg.smoothing.cpp_backend import CppSweepSession
    from egg.smoothing.solver import build_sweep_context
    from egg.smoothing.targets import IdentityTarget

    t_setup0 = time.perf_counter()
    topo, _ents = build_circle_in_rectangle(rough=False, R=args.R)
    grid = topo.initialize_grid()
    ctx = build_sweep_context(grid, IdentityTarget(d=topo.d))
    X0 = grid.global_nodes.copy()
    sess = CppSweepSession(ctx, X0, device="cpu")  # flatten_context + USM upload
    t_setup = time.perf_counter() - t_setup0

    M = X0.shape[0]
    omp = os.environ.get("OMP_NUM_THREADS", "<unset>")
    print(
        f"grid: R={args.R}  nodes={M}  colours={ctx.num_colours}  "
        f"OMP_NUM_THREADS={omp}   (setup {t_setup:.2f}s)",
        flush=True,
    )

    sess.run(args.warmup)  # JIT + OpenMP thread-pool spin-up (untimed)

    t0 = time.perf_counter()
    e, m = sess.run(args.sweeps)
    dt = time.perf_counter() - t0
    print(
        f"sweeps={args.sweeps}  sweep_wall={dt:.3f}s  {1e3 * dt / args.sweeps:.3f} ms/sweep"
        f"   energy[-1]={float(np.asarray(e)[-1]):.6e}  mindet[-1]={float(np.asarray(m)[-1]):.3e}",
        flush=True,
    )
    # Fraction of total wall the (multi-threaded) sweep represents — the closer to
    # 1, the more "Percent of CPU" reflects the sweep rather than setup.
    print(
        f"sweep fraction of timed-region wall ≈ {dt / (dt + t_setup):.2f}", flush=True
    )


if __name__ == "__main__":
    main()
