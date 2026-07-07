# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

import sys
import os
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "examples", "sphere3d"))

# ruff: noqa: E402  (sys.path setup must precede the example import)
from egg._cpp import cpp_core
from sphere_in_cube import build_grid, classify, build_context
from egg.smoothing.cpp_backend import (
    build_structured_context_from_block_maps,
    structured_arrays,
)

n = 15
X, blocks = build_grid(n, 4, 3, 0.5, 0.7)
dof_entities, tags, fixed = classify(X, 0.5)
ctx = build_context(X, blocks, dof_entities, tags, fixed)
bsc = build_structured_context_from_block_maps(3, blocks, X.shape[0])
structured = structured_arrays(bsc)
print(f"nodes={X.shape[0]}")


def bench(device, smoother):
    sess = cpp_core.CppStructuredSweepSession(
        ctx, structured, X.ravel(), device=device, dim=3
    )
    sess.run(5, smoother=smoother)  # warm-up (discard)
    t0 = time.perf_counter()
    sess.run(30, smoother=smoother)
    dt = time.perf_counter() - t0
    return dt / 30 * 1000


for device in ["cpu", "gpu"]:
    for sm in ["colored-gs", "block-jacobi"]:
        try:
            ms = bench(device, sm)
            print(f"{device:4s} {sm:12s} {ms:8.1f} ms/sweep")
        except Exception as ex:
            print(f"{device} {sm} FAILED: {ex}")
