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

"""Block-boundary orthogonality / continuity metric — end-to-end demo.

Two sheared blocks share an oblique seam. Plain shape smoothing crosses the
seam somewhat obliquely; the composed interface term (``mode="normal"``) pulls
the crossing grid lines perpendicular to the seam, while ``mode="continuous"``
keeps them straight at their natural angle. Runs the real C++ structured
block-Jacobi sweep and, with matplotlib, writes a before/after wireframe.

    uv run --no-sync python examples/2D/block-ortho/block_ortho.py \
        --mode normal --weight 8 --out /tmp/block_ortho.png
"""

from __future__ import annotations

import argparse

import numpy as np

from egg.smoothing.cpp_backend import (
    CppStructuredSweepSession,
    build_block_structured_context,
)
from egg.smoothing.interface_ortho import _side_frames
from egg.smoothing.solver import build_sweep_context
from egg.smoothing.targets import IdentityTarget
from egg.topology.builder import TopologyBuilder


def pgram_grid(res=(10, 10), shear=1.0):
    """Two sheared blocks sharing an oblique x≈2 seam."""
    b = TopologyBuilder(d=2)
    for n, p in [
        ("A", (0, 0)),
        ("D", (shear, 2)),
        ("B", (2, 0)),
        ("C", (2 + shear, 2)),
        ("E", (4, 0)),
        ("F", (4 + shear, 2)),
    ]:
        b.add_corner(n, p, fixed=True)
    b.add_block("L", ("A", "D", "B", "C"), res)
    b.add_block("R", ("B", "C", "E", "F"), res)
    b.connect("L", 0, 1, "R", 0, 0)
    return b.build().initialize_grid()


def seam_obliquity(grid, X):
    """Mean |cos∠(crossing edge, seam tangent)| over seam nodes; 0 = orthogonal."""
    topo = grid.topology
    bn = list(topo.block_specs.keys())
    vals = []
    for conn in topo.interface_connections:
        fa = _side_frames(grid, bn, conn.face_a)
        fb = _side_frames(grid, bn, conn.face_b)
        for P, frA in fa.items():
            frB = fb.get(P)
            if frB is None:
                continue
            c = X[frA["Q"]] - X[frB["Q"]]
            t = X[frA["Rp"]] - X[frA["Rm"]]
            vals.append(
                abs(float(np.dot(c, t)) / (np.linalg.norm(c) * np.linalg.norm(t)))
            )
    return float(np.mean(vals)) if vals else float("nan")


def smooth(grid, interface_ortho, device, sweeps):
    ctx = build_sweep_context(grid, IdentityTarget(2), interface_ortho=interface_ortho)
    bsc = build_block_structured_context(grid)
    sess = CppStructuredSweepSession(ctx, bsc, grid.global_nodes, device=device)
    _e, m = sess.run(sweeps, phase="barrier", omega=0.8, report_every=0)
    return np.asarray(sess.get_X()).reshape(-1, 2), float(m[-1])


def draw(ax, grid, X, title):
    for dm in grid.block_dof_maps:
        dm = np.asarray(dm)
        P = X[dm]  # (n0, n1, 2)
        for i in range(P.shape[0]):
            ax.plot(P[i, :, 0], P[i, :, 1], color="0.3", lw=0.6)
        for j in range(P.shape[1]):
            ax.plot(P[:, j, 0], P[:, j, 1], color="0.3", lw=0.6)
    # Highlight the seam (shared nodes appear in both block maps).
    seam = grid.block_dof_maps[0][-1, :]
    ax.plot(X[seam, 0], X[seam, 1], color="crimson", lw=2.0, label="seam")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.axis("off")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["normal", "continuous"], default="normal")
    ap.add_argument("--weight", type=float, default=8.0)
    ap.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    ap.add_argument("--sweeps", type=int, default=300)
    ap.add_argument("--out", default="block_ortho.png")
    args = ap.parse_args(argv)

    Xb, mb = smooth(pgram_grid(), None, args.device, args.sweeps)
    Xi, mi = smooth(
        pgram_grid(),
        {"mode": args.mode, "weight": args.weight},
        args.device,
        args.sweeps,
    )
    gb = pgram_grid()
    ob_b = seam_obliquity(gb, Xb)
    ob_i = seam_obliquity(gb, Xi)
    print(f"base smoothing : seam obliquity={ob_b:.4f}  min det={mb:.3e}")
    print(
        f"{args.mode:<10} w={args.weight:<4g}: seam obliquity={ob_i:.4f}  min det={mi:.3e}"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    draw(axes[0], gb, Xb, f"shape only\nseam obliquity {ob_b:.3f}")
    draw(
        axes[1], gb, Xi, f"+ {args.mode} (w={args.weight:g})\nseam obliquity {ob_i:.3f}"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
