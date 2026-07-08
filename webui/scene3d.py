# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""3D scene payload for the web UI's WebGL viewer.

The 2D UI renders SVG; the 3D UI hands a plain JSON payload to a three.js /
three-cad-viewer scene. This module builds that payload and stays framework-free
(no fasthtml, no browser): per-block hexahedral boundary-surface meshes (the
grid the solver produced) plus feature-curve polylines, with the overall bounds.
Surfaces are shown implicitly by the block faces that lie on them.
"""

from __future__ import annotations

import numpy as np

__all__ = ["topology_scene3d", "geometry_scene3d", "scene3d_payload"]


def _hex_surface(nodes: np.ndarray):
    """Boundary-face quads of a hex block: (verts (N,3), quad index list)."""
    ni, nj, nk = nodes.shape[:3]
    verts = nodes.reshape(-1, 3)

    def vid(i, j, k):
        return (i * nj + j) * nk + k

    quads: list[list[int]] = []
    for i in (0, ni - 1):
        for j in range(nj - 1):
            for k in range(nk - 1):
                quads.append(
                    [
                        vid(i, j, k),
                        vid(i, j + 1, k),
                        vid(i, j + 1, k + 1),
                        vid(i, j, k + 1),
                    ]
                )
    for j in (0, nj - 1):
        for i in range(ni - 1):
            for k in range(nk - 1):
                quads.append(
                    [
                        vid(i, j, k),
                        vid(i + 1, j, k),
                        vid(i + 1, j, k + 1),
                        vid(i, j, k + 1),
                    ]
                )
    for k in (0, nk - 1):
        for i in range(ni - 1):
            for j in range(nj - 1):
                quads.append(
                    [
                        vid(i, j, k),
                        vid(i + 1, j, k),
                        vid(i + 1, j + 1, k),
                        vid(i, j + 1, k),
                    ]
                )
    return verts, quads


def topology_scene3d(grid) -> dict:
    """Per-block hex boundary meshes + overall bounds from a MultiBlockGrid."""
    topo = getattr(grid, "topology", None)
    names = list(topo.block_specs) if topo is not None else []
    blocks, allv = [], []
    for bi, block in enumerate(grid.blocks):
        nodes = np.asarray(block.nodes, dtype=float)
        if nodes.ndim != 4:  # 3D blocks only
            continue
        verts, quads = _hex_surface(nodes)
        blocks.append(
            {
                "name": names[bi] if bi < len(names) else str(bi),
                "verts": verts.tolist(),
                "quads": quads,
            }
        )
        allv.append(verts)
    bounds = None
    if allv:
        v = np.vstack(allv)
        bounds = {"min": v.min(0).tolist(), "max": v.max(0).tolist()}
    return {"blocks": blocks, "bounds": bounds}


def geometry_scene3d(geometry: dict, n: int = 48) -> list[dict]:
    """Feature curves in the geometry, sampled to 3D polylines."""
    out = []
    for name, ent in (geometry or {}).items():
        e = getattr(ent, "entity", ent)  # unwrap frontend Edge wrappers
        if getattr(e, "dim", 1) != 1 or not hasattr(e, "eval_frac"):
            continue  # surfaces (dim 2) are shown by the block faces on them
        try:
            pts = np.array(
                [
                    np.asarray(e.eval_frac(t), dtype=float)[:3]
                    for t in np.linspace(0.0, 1.0, n)
                ]
            )
        except Exception:
            continue
        if pts.ndim == 2 and pts.shape[1] == 3:
            out.append({"name": name, "kind": "curve", "line": pts.tolist()})
    return out


def scene3d_payload(grid, geometry: dict | None = None) -> dict:
    """The full 3D scene: block meshes, feature curves, and bounds."""
    payload = topology_scene3d(grid)
    payload["curves"] = geometry_scene3d(geometry or {})
    return payload
