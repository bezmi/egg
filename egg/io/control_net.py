# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Control-net persistence.

Saves the state of a :class:`~egg.smoothing.control_topology.ControlTopology`
(the reduced control vector ``q``, per-block control counts, and the
Boolean-sum corrections ``b``) to a compressed ``.npz``. The symbolic
structure (seams, union-find roots, the reduction ``R``) is NOT stored — it
is deterministic given the grid's topology, so loading rebuilds it with
:func:`~egg.smoothing.control_topology.build_control_topology` and restores
the saved state onto it. This keeps the file format tiny and immune to
internal representation changes; the cost is that loading needs the same
grid (same blocks, resolutions, and constraints) the net was solved on.
"""

from __future__ import annotations

import numpy as np

__all__ = ["save_control_net", "load_control_net"]

_FORMAT_VERSION = 1


def save_control_net(topo, path) -> None:
    """Write a solved control topology's state to ``path`` (``.npz``)."""
    payload = {
        "version": np.asarray(_FORMAT_VERSION),
        "d": np.asarray(topo.d),
        "n_blocks": np.asarray(len(topo.ctrl_shapes)),
        "q": np.asarray(topo.q, dtype=np.float64),
        "walls": np.asarray(bool(topo.wall_faces)),
        "fit_residual": np.asarray(float(topo.fit_residual)),
    }
    for bi, cs in enumerate(topo.ctrl_shapes):
        payload[f"ctrl_shape_{bi}"] = np.asarray(cs, dtype=np.int64)
        payload[f"b_{bi}"] = np.asarray(topo.b_fields[bi], dtype=np.float64)
        for k, p in enumerate(topo.axis_params[bi]):
            payload[f"params_{bi}_{k}"] = np.asarray(p, dtype=np.float64)
        for k, u in enumerate(topo.cmaps[bi].knots):
            payload[f"knots_{bi}_{k}"] = np.asarray(u, dtype=np.float64)
    np.savez_compressed(path, **payload)


def load_control_net(grid, path):
    """Rebuild the control topology for ``grid`` and restore a saved state.

    ``grid`` must be the same multi-block grid (topology, per-block
    resolutions, constraints) the net was solved on; mismatches raise. The
    returned topology carries the saved ``q`` / ``b`` — evaluate it with
    ``topo.write_to_grid()`` or resample it with
    :func:`~egg.smoothing.control_topology.resample_block`.
    """
    from egg.smoothing.control_topology import build_control_topology

    data = np.load(path)
    if int(data["version"]) != _FORMAT_VERSION:
        raise ValueError(
            f"control-net file version {int(data['version'])} != {_FORMAT_VERSION}"
        )
    d = int(data["d"])
    if d != grid.topology.d:
        raise ValueError(
            f"control-net dimension {d} != grid dimension {grid.topology.d}"
        )
    nb = int(data["n_blocks"])
    if nb != len(grid.blocks):
        raise ValueError(f"control net has {nb} blocks, grid has {len(grid.blocks)}")
    ctrl_shapes = [tuple(int(x) for x in data[f"ctrl_shape_{bi}"]) for bi in range(nb)]
    axis_params = [
        [np.asarray(data[f"params_{bi}_{k}"], dtype=float) for k in range(d)]
        for bi in range(nb)
    ]

    # Knots define the net; older files without them re-derive from the
    # stored parameters (those nets carried no fan refinement).
    knots = None
    if "knots_0_0" in data:
        knots = [
            [np.asarray(data[f"knots_{bi}_{k}"], dtype=float) for k in range(d)]
            for bi in range(nb)
        ]
    topo = build_control_topology(
        grid,
        ctrl_shapes,
        walls=bool(data["walls"]),
        fit_spacing=axis_params,
        knots=knots,
    )
    q = np.asarray(data["q"], dtype=float)
    if q.shape != topo.q.shape:
        raise ValueError(
            f"saved reduced state has shape {q.shape}, the rebuilt topology "
            f"expects {topo.q.shape} — the grid does not match the one the "
            "net was solved on"
        )
    topo.q = q
    for bi in range(nb):
        b = np.asarray(data[f"b_{bi}"], dtype=float)
        if b.shape != topo.b_fields[bi].shape:
            raise ValueError(f"saved b field for block {bi} has shape {b.shape}")
        topo.b_fields[bi] = b
    topo.fit_residual = float(data["fit_residual"])
    return topo
