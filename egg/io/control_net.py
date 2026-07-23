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

import os

import numpy as np

__all__ = [
    "save_control_net",
    "load_control_net",
    "try_load_control_net",
    "retabulate_control_net",
]

_FORMAT_VERSION = 1


def save_control_net(topo, path, *, residual=None) -> None:
    """Write a solved control topology's state to ``path`` (``.npz``).

    With ``residual`` the file also carries an exact-reproduction layer: the
    per-node difference between a solved grid and the net evaluation, so a
    later load at the SAME sampling reproduces that grid bit-for-bit. Pass the
    solved grid's global-node array, or ``True`` to use ``topo.grid``. The
    residual is tied to the stored sampling and is only meaningful there (a
    load at a different resolution must interpolate it, which is lossy).
    """
    payload = {
        "version": np.asarray(_FORMAT_VERSION),
        "d": np.asarray(topo.d),
        "n_blocks": np.asarray(len(topo.ctrl_shapes)),
        "q": np.asarray(topo.q, dtype=np.float64),
        "walls": np.asarray(bool(topo.wall_faces)),
        "fit_residual": np.asarray(float(topo.fit_residual)),
    }
    X_solved: np.ndarray | None = None
    if residual is not None:
        X_solved = (
            np.asarray(topo.grid.global_nodes, dtype=float)
            if residual is True
            else np.asarray(residual, dtype=float)
        )
        payload["has_residual"] = np.asarray(True)
    for bi, cs in enumerate(topo.ctrl_shapes):
        payload[f"ctrl_shape_{bi}"] = np.asarray(cs, dtype=np.int64)
        payload[f"b_{bi}"] = np.asarray(topo.b_fields[bi], dtype=np.float64)
        if X_solved is not None:
            # Per block, at the stored sampling: the exact grid minus the net
            # evaluation. Per block so a re-tabulating load can interpolate it.
            dm = topo.grid.block_dof_maps[bi]
            net_bi = topo.cmaps[bi].prolong(topo.block_C(bi)) + topo.b_fields[bi]
            payload[f"residual_{bi}"] = (X_solved[dm] - net_bi).astype(np.float64)
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
    if "has_residual" in data:
        # Exact-reproduction layer: the per-node difference between the saved
        # (post-processed) grid and the net evaluation. Carry the raw source
        # keyed to its original sampling so a later remesh can re-interpolate
        # it; here the sampling matches, so residual_global() returns it
        # exactly. Apply with topo.write_to_grid(exact=True).
        topo.residual_src = (
            [
                [np.asarray(data[f"params_{bi}_{k}"], dtype=float) for k in range(d)]
                for bi in range(nb)
            ],
            [np.asarray(data[f"residual_{bi}"], dtype=float) for bi in range(nb)],
        )
        topo.residual = topo.residual_global()
    return topo


def try_load_control_net(grid, path):
    """Load a saved net for ``grid`` if one is usable, else return ``None``.

    A soft version of :func:`load_control_net` for the warm-start cache: it
    returns ``None`` (instead of raising) when the file is missing or does not
    match ``grid`` (a different resolution or topology), so the caller simply
    solves from scratch. Any other error still propagates.
    """
    if path is None or not os.path.exists(path):
        return None
    try:
        return load_control_net(grid, path)
    except (ValueError, KeyError):
        return None


def retabulate_control_net(grid, path, *, spacing=None):
    """Put a saved net onto ``grid`` at a DIFFERENT resolution or clustering.

    The net's shape lives in its control points and knots, which do not depend
    on the grid resolution; only the basis sampling and the boundary correction
    ``b`` do. So this rebuilds the net's structure on ``grid`` (which may have a
    new resolution or clustering, but the SAME topology: blocks, connectivity,
    constraints), restores the saved control state, and re-extends ``b`` at the
    new sampling. It does NOT re-fit, so there is no sample-density limit; the
    result is the stored shape re-evaluated on the new grid, a strong start for
    a short polish.

    Parameters
    ----------
    grid : MultiBlockGrid
        The new grid to put the net on (same topology, new resolution).
    path : str
        A saved net (.npz).
    spacing : list, optional
        Per-block list of per-axis node parameters (each in [0, 1]) for
        clustering. ``None`` samples each axis evenly.

    Returns
    -------
    ControlTopology or None
        With the restored control state and re-extended ``b``. ``None`` when
        the saved net does not fit this grid's topology (a changed block /
        seam / fan layout, not just a changed resolution) or the re-evaluated
        net folds.

    Raises
    ------
    ValueError
        On a saved-file version this loader cannot read.
    """
    from egg.smoothing.control_fit import _net_min_det
    from egg.smoothing.control_topology import build_control_topology, wall_b_field

    data = np.load(path)
    if int(data["version"]) != _FORMAT_VERSION:
        raise ValueError(
            f"control-net file version {int(data['version'])} != {_FORMAT_VERSION}"
        )
    d = int(data["d"])
    nb = int(data["n_blocks"])
    if d != grid.topology.d or nb != len(grid.blocks):
        return None
    ctrl_shapes = [tuple(int(x) for x in data[f"ctrl_shape_{bi}"]) for bi in range(nb)]
    knots = None
    if "knots_0_0" in data:
        knots = [
            [np.asarray(data[f"knots_{bi}_{k}"], dtype=float) for k in range(d)]
            for bi in range(nb)
        ]
    fit_spacing = spacing if spacing is not None else "uniform"
    try:
        topo = build_control_topology(
            grid,
            ctrl_shapes,
            walls=bool(data["walls"]),
            fit_spacing=fit_spacing,
            knots=knots,
        )
    except ValueError:
        # A resolution the topology cannot represent conformingly.
        return None
    q = np.asarray(data["q"], dtype=float)
    if q.shape != topo.q.shape:
        # The reduced layout differs: the topology changed, not just the
        # resolution. Refuse rather than restore a mismatched state.
        return None
    topo.q = q
    # Re-extend b against the restored spline at the new sampling.
    for bi in range(nb):
        topo.b_fields[bi] = wall_b_field(
            topo, bi, np.asarray(grid.blocks[bi].nodes, dtype=float)
        )
    if _net_min_det(topo) <= 0.0:
        return None
    topo.fit_residual = float(data["fit_residual"]) if "fit_residual" in data else 0.0
    if "has_residual" in data:
        _attach_interpolated_residual(topo, grid, data, d, nb)
    return topo


def _attach_interpolated_residual(topo, grid, data, d, nb):
    """Attach a saved residual's raw source and interpolate it to the new grid.

    The residual is resolution-locked, so at a new resolution it is
    interpolated in parameter space (exact at nodes that coincide with the
    original ones, e.g. a 2x refinement; smoothly filled in between). The raw
    source is stored on the topology so a later remesh can re-interpolate it to
    yet another sampling; the interpolated ``residual`` array is kept only if
    the net plus residual makes a valid grid, so it can seed a shorter polish
    (otherwise the net alone is used).
    """
    topo.residual_src = (
        [
            [np.asarray(data[f"params_{bi}_{k}"], dtype=float) for k in range(d)]
            for bi in range(nb)
        ],
        [np.asarray(data[f"residual_{bi}"], dtype=float) for bi in range(nb)],
    )
    from egg.smoothing.control_topology import _corner_mindet

    res = topo.residual_global()
    m = np.inf
    for bi in range(nb):
        net_bi = topo.cmaps[bi].prolong(topo.block_C(bi)) + topo.b_fields[bi]
        m = min(m, _corner_mindet(net_bi + res[grid.block_dof_maps[bi]]))
    topo.residual = res if m > 0.0 else None
