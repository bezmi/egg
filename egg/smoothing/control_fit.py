# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Fit a control net to a grid that already holds node coordinates.

The control smoother builds a coarse per-block B-spline net and then MOVES
it. This module does only the first half: it fits a net to whatever
coordinates the grid currently holds and checks that the fitted net still
makes a valid grid. It never runs the smoother and never changes the grid.

The fitted net is a small, resolution-independent summary of the grid. A grid
made by a node smoother (Jacobi or FAS) has no net of its own; fitting one to
the final grid gives every run a net to store, regrid, and cache.

A fitted net only approximates its grid (to ``topo.fit_residual``), unlike a
net the smoother solved from scratch, which the grid matches exactly. So a
regrid from a fitted net leans a little harder on the polish pass.
"""

from __future__ import annotations

import numpy as np

from egg.smoothing.control_topology import (
    _corner_mindet,
    build_control_topology,
    default_ctrl_shapes,
)

__all__ = ["fit_control_net", "ControlFitError"]


class ControlFitError(ValueError):
    """No fitted net keeps the evaluated grid valid.

    The grid itself stays valid and usable; there is simply no net to store.
    """


def _net_min_det(topo) -> float:
    """Smallest cell corner determinant of the net evaluated onto its grid.

    Pure geometry (no metric), so it is a cheap check that the fitted net
    produces a fold-free grid.
    """
    m = np.inf
    for bi, cmap in enumerate(topo.cmaps):
        X = cmap.prolong(topo.block_C(bi)) + topo.b_fields[bi]
        m = min(m, _corner_mindet(X))
    return m


def fit_control_net(
    grid,
    *,
    ratio: int = 4,
    walls: bool = True,
    fit_spacing: str = "chord",
    fan_refine: int = 2,
):
    """Fit a control net to the grid's current node coordinates.

    Solves the same small least-squares fit the control smoother starts from,
    but stops there: the smoother does not run and ``grid`` is not touched.

    Parameters
    ----------
    grid : MultiBlockGrid
        Read only. Its current node coordinates are the fit target.
    ratio : int
        Cells per control point per axis. Larger is coarser: cheaper to store
        and a looser fit. 2 tracks curved walls closely; 4 suits smooth
        interiors.
    walls : bool
        Detect outer faces riding one CAD entity and let their controls slide
        on it (see
        :func:`~egg.smoothing.control_topology.build_control_topology`).
    fit_spacing : str
        ``"chord"`` samples the grid's own spacing so a clustered grid stays
        representable; ``"uniform"`` samples evenly.
    fan_refine : int
        Extra knots at net axis ends landing on a singular fan, where a coarse
        net underfits the tight turning. Backed off automatically if the
        refined fit folds.

    Returns
    -------
    ControlTopology
        Carrying the fitted ``q`` and ``b`` and the recorded ``fit_residual``.

    Raises
    ------
    ControlFitError
        When even the coarsest fan setting leaves the evaluated net folded.
    """
    shapes = default_ctrl_shapes(grid, r=ratio)
    fr = max(int(fan_refine), 0)
    while True:
        topo = build_control_topology(
            grid, shapes, walls=walls, fit_spacing=fit_spacing, fan_refine=fr
        )
        if _net_min_det(topo) > 0.0:
            return topo
        if fr == 0:
            raise ControlFitError(
                f"the fitted control net folds the grid (ratio={ratio}); the "
                "grid stays valid but has no net to store"
            )
        fr -= 1
