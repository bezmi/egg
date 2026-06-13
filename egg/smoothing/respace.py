"""Hard wall-normal respacing: exact first-cell height and growth rate.

TMOP targets are soft — the optimiser trades them off against grid quality,
so a :class:`~egg.smoothing.targets.BoundaryLayerTarget` only approximates
the requested ``first_height``/``growth``. This post-pass enforces them
exactly: for every block face recorded via ``builder.set_boundary_layer``,
each wall-normal grid line is re-sampled along its own polyline so the
first ``n_layers`` cells follow ``s(k) = first_height · growth**k`` to
machine precision, and the remaining cells continue with a single stretch
ratio solved so the line's far endpoint (shared with the neighbouring
block) does not move.

Lines keep their shape — nodes only slide along the existing polyline — so
boundary-snapped nodes stay on their geometry and interfaces shared between
adjacent wall blocks (which are respaced identically from either side)
remain conforming.
"""

from __future__ import annotations

import numpy as np

from .targets import _neighbour_across

__all__ = ["enforce_boundary_layer_spacing"]


def _solve_stretch_ratio(s0: float, n: int, length: float) -> float:
    """Ratio ``r`` so ``s0·(r + r² + … + rⁿ)`` equals ``length`` (bisection).

    The sum is strictly increasing in ``r``; ``r`` is bracketed in
    ``[1e-3, 1e3]`` which covers any sane cell-count/length combination.
    """
    def total(r: float) -> float:
        if abs(r - 1.0) < 1e-12:
            return s0 * n
        return s0 * r * (r**n - 1.0) / (r - 1.0)

    lo, hi = 1e-3, 1e3
    if not (total(lo) <= length <= total(hi)):
        raise ValueError(
            "boundary-layer respacing: cannot fit remaining cells "
            f"(need {length:.3e} over {n} cells after a {s0:.3e} cell)")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if total(mid) < length:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _respace_line(points: np.ndarray, first_height: float, growth: float,
                  n_layers: int, max_height: float | None,
                  entity=None) -> np.ndarray:
    """Re-sample one wall-normal polyline (wall first) to the geometric law.

    With ``entity`` given, the law is enforced in perpendicular wall
    distance: node ``k`` is placed where the line's distance to the wall
    equals the cumulative layer height. Layers then sit at the same height
    above the wall on every line, so oblique lines (e.g. along a domain
    boundary that meets the wall at an angle) cannot lift or fan the
    boundary layer. Without ``entity`` the law is applied to the line's own
    arc length. Returns the new points; endpoints are preserved exactly.
    """
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    n_cells = len(points) - 1

    if entity is not None:
        # Perpendicular wall distance at each original node, regularised to
        # something strictly increasing so it is invertible against cum.
        dist = np.array([
            np.linalg.norm(p - np.asarray(entity.project(p))) for p in points
        ])
        dist[0] = 0.0
        dist = np.maximum.accumulate(dist)
        dist += np.arange(len(dist)) * 1e-15
    else:
        dist = cum
    total = float(dist[-1])

    m = min(n_layers, n_cells - 1)
    spacings = first_height * growth ** np.arange(m)
    if max_height is not None:
        spacings = np.minimum(spacings, max_height)
    s_geo = float(spacings.sum())
    if s_geo >= total:
        raise ValueError(
            "boundary-layer respacing: geometric layers "
            f"(height {s_geo:.3e}) do not fit in the block "
            f"(available height {total:.3e}); reduce "
            "first_height/growth/n_layers")

    last = float(spacings[-1]) if m > 0 else first_height
    r = _solve_stretch_ratio(last, n_cells - m, total - s_geo)
    tail = last * r ** np.arange(1, n_cells - m + 1)
    pos = np.concatenate([[0.0], np.cumsum(np.concatenate([spacings, tail]))])
    pos[-1] = total
    arc = np.interp(pos, dist, cum)

    new = np.empty_like(points)
    for c in range(points.shape[1]):
        new[:, c] = np.interp(arc, cum, points[:, c])
    new[0], new[-1] = points[0], points[-1]
    return new


def _oriented_dof_lines(grid, topology, block_name: str, axis: int,
                        side: int) -> np.ndarray:
    """A block's DOF map with ``axis`` moved to the front, ``side`` at row 0."""
    block_names = list(topology.block_specs.keys())
    dm = np.moveaxis(grid.block_dof_maps[block_names.index(block_name)],
                     axis, 0)
    return dm[::-1] if side == 1 else dm


def enforce_boundary_layer_spacing(grid, topology=None,
                                   extend_through_neighbours: bool = True) -> None:
    """Exactly enforce recorded boundary-layer specs on ``grid`` (in place).

    @param grid      A (smoothed) MultiBlockGrid; ``grid.global_nodes`` is
                     updated in place.
    @param topology  Defaults to ``grid.topology``; must carry
                     ``boundary_layer_specs`` recorded by
                     ``builder.set_boundary_layer`` and the wall associations.
    @param extend_through_neighbours
                     Continue each respaced line into the block behind the
                     wall block (when one is glued there), so the stretch
                     tail relaxes over both blocks instead of compressing
                     against the first shared interface.
    """
    topology = topology if topology is not None else grid.topology
    specs = getattr(topology, "boundary_layer_specs", {})
    if not specs:
        return

    for assoc in topology.associations:
        spec = specs.get(id(assoc.entity))
        if spec is None:
            continue
        face = assoc.face
        dm = _oriented_dof_lines(grid, topology, face.block_name,
                                 face.axis, face.side)
        flat = dm.reshape(dm.shape[0], -1)

        if extend_through_neighbours:
            nb = _neighbour_across(topology, face.block_name,
                                   face.axis, 1 - face.side)
            if nb is not None:
                # Orient the neighbour with the shared face at row 0 and its
                # transverse ordering matched to the wall block's outer face.
                ndm = _oriented_dof_lines(grid, topology, *nb)
                nflat = ndm.reshape(ndm.shape[0], -1)
                if np.array_equal(nflat[0], flat[-1]):
                    flat = np.concatenate([flat, nflat[1:]], axis=0)
                elif np.array_equal(nflat[0][::-1], flat[-1]):
                    flat = np.concatenate([flat, nflat[1:, ::-1]], axis=0)
                # Unmatched transverse ordering (3D permutations): fall back
                # to respacing within the wall block only.

        for col in range(flat.shape[1]):
            dofs = flat[:, col]
            pts = grid.global_nodes[dofs]
            grid.global_nodes[dofs] = _respace_line(
                pts, spec["first_height"], spec["growth"],
                spec["n_layers"], spec["max_height"], entity=assoc.entity)
