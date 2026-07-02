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

from egg.projection.associate import build_dof_constraints

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

    When the geometric band grows past the mean spacing of the remainder,
    the solved tail ratio drops below 1 and the cells grade smoothly finer
    towards the far boundary. That double-sided grading is intentional:
    capping the growth instead (Eilmer GeometricFunction style) plateaus
    the wall-normal spacing, which fights the smoothed equilibrium and was
    measured to shear cells above curved walls (phoebus shoulder: interior
    angles collapsed from ~50 deg to 5 deg).
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


def _smoothstep(t: float) -> float:
    """Hermite smoothstep on ``[0, 1]`` (0 at 0, 1 at 1, zero end-slopes)."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _boundary_shear_direction(grid, entity, flat: np.ndarray, p: int,
                              n_layers: int) -> np.ndarray | None:
    """Unit direction of the pinned (domain-boundary) column off the wall."""
    pts = grid.global_nodes[flat[:, p]]
    k_ref = min(max(n_layers, 1), len(pts) - 1)
    v = pts[k_ref] - pts[0]
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else None


def _orthogonalise_columns(grid, entity, flat: np.ndarray,
                           pinned: list[bool], n_layers: int,
                           done: set, scale: float = 1.0,
                           shear_taper: int = 0,
                           dof_constraints: dict | None = None,
                           fixed_dofs: frozenset = frozenset()) -> None:
    """Straighten near-wall columns, relaxing towards oblique boundaries.

    TMOP equilibrium lets wall-normal grid lines lean to follow oblique
    domain boundaries, fanning the boundary-layer columns near the wall's
    ends. This pass re-anchors each node at its already-enforced
    perpendicular height, laterally on a straight ray through the
    column's foot — the wall normal in the interior, blending into the
    *boundary's own direction* over ``shear_taper`` columns next to a
    ``pinned`` (domain-boundary) column. Near an oblique boundary the
    cells therefore become uniformly sheared parallelograms that follow
    it instead of fanning, while layer heights stay exact (the ray length
    is rescaled by the shear angle).

    Full strength for the first ``n_layers`` layers, fading to nothing by
    ``2·n_layers`` so the lean above the clustered region stays with the
    smoother. On coarse grids where the band reaches the line's end, nodes
    that carry a geometry constraint (``dof_constraints``: sliding DOFs ->
    entity) are only redistributed *along* their entity — the straightened
    position is projected back onto it — so the far boundary row follows
    the fan tangentially without leaving its curve; ``fixed_dofs`` never
    move. ``pinned`` columns stay on their boundary path. ``scale``
    uniformly weakens the pass (used to back off if a cell would invert).
    """
    n_cols = flat.shape[1]
    k_max = min(2 * n_layers, flat.shape[0] - 1)

    # Per-column blend towards each pinned boundary's direction.
    shear = [None] * n_cols  # (lambda, b_hat) of the nearest pinned side
    if shear_taper > 0:
        for p in (0, n_cols - 1):
            if not pinned[p]:
                continue
            b_hat = _boundary_shear_direction(grid, entity, flat, p, n_layers)
            if b_hat is None:
                continue
            for col in range(n_cols):
                lam = 1.0 - _smoothstep(abs(col - p) / shear_taper)
                if lam > 0 and (shear[col] is None or lam > shear[col][0]):
                    shear[col] = (lam, b_hat)

    for col in range(n_cols):
        if pinned[col]:
            continue
        dofs = flat[:, col]
        key = int(dofs[1])
        if key in done:
            continue
        done.add(key)
        pts = grid.global_nodes[dofs]
        foot = pts[0]
        n_hat = np.asarray(entity.normal(np.asarray(entity.project(foot))),
                           dtype=float)
        if np.dot(n_hat, pts[1] - foot) < 0:
            n_hat = -n_hat
        d_hat = n_hat
        if shear[col] is not None:
            lam, b_hat = shear[col]
            d_hat = (1.0 - lam) * n_hat + lam * b_hat
            d_hat = d_hat / np.linalg.norm(d_hat)
        # Stretch the ray so the perpendicular layer height stays exact.
        stretch = 1.0 / max(float(np.dot(d_hat, n_hat)), 0.2)
        for k in range(1, k_max + 1):
            dof = int(dofs[k])
            if dof in fixed_dofs:
                continue
            height = np.linalg.norm(
                pts[k] - np.asarray(entity.project(pts[k])))
            w = 1.0 if k <= n_layers else \
                1.0 - _smoothstep((k - n_layers) / max(n_layers, 1))
            w *= scale
            new = w * (foot + height * stretch * d_hat) + (1.0 - w) * pts[k]
            ent_k = dof_constraints.get(dof) if dof_constraints else None
            if ent_k is not None and ent_k is not entity:
                new = np.asarray(ent_k.project(new), dtype=float)
            grid.global_nodes[dof] = new


def _region_orientation(nodes: np.ndarray, k_max: int) -> int:
    """Sign of the cell orientation in rows 0..k_max (0 if mixed/degenerate)."""
    region = nodes[:k_max + 2]
    det = np.cross(region[1:, :-1] - region[:-1, :-1],
                   region[:-1, 1:] - region[:-1, :-1])
    if np.all(det > 0):
        return 1
    if np.all(det < 0):
        return -1
    return 0


def _oriented_dof_lines(grid, topology, block_name: str, axis: int,
                        side: int) -> np.ndarray:
    """A block's DOF map with ``axis`` moved to the front, ``side`` at row 0."""
    block_names = list(topology.block_specs.keys())
    dm = np.moveaxis(grid.block_dof_maps[block_names.index(block_name)],
                     axis, 0)
    return dm[::-1] if side == 1 else dm


def enforce_boundary_layer_spacing(grid, topology=None,
                                   extend_through_neighbours: bool = True,
                                   relax_boundary_normals: bool = True) -> None:
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
    @param relax_boundary_normals
                     Near a domain boundary that meets the wall obliquely,
                     shear the straightened columns along the boundary's
                     direction instead of forcing them onto the wall normal
                     (heights stay exact). Off, columns stay wall-normal up
                     to the boundary, which can fan or invert cells against
                     an inward-leaning boundary.
    """
    topology = topology if topology is not None else grid.topology
    specs = getattr(topology, "boundary_layer_specs", {})
    if not specs:
        return

    dof_constraints, fixed_dofs = build_dof_constraints(topology, grid)
    fixed_dofs = frozenset(fixed_dofs)

    ortho_done: set = set()
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

        # Columns lying on an unconnected (domain-boundary) transverse face
        # must stay on their boundary path; all others are straightened onto
        # the wall normal through the clustered layers. If straightening
        # would invert a cell despite the taper, back off and finally give
        # up rather than break the grid.
        t_axis = 1 - face.axis if topology.d == 2 else None
        if t_axis is not None:
            pinned = [False] * flat.shape[1]
            if _neighbour_across(topology, face.block_name, t_axis, 0) is None:
                pinned[0] = True
            if _neighbour_across(topology, face.block_name, t_axis, 1) is None:
                pinned[-1] = True
            k_max = min(2 * spec["n_layers"], flat.shape[0] - 1)
            taper = max(3, spec["n_layers"]) if relax_boundary_normals else 0
            saved = grid.global_nodes[flat].copy()
            sign = _region_orientation(grid.global_nodes[flat], k_max)
            for attempt in range(4):
                done_try = set(ortho_done)
                _orthogonalise_columns(grid, assoc.entity, flat, pinned,
                                       spec["n_layers"], done_try,
                                       scale=0.5 ** attempt,
                                       shear_taper=taper,
                                       dof_constraints=dof_constraints,
                                       fixed_dofs=fixed_dofs)
                new_sign = _region_orientation(grid.global_nodes[flat], k_max)
                if sign == 0 or new_sign == sign:
                    ortho_done = done_try
                    break
                grid.global_nodes[flat] = saved
            else:
                grid.global_nodes[flat] = saved

    # Keep the per-block node views consistent with the updated globals
    # (plotting and export read block.nodes).
    for bi, blk in enumerate(grid.blocks):
        blk.nodes[...] = grid.global_nodes[grid.block_dof_maps[bi]]
