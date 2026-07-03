"""TMOP target-matrix (W) constructors.

All targets return a d x d matrix W with det(W) > 0, encoding the desired
local cell shape/orientation at each sample point. Called as
``target(bi, block, cell_base, corner_offset) -> (d, d)``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "IdentityTarget",
    "AnisotropicTarget",
    "BoundaryLayerTarget",
    "MultiBlockTarget",
    "build_boundary_layer_target",
]


def IdentityTarget(d: int) -> Callable[..., np.ndarray]:
    """Target function that always yields W = I (isotropic).

    Parameters
    ----------
    d : int
        Spatial dimension.

    Returns
    -------
    target_fn : callable
    """
    W = np.eye(d)

    def target(*args, **kwargs) -> np.ndarray:
        return W.copy()

    return target


def _smoothstep(t: float) -> float:
    """Hermite smoothstep on ``[0, 1]`` (0 at 0, 1 at 1, zero end-slopes)."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class BoundaryLayerTarget:
    """Oriented, layer-indexed wall-clustering target W for one ring block.

    Realises boundary-layer clustering purely through the TMOP target: the
    wall-normal target spacing grows geometrically away from the wall
    (``s_n(k) = first_height * growth**k``) until capped at the
    wall-tangential spacing ``s_t`` — far-from-wall cells then ask for an
    isotropic ``W`` matching the :func:`IdentityTarget` of neighbouring
    blocks (the shape metric is scale-invariant, so only the ``s_n/s_t``
    ratio acts). Orientation comes from the geometry *entity*, which stays
    stable even while the working mesh is folded during untangling.
    ``det W = s_t * s_n > 0``.

    Parameters
    ----------
    entity : GeometryEntity
        The wall the block face lies on (provides ``normal`` / ``tangent_space``).
    first_height : float
        Wall-normal height of the first off-wall layer (k=0). Under a
        scale-invariant shape metric only the ``first_height/tangential_spacing``
        ratio is meaningful.
    growth : float
        Geometric growth ratio between successive near-wall layers.
    wall_axis : int
        Logical axis crossing the wall (the wall-normal logical direction).
    wall_side : int
        0 = low face, 1 = high face of ``wall_axis``.
    n_layers : int
        Unused; retained for backwards compatibility (the geometric growth now
        runs until it is capped by ``interior_spacing``).
    interior_spacing : float
        Cap on the wall-normal spacing; defaults to ``tangential_spacing`` so
        the far field is isotropic.
    max_height : float, optional
        Clamp on ``s_n``.
    tangential_spacing : float, optional
        Wall-tangential target spacing ``s_t`` (defaults to ``interior_spacing``).
    k_offset : int
        Added to the layer index; lets a neighbouring block continue the
        clustering profile of a wall block across their shared interface
        (set it to the wall block's cell count along ``wall_axis``).
    boundary_shear : dict, optional
        Relax wall-normal orthogonality towards an oblique transverse
        boundary: maps transverse side (0/1) -> ``(b_hat, taper)`` where
        ``b_hat`` is the boundary's off-wall unit direction and ``taper``
        the tangential fade width in cells. Within the taper the target's
        wall-normal column is blended from the wall normal into ``b_hat``
        (length rescaled so the perpendicular layer height stays exact), so
        the metric prefers uniformly sheared parallelograms that follow the
        boundary instead of trading layer heights for orthogonality to it.
        Usually set via ``build_boundary_layer_target(relax_orthogonality=…)``.
    """

    def __init__(
        self,
        entity,
        *,
        first_height,
        growth,
        wall_axis,
        wall_side,
        n_layers=8,
        interior_spacing=None,
        max_height=None,
        tangential_spacing=None,
        k_offset=0,
        boundary_shear=None,
    ):
        if interior_spacing is None and tangential_spacing is None:
            raise ValueError(
                "give at least one of interior_spacing / tangential_spacing"
            )
        self.entity = entity
        self.first_height = float(first_height)
        self.growth = float(growth)
        self.wall_axis = int(wall_axis)
        self.wall_side = int(wall_side)
        self.n_layers = int(n_layers)
        self.tangential_spacing = float(
            tangential_spacing if tangential_spacing is not None else interior_spacing
        )
        self.interior_spacing = float(
            interior_spacing
            if interior_spacing is not None
            else self.tangential_spacing
        )
        self.max_height = None if max_height is None else float(max_height)
        self.k_offset = int(k_offset)
        self.boundary_shear = dict(boundary_shear or {})
        # Layer index where the geometric growth reaches the isotropic
        # interior spacing — the extent of the anisotropic band. Boundary
        # shear fades above it (see __call__).
        if self.growth > 1.0 and self.first_height < self.interior_spacing:
            self._k_iso = max(
                1,
                int(
                    np.ceil(
                        np.log(self.interior_spacing / self.first_height)
                        / np.log(self.growth)
                    )
                ),
            )
        else:
            self._k_iso = max(1, self.n_layers)

    def normal_spacing(self, k: int) -> float:
        """Wall-normal target spacing at layer index ``k`` (capped growth)."""
        s_geo = self.first_height * self.growth**k
        if self.max_height is not None:
            s_geo = min(s_geo, self.max_height)
        return min(s_geo, self.interior_spacing)

    def _layer_index(self, block, cell_base) -> int:
        """Off-wall layer index of a cell (plus ``k_offset``)."""
        n_cells = block.logical_shape[self.wall_axis] - 1
        c = int(cell_base[self.wall_axis])
        k = c if self.wall_side == 0 else (n_cells - 1 - c)
        return k + self.k_offset

    def _wall_anchor(self, block, cell_base):
        """Physical position of the cell corner that sits on the wall face."""
        d = block.nodes.ndim - 1
        idx = list(int(cell_base[a]) for a in range(d))
        if self.wall_side == 1:
            idx[self.wall_axis] += (
                block.logical_shape[self.wall_axis] - 1 - int(cell_base[self.wall_axis])
            )
        # else: low side, cell_base already on/under the wall face
        # Clamp to valid range.
        for a in range(d):
            idx[a] = min(max(idx[a], 0), block.logical_shape[a] - 1)
        return np.asarray(block.nodes[tuple(idx)], dtype=float)

    def __call__(self, bi, block, cell_base, corner_offset) -> np.ndarray:
        """Target matrix W for one (cell, corner) sample. Shape (d, d)."""
        k = self._layer_index(block, cell_base)
        s_n = self.normal_spacing(k)
        s_t = self.tangential_spacing

        anchor = self._wall_anchor(block, cell_base)
        q = np.asarray(self.entity.project(anchor), dtype=float)
        n_hat = np.asarray(self.entity.normal(q), dtype=float)
        t_hat = np.asarray(self.entity.tangent_space(q), dtype=float)[:, 0]

        d = n_hat.shape[0]
        # FIXME(3D): only ONE tangential column is filled below, so in 3D the
        # third column of the np.empty W is uninitialized garbage. Fill every
        # tangential column from the (d, d-1) ``tangent_space`` basis with its
        # own spacing (t_hat_i * s_t_i) and keep the det-sign fixup.
        other = (
            1 - self.wall_axis
            if d == 2
            else [a for a in range(d) if a != self.wall_axis][0]
        )

        # Optional shear towards an oblique transverse boundary: within the
        # taper the wall-normal column leans into the boundary's direction,
        # rescaled so the perpendicular layer height stays exact.
        # TODO(3D): the taper distance below is an index along the single 2D
        # tangential axis; in 3D the oblique boundary is a face and the
        # distance is to that face's edge in the wall face's 2D index space.
        d_hat = n_hat
        if self.boundary_shear:
            n_t = block.logical_shape[other] - 1  # cells along the wall
            j = int(cell_base[other])
            lam, b_lam = 0.0, None
            for side, (b_hat, taper) in self.boundary_shear.items():
                dist = j if side == 0 else n_t - 1 - j
                w = 1.0 - _smoothstep(dist / max(taper, 1))
                if w > lam:
                    lam, b_lam = w, b_hat
            # Fade the shear with height: full strength through the
            # anisotropic band, gone by three times its extent (a gentler
            # turn keeps the band heights from being dragged by the
            # orthogonal far field). Above that the profile is isotropic, so
            # the lean stays with the smoother and the target matches
            # neighbouring identity-target blocks at the shared interface
            # (no cusp in the grid lines there).
            lam *= 1.0 - _smoothstep((k - self._k_iso) / (2 * self._k_iso))
            if lam > 0.0 and b_lam is not None:
                b = np.asarray(b_lam, dtype=float)
                if float(np.dot(b, n_hat)) < 0.0:
                    b = -b
                v = (1.0 - lam) * n_hat + lam * b
                v /= np.linalg.norm(v)
                d_hat = v / max(float(np.dot(v, n_hat)), 0.2)

        W = np.empty((d, d))
        W[:, self.wall_axis] = d_hat * s_n
        W[:, other] = t_hat * s_t
        # Guarantee det W > 0 (flip the tangential column if needed).
        if np.linalg.det(W) < 0:
            W[:, other] = -t_hat * s_t
        return W


class MultiBlockTarget:
    """Per-block target dispatch.

    ``default`` (a ``target_fn``) is used for any block not in ``per_block``;
    ``per_block`` maps block index → its own ``target_fn`` (e.g. a
    :class:`BoundaryLayerTarget` for each O-ring ring block).
    """

    def __init__(self, default, per_block: dict | None = None):
        self.default = default
        self.per_block = dict(per_block or {})

    def __call__(self, bi, block, cell_base, corner_offset) -> np.ndarray:
        """Dispatch to block ``bi``'s target (or ``default``)."""
        fn = self.per_block.get(bi, self.default)
        return fn(bi, block, cell_base, corner_offset)


def _face_tangential_spacing(topology, block_name: str, axis: int, side: int) -> float:
    """Average corner-to-corner spacing along a block face (2D).

    Distance between the face's two corner positions divided by the cell
    count along the face — the block's natural tangential spacing.

    TODO(3D): a 3D face has two tangential axes; return both spacings
    (from the face's four corners) so ``BoundaryLayerTarget`` can scale
    each tangential column of W independently.
    """
    spec = topology.block_specs[block_name]
    names = spec.face_corner_names(axis, side, topology.d)
    p0 = topology.corners[names[0]].position
    p1 = topology.corners[names[1]].position
    return float(np.linalg.norm(p1 - p0)) / spec.resolutions[1 - axis]


def _neighbour_across(topology, block_name: str, axis: int, side: int):
    """The (block_name, axis, side) face glued to the given face, or None."""
    for conn in topology.interface_connections:
        a, b = conn.face_a, conn.face_b
        if (a.block_name, a.axis, a.side) == (block_name, axis, side):
            return b.block_name, b.axis, b.side
        if (b.block_name, b.axis, b.side) == (block_name, axis, side):
            return a.block_name, a.axis, a.side
    return None


def _boundary_direction(
    topology, entity, block_name: str, axis: int, side: int, wall_entity
) -> np.ndarray | None:
    """Off-wall unit direction of a transverse boundary face's entity.

    Evaluated as the entity tangent at the face corner nearest the wall
    (sign is resolved against the local wall normal at target-evaluation
    time).
    """
    spec = topology.block_specs[block_name]
    names = spec.face_corner_names(axis, side, topology.d)
    pts = [np.asarray(topology.corners[n].position, dtype=float) for n in names]
    p = min(
        pts, key=lambda x: float(np.linalg.norm(x - np.asarray(wall_entity.project(x))))
    )
    q = np.asarray(entity.project(p), dtype=float)
    t = np.asarray(entity.tangent_space(q), dtype=float)[:, 0]
    norm = float(np.linalg.norm(t))
    return t / norm if norm > 0 else None


def build_boundary_layer_target(
    topology,
    grid=None,
    default=None,
    interior_spacing: float | None = None,
    blend_neighbours: bool = True,
    relax_orthogonality=(),
):
    """Build a :class:`MultiBlockTarget` from a topology's boundary-layer specs.

    Every block whose face association carries a spec recorded via
    ``builder.set_boundary_layer`` gets a :class:`BoundaryLayerTarget`
    oriented on that entity with the recorded ``first_height``/``growth``.

    Parameters
    ----------
    topology : BlockTopology
    grid : MultiBlockGrid, optional
    default : callable, optional
        Target for blocks without a spec; defaults to
        :func:`IdentityTarget` of the topology's dimension.
    interior_spacing : float, optional
        Cap on the wall-normal growth (isotropic far field).
    blend_neighbours : bool, optional
        A block sitting behind a wall block continues the clustering
        profile across their shared interface (via ``k_offset``) instead
        of jumping straight to ``default`` — removes the cell-size
        discontinuity there when the clustering has not decayed to
        isotropic within the wall block. Default on.
    relax_orthogonality : sequence of entities or Edges, optional
        Domain boundaries that meet the wall obliquely: near each, the
        per-block targets shear the wall-normal column into the boundary's
        own direction (see ``BoundaryLayerTarget.boundary_shear``), so the
        optimiser follows the boundary with uniformly sheared
        parallelograms instead of rotating the near-wall cells orthogonal
        to it and losing the layer heights. Unlisted boundaries keep the
        plain orthogonal target. Defaults to the entities declared via
        ``builder.set_boundary_layer(relax_orthogonality=...)``; passing
        the argument here overrides that declaration.

    Returns
    -------
    target_fn : callable
        ``default`` itself when the topology has no specs (so callers can
        use this unconditionally).

    Notes
    -----
    Unless a spec pins ``tangential_spacing``, each wall block uses its
    own natural face spacing (face corner separation / cell count), so
    blocks of different sizes along the wall keep their node distribution
    instead of fighting over a global value; the wall-normal growth is
    capped at that spacing, making far-from-wall cells isotropic like the
    default target.
    """
    d = topology.d
    if default is None:
        default = IdentityTarget(d)
    specs = getattr(topology, "boundary_layer_specs", {})
    if not specs:
        return default

    block_names = list(topology.block_specs.keys())
    per_block: dict[int, BoundaryLayerTarget] = {}
    for assoc in topology.associations:
        spec = specs.get(id(assoc.entity))
        if spec is None:
            continue
        face = assoc.face
        bi = block_names.index(face.block_name)
        s_t = spec["tangential_spacing"]
        if s_t is None:
            s_t = _face_tangential_spacing(
                topology, face.block_name, face.axis, face.side
            )
        per_block[bi] = BoundaryLayerTarget(
            assoc.entity,
            first_height=spec["first_height"],
            growth=spec["growth"],
            wall_axis=face.axis,
            wall_side=face.side,
            n_layers=spec["n_layers"],
            interior_spacing=interior_spacing,
            max_height=spec["max_height"],
            tangential_spacing=s_t,
        )

    if blend_neighbours:
        for bi, blt in list(per_block.items()):
            bname = block_names[bi]
            nb = _neighbour_across(topology, bname, blt.wall_axis, 1 - blt.wall_side)
            if nb is None:
                continue
            nbi = block_names.index(nb[0])
            if nbi in per_block:
                continue
            wall_cells = topology.block_specs[bname].resolutions[blt.wall_axis]
            # The profile is already isotropic at the interface — the
            # neighbour can keep its default target.
            if blt.normal_spacing(wall_cells) >= blt.interior_spacing:
                continue
            per_block[nbi] = BoundaryLayerTarget(
                blt.entity,
                first_height=blt.first_height,
                growth=blt.growth,
                wall_axis=nb[1],
                wall_side=nb[2],
                n_layers=blt.n_layers,
                interior_spacing=interior_spacing,
                max_height=blt.max_height,
                tangential_spacing=_face_tangential_spacing(
                    topology, nb[0], nb[1], nb[2]
                ),
                k_offset=wall_cells,
            )

    if relax_orthogonality:
        relax_ids = {id(getattr(e, "entity", e)) for e in relax_orthogonality}
    else:
        relax_ids = {
            id(getattr(e, "entity", e))
            for spec in specs.values()
            for e in spec.get("relax_orthogonality", ())
        }
    # TODO(3D): relax_orthogonality is 2D-only. In 3D the oblique boundary is
    # a face: the taper becomes a distance-to-edge in the wall face's 2D index
    # space, and b_hat varies along the wall-boundary edge — evaluate the
    # boundary entity's tangent at each anchor's projection (per sample)
    # instead of once per block in _boundary_direction.
    if relax_ids and d == 2:
        face_entities = {
            (a.face.block_name, a.face.axis, a.face.side): a.entity
            for a in topology.associations
        }
        for bi, blt in per_block.items():
            bname = block_names[bi]
            t_axis = 1 - blt.wall_axis
            for side in (0, 1):
                if _neighbour_across(topology, bname, t_axis, side) is not None:
                    continue  # interior interface, not a domain boundary
                ent = face_entities.get((bname, t_axis, side))
                if ent is None or id(ent) not in relax_ids:
                    continue
                b_hat = _boundary_direction(
                    topology, ent, bname, t_axis, side, blt.entity
                )
                if b_hat is not None:
                    blt.boundary_shear[side] = (b_hat, max(3, blt.n_layers))

    return MultiBlockTarget(default, per_block)


def AnisotropicTarget(spacings: tuple[float, ...]) -> Callable[..., np.ndarray]:
    """Target yielding a constant diagonal W with prescribed cell spacings.

    ``W = diag(s0, ..., s_{d-1})``, ``s_i`` the desired physical cell edge
    length along logical axis i.

    Parameters
    ----------
    spacings : tuple of float
        Length d.

    Returns
    -------
    target_fn : callable
    """
    W = np.diag(list(spacings))

    def target(*args, **kwargs) -> np.ndarray:
        return W.copy()

    return target
