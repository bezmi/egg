"""W constructors (identity, anisotropic, reference-mesh).

All targets return a d×d matrix W with det(W) > 0, encoding the desired
local cell shape/orientation at each sample point.
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
    """Return a target function that always yields W = I (isotropic).

    Parameters
    ----------
    d : int
        Spatial dimension.

    Returns
    -------
    target_fn : callable
        Signature: target_fn(cell_base, corner_offset) -> ndarray of shape (d, d)
    """
    W = np.eye(d)

    def target(*args, **kwargs) -> np.ndarray:
        return W.copy()

    return target


class BoundaryLayerTarget:
    """Oriented, layer-indexed wall-clustering target ``W`` for one ring block.

    Realises boundary-layer clustering purely through the TMOP target: the
    wall-normal target spacing grows geometrically away from the wall
    (``s_n(k) = first_height · growth**k``) until it reaches the wall-tangential
    spacing ``s_t``, where it is capped — so far-from-wall cells ask for an
    isotropic ``W ∝ I`` that matches the default :func:`IdentityTarget` of
    neighbouring blocks (the shape metric is scale-invariant, so only the
    ``s_n/s_t`` ratio acts). Orientation comes from the geometry **entity**
    (stable even while the working mesh is folded during untangling).

    Called as ``target(bi, block, cell_base, corner_offset) -> (d, d)`` with
    ``det W = s_t · s_n > 0``.

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
    """

    def __init__(self, entity, *, first_height, growth, wall_axis, wall_side,
                 n_layers=8, interior_spacing=None, max_height=None,
                 tangential_spacing=None, k_offset=0):
        if interior_spacing is None and tangential_spacing is None:
            raise ValueError(
                "give at least one of interior_spacing / tangential_spacing")
        self.entity = entity
        self.first_height = float(first_height)
        self.growth = float(growth)
        self.wall_axis = int(wall_axis)
        self.wall_side = int(wall_side)
        self.n_layers = int(n_layers)
        self.tangential_spacing = float(
            tangential_spacing if tangential_spacing is not None
            else interior_spacing)
        self.interior_spacing = float(
            interior_spacing if interior_spacing is not None
            else self.tangential_spacing)
        self.max_height = None if max_height is None else float(max_height)
        self.k_offset = int(k_offset)

    def normal_spacing(self, k: int) -> float:
        """Wall-normal target spacing at layer index ``k`` (capped growth)."""
        s_geo = self.first_height * self.growth ** k
        if self.max_height is not None:
            s_geo = min(s_geo, self.max_height)
        return min(s_geo, self.interior_spacing)

    def _layer_index(self, block, cell_base) -> int:
        n_cells = block.logical_shape[self.wall_axis] - 1
        c = int(cell_base[self.wall_axis])
        k = c if self.wall_side == 0 else (n_cells - 1 - c)
        return k + self.k_offset

    def _wall_anchor(self, block, cell_base):
        """Physical position of the cell corner that sits on the wall face."""
        d = block.nodes.ndim - 1
        idx = list(int(cell_base[a]) for a in range(d))
        if self.wall_side == 1:
            idx[self.wall_axis] += block.logical_shape[self.wall_axis] - 1 \
                - int(cell_base[self.wall_axis])
        # else: low side, cell_base already on/under the wall face
        # Clamp to valid range.
        for a in range(d):
            idx[a] = min(max(idx[a], 0), block.logical_shape[a] - 1)
        return np.asarray(block.nodes[tuple(idx)], dtype=float)

    def __call__(self, bi, block, cell_base, corner_offset) -> np.ndarray:
        k = self._layer_index(block, cell_base)
        s_n = self.normal_spacing(k)
        s_t = self.tangential_spacing

        anchor = self._wall_anchor(block, cell_base)
        q = np.asarray(self.entity.project(anchor), dtype=float)
        n_hat = np.asarray(self.entity.normal(q), dtype=float)
        t_hat = np.asarray(self.entity.tangent_space(q), dtype=float)[:, 0]

        d = n_hat.shape[0]
        other = 1 - self.wall_axis if d == 2 else \
            [a for a in range(d) if a != self.wall_axis][0]
        W = np.empty((d, d))
        W[:, self.wall_axis] = n_hat * s_n
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
        fn = self.per_block.get(bi, self.default)
        return fn(bi, block, cell_base, corner_offset)


def _face_tangential_spacing(topology, block_name: str, axis: int,
                             side: int) -> float:
    """Average corner-to-corner spacing along a block face (2D).

    Distance between the face's two corner positions divided by the cell
    count along the face — the block's natural tangential spacing.
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


def build_boundary_layer_target(topology, grid=None, default=None,
                                interior_spacing: float | None = None,
                                blend_neighbours: bool = True):
    """Build a :class:`MultiBlockTarget` from a topology's boundary-layer specs.

    For every association whose entity carries a spec recorded via
    ``builder.set_boundary_layer``, the corresponding block gets a
    :class:`BoundaryLayerTarget` oriented on that entity with the recorded
    ``first_height``/``growth``. Blocks without a spec fall back to ``default``
    (an :class:`IdentityTarget` of the topology's dimension if not given).

    Unless the spec pins ``tangential_spacing``, each wall block uses its own
    natural face spacing (face corner separation / cell count), so blocks of
    different sizes along the wall keep their node distribution instead of
    fighting over a global value. The wall-normal growth is capped at that
    spacing, making far-from-wall cells isotropic like the default target.

    With ``blend_neighbours`` (default on), a block sitting behind a wall
    block continues the clustering profile across their shared interface
    (via ``k_offset``) instead of jumping straight to the default target —
    this removes the cell-size discontinuity at that interface when the
    clustering has not decayed to isotropic within the wall block.

    Returns the assembled ``target_fn``; if the topology has no specs it
    returns ``default`` (so callers can use it unconditionally).
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
                topology, face.block_name, face.axis, face.side)
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
            nb = _neighbour_across(
                topology, bname, blt.wall_axis, 1 - blt.wall_side)
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
                    topology, nb[0], nb[1], nb[2]),
                k_offset=wall_cells,
            )
    return MultiBlockTarget(default, per_block)


def AnisotropicTarget(spacings: tuple[float, ...]) -> Callable[..., np.ndarray]:
    """Return a target that yields a diagonal W with prescribed cell spacing.

    W = diag(s0, s1, ..., s_{d-1})  where s_i is the desired spacing
    (cell edge length in physical space) along logical axis i.

    Parameters
    ----------
    spacings : tuple of float
        Desired cell edge lengths per logical axis. Length = d.

    Returns
    -------
    target_fn : callable
        Signature: target_fn(cell_base, corner_offset) -> ndarray of shape (d, d)
    """
    d = len(spacings)
    W = np.diag(list(spacings))

    def target(*args, **kwargs) -> np.ndarray:
        return W.copy()

    return target
