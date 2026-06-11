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


def _smoothstep(t: float) -> float:
    """Hermite smoothstep on ``[0, 1]`` (0 at 0, 1 at 1, zero end-slopes)."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class BoundaryLayerTarget:
    """Oriented, layer-indexed wall-clustering target ``W`` for one ring block.

    Realises boundary-layer clustering purely through the TMOP target: the
    wall-normal target spacing grows geometrically away from the wall
    (``s_n(k) = first_height · growth**k``), is clamped to ``max_height`` and
    blended toward the isotropic interior spacing over the remaining layers, and
    is oriented from the geometry **entity** (stable even while the working mesh
    is folded during untangling).

    Called as ``target(bi, block, cell_base, corner_offset) -> (d, d)`` with
    ``det W = s_t · s_n > 0``.

    Parameters
    ----------
    entity : GeometryEntity
        The wall the block face lies on (provides ``normal`` / ``tangent_space``).
    first_height : float
        Wall-normal height of the first off-wall layer (k=0).
    growth : float
        Geometric growth ratio between successive near-wall layers.
    wall_axis : int
        Logical axis crossing the wall (the wall-normal logical direction).
    wall_side : int
        0 = low face, 1 = high face of ``wall_axis``.
    n_layers : int
        Number of geometrically-clustered near-wall layers before blending to
        the isotropic interior spacing.
    interior_spacing : float
        Target isotropic spacing the outer layers blend toward.
    max_height : float, optional
        Clamp on ``s_n``.
    tangential_spacing : float, optional
        Wall-tangential target spacing ``s_t`` (defaults to ``interior_spacing``).
    """

    def __init__(self, entity, *, first_height, growth, wall_axis, wall_side,
                 n_layers=8, interior_spacing=1.0, max_height=None,
                 tangential_spacing=None):
        self.entity = entity
        self.first_height = float(first_height)
        self.growth = float(growth)
        self.wall_axis = int(wall_axis)
        self.wall_side = int(wall_side)
        self.n_layers = int(n_layers)
        self.interior_spacing = float(interior_spacing)
        self.max_height = None if max_height is None else float(max_height)
        self.tangential_spacing = (
            float(tangential_spacing) if tangential_spacing is not None
            else float(interior_spacing))

    def normal_spacing(self, k: int) -> float:
        """Wall-normal target spacing at layer index ``k`` (clamp + blend)."""
        s_geo = self.first_height * self.growth ** k
        if self.max_height is not None:
            s_geo = min(s_geo, self.max_height)
        if k <= self.n_layers:
            return s_geo
        # Blend geometrically-grown spacing toward the isotropic interior.
        # (k just past n_layers starts the blend; far layers are isotropic.)
        span = max(self.n_layers, 1)
        w = _smoothstep((k - self.n_layers) / span)
        return (1.0 - w) * s_geo + w * self.interior_spacing

    def _layer_index(self, block, cell_base) -> int:
        n_cells = block.logical_shape[self.wall_axis] - 1
        c = int(cell_base[self.wall_axis])
        return c if self.wall_side == 0 else (n_cells - 1 - c)

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


def build_boundary_layer_target(topology, grid=None, default=None,
                                interior_spacing: float = 1.0):
    """Build a :class:`MultiBlockTarget` from a topology's boundary-layer specs.

    For every association whose entity carries a spec recorded via
    ``builder.set_boundary_layer``, the corresponding block gets a
    :class:`BoundaryLayerTarget` oriented on that entity with the recorded
    ``first_height``/``growth``. Blocks without a spec fall back to ``default``
    (an :class:`IdentityTarget` of the topology's dimension if not given).

    Returns the assembled ``target_fn``; if the topology has no specs it returns
    ``default`` (so callers can use it unconditionally).
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
        bi = block_names.index(assoc.face.block_name)
        per_block[bi] = BoundaryLayerTarget(
            assoc.entity,
            first_height=spec["first_height"],
            growth=spec["growth"],
            wall_axis=assoc.face.axis,
            wall_side=assoc.face.side,
            n_layers=spec["n_layers"],
            interior_spacing=interior_spacing,
            max_height=spec["max_height"],
            tangential_spacing=spec["tangential_spacing"],
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
