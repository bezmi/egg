"""Programmatic API to declare a rough topology."""

from __future__ import annotations

from typing import Any

import numpy as np

from .block_topology import (
    Association,
    BlockSpec,
    BlockTopology,
    Corner,
    FaceSpec,
    InterfaceConnection,
)

__all__ = ["TopologyBuilder"]


class TopologyBuilder:
    """Fluent API for declaring a multiblock topology."""

    def __init__(self, d: int = 2):
        self._d = d
        self._corners: dict[str, Corner] = {}
        self._block_specs: dict[str, BlockSpec] = {}
        self._associations: list[Association] = []
        self._connections: list[InterfaceConnection] = []
        # id(entity) -> BoundaryLayerTarget kwargs
        self._boundary_layer_specs: dict[int, dict] = {}
        self._bl_entities: dict[int, Any] = {}

    def add_corner(
        self, name: str, position: Any, *, fixed: bool = True
    ) -> "TopologyBuilder":
        """Register a named corner point."""
        pos = np.asarray(position, dtype=float)
        if pos.shape != (self._d,):
            raise ValueError(
                f"Corner '{name}' position has shape {pos.shape}, expected ({self._d},)"
            )
        self._corners[name] = Corner(name=name, position=pos, fixed=fixed)
        return self

    def add_block(
        self,
        name: str,
        corners: tuple[str, ...],
        resolutions: tuple[int, ...],
    ) -> "TopologyBuilder":
        """Declare a structured block.

        Parameters
        ----------
        name : str
            Block name.
        corners : tuple of str
            2**d corner names in product((0,1), repeat=d) order.
        resolutions : tuple of int
            Per-axis cell counts: (n_cells_0, ..., n_cells_{d-1}).
        """
        if len(corners) != 2 ** self._d:
            raise ValueError(
                f"Block '{name}' needs {2 ** self._d} corners "
                f"(2**d for d={self._d}), got {len(corners)}"
            )
        if len(resolutions) != self._d:
            raise ValueError(
                f"Block '{name}' needs {self._d} resolution values, "
                f"got {len(resolutions)}"
            )
        for cname in corners:
            if cname not in self._corners:
                raise ValueError(
                    f"Block '{name}' references unknown corner '{cname}'"
                )
        self._block_specs[name] = BlockSpec(
            name=name,
            corner_names=corners,
            resolutions=resolutions,
        )
        return self

    def connect(
        self,
        block_a: str,
        axis_a: int,
        side_a: int,
        block_b: str,
        axis_b: int,
        side_b: int,
    ) -> "TopologyBuilder":
        """Declare that two block faces are shared.

        Parameters
        ----------
        block_a, block_b : str
            Block names.
        axis_a, axis_b : int
            Which logical axis the face lies on (0..d-1).
        side_a, side_b : int
            0 = low side, 1 = high side.

        Orientation is auto-detected from corner name matching at build() time.
        """
        for name in (block_a, block_b):
            if name not in self._block_specs:
                raise ValueError(f"connect() references unknown block '{name}'")

        self._connections.append(
            InterfaceConnection(
                face_a=FaceSpec(block_a, axis_a, side_a),
                face_b=FaceSpec(block_b, axis_b, side_b),
            )
        )
        return self

    def associate(
        self,
        block_name: str,
        axis: int,
        side: int,
        entity: Any,
    ) -> "TopologyBuilder":
        """Tag a block face as lying on a geometry entity.

        Parameters
        ----------
        block_name : str
        axis : int
        side : int
        entity : GeometryEntity
        """
        if block_name not in self._block_specs:
            raise ValueError(f"associate() references unknown block '{block_name}'")
        self._associations.append(
            Association(face=FaceSpec(block_name, axis, side), entity=entity)
        )
        return self

    def set_boundary_layer(
        self,
        entity: Any,
        *,
        first_height: float,
        growth: float,
        n_layers: int = 8,
        max_height: float | None = None,
        tangential_spacing: float | None = None,
    ) -> "TopologyBuilder":
        """Request wall-normal clustering on every block face lying on ``entity``.

        Consumed later (e.g. by ``targets.build_boundary_layer_target``) to build
        a :class:`~egg.smoothing.targets.BoundaryLayerTarget` per ring block,
        oriented from ``entity`` with the geometric near-wall spacing
        ``s_n(k) = first_height · growth**k``.
        """
        self._boundary_layer_specs[id(entity)] = dict(
            first_height=first_height, growth=growth, n_layers=n_layers,
            max_height=max_height, tangential_spacing=tangential_spacing,
        )
        self._bl_entities[id(entity)] = entity
        return self

    def build(self) -> BlockTopology:
        """Construct the validated BlockTopology with DOF map and singularities."""
        return BlockTopology(
            d=self._d,
            corners=self._corners,
            block_specs=self._block_specs,
            connections=self._connections,
            associations=self._associations,
            boundary_layer_specs=self._boundary_layer_specs,
        )
