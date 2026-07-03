"""Programmatic API to declare a rough multiblock topology."""

from __future__ import annotations

from typing import Any

import numpy as np

from egg.geometry.frontend2d import Edge, Node, Vector3

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
    """Fluent API for declaring a multiblock topology.

    Declare corners and blocks (:meth:`add_corner`, :meth:`add_block`,
    :meth:`add_block_array`), geometry attachments (:meth:`associate`,
    :meth:`set_boundary_layer`, :meth:`tag_boundary`), and explicit face
    sharing (:meth:`connect`); :meth:`build` validates everything, infers
    undeclared connections/associations, and returns the
    :class:`~egg.topology.block_topology.BlockTopology`.

    Parameters
    ----------
    d : int
        Spatial dimension (default 2).
    """

    def __init__(self, d: int = 2):
        self._d = d
        self._corners: dict[str, Corner] = {}
        self._block_specs: dict[str, BlockSpec] = {}
        self._associations: list[Association] = []
        self._connections: list[InterfaceConnection] = []
        # id(entity) -> BoundaryLayerTarget kwargs
        self._boundary_layer_specs: dict[int, dict] = {}
        self._bl_entities: dict[int, Any] = {}
        # Corner-object identity: id(Vector3/Node) -> corner name, and the
        # original object per name (Node provenance drives associate inference).
        self._corner_ids: dict[int, str] = {}
        self._corner_objs: dict[str, Any] = {}
        self._boundary_tags: dict[str, list[FaceSpec]] = {}

    def add_corner(
        self, name: str, position: Any, *, fixed: bool = True
    ) -> "TopologyBuilder":
        """Register a named corner point.

        ``position`` may be a coordinate sequence, a
        :class:`~egg.geometry.frontend2d.Vector3`, or a
        :class:`~egg.geometry.frontend2d.Node` placed on an
        :class:`~egg.geometry.frontend2d.Edge`.
        """
        obj = None
        if isinstance(position, Vector3):
            obj = position
            position = (position.x, position.y, position.z)[: self._d]
        pos = np.asarray(position, dtype=float)
        if pos.shape != (self._d,):
            raise ValueError(
                f"Corner '{name}' position has shape {pos.shape}, expected ({self._d},)"
            )
        self._corners[name] = Corner(name=name, position=pos, fixed=fixed)
        if obj is not None:
            self._corner_ids[id(obj)] = name
            self._corner_objs[name] = obj
        return self

    def _corner_ref(self, corner: Any) -> str:
        """Resolve a block-corner reference to a registered corner name.

        Accepts a registered corner name (str) or a :class:`Vector3` /
        :class:`Node` object. Objects are deduplicated by identity — passing
        the same object to two blocks means the same grid corner — and are
        auto-registered on first use (``fixed`` taken from the object).
        """
        if isinstance(corner, str):
            if corner not in self._corners:
                raise ValueError(f"unknown corner '{corner}'")
            return corner
        if isinstance(corner, Vector3):
            name = self._corner_ids.get(id(corner))
            if name is None:
                k = len(self._corner_ids)
                while f"_c{k}" in self._corners:
                    k += 1
                name = f"_c{k}"
                self.add_corner(name, corner, fixed=corner.fixed)
            return name
        raise TypeError(
            f"block corner must be a corner name or Vector3/Node, "
            f"got {type(corner).__name__}"
        )

    def add_block(
        self,
        name: str | None = None,
        corners: tuple | None = None,
        resolutions: tuple[int, ...] | None = None,
        *,
        sw: Any = None,
        se: Any = None,
        nw: Any = None,
        ne: Any = None,
        res: tuple[int, ...] | None = None,
    ) -> "TopologyBuilder":
        """Declare a structured block.

        Two forms: positional ``add_block(name, corners, resolutions)``, or
        compass (2D only) ``add_block(sw=, se=, nw=, ne=, res=, name=...)``.

        Parameters
        ----------
        name : str, optional
            Auto-generated as ``blk<i>`` if omitted.
        corners : tuple of str or Vector3 or Node, optional
            2**d corner references in ``product((0,1), repeat=d)`` order.
            Objects are deduplicated by identity and auto-registered, so
            sharing an object between two blocks makes it the same grid
            corner; strings must name registered corners.
        resolutions, res : tuple of int
            Per-axis cell counts ``(n_cells_0, ..., n_cells_{d-1})``;
            ``res`` is an alias for the compass form.
        sw, se, nw, ne : str or Vector3 or Node, optional
            Compass-form corner references.
        """
        compass = (sw, nw, se, ne)  # product((0,1), repeat=2) order
        if corners is None:
            if self._d != 2:
                raise ValueError("compass corners (sw/se/nw/ne) are 2D-only")
            if any(c is None for c in compass):
                raise ValueError(
                    "add_block needs either `corners` or all of sw/se/nw/ne"
                )
            corners = compass
        elif any(c is not None for c in compass):
            raise ValueError("pass either `corners` or sw/se/nw/ne, not both")
        resolutions = res if resolutions is None else resolutions
        if resolutions is None:
            raise ValueError("add_block needs `resolutions` (or `res`)")
        if name is None:
            name = f"blk{len(self._block_specs)}"
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
        corner_names = tuple(self._corner_ref(c) for c in corners)
        self._block_specs[name] = BlockSpec(
            name=name,
            corner_names=corner_names,
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

        Usually unnecessary: :meth:`build` infers connections from shared
        corners. Orientation is auto-detected by corner-name matching at
        build time.

        Parameters
        ----------
        block_a, block_b : str
            Block names.
        axis_a, axis_b : int
            Logical axis the face lies on (0..d-1).
        side_a, side_b : int
            0 = low side, 1 = high side.
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

        Face nodes are snapped/slid on the entity during initialisation and
        smoothing.

        Parameters
        ----------
        block_name : str
        axis, side : int
            Face selector (axis 0..d-1; side 0 = low, 1 = high).
        entity : GeometryEntity or Edge
            An :class:`~egg.geometry.frontend2d.Edge` is unwrapped to its
            underlying entity, so the pure-Python wrapper never reaches the
            entity-encoding path.
        """
        if isinstance(entity, Edge):
            entity = entity.entity
        if block_name not in self._block_specs:
            raise ValueError(f"associate() references unknown block '{block_name}'")
        self._associations.append(
            Association(face=FaceSpec(block_name, axis, side), entity=entity)
        )
        return self

    def tag_boundary(
        self,
        name: str,
        block_name: str,
        axis: int,
        side: int,
    ) -> "TopologyBuilder":
        """Tag a block face with a named boundary marker (e.g. for export).

        Several faces may carry the same tag; exporters group them into one
        marker (an SU2 ``MARKER_TAG``, say) for boundary-condition
        assignment.

        Parameters
        ----------
        name : str
            Marker name, e.g. ``"inlet"`` or ``"wall"``.
        block_name : str
        axis, side : int
            Face selector (side 0 = low, 1 = high).
        """
        if block_name not in self._block_specs:
            raise ValueError(f"tag_boundary() references unknown block '{block_name}'")
        self._boundary_tags.setdefault(name, []).append(
            FaceSpec(block_name, axis, side)
        )
        return self

    def add_block_array(
        self,
        *,
        south: Edge,
        north: Edge,
        west: Edge,
        east: Edge,
        nib: int,
        njb: int,
        res: tuple[int, int],
        fixed_corners: bool = True,
        corner_prefix: str = "c",
        block_prefix: str = "b",
    ) -> tuple[dict, list[list[str]]]:
        """Add an ``nib x njb`` array of blocks over a four-edge patch.

        The Eilmer ``registerFluidGridArray`` analogue: sub-block corners
        are placed parametrically on the bounding edges (sliding nodes) and
        by bilinear TFI in the interior. Blocks share the corner objects,
        so block-to-block connectivity is inferred at :meth:`build`, and
        the outer block faces are associated with their bounding edges.

        Parameters
        ----------
        south, north : Edge
            Parameterised west -> east; axis 0 of every block runs west ->
            east.
        west, east : Edge
            Parameterised south -> north; axis 1 runs south -> north.
        nib, njb : int
            Block counts along axis 0 / axis 1.
        res : tuple of int
            TOTAL cell count across the array ``(n_axis0, n_axis1)``, split
            as evenly as possible into per-block resolutions.
        fixed_corners : bool, optional
            Pin the four patch-corner nodes (default True).
        corner_prefix, block_prefix : str, optional
            Corners register as ``c{i}_{j}`` (i: 0=west..nib=east, j:
            0=south..njb=north) and blocks as ``b{i}_{j}``, keeping
            ``--plot-topology`` labels readable.

        Returns
        -------
        corner : dict
            Shared corner objects keyed ``(i, j)``.
        block_names : list of list of str
            Block-name grid ``block_names[i][j]``, for tagging boundaries
            or attaching further structure.

        Notes
        -----
        TODO(3D): the hexahedral analogue is a block array over a patch
        bounded by six faces, with edge/face/interior corners placed by the
        corresponding 1D/2D/3D transfinite interpolations.
        """
        from egg.geometry.frontend2d import split_cells, tfi_point

        nx, ny = split_cells(res[0], nib), split_cells(res[1], njb)
        corner: dict[tuple[int, int], Any] = {}
        for i in range(nib + 1):
            u = i / nib
            for j in range(njb + 1):
                v = j / njb
                patch_corner = fixed_corners and i in (0, nib)
                if j == 0:
                    corner[i, j] = south.place_node(u, fixed=patch_corner)
                elif j == njb:
                    corner[i, j] = north.place_node(u, fixed=patch_corner)
                elif i == 0:
                    corner[i, j] = west.place_node(v)
                elif i == nib:
                    corner[i, j] = east.place_node(v)
                else:
                    corner[i, j] = tfi_point(u, v, south, north, west, east)
        for (i, j), obj in sorted(corner.items()):
            self.add_corner(f"{corner_prefix}{i}_{j}", obj,
                            fixed=getattr(obj, "fixed", False))

        names = [[f"{block_prefix}{i}_{j}" for j in range(njb)]
                 for i in range(nib)]
        for i in range(nib):
            for j in range(njb):
                self.add_block(
                    names[i][j],
                    sw=corner[i, j],
                    se=corner[i + 1, j],
                    nw=corner[i, j + 1],
                    ne=corner[i + 1, j + 1],
                    res=(nx[i], ny[j]),
                )
        for i in range(nib):
            self.associate(names[i][0], 1, 0, south)
            self.associate(names[i][njb - 1], 1, 1, north)
        for j in range(njb):
            self.associate(names[0][j], 0, 0, west)
            self.associate(names[nib - 1][j], 0, 1, east)
        return corner, names

    def set_boundary_layer(
        self,
        entity: Any,
        *,
        first_height: float,
        growth: float,
        n_layers: int = 8,
        max_height: float | None = None,
        tangential_spacing: float | None = None,
        n_fixed: int = 1,
        relax_orthogonality: tuple = (),
    ) -> "TopologyBuilder":
        """Request wall-normal clustering on every block face lying on ``entity``.

        Consumed later (e.g. by
        :func:`~egg.smoothing.targets.build_boundary_layer_target`) to
        build a :class:`~egg.smoothing.targets.BoundaryLayerTarget` per
        ring block, oriented from ``entity``.

        Parameters
        ----------
        entity : GeometryEntity or Edge
            The wall; an Edge is unwrapped to its underlying entity
            (matching :meth:`associate`).
        first_height : float
            Wall-normal height of the first off-wall layer.
        growth : float
            Geometric growth ratio: ``s_n(k) = first_height * growth**k``.
        n_layers : int, optional
        max_height : float, optional
            Clamp on the wall-normal spacing.
        tangential_spacing : float, optional
            Wall-tangential target spacing; defaults to each block's
            natural face spacing.
        n_fixed : int, optional
            Number of near-wall layers to pin at their exact geometric
            heights, consumed by
            :func:`egg.smoothing.respace_first_layers`: after a TMOP pass,
            rows ``1..n_fixed`` slide along their smoothed columns to the
            exact cumulative heights and leave the optimisation, so a
            follow-up TMOP pass smooths only the grid above them.
        relax_orthogonality : tuple of entities or Edges, optional
            Domain-boundary entities that meet this wall obliquely: near
            each, the clustering target shears the wall-normal direction
            into the boundary's own direction, so the smoother follows it
            with sheared parallelograms instead of rotating the near-wall
            cells orthogonal to it and trading away the layer heights (see
            :func:`~egg.smoothing.targets.build_boundary_layer_target`).
        """
        if isinstance(entity, Edge):
            entity = entity.entity
        self._boundary_layer_specs[id(entity)] = dict(
            first_height=first_height, growth=growth, n_layers=n_layers,
            max_height=max_height, tangential_spacing=tangential_spacing,
            n_fixed=int(n_fixed),
            relax_orthogonality=tuple(
                e.entity if isinstance(e, Edge) else e
                for e in relax_orthogonality
            ),
        )
        self._bl_entities[id(entity)] = entity
        return self

    def _infer_connections(self) -> list[InterfaceConnection]:
        """Connections implied by two block faces sharing all corner names.

        Two corners are shared only when the same corner (name or object) was
        used for both blocks, so coincident-but-distinct points (e.g. the two
        sides of a slit) are never joined. Explicitly declared connections are
        kept and not duplicated.
        """
        declared = {
            frozenset((
                (c.face_a.block_name, c.face_a.axis, c.face_a.side),
                (c.face_b.block_name, c.face_b.axis, c.face_b.side),
            ))
            for c in self._connections
        }
        groups: dict[frozenset, list[FaceSpec]] = {}
        for spec in self._block_specs.values():
            for axis in range(self._d):
                for side in (0, 1):
                    names = spec.face_corner_names(axis, side, self._d)
                    key = frozenset(names)
                    if len(key) != len(names):  # degenerate face
                        continue
                    groups.setdefault(key, []).append(
                        FaceSpec(spec.name, axis, side)
                    )
        inferred = []
        for key, faces in groups.items():
            if len(faces) > 2:
                blocks = sorted({f.block_name for f in faces})
                raise ValueError(
                    f"corners {sorted(key)} bound faces of more than two "
                    f"blocks ({blocks}); topology is ambiguous"
                )
            if len(faces) != 2:
                continue
            fa, fb = faces
            pair = frozenset((
                (fa.block_name, fa.axis, fa.side),
                (fb.block_name, fb.axis, fb.side),
            ))
            if pair not in declared:
                inferred.append(InterfaceConnection(face_a=fa, face_b=fb))
        return inferred

    def _infer_associations(
        self, connections: list[InterfaceConnection]
    ) -> list[Association]:
        """Associations implied by Node provenance on boundary faces.

        A face whose corners are all Nodes placed on the same Edge lies on
        that edge's entity. Shared (connected) faces and explicitly declared
        associations are skipped.
        """
        connected = set()
        for c in connections:
            for f in (c.face_a, c.face_b):
                connected.add((f.block_name, f.axis, f.side))
        declared = {
            (a.face.block_name, a.face.axis, a.face.side, id(a.entity))
            for a in self._associations
        }
        inferred = []
        for spec in self._block_specs.values():
            for axis in range(self._d):
                for side in (0, 1):
                    if (spec.name, axis, side) in connected:
                        continue
                    objs = [
                        self._corner_objs.get(n)
                        for n in spec.face_corner_names(axis, side, self._d)
                    ]
                    if not objs or not all(isinstance(o, Node) for o in objs):
                        continue
                    edge = objs[0].edge
                    if not all(o.edge is edge for o in objs[1:]):
                        continue
                    key = (spec.name, axis, side, id(edge.entity))
                    if key in declared:
                        continue
                    inferred.append(Association(
                        face=FaceSpec(spec.name, axis, side),
                        entity=edge.entity,
                    ))
        return inferred

    def build(self) -> BlockTopology:
        """Construct the validated :class:`BlockTopology` (DOF map, singularities).

        Undeclared connections and associations are inferred: faces of two
        blocks sharing all corners are connected, and boundary faces whose
        corners are Nodes on a common Edge are associated with that edge's
        entity.
        """
        connections = self._connections + self._infer_connections()
        associations = self._associations + self._infer_associations(connections)
        return BlockTopology(
            d=self._d,
            corners=self._corners,
            block_specs=self._block_specs,
            connections=connections,
            associations=associations,
            boundary_layer_specs=self._boundary_layer_specs,
            boundary_tags=self._boundary_tags,
        )
