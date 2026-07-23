# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Programmatic API to declare a rough multiblock topology."""

from __future__ import annotations

from typing import Any

import numpy as np

from egg.geometry.frontend2d import Edge, Node

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
        # Boundaries declared via relax_orthogonality() — entities, Edges, or
        # names; resolved at build() against the associated entities.
        self._relax_orthogonality: list = []
        # Fan frames declared via fan_frame() — corner-name dicts, resolved
        # and validated by BlockTopology at build().
        self._fan_frames: list[dict] = []
        # Parallel chains declared via parallel_to() — corner-name dicts with
        # the boundary as given (entity, Edge, or name; names resolve at
        # build()), resolved and validated by BlockTopology.
        self._parallel_chains: list[dict] = []

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
        if hasattr(position, "x") and hasattr(position, "y"):
            obj = position
            coords = (position.x, position.y, getattr(position, "z", 0.0))
            position = coords[: self._d]
        pos = np.asarray(position, dtype=float)
        if pos.shape != (self._d,):
            raise ValueError(
                f"Corner '{name}' position has shape {pos.shape}, expected ({self._d},)"
            )
        self._corners[name] = Corner(name=name, position=pos, fixed=fixed, source=obj)
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
        if hasattr(corner, "x") and hasattr(corner, "y"):
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
        if len(corners) != 2**self._d:
            raise ValueError(
                f"Block '{name}' needs {2**self._d} corners "
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
        self._check_face(axis, side, "associate")
        self._associations.append(
            Association(face=FaceSpec(block_name, axis, side), entity=entity)
        )
        return self

    def _check_face(self, axis: int, side: int, who: str) -> None:
        """Reject an out-of-range face selector before it reaches the grid build."""
        if not isinstance(axis, int) or not 0 <= axis < self._d:
            raise ValueError(
                f"{who}() axis must be in 0..{self._d - 1}, got {axis!r}"
            )
        if side not in (0, 1):
            raise ValueError(f"{who}() side must be 0 (low) or 1 (high), got {side!r}")

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
        self._check_face(axis, side, "tag_boundary")
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
        skip: set[tuple[int, int]] | None = None,
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
        skip : set of (i, j), optional
            Array cells to leave empty — their corners are still created
            and returned, so hand-built replacement blocks (e.g. a
            singularity insertion) can share them; the caller owns the
            skipped cells' boundary associations and tags.

        Returns
        -------
        corner : dict
            Shared corner objects keyed ``(i, j)``.
        block_names : list of list of str
            Block-name grid ``block_names[i][j]``, for tagging boundaries
            or attaching further structure. Skipped cells keep their name
            in the grid but no block is declared for them.

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
            self.add_corner(
                f"{corner_prefix}{i}_{j}", obj, fixed=getattr(obj, "fixed", False)
            )

        skip = skip or set()
        names = [[f"{block_prefix}{i}_{j}" for j in range(njb)] for i in range(nib)]
        for i in range(nib):
            for j in range(njb):
                if (i, j) in skip:
                    continue
                self.add_block(
                    names[i][j],
                    sw=corner[i, j],
                    se=corner[i + 1, j],
                    nw=corner[i, j + 1],
                    ne=corner[i + 1, j + 1],
                    res=(nx[i], ny[j]),
                )
        for i in range(nib):
            if (i, 0) not in skip:
                self.associate(names[i][0], 1, 0, south)
            if (i, njb - 1) not in skip:
                self.associate(names[i][njb - 1], 1, 1, north)
        for j in range(njb):
            if (0, j) not in skip:
                self.associate(names[0][j], 0, 0, west)
            if (nib - 1, j) not in skip:
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
        blocks: tuple | None = None,
    ) -> "TopologyBuilder":
        """Request wall-normal clustering on every block face lying on ``entity``.

        Consumed later (e.g. by
        :func:`~egg.smoothing.targets.build_topology_target`) to
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
            Deprecated — declare relaxed boundaries on the topology itself
            via :meth:`relax_orthogonality`; entries given here are
            forwarded there (with a :class:`DeprecationWarning`).
        blocks : tuple of str, optional
            Restrict the spec to the faces of these blocks. By default the
            clustering applies to every face associated with ``entity`` —
            on an internal interface (associated from both flanking
            strips) that means both sides; name the blocks of one side to
            cluster only there.
        """
        if isinstance(entity, Edge):
            entity = entity.entity
        if not (isinstance(first_height, (int, float)) and first_height > 0):
            raise ValueError(
                f"set_boundary_layer() first_height must be > 0, got {first_height!r}"
            )
        if not (isinstance(growth, (int, float)) and growth >= 1):
            raise ValueError(
                f"set_boundary_layer() growth must be >= 1, got {growth!r}"
            )
        if not (isinstance(n_layers, int) and n_layers > 0):
            raise ValueError(
                f"set_boundary_layer() n_layers must be an int > 0, got {n_layers!r}"
            )
        if not (isinstance(n_fixed, int) and 0 <= n_fixed <= n_layers):
            raise ValueError(
                f"set_boundary_layer() n_fixed must be an int in 0..n_layers "
                f"({n_layers}), got {n_fixed!r}"
            )
        if relax_orthogonality:
            import warnings

            warnings.warn(
                "set_boundary_layer(relax_orthogonality=...) is deprecated; "
                "use TopologyBuilder.relax_orthogonality(...)",
                DeprecationWarning,
                stacklevel=2,
            )
            self._relax_orthogonality.extend(relax_orthogonality)
        self._boundary_layer_specs[id(entity)] = dict(
            first_height=first_height,
            growth=growth,
            n_layers=n_layers,
            max_height=max_height,
            tangential_spacing=tangential_spacing,
            n_fixed=int(n_fixed),
            relax_orthogonality=(),
            blocks=tuple(blocks) if blocks is not None else None,
        )
        self._bl_entities[id(entity)] = entity
        return self

    def relax_orthogonality(self, *boundaries: Any) -> "TopologyBuilder":
        """Relax the smoother's orthogonality requirements at these domain
        boundaries; return ``self`` (fluent).

        Near a listed boundary the mesh may follow the boundary's own
        direction instead of being held orthogonal to whatever structure
        (e.g. a boundary-layer clustering target) would otherwise rotate
        cells against it — a wall-normal column shears into the boundary
        rather than trading away its layer heights (see
        :func:`~egg.smoothing.targets.build_topology_target`). The
        declaration is independent of any clustering spec; consumers that
        need it pick it up from the built topology's
        ``relax_orthogonality`` tuple.

        ``boundaries`` may be geometry entities, :class:`Edge`\\ s, or the
        *names* of named associated geometry (resolved at :meth:`build`;
        an unknown name raises there).
        """
        self._relax_orthogonality.extend(boundaries)
        return self

    def fan_frame(
        self, corner: Any, *, through: tuple, normal: Any
    ) -> "TopologyBuilder":
        """Frame a singular fan corner; return ``self`` (fluent).

        The two ``through`` legs are held smoothly collinear across the fan
        — one continuous rail passing through the singular node — and the
        ``normal`` leg orthogonal to them, so the fan's angular defect lands
        in the remaining legs. ``corner`` and each leg's far corner may be
        registered names or the Vector3/Node objects the corners were
        declared with. Validation happens at :meth:`build`: the corner must
        be a singular node and every leg one of its block edges.
        """
        thr = tuple(through)
        if len(thr) != 2:
            raise ValueError("fan_frame: 'through' must name exactly two corners")
        self._fan_frames.append(
            dict(
                corner=self._corner_ref(corner),
                through=(self._corner_ref(thr[0]), self._corner_ref(thr[1])),
                normal=self._corner_ref(normal),
            )
        )
        return self

    def parallel_to(
        self,
        boundary: Any,
        *,
        chain: tuple,
        weight: float = 1.0,
        taper: float | None = None,
    ) -> "TopologyBuilder":
        """Hold a line of block edges loosely parallel to a boundary; return
        ``self`` (fluent).

        ``chain`` is an ordered corner list whose consecutive pairs must be
        block edges (corners may be registered names or the Vector3/Node
        objects they were declared with); ``boundary`` a geometry entity, an
        :class:`Edge`, or the *name* of a named associated entity (resolved
        at :meth:`build`). The smoother adds a soft directional energy on
        each chain segment against the boundary's frozen normal — orientation
        only, spacing stays free. ``weight`` scales the term; ``taper``, when
        given, is the arclength scale of a Gaussian falloff from the chain's
        singular vertices (default: uniform over the whole chain).
        Validation happens at :meth:`build`: every consecutive pair must be a
        block edge and the boundary must be associated to at least one face.
        """
        ch = tuple(self._corner_ref(c) for c in chain)
        if len(ch) < 2:
            raise ValueError("parallel_to: 'chain' needs at least two corners")
        w = float(weight)
        if not w > 0.0:
            raise ValueError("parallel_to: weight must be positive")
        t = None if taper is None else float(taper)
        if t is not None and not t > 0.0:
            raise ValueError("parallel_to: taper must be positive")
        self._parallel_chains.append(dict(to=boundary, chain=ch, weight=w, taper=t))
        return self

    def _infer_connections(self) -> list[InterfaceConnection]:
        """Connections implied by two block faces sharing all corner names.

        Two corners are shared only when the same corner (name or object) was
        used for both blocks, so coincident-but-distinct points (e.g. the two
        sides of a slit) are never joined. Explicitly declared connections are
        kept and not duplicated.
        """
        declared = {
            frozenset(
                (
                    (c.face_a.block_name, c.face_a.axis, c.face_a.side),
                    (c.face_b.block_name, c.face_b.axis, c.face_b.side),
                )
            )
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
                    groups.setdefault(key, []).append(FaceSpec(spec.name, axis, side))
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
            pair = frozenset(
                (
                    (fa.block_name, fa.axis, fa.side),
                    (fb.block_name, fb.axis, fb.side),
                )
            )
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
                    inferred.append(
                        Association(
                            face=FaceSpec(spec.name, axis, side),
                            entity=edge.entity,
                        )
                    )
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
        boundary_tags = self._auto_boundary_tags(associations, connections)
        boundary_layer_specs, carried_relax = self._collect_boundary_layers(
            associations
        )
        relax: list = []
        for e in self._resolve_relax(associations) + carried_relax:
            if not any(e is r for r in relax):
                relax.append(e)
        return BlockTopology(
            d=self._d,
            corners=self._corners,
            block_specs=self._block_specs,
            connections=connections,
            associations=associations,
            boundary_layer_specs=boundary_layer_specs,
            boundary_tags=boundary_tags,
            relax_orthogonality=tuple(relax),
            fan_frames=tuple(self._fan_frames),
            parallel_chains=tuple(
                dict(p, to=self._resolve_boundary(p["to"], associations))
                for p in self._parallel_chains
            ),
        )

    def _resolve_relax(self, associations: list[Association]) -> list:
        """The relax_orthogonality() declarations as resolved entities: Edges
        unwrap, names look up the named associated entities (unknown -> raise)."""
        by_name: dict[str, Any] = {}
        for assoc in associations:
            nm = getattr(assoc.entity, "name", None)
            if nm is not None:
                by_name.setdefault(nm, assoc.entity)
        out = []
        for o in self._relax_orthogonality:
            if isinstance(o, str):
                if o not in by_name:
                    raise ValueError(
                        f"relax_orthogonality: no associated entity is named {o!r}"
                    )
                out.append(by_name[o])
            else:
                out.append(o.entity if isinstance(o, Edge) else o)
        return out

    @staticmethod
    def _resolve_boundary(o: Any, associations: list[Association]) -> Any:
        """A parallel_to boundary reference as a resolved entity: Edges
        unwrap, names look up the named associated entities (unknown -> raise)."""
        if isinstance(o, str):
            for assoc in associations:
                if getattr(assoc.entity, "name", None) == o:
                    return assoc.entity
            raise ValueError(f"parallel_to: no associated entity is named {o!r}")
        return o.entity if isinstance(o, Edge) else o

    def _collect_boundary_layers(
        self, associations: list[Association]
    ) -> tuple[dict[int, dict], list]:
        """Explicit :meth:`set_boundary_layer` specs plus any carried on an
        associated entity via :meth:`~egg.geometry.base.GeometryEntity.clustered`,
        and the (deprecated) ``relax_orthogonality`` entries those carried specs
        declare — resolved against the named associated entities, returned
        separately so :meth:`build` folds them into the topology-level set.

        Keyed by ``id(entity)`` like the explicit specs (the smoother matches
        them against the associations the same way). An explicit
        ``set_boundary_layer`` for an entity wins over its carried spec, so the
        builder method stays the override.
        """
        specs = dict(self._boundary_layer_specs)
        carried_relax: list = []
        by_name: dict[str, Any] = {}
        for assoc in associations:
            nm = getattr(assoc.entity, "name", None)
            if nm is not None:
                by_name.setdefault(nm, assoc.entity)

        def _resolve(o):
            if isinstance(o, str):
                return by_name.get(o)
            return o.entity if isinstance(o, Edge) else o

        for assoc in associations:
            ent = assoc.entity
            bl = getattr(ent, "boundary_layer", None)
            if bl is None or id(ent) in specs:  # explicit set_boundary_layer wins
                continue
            spec = dict(bl)
            carried_relax += [
                r
                for r in map(_resolve, bl.get("relax_orthogonality", ()))
                if r is not None
            ]
            spec["relax_orthogonality"] = ()
            specs[id(ent)] = spec
            self._bl_entities.setdefault(id(ent), ent)
        return specs, carried_relax

    def _auto_boundary_tags(
        self, associations: list[Association], connections: list
    ) -> dict[str, list[FaceSpec]]:
        """Explicit :meth:`tag_boundary` tags plus an auto-derived marker for
        every associated face whose entity carries a
        :attr:`~egg.geometry.base.GeometryEntity.tag` (which defaults to its
        :attr:`~egg.geometry.base.GeometryEntity.name`).

        A face already carrying an explicit tag keeps it — a hand-written
        ``tag_boundary`` overrides the entity's tag on that face. An entity
        whose ``tag`` is ``None`` emits no marker (associated/drawn only).
        A face that is a shared interface (connected to another block) never
        receives an auto-derived marker: markers describe domain-boundary
        faces, while an associated internal interface (e.g. a shock-fitted
        band sliding on its spline) is a constraint only. An explicit
        ``tag_boundary`` on such a face still fails validation.
        """
        tags = {name: list(faces) for name, faces in self._boundary_tags.items()}
        tagged = {(f.block_name, f.axis, f.side) for fs in tags.values() for f in fs}
        shared = {
            (f.block_name, f.axis, f.side)
            for conn in connections
            for f in (conn.face_a, conn.face_b)
        }
        for assoc in associations:
            tag = getattr(assoc.entity, "tag", None)
            face = assoc.face
            key = (face.block_name, face.axis, face.side)
            if tag is not None and key not in tagged and key not in shared:
                tags.setdefault(tag, []).append(face)
                tagged.add(key)
        return tags
