# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Multiblock topology data model: corners, blocks, connectivity, associations.

Usually constructed via :class:`egg.topology.builder.TopologyBuilder` rather
than directly.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product
from collections.abc import Sequence
from typing import Any

import numpy as np

from egg.core.types import Block, MultiBlockGrid
from egg.errors import EggValidationError

__all__ = [
    "Corner",
    "BlockSpec",
    "FaceSpec",
    "InterfaceConnection",
    "Association",
    "Singularity",
    "FanFrame",
    "ParallelChain",
    "ParallelWalkWarning",
    "BlockTopology",
]


class ParallelWalkWarning(UserWarning):
    """Some parallel-chain nodes have no logical column to their boundary;
    the sample builder will fall back to frozen nearest-foot projection."""


@dataclass
class Corner:
    """A named point in R^d.

    ``source`` keeps the object the corner was declared with (a
    :class:`~egg.geometry.frontend2d.Vector3` or
    :class:`~egg.geometry.frontend2d.Node`), when there was one: a Node's
    edge/parameter provenance lets :meth:`BlockTopology.initialize_grid`
    space wall nodes along the curve instead of along the chord.
    """

    name: str
    position: np.ndarray
    fixed: bool = True
    source: Any = None

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)


@dataclass
class FaceSpec:
    """Identifies a boundary face of a block by (axis, side)."""

    block_name: str
    axis: int
    side: int  # 0 = low, 1 = high


@dataclass
class BlockSpec:
    """Specification for a structured block: corners and per-axis cell counts."""

    name: str
    corner_names: tuple[str, ...]  # 2**d names in product((0,1), repeat=d) order
    resolutions: tuple[int, ...]  # cells per axis: (n0, ..., n_{d-1})

    @property
    def logical_shape(self) -> tuple[int, ...]:
        """Node count per axis: (n0+1, n1+1, ...)."""
        return tuple(r + 1 for r in self.resolutions)

    @property
    def node_count(self) -> int:
        """Total node count of the block."""
        return int(np.prod(self.logical_shape))

    def _corner_logical_indices(self, d: int) -> list[tuple[int, ...]]:
        """Return logical indices for all 2**d corners in product order."""
        return list(product((0, 1), repeat=d))

    def face_corner_names(self, axis: int, side: int, d: int) -> list[str]:
        """Return the 2**(d-1) corner names on the given face.

        Filters corners to those where logical_index[axis] == side.
        """
        indices = self._corner_logical_indices(d)
        result = []
        for idx, cname in zip(indices, self.corner_names):
            if idx[axis] == side:
                result.append(cname)
        return result

    def face_corner_indices(self, axis: int, side: int, d: int) -> list[int]:
        """Return the indices (into corner_names) of corners on the given face."""
        indices = self._corner_logical_indices(d)
        result = []
        for i, idx in enumerate(indices):
            if idx[axis] == side:
                result.append(i)
        return result


@dataclass
class InterfaceConnection:
    """A declared shared face between two blocks."""

    face_a: FaceSpec
    face_b: FaceSpec
    orientation: dict[str, str] = field(default_factory=dict)
    # orientation maps face_a corner name → face_b corner name


@dataclass
class Association:
    """Tags a block face as lying on a geometry entity."""

    face: FaceSpec
    entity: Any  # GeometryEntity — imported lazily to avoid circular imports


@dataclass
class Singularity:
    """A node where incident edge count ≠ 2·d."""

    block_name: str
    logical_idx: tuple[int, ...]
    valence: int


@dataclass
class FanFrame:
    """A user-directed frame at a singular fan corner.

    The two ``through`` legs are held smoothly collinear across the fan
    (one continuous rail passing through the singular node) and the
    ``normal`` leg orthogonal to them; the remaining legs absorb the fan's
    angular defect. Declared with corner names; resolution at
    :class:`BlockTopology` construction fills in the fan corner's global
    DOF and, per leg, the ordered fine-node DOF ids along its block edge
    (the fan corner's own DOF first).
    """

    corner: str
    through: tuple[str, str]
    normal: str
    dof: int = -1
    through_rails: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    normal_rail: tuple[int, ...] = ()


@dataclass
class ParallelChain:
    """A line of block edges held loosely parallel to an associated boundary.

    Declared with an ordered corner list whose consecutive pairs are block
    edges (via :meth:`~egg.topology.builder.TopologyBuilder.parallel_to`).
    Resolution at :class:`BlockTopology` construction fills in ``dofs`` — the
    ordered fine-node DOF ids along the whole chain — and ``wall_dofs``: per
    chain node, the boundary fine-node DOF its block column reaches when
    walked perpendicular to the chain (crossing interfaces), or ``-1`` where
    no column reaches the boundary and the sample builder must fall back to
    frozen nearest-foot projection.
    """

    to: Any
    chain: tuple[str, ...]
    weight: float = 1.0
    taper: float | None = None
    dofs: tuple[int, ...] = ()
    wall_dofs: tuple[int, ...] = ()


class BlockTopology:
    """Validated topology: corners, blocks, connectivity, associations.

    Construction validates connections and boundary tags, resolves
    interface orientations, builds the shared-DOF map (union-find over
    interface nodes), and detects singularities. :meth:`initialize_grid`
    then produces the TFI-initialised :class:`~egg.core.types.MultiBlockGrid`.
    """

    def __init__(
        self,
        d: int,
        corners: dict[str, Corner],
        block_specs: dict[str, BlockSpec],
        connections: list[InterfaceConnection],
        associations: list[Association],
        boundary_layer_specs: dict | None = None,
        boundary_tags: dict[str, list[FaceSpec]] | None = None,
        relax_orthogonality: Sequence = (),
        fan_frames: tuple = (),
        parallel_chains: tuple = (),
    ):
        self.d = d
        self.corners = dict(corners)
        self.block_specs = dict(block_specs)
        self.interface_connections = list(connections)
        self.associations = list(associations)
        # entity-id -> dict of BoundaryLayerTarget kwargs (first_height, growth, …)
        self.boundary_layer_specs = dict(boundary_layer_specs or {})
        # Domain boundaries whose orthogonality requirements are relaxed —
        # resolved entities, declared via TopologyBuilder.relax_orthogonality
        # (or the deprecated clustering-spec kwargs, forwarded at build).
        # Consumed by egg.smoothing.targets.build_topology_target.
        self.relax_orthogonality: tuple = tuple(relax_orthogonality)
        # marker name -> block faces carrying that boundary-condition tag
        self.boundary_tags: dict[str, list[FaceSpec]] = {
            name: list(faces) for name, faces in (boundary_tags or {}).items()
        }
        self.singularities: list[Singularity] = []

        self._validate()
        self._validate_boundary_tags()
        self._resolve_orientations()
        self._validate_conformance()
        self.grid = self._build_dof_map()
        self.singularities = self._detect_singularities()
        # Declared fan frames (dicts of corner names, via
        # TopologyBuilder.fan_frame) resolved against the DOF map: each must
        # name a singular corner and three of its block edges.
        self.fan_frames: list[FanFrame] = self._resolve_fan_frames(fan_frames)
        # Declared parallel chains (dicts with a resolved boundary entity,
        # via TopologyBuilder.parallel_to) resolved against the DOF map:
        # ordered fine-node ids plus the per-node boundary correspondence.
        self.parallel_chains: list[ParallelChain] = self._resolve_parallel_chains(
            parallel_chains
        )

    @property
    def entities(self) -> dict[str, Any]:
        """The whole-domain ``{name: entity}`` map of every
        :meth:`~egg.geometry.base.GeometryEntity.named` associated entity
        (deduped by name, first association wins).

        Derived from the associations, so it needs no hand-maintained
        companion dict: name a curve once and it appears here for rendering
        and export.
        """
        out: dict[str, Any] = {}
        for assoc in self.associations:
            name = getattr(assoc.entity, "name", None)
            if name is not None and name not in out:
                out[name] = assoc.entity
        return out

    def _validate(self) -> None:
        """Check that all connections reference valid blocks and matching faces."""
        for conn in self.interface_connections:
            for face_spec in (conn.face_a, conn.face_b):
                if face_spec.block_name not in self.block_specs:
                    raise ValueError(
                        f"Connection references unknown block '{face_spec.block_name}'"
                    )
                if not (0 <= face_spec.axis < self.d):
                    raise ValueError(
                        f"Face axis {face_spec.axis} out of range for d={self.d}"
                    )
                if face_spec.side not in (0, 1):
                    raise ValueError(f"Face side must be 0 or 1, got {face_spec.side}")

            # Check matching face corner names
            spec_a = self.block_specs[conn.face_a.block_name]
            spec_b = self.block_specs[conn.face_b.block_name]
            names_a = set(
                spec_a.face_corner_names(conn.face_a.axis, conn.face_a.side, self.d)
            )
            names_b = set(
                spec_b.face_corner_names(conn.face_b.axis, conn.face_b.side, self.d)
            )
            if names_a != names_b:
                raise ValueError(
                    f"Face corners do not match for connection "
                    f"{conn.face_a.block_name}[a={conn.face_a.axis},s={conn.face_a.side}]"
                    f" ↔ "
                    f"{conn.face_b.block_name}[a={conn.face_b.axis},s={conn.face_b.side}]"
                    f": {names_a} vs {names_b}"
                )

    def _validate_boundary_tags(self) -> None:
        """Check that boundary tags reference valid, non-interface block faces."""
        shared = self._get_shared_face_set()
        for name, faces in self.boundary_tags.items():
            for face in faces:
                if face.block_name not in self.block_specs:
                    raise ValueError(
                        f"Boundary tag '{name}' references unknown block "
                        f"'{face.block_name}'"
                    )
                if not (0 <= face.axis < self.d) or face.side not in (0, 1):
                    raise ValueError(
                        f"Boundary tag '{name}' has invalid face "
                        f"(axis={face.axis}, side={face.side}) for d={self.d}"
                    )
                if (face.block_name, face.axis, face.side) in shared:
                    raise ValueError(
                        f"Boundary tag '{name}' targets shared interface face "
                        f"{face.block_name}[axis={face.axis}, side={face.side}]"
                    )

    def _resolve_orientations(self) -> None:
        """For each InterfaceConnection, build the corner-name orientation dict."""
        for conn in self.interface_connections:
            spec_a = self.block_specs[conn.face_a.block_name]
            spec_b = self.block_specs[conn.face_b.block_name]
            names_a = spec_a.face_corner_names(
                conn.face_a.axis, conn.face_a.side, self.d
            )
            names_b = spec_b.face_corner_names(
                conn.face_b.axis, conn.face_b.side, self.d
            )

            orientation: dict[str, str] = {}
            unmatched_b = set(names_b)
            for name_a in names_a:
                for name_b in list(unmatched_b):
                    if name_a == name_b:
                        orientation[name_a] = name_b
                        unmatched_b.discard(name_b)
                        break
            conn.orientation = orientation

    def _validate_conformance(self) -> None:
        """Reject connected faces whose shared axes have mismatched resolutions.

        Two blocks joined on a face must have the same number of nodes along
        every shared face axis; otherwise the node-matching walk in
        ``_build_dof_map`` indexes the smaller block out of range (a raw
        IndexError deep in the build). Catch it here with a clear message.
        """
        problems: list[str] = []
        for conn in self.interface_connections:
            spec_a = self.block_specs[conn.face_a.block_name]
            spec_b = self.block_specs[conn.face_b.block_name]
            shape_a = spec_a.logical_shape
            shape_b = spec_b.logical_shape
            free_axes_a, free_axes_b, axis_map = self._interface_axis_map(conn)
            for p, (q, _rev) in enumerate(axis_map):
                na = shape_a[free_axes_a[p]]
                nb = shape_b[free_axes_b[q]]
                if na != nb:
                    problems.append(
                        f"non-conforming interface "
                        f"{conn.face_a.block_name}[{conn.face_a.axis},{conn.face_a.side}]"
                        f" <-> "
                        f"{conn.face_b.block_name}[{conn.face_b.axis},{conn.face_b.side}]:"
                        f" {na} vs {nb} nodes along a shared axis"
                    )
        if problems:
            raise EggValidationError(
                "connected block faces must have matching resolutions:\n  "
                + "\n  ".join(problems)
            )

    def _interface_axis_map(
        self, conn: InterfaceConnection
    ) -> tuple[list[int], list[int], list[tuple[int, bool]]]:
        """Map a shared face's free axes from block A to block B.

        Returns ``(free_axes_a, free_axes_b, axis_map)`` where
        ``axis_map[p] = (q, rev)`` means face_a free axis ``free_axes_a[p]``
        corresponds to face_b free axis ``free_axes_b[q]``, reversed when
        ``rev``. Derived from the corner-name correspondence
        (``conn.orientation``), so it resolves every rotation and reflection of
        the shared quad, for one free axis (2D) or two (3D).
        """
        spec_a = self.block_specs[conn.face_a.block_name]
        spec_b = self.block_specs[conn.face_b.block_name]
        axis_a, side_a = conn.face_a.axis, conn.face_a.side
        axis_b, side_b = conn.face_b.axis, conn.face_b.side
        free_axes_a = [k for k in range(self.d) if k != axis_a]
        free_axes_b = [k for k in range(self.d) if k != axis_b]

        def free_corner_map(spec, axis, side, free_axes):
            # product-space free coords ({0,1}^(d-1)) -> corner name
            out: dict[tuple[int, ...], str] = {}
            for idx, cname in zip(product((0, 1), repeat=self.d), spec.corner_names):
                if idx[axis] == side:
                    out[tuple(idx[k] for k in free_axes)] = cname
            return out

        cmap_a = free_corner_map(spec_a, axis_a, side_a, free_axes_a)
        b_coords_of = {
            name: coords
            for coords, name in free_corner_map(
                spec_b, axis_b, side_b, free_axes_b
            ).items()
        }

        def matched_b(coords):
            name = cmap_a[coords]
            return b_coords_of[conn.orientation.get(name, name)]

        nfree = len(free_axes_a)
        b_origin = matched_b((0,) * nfree)
        axis_map: list[tuple[int, bool]] = []
        for p in range(nfree):
            b_flip = matched_b(tuple(1 if i == p else 0 for i in range(nfree)))
            changed = [q for q in range(nfree) if b_flip[q] != b_origin[q]]
            if len(changed) != 1:
                raise ValueError(
                    f"interface {conn.face_a.block_name}[{axis_a},{side_a}] ↔ "
                    f"{conn.face_b.block_name}[{axis_b},{side_b}] has a degenerate "
                    "corner correspondence (not a consistent face map)"
                )
            q = changed[0]
            axis_map.append((q, bool(b_origin[q])))
        return free_axes_a, free_axes_b, axis_map

    def _build_dof_map(self) -> MultiBlockGrid:
        """Build the shared-DOF index map using union-find.

        Returns a MultiBlockGrid with block_dof_maps populated.
        """
        block_names = list(self.block_specs.keys())

        # 1. Compute per-block node counts and provisional offsets
        block_node_counts = [self.block_specs[n].node_count for n in block_names]
        offsets = [0]
        for cnt in block_node_counts[:-1]:
            offsets.append(offsets[-1] + cnt)
        total_nodes = sum(block_node_counts)

        # 2. Union-find: set up provisional DOF indices
        parent = {i: i for i in range(total_nodes)}
        rank = {i: 0 for i in range(total_nodes)}

        def find(x: int) -> int:
            # Iterative path-compression: recursion would risk Python's recursion
            # limit on long union chains.
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                parent[rx] = ry
            elif rank[rx] > rank[ry]:
                parent[ry] = rx
            else:
                parent[ry] = rx
                rank[rx] += 1

        # 3. Helper: convert logical index → provisional DOF index
        def logical_to_provisional(block_idx: int, logical_idx: tuple[int, ...]) -> int:
            spec = self.block_specs[block_names[block_idx]]
            shape = spec.logical_shape
            flat = 0
            stride = 1
            for idx, size in zip(reversed(logical_idx), reversed(shape)):
                flat += idx * stride
                stride *= size
            return offsets[block_idx] + flat

        # 4. For each connection, walk the shared face and union matching nodes
        for conn in self.interface_connections:
            idx_a = block_names.index(conn.face_a.block_name)
            idx_b = block_names.index(conn.face_b.block_name)
            spec_a = self.block_specs[conn.face_a.block_name]
            spec_b = self.block_specs[conn.face_b.block_name]

            axis_a, side_a = conn.face_a.axis, conn.face_a.side
            axis_b, side_b = conn.face_b.axis, conn.face_b.side

            shape_a = spec_a.logical_shape
            shape_b = spec_b.logical_shape

            # Fixed index values for each face
            fixed_a = 0 if side_a == 0 else shape_a[axis_a] - 1
            fixed_b = 0 if side_b == 0 else shape_b[axis_b] - 1

            free_axes_a, free_axes_b, axis_map = self._interface_axis_map(conn)

            for free_idx_tuple in product(*[range(shape_a[k]) for k in free_axes_a]):
                logical_a = [0] * self.d
                logical_a[axis_a] = fixed_a
                for j, k in enumerate(free_axes_a):
                    logical_a[k] = free_idx_tuple[j]

                logical_b = [0] * self.d
                logical_b[axis_b] = fixed_b
                for p, (q, rev) in enumerate(axis_map):
                    kb = free_axes_b[q]
                    logical_b[kb] = (
                        shape_b[kb] - 1 - free_idx_tuple[p]
                        if rev
                        else free_idx_tuple[p]
                    )

                dof_a = logical_to_provisional(idx_a, tuple(logical_a))
                dof_b = logical_to_provisional(idx_b, tuple(logical_b))
                union(dof_a, dof_b)

        # 5. Collapse unions to sequential global indices
        root_to_global: dict[int, int] = {}
        next_global = 0
        for i in range(total_nodes):
            root = find(i)
            if root not in root_to_global:
                root_to_global[root] = next_global
                next_global += 1

        # 6. Build per-block dof maps and fixed-node set
        block_dof_maps: list[np.ndarray] = []
        fixed_global_dofs: set[int] = set()

        for bi, name in enumerate(block_names):
            spec = self.block_specs[name]
            shape = spec.logical_shape
            dof_map = np.full(shape, -1, dtype=int)

            for logical_idx in product(*[range(s) for s in shape]):
                logical_idx_t = tuple(logical_idx)
                prov = logical_to_provisional(bi, logical_idx_t)
                global_idx = root_to_global[find(prov)]
                dof_map[logical_idx_t] = global_idx

                # Check if this node is a named corner and is fixed
                corner_indices = spec._corner_logical_indices(self.d)
                for ci, ci_logical in enumerate(corner_indices):
                    # Map product-space (0,1) to actual index (0, n-1)
                    actual = tuple(
                        0 if c == 0 else shape[dim] - 1
                        for dim, c in enumerate(ci_logical)
                    )
                    if actual == logical_idx_t:
                        cname = spec.corner_names[ci]
                        if cname in self.corners and self.corners[cname].fixed:
                            fixed_global_dofs.add(global_idx)
                        break

            block_dof_maps.append(dof_map)

        # 7. Build free_mask
        global_node_count = next_global
        free_mask = np.ones(global_node_count, dtype=bool)
        for gidx in fixed_global_dofs:
            free_mask[gidx] = False

        # 8. Create placeholder blocks (NaN node arrays)
        blocks = []
        for bi, name in enumerate(block_names):
            spec = self.block_specs[name]
            shape = spec.logical_shape + (self.d,)  # type: ignore[operator]
            node_array = np.full(shape, np.nan)
            blocks.append(Block(node_array))

        return MultiBlockGrid(
            topology=self,
            blocks=blocks,
            block_dof_maps=block_dof_maps,
            global_node_count=global_node_count,
            free_mask=free_mask,
        )

    def _get_shared_face_set(self) -> set[tuple[str, int, int]]:
        """Return set of (block_name, axis, side) tuples for all shared faces."""
        shared: set[tuple[str, int, int]] = set()
        for conn in self.interface_connections:
            shared.add((conn.face_a.block_name, conn.face_a.axis, conn.face_a.side))
            shared.add((conn.face_b.block_name, conn.face_b.axis, conn.face_b.side))
        return shared

    def _detect_singularities(self) -> list[Singularity]:
        """Find interior nodes where incident edge count != 2*d.

        A node on an external (non-shared) block face is a regular boundary
        node — missing neighbours are expected there.  Only interior nodes
        with anomalous neighbour counts, and nodes with *extra* neighbours
        (valence > 2*d) on shared interfaces, are flagged as singularities.
        """
        singularities: list[Singularity] = []
        shared_faces = self._get_shared_face_set()

        # Build cross-face neighbour map
        cross_map: dict[tuple[str, int, int, tuple], int] = {}
        # cross_map[(block, axis, side, logical_idx)] = global_dof of inward
        # neighbour in the adjacent block, one step past the shared face

        block_names = list(self.block_specs.keys())
        for conn in self.interface_connections:
            spec_a = self.block_specs[conn.face_a.block_name]
            spec_b = self.block_specs[conn.face_b.block_name]
            shape_a = spec_a.logical_shape
            shape_b = spec_b.logical_shape
            axis_a, side_a = conn.face_a.axis, conn.face_a.side
            axis_b, side_b = conn.face_b.axis, conn.face_b.side
            fixed_a = 0 if side_a == 0 else shape_a[axis_a] - 1
            fixed_b = 0 if side_b == 0 else shape_b[axis_b] - 1
            free_axes_a = [k for k in range(self.d) if k != axis_a]
            free_axes_b = [k for k in range(self.d) if k != axis_b]

            _, _, axis_map = self._interface_axis_map(conn)

            # Lookup for bidirectional mapping
            dof_a = self.grid.block_dof_maps[block_names.index(conn.face_a.block_name)]
            dof_b = self.grid.block_dof_maps[block_names.index(conn.face_b.block_name)]

            for free_idx_tuple in product(*[range(shape_a[k]) for k in free_axes_a]):
                logical_a = [0] * self.d
                logical_a[axis_a] = fixed_a
                for j, k in enumerate(free_axes_a):
                    logical_a[k] = free_idx_tuple[j]
                la_t = tuple(logical_a)

                logical_b = [0] * self.d
                logical_b[axis_b] = fixed_b
                for p, (q, rev) in enumerate(axis_map):
                    kb = free_axes_b[q]
                    logical_b[kb] = (
                        shape_b[kb] - 1 - free_idx_tuple[p]
                        if rev
                        else free_idx_tuple[p]
                    )

                # Inward neighbor in block B (step away from shared face)
                inward_b = list(logical_b)
                inward_b[axis_b] += 1 if side_b == 0 else -1
                if 0 <= inward_b[axis_b] < shape_b[axis_b]:
                    cross_map[(conn.face_a.block_name, axis_a, side_a, la_t)] = int(
                        dof_b[tuple(inward_b)]
                    )

                # Inward neighbor in block A
                inward_a = list(logical_a)
                inward_a[axis_a] += 1 if side_a == 0 else -1
                if 0 <= inward_a[axis_a] < shape_a[axis_a]:
                    lb_t = tuple(logical_b)
                    cross_map[(conn.face_b.block_name, axis_b, side_b, lb_t)] = int(
                        dof_a[tuple(inward_a)]
                    )

        # Accumulate neighbours across ALL blocks (a shared DOF may live in
        # multiple blocks).  Also record per-DOF metadata.
        neighbour_sets: dict[int, set[int]] = {}
        dof_on_boundary: dict[int, bool] = {}
        dof_ref: dict[int, tuple[str, tuple[int, ...]]] = {}

        for bi, (name, spec) in enumerate(self.block_specs.items()):
            dof_map = self.grid.block_dof_maps[bi]
            shape = spec.logical_shape

            for logical_idx_t in product(*[range(s) for s in shape]):
                logical_idx = tuple(logical_idx_t)
                dof = int(dof_map[logical_idx])

                if not self.grid.free_mask[dof]:
                    continue

                if dof not in neighbour_sets:
                    neighbour_sets[dof] = set()
                    dof_ref[dof] = (name, logical_idx)
                    # A node is on a boundary if ANY block places it on a
                    # non-shared face.
                    on_bound = False
                    for axis in range(self.d):
                        for side in (0, 1):
                            sv = 0 if side == 0 else shape[axis] - 1
                            if (
                                logical_idx[axis] == sv
                                and (name, axis, side) not in shared_faces
                            ):
                                on_bound = True
                    dof_on_boundary[dof] = on_bound
                elif not dof_on_boundary[dof]:
                    # Already seen in another block, but check boundary status
                    # in *this* block as well.
                    for axis in range(self.d):
                        for side in (0, 1):
                            sv = 0 if side == 0 else shape[axis] - 1
                            if (
                                logical_idx[axis] == sv
                                and (name, axis, side) not in shared_faces
                            ):
                                dof_on_boundary[dof] = True

                # Within-block neighbours
                for axis in range(self.d):
                    for offset in (-1, 1):
                        ni = list(logical_idx)
                        ni[axis] += offset
                        if 0 <= ni[axis] < shape[axis]:
                            neighbour_sets[dof].add(int(dof_map[tuple(ni)]))

                # Cross-interface neighbours
                for axis in range(self.d):
                    for side in (0, 1):
                        sv = 0 if side == 0 else shape[axis] - 1
                        if (
                            logical_idx[axis] == sv
                            and (name, axis, side) in shared_faces
                        ):
                            cross_dof = cross_map.get((name, axis, side, logical_idx))
                            if cross_dof is not None:
                                neighbour_sets[dof].add(cross_dof)

        # Classify each free DOF
        for dof, neighbours in neighbour_sets.items():
            valence = len(neighbours)
            block_name, logical_idx = dof_ref[dof]

            if valence > 2 * self.d:
                singularities.append(
                    Singularity(
                        block_name=block_name,
                        logical_idx=logical_idx,
                        valence=valence,
                    )
                )
            elif valence < 2 * self.d and not dof_on_boundary[dof]:
                singularities.append(
                    Singularity(
                        block_name=block_name,
                        logical_idx=logical_idx,
                        valence=valence,
                    )
                )
            # valence == 2*d, or valence < 2*d on boundary -> regular node

        return singularities

    def _corner_leg_rails(self, corner: str) -> dict[str, tuple[int, ...]]:
        """Per block-edge neighbour of ``corner``: the fine-node DOF ids along
        that edge, ordered from the corner outward (its own DOF first).

        Blocks sharing the edge produce identical rails through the shared-DOF
        map, so the first block found wins.
        """
        rails: dict[str, tuple[int, ...]] = {}
        for bi, spec in enumerate(self.block_specs.values()):
            if corner not in spec.corner_names:
                continue
            dof_map = self.grid.block_dof_maps[bi]
            shape = spec.logical_shape
            idxs = spec._corner_logical_indices(self.d)
            slot_of = {t: s for s, t in enumerate(idxs)}
            for slot, cname in enumerate(spec.corner_names):
                if cname != corner:
                    continue
                cidx = idxs[slot]
                for axis in range(self.d):
                    nidx = tuple(1 - v if k == axis else v for k, v in enumerate(cidx))
                    nname = spec.corner_names[slot_of[nidx]]
                    if nname == corner or nname in rails:
                        continue
                    ix = [0 if v == 0 else shape[k] - 1 for k, v in enumerate(cidx)]
                    n = shape[axis]
                    span = range(n) if cidx[axis] == 0 else range(n - 1, -1, -1)
                    line = []
                    for t in span:
                        ix[axis] = t
                        line.append(int(dof_map[tuple(ix)]))
                    rails[nname] = tuple(line)
        return rails

    def _resolve_fan_frames(self, declared: tuple) -> list[FanFrame]:
        """Validate and resolve fan-frame declarations against the DOF map.

        Each declaration must name a singular corner and three distinct block
        edges incident to it (two through, one normal); anything else raises
        with the offending corner/leg named.
        """
        if not declared:
            return []
        shared = self._get_shared_face_set()
        frames: list[FanFrame] = []
        seen: set[str] = set()
        for f in declared:
            corner = str(f["corner"])
            through = tuple(str(c) for c in f["through"])
            normal = str(f["normal"])
            if corner in seen:
                raise ValueError(f"fan_frame: {corner!r} is framed twice")
            seen.add(corner)
            if corner not in self.corners:
                raise ValueError(f"fan_frame: unknown corner {corner!r}")
            if len(through) != 2:
                raise ValueError(
                    f"fan_frame at {corner!r}: 'through' must name exactly two corners"
                )
            legs = (*through, normal)
            if len(set(legs)) != 3:
                raise ValueError(
                    f"fan_frame at {corner!r}: through and normal must be "
                    "three distinct edges"
                )
            rails = self._corner_leg_rails(corner)
            for leg in legs:
                if leg not in rails:
                    raise ValueError(
                        f"fan_frame at {corner!r}: no block edge {corner!r}-{leg!r}"
                    )
            # Valence = incident fine edges (distinct first-step DOFs); the
            # regular count is 2d for an interior node, 2d-1 on the boundary.
            valence = len({r[1] for r in rails.values() if len(r) > 1})
            on_boundary = any(
                (name, axis, side) not in shared
                for name, spec in self.block_specs.items()
                if corner in spec.corner_names
                for axis in range(self.d)
                for side in (0, 1)
                if corner in spec.face_corner_names(axis, side, self.d)
            )
            regular = 2 * self.d - 1 if on_boundary else 2 * self.d
            singular = valence > regular or (not on_boundary and valence != regular)
            if not singular:
                raise ValueError(
                    f"fan_frame: {corner!r} is a regular node (valence "
                    f"{valence}); frames apply only at singular fans"
                )
            frames.append(
                FanFrame(
                    corner=corner,
                    through=(through[0], through[1]),
                    normal=normal,
                    dof=int(rails[legs[0]][0]),
                    through_rails=(rails[through[0]], rails[through[1]]),
                    normal_rail=rails[normal],
                )
            )
        return frames

    def _face_twin_map(self) -> dict:
        """``(block_name, axis, side) -> (other_block_idx, other_axis,
        other_side, mapper)`` for every shared face, where ``mapper`` carries a
        full logical index across the interface (both directions covered)."""
        block_names = list(self.block_specs.keys())
        twin: dict = {}
        for conn in self.interface_connections:
            spec_a = self.block_specs[conn.face_a.block_name]
            spec_b = self.block_specs[conn.face_b.block_name]
            axis_a, side_a = conn.face_a.axis, conn.face_a.side
            axis_b, side_b = conn.face_b.axis, conn.face_b.side
            free_a = [k for k in range(self.d) if k != axis_a]
            free_b = [k for k in range(self.d) if k != axis_b]
            _, _, axis_map = self._interface_axis_map(conn)
            shape_a, shape_b = spec_a.logical_shape, spec_b.logical_shape
            fixed_a = 0 if side_a == 0 else shape_a[axis_a] - 1
            fixed_b = 0 if side_b == 0 else shape_b[axis_b] - 1

            def fwd(
                logical,
                *,
                _free_a=free_a,
                _free_b=free_b,
                _map=axis_map,
                _shape_b=shape_b,
                _axis_b=axis_b,
                _fixed_b=fixed_b,
                _d=self.d,
            ):
                out = [0] * _d
                out[_axis_b] = _fixed_b
                for p, (q, rev) in enumerate(_map):
                    kb = _free_b[q]
                    v = logical[_free_a[p]]
                    out[kb] = _shape_b[kb] - 1 - v if rev else v
                return tuple(out)

            def bwd(
                logical,
                *,
                _free_a=free_a,
                _free_b=free_b,
                _map=axis_map,
                _shape_b=shape_b,
                _axis_a=axis_a,
                _fixed_a=fixed_a,
                _d=self.d,
            ):
                out = [0] * _d
                out[_axis_a] = _fixed_a
                for p, (q, rev) in enumerate(_map):
                    kb = _free_b[q]
                    v = logical[kb]
                    out[_free_a[p]] = _shape_b[kb] - 1 - v if rev else v
                return tuple(out)

            bi_a = block_names.index(conn.face_a.block_name)
            bi_b = block_names.index(conn.face_b.block_name)
            twin[(conn.face_a.block_name, axis_a, side_a)] = (bi_b, axis_b, side_b, fwd)
            twin[(conn.face_b.block_name, axis_b, side_b)] = (bi_a, axis_a, side_a, bwd)
        return twin

    def _walk_column(
        self, bi: int, logical, axis: int, direction: int, target_faces, twin
    ):
        """From node ``logical`` of block ``bi``, walk along ``axis`` in
        ``direction`` to the block face, crossing shared interfaces, until a
        face in ``target_faces``. Returns ``(boundary_dof, steps)`` with
        ``steps`` the fine cells traversed, or ``None`` (dead end / cycle)."""
        block_names = list(self.block_specs.keys())
        cur = list(logical)
        steps = 0
        seen: set = set()
        while True:
            name = block_names[bi]
            shape = self.block_specs[name].logical_shape
            side = 1 if direction > 0 else 0
            face = shape[axis] - 1 if direction > 0 else 0
            steps += abs(face - cur[axis])
            cur[axis] = face
            key = (name, axis, side)
            if key in target_faces:
                return int(self.grid.block_dof_maps[bi][tuple(cur)]), steps
            hop = twin.get(key)
            if hop is None or key in seen:
                return None
            seen.add(key)
            bi, axis, other_side, mapper = hop
            cur = list(mapper(tuple(cur)))
            direction = -1 if other_side == 1 else 1

    def _resolve_parallel_chains(self, declared: tuple) -> list[ParallelChain]:
        """Validate and resolve parallel-chain declarations against the DOF map.

        Each consecutive corner pair must be a block edge; the boundary must
        be associated to at least one face. Correspondence is the logical
        column walk: from every block home of a chain node, walk each
        non-chain direction to a boundary face, crossing interfaces; the
        nearest hit wins. Nodes no walk reaches get ``-1`` (projection
        fallback) and a :class:`ParallelWalkWarning`.
        """
        if not declared:
            return []
        block_names = list(self.block_specs.keys())

        resolved: list[tuple[dict, tuple[int, ...]]] = []
        for p in declared:
            chain = tuple(str(c) for c in p["chain"])
            if len(chain) < 2:
                raise ValueError("parallel_to: 'chain' needs at least two corners")
            dofs: list[int] = []
            for ca, cb in zip(chain[:-1], chain[1:]):
                if ca not in self.corners or cb not in self.corners:
                    missing = ca if ca not in self.corners else cb
                    raise ValueError(f"parallel_to: unknown corner {missing!r}")
                rail = self._corner_leg_rails(ca).get(cb)
                if rail is None:
                    raise ValueError(f"parallel_to: no block edge {ca!r}-{cb!r}")
                dofs.extend(rail if not dofs else rail[1:])
            resolved.append((dict(p, chain=chain), tuple(dofs)))

        # Occurrences of every chain node: global dof -> [(block_idx, logical)]
        all_dofs = sorted({g for _, dofs in resolved for g in dofs})
        occ: dict[int, list[tuple[int, tuple[int, ...]]]] = {g: [] for g in all_dofs}
        lookup = np.array(all_dofs)
        for bi in range(len(block_names)):
            dof_map = self.grid.block_dof_maps[bi]
            for idx in np.argwhere(np.isin(dof_map, lookup)):
                t = tuple(int(v) for v in idx)
                occ[int(dof_map[t])].append((bi, t))
        twin = self._face_twin_map()

        def faces_of(to):
            tname = getattr(to, "name", None)
            return {
                (a.face.block_name, a.face.axis, a.face.side)
                for a in self.associations
                if a.entity is to
                or (tname is not None and getattr(a.entity, "name", None) == tname)
            }

        chains: list[ParallelChain] = []
        for p, chain_dofs in resolved:
            to = p["to"]
            target = faces_of(to)
            label = getattr(to, "name", None) or repr(to)
            if not target:
                raise ValueError(f"parallel_to: {label} is not an associated boundary")
            dof_set = set(chain_dofs)
            wall: list[int] = []
            for g in chain_dofs:
                best = None  # (steps, tie-break order, wall_dof)
                order = 0
                for bi, logical in occ[g]:
                    dof_map = self.grid.block_dof_maps[bi]
                    shape = self.block_specs[block_names[bi]].logical_shape
                    for axis in range(self.d):
                        for direction in (-1, 1):
                            ni = logical[axis] + direction
                            if 0 <= ni < shape[axis]:
                                step = list(logical)
                                step[axis] = ni
                                if int(dof_map[tuple(step)]) in dof_set:
                                    continue  # runs along the chain itself
                            hit = self._walk_column(
                                bi, logical, axis, direction, target, twin
                            )
                            order += 1
                            if hit is not None and (
                                best is None or (hit[1], order) < best[:2]
                            ):
                                best = (hit[1], order, hit[0])
                wall.append(-1 if best is None else best[2])
            n_missing = sum(1 for w in wall if w < 0)
            if n_missing:
                warnings.warn(
                    f"parallel_to {label}: {n_missing} of {len(wall)} chain "
                    "nodes have no logical column to the boundary; frozen "
                    "nearest-foot projection will be used for them",
                    ParallelWalkWarning,
                    stacklevel=2,
                )
            chains.append(
                ParallelChain(
                    to=to,
                    chain=p["chain"],
                    weight=float(p.get("weight", 1.0)),
                    taper=p.get("taper"),
                    dofs=chain_dofs,
                    wall_dofs=tuple(wall),
                )
            )
        return chains

    # ---- Grid initialization ----

    @staticmethod
    def _project_batch(entity, pts: np.ndarray) -> np.ndarray:
        """Feet of ``pts`` on ``entity``, batched when the entity supports it."""
        if hasattr(entity, "project_many"):
            return np.asarray(entity.project_many(pts))
        return np.stack([entity.project(p) for p in pts])

    @staticmethod
    def _row_defective(P: np.ndarray) -> bool:
        """A sampled node line is defective when it folds back on itself or
        its spacing collapses/blows up far beyond the median — the signature
        of nearest-foot projection across a curve bending away from the
        sampling chord. Mild monotone non-uniformity is not defective."""
        v = np.diff(np.asarray(P, dtype=float), axis=0)
        L = np.linalg.norm(v, axis=1)
        if L.size < 2 or not np.all(np.isfinite(L)):
            return False
        med = float(np.median(L))
        if med <= 0.0:
            return True
        if float(L.min()) < 0.25 * med or float(L.max()) > 4.0 * med:
            return True
        return bool(np.any(np.einsum("ij,ij->i", v[1:], v[:-1]) < 0.0))

    def initialize_grid(self) -> MultiBlockGrid:
        """Create the initial node array for the multiblock grid.

        Steps per block:
        1. Place corner positions from Corner.position
        2. Space edge nodes between corners (uniform in the edge parameter
           when both corners are Nodes on one Edge, chord otherwise)
        3. Project geometry-associated face nodes onto their entities
        4. Fill interior nodes via TFI

        Shared interface nodes are filled once via the global DOF array.
        """
        from egg.init.tfi import tfi_fill_interior

        M = self.grid.global_node_count
        global_nodes = np.full((M, self.d), np.nan)

        # Per-block logic for placing nodes into the global array
        block_names = list(self.block_specs.keys())
        # Nodes placed by evaluating a curve between corner parameters
        # (step 2's place_node path). Their spacing is deliberate — step 3's
        # chord-projection refinement must leave them alone.
        curve_sampled: set[int] = set()

        for bi, name in enumerate(block_names):
            spec = self.block_specs[name]
            shape = spec.logical_shape
            dof_map = self.grid.block_dof_maps[bi]

            # Step 1: Place corners
            corner_indices = spec._corner_logical_indices(self.d)
            for ci, ci_logical in enumerate(corner_indices):
                actual = tuple(
                    0 if c == 0 else shape[dim] - 1 for dim, c in enumerate(ci_logical)
                )
                cname = spec.corner_names[ci]
                gidx = int(dof_map[actual])
                global_nodes[gidx] = self.corners[cname].position.copy()

            # Step 2: space nodes along each block edge. When both endpoint
            # corners are Nodes on a common curve, interpolate the edge
            # parameter (shortest way around a closed curve) so a strongly
            # curved wall is sampled along the curve rather than its chord,
            # which can leave a sharp apex empty. Otherwise fall back to the
            # chord; step 3 projects it onto any associated entity.
            corner_of: dict[tuple[int, ...], str] = dict(
                zip(product((0, 1), repeat=self.d), spec.corner_names)
            )
            for edge_axis in range(self.d):
                n_edge = shape[edge_axis]
                if n_edge < 3:
                    continue
                others = [a for a in range(self.d) if a != edge_axis]
                for combo in product((0, 1), repeat=len(others)):
                    end_ci = [[0] * self.d, [0] * self.d]
                    for e in (0, 1):
                        end_ci[e][edge_axis] = e
                        for a, s in zip(others, combo):
                            end_ci[e][a] = s
                    c0 = self.corners[corner_of[tuple(end_ci[0])]]
                    c1 = self.corners[corner_of[tuple(end_ci[1])]]

                    edge = getattr(c0.source, "edge", None)
                    param = None
                    if edge is not None and getattr(c1.source, "edge", None) is edge:
                        t0 = float(c0.source.t)
                        dt = float(c1.source.t) - t0
                        p0, p1 = edge.point_at(0.0), edge.point_at(1.0)
                        closed = bool(getattr(edge.entity, "closed", False)) or (
                            abs(p1 - p0) <= 1e-9 * (1.0 + abs(p0))
                        )
                        if closed and abs(dt) > 0.5:
                            dt -= np.copysign(1.0, dt)
                        param = (t0, dt, closed)

                    for k in range(1, n_edge - 1):
                        logical = [0] * self.d
                        logical[edge_axis] = k
                        for a, s in zip(others, combo):
                            logical[a] = 0 if s == 0 else shape[a] - 1
                        gidx = int(dof_map[tuple(logical)])
                        if not np.any(np.isnan(global_nodes[gidx])):
                            continue
                        t = k / (n_edge - 1)
                        if param is not None:
                            assert edge is not None  # param is set only when edge is
                            t0, dt, closed = param
                            tt = t0 + t * dt
                            if closed:
                                tt %= 1.0
                            p = edge.point_at(tt)
                            global_nodes[gidx] = np.array(list(p)[: self.d])
                            curve_sampled.add(gidx)
                        else:
                            global_nodes[gidx] = (
                                1.0 - t
                            ) * c0.position + t * c1.position

            # Step 3: Project geometry-associated face nodes
            for assoc in self.associations:
                if assoc.face.block_name != name:
                    continue
                axis, side = assoc.face.axis, assoc.face.side
                entity = assoc.entity
                fixed = 0 if side == 0 else shape[axis] - 1
                free_axes = [a for a in range(self.d) if a != axis]
                # A 3D+ face is a (d-1)-manifold whose interior nodes are NaN
                # until the volume TFI, so fill it from its already-set edges
                # here (TFI over the face) before projecting them onto the entity.
                fgidx = None
                if len(free_axes) >= 2:
                    face_shape = tuple(shape[a] for a in free_axes)
                    fnodes = np.full(face_shape + (self.d,), np.nan)
                    fgidx = np.empty(face_shape, dtype=int)
                    for fi in product(*[range(s) for s in face_shape]):
                        logical = [0] * self.d
                        logical[axis] = fixed
                        for j, a in enumerate(free_axes):
                            logical[a] = fi[j]
                        g = int(dof_map[tuple(logical)])
                        fgidx[fi] = g
                        fnodes[fi] = global_nodes[g]
                    fb = Block(fnodes)
                    tfi_fill_interior(fb)
                    for fi in product(*[range(s) for s in face_shape]):
                        g = int(fgidx[fi])
                        if np.any(np.isnan(global_nodes[g])):
                            global_nodes[g] = fb.nodes[fi]
                # Project the whole face as one batch — a scalar project() per
                # node makes many-segment composite curves (splines through
                # dense point files) dominate initialization.
                gidxs = []
                for free_idx in product(*[range(shape[a]) for a in free_axes]):
                    logical = [0] * self.d
                    logical[axis] = fixed
                    for j, a in enumerate(free_axes):
                        logical[a] = free_idx[j]
                    gidxs.append(int(dof_map[tuple(logical)]))
                gidxs = np.asarray(gidxs, dtype=int)
                pos = global_nodes[gidxs]
                set_rows = ~np.isnan(pos).any(axis=1)
                if set_rows.any():
                    feet = self._project_batch(entity, pos[set_rows])
                    global_nodes[gidxs[set_rows]] = feet

                # Chord-sampled faces (no place_node parameter provenance —
                # drawn/explicit topologies): a single nearest-foot projection
                # keeps nodes ON the entity but bunches them where it bends
                # away from the chord, and can fold them across a corner.
                # When the projected row is actually defective (a fold, or
                # spacing collapsed/blown far beyond its median), fair it by
                # re-spacing along the projected shape and re-projecting until
                # it settles — the parameter-free equivalent of sampling along
                # the curve between the corners. Mildly non-uniform but
                # monotone projections are left as-is, so programmatic
                # topologies keep their established init distribution.
                if len(free_axes) == 1 and len(gidxs) >= 3:
                    interior = gidxs[1:-1]
                    if not all(int(g) in curve_sampled for g in interior):
                        for _ in range(3):
                            P = global_nodes[gidxs]
                            if not self._row_defective(P):
                                break
                            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
                            s = np.concatenate([[0.0], np.cumsum(seg)])
                            if s[-1] <= 0.0:
                                break
                            si = np.linspace(0.0, s[-1], len(gidxs))[1:-1]
                            Q = np.stack(
                                [np.interp(si, s, P[:, k]) for k in range(self.d)],
                                axis=1,
                            )
                            global_nodes[interior] = self._project_batch(entity, Q)
                elif fgidx is not None:
                    # The (d-1)-face analogue: re-fill the interior by TFI from
                    # the projected boundary rings, then re-project — only when
                    # a face row is defective, by the same criterion.
                    inner = tuple(slice(1, -1) for _ in fgidx.shape)
                    if all(s > 2 for s in fgidx.shape):

                        def _face_defective(F):
                            for axf in range(F.ndim - 1):
                                rows = np.moveaxis(F, axf, 0)
                                rows = rows.reshape(rows.shape[0], -1, self.d)
                                for c in range(rows.shape[1]):
                                    if self._row_defective(rows[:, c]):
                                        return True
                            return False

                        for _ in range(2):
                            fnodes = global_nodes[fgidx]
                            if not _face_defective(fnodes):
                                break
                            fnodes = fnodes.copy()
                            fnodes[inner] = np.nan
                            fb = Block(fnodes)
                            tfi_fill_interior(fb)
                            Q = fb.nodes[inner].reshape(-1, self.d)
                            global_nodes[fgidx[inner].reshape(-1)] = (
                                self._project_batch(entity, Q)
                            )

        # Step 4: Copy global nodes into block node arrays and fill interiors
        for bi, name in enumerate(block_names):
            spec = self.block_specs[name]
            shape = spec.logical_shape
            dof_map = self.grid.block_dof_maps[bi]

            nodes = np.full(shape + (self.d,), np.nan)
            for logical_idx in product(*[range(s) for s in shape]):
                gidx = int(dof_map[logical_idx])
                nodes[logical_idx] = global_nodes[gidx]

            block = Block(nodes)
            tfi_fill_interior(block)
            self.grid.blocks[bi] = block

            # Sync TFI-filled interior nodes back to global_nodes
            for logical_idx in product(*[range(s) for s in shape]):
                gidx = int(dof_map[logical_idx])
                if np.any(np.isnan(global_nodes[gidx])):
                    global_nodes[gidx] = block.nodes[logical_idx]

        # ---- Geometry boundary constraints: free / sliding / fixed ----
        # A node on one associated entity slides along it; a node where two
        # entities meet (e.g. a domain corner) is fixed. Sliding nodes stay in
        # free_mask and are projected back onto their entity by the solver.
        from egg.projection.associate import build_dof_constraints

        dof_constraints, fixed_dofs = build_dof_constraints(self, self.grid)
        for gidx in fixed_dofs:
            self.grid.free_mask[gidx] = False
        self.grid.dof_constraints = dof_constraints

        self.grid.global_nodes = global_nodes
        return self.grid
