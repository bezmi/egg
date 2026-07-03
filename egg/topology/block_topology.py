"""Multiblock topology data model: corners, blocks, connectivity, associations.

Usually constructed via :class:`egg.topology.builder.TopologyBuilder` rather
than directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from egg.core.types import Block, MultiBlockGrid

__all__ = [
    "Corner",
    "BlockSpec",
    "FaceSpec",
    "InterfaceConnection",
    "Association",
    "Singularity",
    "BlockTopology",
]


@dataclass
class Corner:
    """A named point in R^d."""

    name: str
    position: np.ndarray
    fixed: bool = True

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
    ):
        self.d = d
        self.corners = dict(corners)
        self.block_specs = dict(block_specs)
        self.interface_connections = list(connections)
        self.associations = list(associations)
        # entity-id -> dict of BoundaryLayerTarget kwargs (first_height, growth, …)
        self.boundary_layer_specs = dict(boundary_layer_specs or {})
        # marker name -> block faces carrying that boundary-condition tag
        self.boundary_tags: dict[str, list[FaceSpec]] = {
            name: list(faces) for name, faces in (boundary_tags or {}).items()
        }
        self.singularities: list[Singularity] = []

        self._validate()
        self._validate_boundary_tags()
        self._resolve_orientations()
        self.grid = self._build_dof_map()
        self.singularities = self._detect_singularities()

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

            # Free axes (everything except the face axis)
            free_axes_a = [k for k in range(self.d) if k != axis_a]
            free_axes_b = [k for k in range(self.d) if k != axis_b]

            # Get corner names on each face and their logical indices
            names_a = spec_a.face_corner_names(axis_a, side_a, self.d)
            corner_indices_a = spec_a.face_corner_indices(axis_a, side_a, self.d)
            names_b = spec_b.face_corner_names(axis_b, side_b, self.d)
            corner_indices_b = spec_b.face_corner_indices(axis_b, side_b, self.d)

            # Map each face_a corner to its corresponding face_b corner via orientation
            name_to_b_corner = {}
            for ci_b, name_b in zip(corner_indices_b, names_b):
                name_to_b_corner[name_b] = ci_b

            corner_a_ci_to_b_ci: dict[int, int] = {}
            for ci_a, name_a in zip(corner_indices_a, names_a):
                b_name = conn.orientation.get(name_a, name_a)
                corner_a_ci_to_b_ci[ci_a] = name_to_b_corner[b_name]

            # Build the mapping from face_a free-axis indices → face_b free-axis indices
            # For 2D: face is an edge with 1 free axis; orientation = direction or reversal
            # For 3D: more complex mapping; we support axis correspondence

            free_shapes_a = [shape_a[k] for k in free_axes_a]
            free_shapes_b = [shape_b[k] for k in free_axes_b]

            for free_idx_tuple in product(*[range(s) for s in free_shapes_a]):
                # Build logical index in block A for this face node
                logical_a = [0] * self.d
                logical_a[axis_a] = fixed_a
                for j, k in enumerate(free_axes_a):
                    logical_a[k] = free_idx_tuple[j]

                # Build logical index in block B
                # Determine the matching free-axis indices on block B
                logical_b = [0] * self.d
                logical_b[axis_b] = fixed_b

                if self.d == 2:
                    # 2D: faces are edges, 1 free axis
                    # Check orientation: does node walk in same or opposite direction?
                    c0_a_idx = corner_a_ci_to_b_ci.get(corner_indices_a[0])
                    if c0_a_idx is not None:
                        # Determine direction: if first corner of A maps to first corner of B, same direction
                        # A node at free index k walks from A's first corner to A's second corner
                        # The matching B index depends on the direction
                        n_free = free_shapes_b[0]
                        if conn.orientation.get(names_a[0]) == names_b[0]:
                            # Same direction
                            logical_b[free_axes_b[0]] = free_idx_tuple[0]
                        else:
                            # Reversed
                            logical_b[free_axes_b[0]] = n_free - 1 - free_idx_tuple[0]
                    else:
                        # Fallback: direct mapping
                        logical_b[free_axes_b[0]] = free_idx_tuple[0]
                else:
                    # 3D: faces are quads, 2 free axes — deferred
                    # For initial 3D support: assume same-orientation direct mapping
                    for j, k in enumerate(free_axes_b):
                        logical_b[k] = free_idx_tuple[min(j, len(free_idx_tuple) - 1)]

                logical_a_t = tuple(logical_a)
                logical_b_t = tuple(logical_b)

                dof_a = logical_to_provisional(idx_a, logical_a_t)
                dof_b = logical_to_provisional(idx_b, logical_b_t)
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

            names_a = spec_a.face_corner_names(axis_a, side_a, self.d)
            names_b = spec_b.face_corner_names(axis_b, side_b, self.d)
            reversed_2d = (
                self.d == 2
                and len(names_a) >= 2
                and conn.orientation.get(names_a[0]) != names_b[0]
            )

            free_shapes_a = [shape_a[k] for k in free_axes_a]
            free_shapes_b = [shape_b[k] for k in free_axes_b]

            # Lookup for bidirectional mapping
            dof_a = self.grid.block_dof_maps[block_names.index(conn.face_a.block_name)]
            dof_b = self.grid.block_dof_maps[block_names.index(conn.face_b.block_name)]

            for free_idx_tuple in product(*[range(s) for s in free_shapes_a]):
                logical_a = [0] * self.d
                logical_a[axis_a] = fixed_a
                for j, k in enumerate(free_axes_a):
                    logical_a[k] = free_idx_tuple[j]
                la_t = tuple(logical_a)

                logical_b = [0] * self.d
                logical_b[axis_b] = fixed_b
                if self.d == 2 and free_axes_b:
                    n_fb = free_shapes_b[0] if free_shapes_b else 1
                    if reversed_2d:
                        logical_b[free_axes_b[0]] = n_fb - 1 - free_idx_tuple[0]
                    else:
                        logical_b[free_axes_b[0]] = free_idx_tuple[0]

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

    # ---- Grid initialization ----

    def initialize_grid(self) -> MultiBlockGrid:
        """Create the initial node array for the multiblock grid.

        Steps per block:
        1. Place corner positions from Corner.position
        2. Linearly space edge nodes between corners
        3. Project geometry-associated face nodes onto their entities
        4. Fill interior nodes via TFI

        Shared interface nodes are filled once via the global DOF array.
        """
        from egg.init.tfi import tfi_fill_interior

        M = self.grid.global_node_count
        global_nodes = np.full((M, self.d), np.nan)

        # Per-block logic for placing nodes into the global array
        block_names = list(self.block_specs.keys())

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

            # Step 2: Linearly space nodes along each edge (1D face)
            for axis in range(self.d):
                for side in (0, 1):
                    fixed = 0 if side == 0 else shape[axis] - 1
                    face_cnames = spec.face_corner_names(axis, side, self.d)
                    if len(face_cnames) != 2:
                        continue
                    c0 = self.corners[face_cnames[0]].position
                    c1 = self.corners[face_cnames[1]].position

                    free_axes = [a for a in range(self.d) if a != axis]
                    if not free_axes:
                        continue
                    free_axis = free_axes[0]
                    n_edge_nodes = shape[free_axis]

                    for k in range(1, n_edge_nodes - 1):
                        t = k / (n_edge_nodes - 1)
                        logical = [0] * self.d
                        logical[axis] = fixed
                        logical[free_axis] = k
                        logical_t = tuple(logical)
                        gidx = int(dof_map[logical_t])
                        if not np.any(np.isnan(global_nodes[gidx])):
                            continue
                        global_nodes[gidx] = (1.0 - t) * c0 + t * c1

            # Step 3: Project geometry-associated face nodes
            for assoc in self.associations:
                if assoc.face.block_name != name:
                    continue
                axis, side = assoc.face.axis, assoc.face.side
                entity = assoc.entity
                fixed = 0 if side == 0 else shape[axis] - 1
                free_axes = [a for a in range(self.d) if a != axis]
                for free_idx in product(*[range(shape[a]) for a in free_axes]):
                    logical = [0] * self.d
                    logical[axis] = fixed
                    for j, a in enumerate(free_axes):
                        logical[a] = free_idx[j]
                    logical_t = tuple(logical)
                    gidx = int(dof_map[logical_t])
                    pos = global_nodes[gidx]
                    if not np.any(np.isnan(pos)):
                        projected = entity.project(pos)
                        global_nodes[gidx] = projected

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
