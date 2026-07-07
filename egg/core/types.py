# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Dimension-agnostic data model: Block, MultiBlockGrid."""

from __future__ import annotations

from itertools import product
from typing import Any, Iterator

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from egg.topology.block_topology import BlockTopology

__all__ = ["Block", "MultiBlockGrid"]


class Block:
    """A single structured block with nodes of shape (n_0, ..., n_{d-1}, d)."""

    def __init__(self, nodes: np.ndarray):
        self.nodes = np.asarray(nodes)
        self.logical_shape: tuple[int, ...] = self.nodes.shape[:-1]
        self.d: int = self.nodes.ndim - 1

    def cell_count(self) -> tuple[int, ...]:
        """Cells per axis: (n_0-1, ..., n_{d-1}-1)."""
        return tuple(n - 1 for n in self.logical_shape)

    def iter_cells(self) -> Iterator[tuple[int, ...]]:
        """Yield (i0, i1, ...) lower-corner logical index for every cell."""
        ranges = [range(n - 1) for n in self.logical_shape]
        yield from product(*ranges)

    def corner_indices(self, cell_base: tuple[int, ...]) -> list[tuple[int, ...]]:
        """Return 2**d corner logical indices for a cell given its lower corner."""
        offsets = product((0, 1), repeat=self.d)
        return [tuple(b + o for b, o in zip(cell_base, offset)) for offset in offsets]

    def iter_boundary_faces(self) -> Iterator[tuple[int, int, tuple[Any, ...]]]:
        """Yield (axis, side, slice_expr) for each boundary face."""
        for axis in range(self.d):
            for side in (0, 1):
                yield axis, side, self._face_node_idx(axis, side)

    def _face_node_idx(self, axis: int, side: int) -> tuple[Any, ...]:
        """Return the numpy index tuple for all nodes on the given face."""
        idx: list[Any] = [slice(None)] * self.d
        idx[axis] = 0 if side == 0 else self.logical_shape[axis] - 1
        return tuple(idx)

    def face_node_slice(self, axis: int, side: int) -> tuple[Any, ...]:
        """Return the numpy index tuple for all nodes on the given face."""
        return self._face_node_idx(axis, side)

    def is_interior(self, idx: tuple[int, ...]) -> bool:
        """True if the node is not on any block boundary."""
        return all(0 < i < n - 1 for i, n in zip(idx, self.logical_shape))

    def node_patch(self, idx: tuple[int, ...]) -> list[tuple[int, ...]]:
        """All cell lower-corner indices incident to this node.

        For an interior node in 2D this returns 4 cells; in 3D, 8 cells.
        """
        cells = []
        for offset in product((0, 1), repeat=self.d):
            base = tuple(i - o for i, o in zip(idx, offset))
            if all(0 <= b < cn for b, cn in zip(base, self.cell_count())):
                cells.append(base)
        return cells


def _find(parent: dict[int, int], x: int) -> int:
    """Union-find root lookup with path compression (iterative).

    Iterative to avoid hitting Python's recursion limit on pathological
    (long-chain) inputs.
    """
    root = x
    while parent[root] != root:
        root = parent[root]
    # Path compression: point every node on the walk straight at the root.
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def _union(parent: dict[int, int], rank: dict[int, int], x: int, y: int) -> None:
    """Union two sets by rank."""
    rx, ry = _find(parent, x), _find(parent, y)
    if rx == ry:
        return
    if rank[rx] < rank[ry]:
        parent[rx] = ry
    elif rank[rx] > rank[ry]:
        parent[ry] = rx
    else:
        parent[ry] = rx
        rank[rx] += 1


class MultiBlockGrid:
    """Collection of blocks with a shared-DOF interface map."""

    def __init__(
        self,
        topology: "BlockTopology",
        blocks: list[Block],
        block_dof_maps: list[np.ndarray],
        global_node_count: int,
        free_mask: np.ndarray,
        global_nodes: np.ndarray | None = None,
    ):
        self.topology = topology
        self.blocks = blocks
        self.block_dof_maps = block_dof_maps
        self.global_node_count = global_node_count
        self.free_mask = free_mask
        self.global_nodes = (
            global_nodes
            if global_nodes is not None
            else np.full((global_node_count, blocks[0].d if blocks else 0), np.nan)
        )
        # Sliding geometry constraints: global DOF index -> GeometryEntity.
        # Populated by BlockTopology.initialize_grid; empty means all free/fixed.
        self.dof_constraints: dict[int, Any] = {}

    @property
    def total_free_dofs(self) -> int:
        """Number of free DOF indices (useful for optimizer vector sizing)."""
        return int(np.sum(self.free_mask))

    def pack(self) -> np.ndarray:
        """Flatten free-DOF coordinates into a 1D optimization vector.

        Delegates to :func:`egg.smoothing.objective.pack_x` (the real
        implementation).
        """
        from egg.smoothing.objective import pack_x

        return pack_x(self)

    def unpack(self, x: np.ndarray) -> None:
        """Write ``x`` back into global_nodes and propagate to block.nodes.

        Delegates to :func:`egg.smoothing.objective.unpack_x`.
        """
        from egg.smoothing.objective import unpack_x

        unpack_x(x, self)
