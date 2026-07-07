# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""SU2 native ASCII mesh export for 2D/3D multiblock structured grids.

Writes the format documented at https://su2code.github.io/docs/Mesh-File/:
``NDIME`` / ``NPOIN`` / ``NELEM`` / ``NMARK`` sections, zero-based VTK element
types (line = 3, quadrilateral = 9, hexahedron = 12) and implied point/element
indices.  Interface nodes shared between blocks are written once — the grid's
global DOF map already deduplicates them.

Boundary markers come from :attr:`BlockTopology.boundary_tags` (see
:meth:`TopologyBuilder.tag_boundary`) or from an explicit ``markers`` mapping.
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping

import numpy as np

from egg.core.types import MultiBlockGrid

__all__ = ["export_su2"]

# VTK element type ids used by the SU2 format.
_VTK_LINE = 3
_VTK_QUAD = 9
_VTK_HEXAHEDRON = 12


def export_su2(
    grid: MultiBlockGrid,
    path: str | os.PathLike,
    markers: Mapping[str, Iterable[tuple[str, int, int]]] | None = None,
) -> None:
    """Write ``grid`` to ``path`` in the SU2 native ASCII mesh format.

    @param grid     Solved (or at least initialized) multiblock grid; node
                    coordinates are taken from ``grid.global_nodes``.
    @param path     Output file path (conventionally ``*.su2``).
    @param markers  Optional mapping of marker name to an iterable of
                    ``(block_name, axis, side)`` block faces.  When omitted,
                    the topology's ``boundary_tags`` are used.

    Cells with negative orientation (clockwise quads / left-handed hexes) are
    reordered so every exported element has positive volume.
    """
    d = grid.blocks[0].d if grid.blocks else 0
    if d not in (2, 3):
        raise ValueError(f"SU2 export supports d=2 or d=3 grids, got d={d}")

    nodes = grid.global_nodes
    if np.any(np.isnan(nodes)):
        raise ValueError(
            "grid.global_nodes contains NaN — initialize the grid before export"
        )

    elements = _gather_volume_elements(grid, d)
    marker_faces = _resolve_markers(grid, markers)
    marker_elements = {
        name: _gather_marker_elements(grid, faces, d)
        for name, faces in marker_faces.items()
    }

    etype = _VTK_QUAD if d == 2 else _VTK_HEXAHEDRON
    btype = _VTK_LINE if d == 2 else _VTK_QUAD

    with open(path, "w") as f:
        f.write(f"NDIME= {d}\n")

        f.write(f"NPOIN= {len(nodes)}\n")
        for p in nodes:
            f.write(" ".join(f"{c:.16g}" for c in p) + "\n")

        f.write(f"NELEM= {len(elements)}\n")
        for conn in elements:
            f.write(f"{etype} " + " ".join(str(i) for i in conn) + "\n")

        f.write(f"NMARK= {len(marker_elements)}\n")
        for name, elems in marker_elements.items():
            f.write(f"MARKER_TAG= {name}\n")
            f.write(f"MARKER_ELEMS= {len(elems)}\n")
            for conn in elems:
                f.write(f"{btype} " + " ".join(str(i) for i in conn) + "\n")


def _resolve_markers(
    grid: MultiBlockGrid,
    markers: Mapping[str, Iterable[tuple[str, int, int]]] | None,
) -> dict[str, list[tuple[str, int, int]]]:
    """Normalize the marker spec to ``{name: [(block, axis, side), ...]}``."""
    if markers is not None:
        return {name: list(faces) for name, faces in markers.items()}
    tags = getattr(grid.topology, "boundary_tags", {}) or {}
    return {
        name: [(fs.block_name, fs.axis, fs.side) for fs in faces]
        for name, faces in tags.items()
    }


def _block_dof_map(grid: MultiBlockGrid, block_name: str) -> np.ndarray:
    """Look up a block's global-DOF map by block name."""
    block_names = list(grid.topology.block_specs.keys())
    try:
        bi = block_names.index(block_name)
    except ValueError:
        raise ValueError(f"Marker references unknown block '{block_name}'") from None
    return grid.block_dof_maps[bi]


def _gather_volume_elements(grid: MultiBlockGrid, d: int) -> np.ndarray:
    """Collect all cells of all blocks as (n_cells, 2**d) VTK-ordered indices."""
    per_block = []
    for dof_map in grid.block_dof_maps:
        conn = _block_cells(dof_map, d)
        _fix_orientation(grid.global_nodes, conn, d)
        per_block.append(conn)
    if not per_block:
        return np.empty((0, 2**d), dtype=int)
    return np.concatenate(per_block, axis=0)


def _block_cells(dof_map: np.ndarray, d: int) -> np.ndarray:
    """Vectorized VTK connectivity for one block's structured cells.

    VTK quad order: (i,j), (i+1,j), (i+1,j+1), (i,j+1).
    VTK hex order: that quad at k, then the same quad at k+1.
    """
    if d == 2:
        offsets = [(0, 0), (1, 0), (1, 1), (0, 1)]
    else:
        offsets = [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ]
    cell_shape = tuple(n - 1 for n in dof_map.shape)
    cols = [
        dof_map[tuple(slice(o, o + n) for o, n in zip(off, cell_shape))].reshape(-1)
        for off in offsets
    ]
    return np.stack(cols, axis=1)


def _fix_orientation(nodes: np.ndarray, conn: np.ndarray, d: int) -> None:
    """Reorder negatively-oriented cells in-place so volumes are positive."""
    if d == 2:
        p = nodes[conn]  # (n, 4, 2)
        # Shoelace signed area of each quad.
        x, y = p[..., 0], p[..., 1]
        area = np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
        flip = area < 0
        conn[flip] = conn[flip][:, ::-1]
    else:
        p = nodes[conn]  # (n, 8, 3)
        # Triple product of the three corner edges approximates the volume sign.
        e1 = p[:, 1] - p[:, 0]
        e2 = p[:, 3] - p[:, 0]
        e3 = p[:, 4] - p[:, 0]
        vol = np.einsum("ij,ij->i", np.cross(e1, e2), e3)
        flip = vol < 0
        # Mirror the hex: swap bottom and top quads.
        conn[flip] = conn[flip][:, [4, 5, 6, 7, 0, 1, 2, 3]]


def _gather_marker_elements(
    grid: MultiBlockGrid,
    faces: Iterable[tuple[str, int, int]],
    d: int,
) -> np.ndarray:
    """Collect boundary elements (lines in 2D, quads in 3D) for one marker."""
    per_face = []
    for block_name, axis, side in faces:
        dof_map = _block_dof_map(grid, block_name)
        face = _take_face(dof_map, axis, side)
        if d == 2:
            seg = np.stack([face[:-1], face[1:]], axis=1)
            per_face.append(seg)
        else:
            per_face.append(_block_cells(face, 2))
    width = 2 if d == 2 else 4
    if not per_face:
        return np.empty((0, width), dtype=int)
    return np.concatenate(per_face, axis=0)


def _take_face(dof_map: np.ndarray, axis: int, side: int) -> np.ndarray:
    """Slice one boundary face out of a block's DOF map."""
    idx: list = [slice(None)] * dof_map.ndim
    idx[axis] = 0 if side == 0 else dof_map.shape[axis] - 1
    return dof_map[tuple(idx)]
