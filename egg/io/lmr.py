# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Export multiblock structured grids for the gdtk/Eilmer ``lmr`` toolchain.

egg writes one native structured-grid file per block plus a ``grid.lua``
registration stub.  The user runs ``lmr prep-grid --job=grid.lua``; gdtk's own
reader ingests the block files and ``identifyGridConnections()`` rediscovers the
block-to-block interfaces from the coincident interface vertices egg already
guarantees (non-conforming interfaces are rejected upstream, so coincidence
detection is reliable).  egg therefore writes *only* block geometry: the grid
metadata, connection table and block list are all produced by ``prep-grid``.

No gdtk code is used or linked; the file layouts below are reproduced from the
published format so the output is plain data interchange.  Three container
formats are supported, all accepted by ``StructuredGrid:new{filename=, fmt=}``:

- ``"gziptext"`` (default): the Eilmer native text format, gzip-compressed.
- ``"rawbinary"``: the Eilmer native little-endian binary format.
- ``"vtk"``: legacy VTK ASCII structured grid (also openable in ParaView).

Vertex order in every format is i-fastest, then j, then k, matching gdtk's
``single_index = i + niv*(j + njv*k)``.  Boundary conditions are not embedded in
the grid files; they are declared per block in ``grid.lua`` via ``bcTags`` and
assigned at registration.
"""

from __future__ import annotations

import gzip
import os
import struct
import textwrap
from typing import TYPE_CHECKING

import numpy as np

from egg.core.types import MultiBlockGrid

if TYPE_CHECKING:
    from egg.topology.block_topology import BlockTopology

__all__ = ["export_lmr", "untagged_external_faces"]

# Marker stem for external faces that carry no boundary tag; the suffix is a
# group index (faces on one geometry entity, or one block edge, share it).
_UNTAGGED_PREFIX = "egg-untagged-"

# gdtk structured-grid face names, indexed by (axis, side); see
# gdtk src/geom/elements/nomenclature.d (west/east = i-/+, south/north = j-/+,
# bottom/top = k-/+).
_FACE_NAMES: dict[tuple[int, int], str] = {
    (0, 0): "west",
    (0, 1): "east",
    (1, 0): "south",
    (1, 1): "north",
    (2, 0): "bottom",
    (2, 1): "top",
}

_EXT = {"gziptext": "gz", "rawbinary": "bin", "vtk": "vtk"}

_FORMAT_VERSION = "1.1"


def export_lmr(
    grid: MultiBlockGrid,
    out_dir: str | os.PathLike,
    fmt: str = "gziptext",
    write_grid_lua: bool = True,
    block_prefix: str = "block",
    grid_lua_instructions: bool | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Write ``grid`` as per-block lmr grid files plus a ``grid.lua`` stub.

    @param grid            Initialized (or solved) multiblock structured grid;
                           coordinates are taken from ``grid.global_nodes``.
    @param out_dir         Directory to write into (created if absent).  The
                           block files and ``grid.lua`` land here together so
                           ``grid.lua``'s relative filenames resolve when
                           ``lmr prep-grid`` runs in this directory.
    @param fmt             Grid-file container: ``"gziptext"`` (default),
                           ``"rawbinary"`` or ``"vtk"``.
    @param write_grid_lua  Also emit a ``grid.lua`` registration stub with
                           ``registerFluidGrid`` calls and
                           ``identifyGridConnections()``.
    @param block_prefix    Basename stem for the per-block files; block ``i``
                           is written as ``{block_prefix}-{i:04d}.{ext}``.
    @param grid_lua_instructions  Prepend the "how to run this in lmr" comment
                           block to ``grid.lua``.  ``None`` (default) consults
                           the ``export.lmr_grid_lua_instructions`` config flag
                           (itself defaulting to ``True``); pass a bool to force.
    @param overwrite       Allow writing into a directory that already holds an
                           lmr export.  ``False`` (default) raises
                           :class:`FileExistsError` rather than clobbering a
                           possibly hand-edited grid.

    @return The list of written file paths (block files, then ``grid.lua``).
    """
    if grid_lua_instructions is None:
        grid_lua_instructions = _instructions_enabled()
    if fmt not in _EXT:
        raise ValueError(f"Unknown lmr grid format {fmt!r}; expected one of {sorted(_EXT)}")

    d = grid.blocks[0].d if grid.blocks else 0
    if d not in (2, 3):
        raise ValueError(f"lmr export supports d=2 or d=3 grids, got d={d}")

    nodes = grid.global_nodes
    if np.any(np.isnan(nodes)):
        raise ValueError(
            "grid.global_nodes contains NaN — initialize the grid before export"
        )

    out_dir = os.fspath(out_dir)
    if not overwrite:
        signature = (
            os.path.join(out_dir, "grid.lua")
            if write_grid_lua
            else os.path.join(out_dir, f"{block_prefix}-0000.{_EXT[fmt]}")
        )
        if os.path.exists(signature):
            raise FileExistsError(
                f"{out_dir!r} already holds an lmr export "
                f"({os.path.basename(signature)}); pass overwrite=True to replace "
                f"it, or choose an empty directory"
            )
    os.makedirs(out_dir, exist_ok=True)

    block_names = list(grid.topology.block_specs.keys())
    ext = _EXT[fmt]
    written: list[str] = []
    filenames: list[str] = []

    for bi, dof_map in enumerate(grid.block_dof_maps):
        label = block_names[bi]
        coords = nodes[dof_map]  # shape == block logical shape + (d,)
        pts = _points_i_fastest(coords, d)
        fname = f"{block_prefix}-{bi:04d}.{ext}"
        path = os.path.join(out_dir, fname)
        dims = _block_dims(dof_map.shape)
        _write_block(path, fmt, label, d, dims, pts)
        written.append(path)
        filenames.append(fname)

    if write_grid_lua:
        lua_path = os.path.join(out_dir, "grid.lua")
        _write_grid_lua(
            lua_path, grid, d, fmt, block_names, filenames, grid_lua_instructions
        )
        written.append(lua_path)

    return written


def _instructions_enabled() -> bool:
    """The ``export.lmr_grid_lua_instructions`` config flag (default ``True``).

    Guarded: a missing/unreadable config falls back to ``True`` so the helpful
    default never depends on the config machinery being importable.
    """
    try:
        from egg.webui.config import load_config

        return bool(
            load_config().get("export", {}).get("lmr_grid_lua_instructions", True)
        )
    except Exception:
        return True


def _block_dims(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Map a block's logical node shape to gdtk (niv, njv, nkv)."""
    niv = shape[0]
    njv = shape[1] if len(shape) > 1 else 1
    nkv = shape[2] if len(shape) > 2 else 1
    return niv, njv, nkv


def _points_i_fastest(coords: np.ndarray, d: int) -> np.ndarray:
    """Flatten node coords to (N, 3) in gdtk i-fastest order, padding z=0 in 2D.

    ``coords`` is indexed ``[i, j(, k), comp]``.  gdtk iterates k outer, j
    middle, i inner (i fastest), so we move the i axis to be last-but-one and
    C-flatten.
    """
    if d == 2:
        flat = np.transpose(coords, (1, 0, 2)).reshape(-1, 2)
        z = np.zeros((flat.shape[0], 1), dtype=flat.dtype)
        return np.concatenate([flat, z], axis=1)
    flat = np.transpose(coords, (2, 1, 0, 3)).reshape(-1, 3)
    return flat


def _write_block(
    path: str,
    fmt: str,
    label: str,
    d: int,
    dims: tuple[int, int, int],
    pts: np.ndarray,
) -> None:
    if fmt == "gziptext":
        _write_gziptext(path, label, d, dims, pts)
    elif fmt == "rawbinary":
        _write_rawbinary(path, label, d, dims, pts)
    else:  # vtk
        _write_vtk(path, label, dims, pts)


def _write_gziptext(
    path: str, label: str, d: int, dims: tuple[int, int, int], pts: np.ndarray
) -> None:
    """Eilmer native text format (``structured_grid 1.1``), gzip-compressed."""
    niv, njv, nkv = dims
    lines = [
        f"structured_grid {_FORMAT_VERSION}\n",
        f"label: {label}\n",
        f"dimensions: {d}\n",
        f"niv: {niv}\n",
        f"njv: {njv}\n",
        f"nkv: {nkv}\n",
    ]
    for x, y, z in pts:
        lines.append(f"{x:.18e} {y:.18e} {z:.18e}\n")
    # bcTags are assigned at registration in grid.lua, not embedded here.
    lines.append("ntags: 0\n")
    with gzip.open(path, "wt", encoding="ascii") as f:
        f.write("".join(lines))


def _write_rawbinary(
    path: str, label: str, d: int, dims: tuple[int, int, int], pts: np.ndarray
) -> None:
    """Eilmer native raw-binary format: ASCII header, int32 sizes, f64 coords.

    Layout: ``"structured_grid 1.1"`` (19 bytes, no terminator), int32
    label length + label bytes, int32[4] {dimensions, niv, njv, nkv}, then
    niv*njv*nkv little-endian f64 triples, then int32 ntags (0 here).
    """
    niv, njv, nkv = dims
    label_bytes = label.encode("ascii")
    with open(path, "wb") as f:
        f.write(f"structured_grid {_FORMAT_VERSION}".encode("ascii"))
        f.write(struct.pack("<i", len(label_bytes)))
        f.write(label_bytes)
        f.write(struct.pack("<4i", d, niv, njv, nkv))
        np.ascontiguousarray(pts, dtype="<f8").tofile(f)
        f.write(struct.pack("<i", 0))  # ntags


def _write_vtk(
    path: str, label: str, dims: tuple[int, int, int], pts: np.ndarray
) -> None:
    """Legacy VTK ASCII structured grid."""
    niv, njv, nkv = dims
    lines = [
        "# vtk DataFile Version 2.0\n",
        f"{label}\n",
        "ASCII\n",
        "\n",
        "DATASET STRUCTURED_GRID\n",
        f"DIMENSIONS {niv} {njv} {nkv}\n",
        f"POINTS {niv * njv * nkv} float\n",
    ]
    for x, y, z in pts:
        lines.append(f"{x:.18e} {y:.18e} {z:.18e}\n")
    with open(path, "w") as f:
        f.write("".join(lines))


def _external_faces(
    topology: "BlockTopology", block_name: str, d: int
) -> list[tuple[int, int]]:
    """External (non-interface) faces of a block, as (axis, side) tuples."""
    shared = set()
    for conn in topology.interface_connections:
        for face in (conn.face_a, conn.face_b):
            shared.add((face.block_name, face.axis, face.side))
    faces = []
    for axis in range(d):
        for side in (0, 1):
            if (block_name, axis, side) not in shared:
                faces.append((axis, side))
    return faces


def _face_tag_map(topology: "BlockTopology") -> dict[tuple[str, int, int], str]:
    """Reverse the topology's boundary tags to (block, axis, side) -> tag name."""
    out: dict[tuple[str, int, int], str] = {}
    for name, faces in (getattr(topology, "boundary_tags", {}) or {}).items():
        for fs in faces:
            out[(fs.block_name, fs.axis, fs.side)] = name
    return out


def _untagged_assignment(
    grid: MultiBlockGrid, d: int
) -> tuple[dict[tuple[str, int, int], str], list[dict]]:
    """Assign an ``egg-untagged-N`` marker to each external face lacking a tag.

    An external face has no boundary tag when it is unassociated, or associated
    with a geometry entity whose ``tag`` is ``None`` (an unnamed curve, or one
    whose marker was explicitly suppressed). Grouping keeps related faces under
    one marker: faces sharing a tag-less geometry entity share a marker; faces
    with no associated geometry get one marker per block edge.

    Returns ``(face_tags, groups)`` where ``face_tags`` maps
    ``(block, axis, side)`` to its marker and ``groups`` is a per-marker report
    (``{"tag", "geometry", "faces"}``) for surfacing in the UI.
    """
    topo = grid.topology
    tag_map = _face_tag_map(topo)
    # First association per face wins; an untagged external face on geometry
    # means that entity's tag is None, and several faces on one entity (e.g. a
    # boundary split across blocks) should share a marker.
    assoc_entity: dict[tuple[str, int, int], object] = {}
    for assoc in getattr(topo, "associations", []):
        f = assoc.face
        assoc_entity.setdefault((f.block_name, f.axis, f.side), assoc.entity)

    group_idx: dict[tuple, int] = {}
    group_entity: dict[int, object] = {}
    group_faces: dict[int, list[tuple[str, int, int]]] = {}
    face_tags: dict[tuple[str, int, int], str] = {}
    for bname in topo.block_specs:
        for axis, side in _external_faces(topo, bname, d):
            key = (bname, axis, side)
            if key in tag_map:
                continue
            ent = assoc_entity.get(key)
            gkey = ("geom", id(ent)) if ent is not None else ("edge", key)
            if gkey not in group_idx:
                idx = len(group_idx)
                group_idx[gkey] = idx
                group_entity[idx] = ent
                group_faces[idx] = []
            idx = group_idx[gkey]
            face_tags[key] = f"{_UNTAGGED_PREFIX}{idx}"
            group_faces[idx].append(key)

    groups: list[dict] = []
    for idx in range(len(group_idx)):
        ent = group_entity[idx]
        name = getattr(ent, "name", None) if ent is not None else None
        if ent is None:
            geometry = None
        else:
            geometry = name if name is not None else type(ent).__name__
        groups.append(
            {
                "tag": f"{_UNTAGGED_PREFIX}{idx}",
                "geometry": geometry,
                "faces": [
                    {"block": b, "face": _FACE_NAMES[(a, s)], "axis": a, "side": s}
                    for (b, a, s) in group_faces[idx]
                ],
            }
        )
    return face_tags, groups


def untagged_external_faces(grid: MultiBlockGrid) -> list[dict]:
    """The ``egg-untagged-N`` marker groups :func:`export_lmr` assigns, for UI
    reporting. Empty when every external face already carries a boundary tag.

    Each entry is ``{"tag", "geometry", "faces"}``: ``geometry`` is the tag-less
    entity's name (or type when unnamed), or ``None`` for a bare block edge;
    ``faces`` lists ``{"block", "face", "axis", "side"}`` members of the group.
    """
    d = grid.blocks[0].d if grid.blocks else 0
    if d not in (2, 3):
        return []
    _face_tags, groups = _untagged_assignment(grid, d)
    return groups


def _grid_lua_instruction_lines(groups: list[dict]) -> list[str]:
    """Terse numbered 'how to run in lmr' comment block for the grid.lua head.

    A leading step naming the ``egg-untagged-*`` placeholders is emitted only
    when there are untagged faces; without any, that step is dropped and the
    remaining steps renumber.
    """
    steps: list[list[str]] = []
    if groups:
        names = ", ".join(g["tag"] for g in groups)
        steps.append(
            textwrap.wrap(
                "bcTags below include untagged boundaries, written as the "
                f"placeholders {names}. Replace each with a real boundary name "
                "(every tag needs a matching entry in the sim bcDict).",
                width=70,
                break_on_hyphens=False,
                break_long_words=False,
            )
        )
    steps.append(["Gas:  lmr prep-gas -i <gas>.lua -o <gas>.gas"])
    steps.append(
        [
            "Sim file: setGasModel, flowDict{initial=...}",
            "(matches fsTag), bcDict{<tag>=...}, makeFluidBlocks(bcDict, flowDict),",
            "then the solver config.",
        ]
    )
    steps.append(
        [
            "Prep+run:  lmr prep-grid --job=grid.lua ; "
            "lmr prep-sim --job=<sim>.lua ; lmr run"
        ]
    )
    out = ["--\n", "-- To run this grid in lmr:\n"]
    for i, block in enumerate(steps, 1):
        first, *rest = block
        out.append(f"--   {i}. {first}\n")
        out.extend(f"--      {line}\n" for line in rest)
    out.append("--\n")
    return out


def _write_grid_lua(
    path: str,
    grid: MultiBlockGrid,
    d: int,
    fmt: str,
    block_names: list[str],
    filenames: list[str],
    instructions: bool = True,
) -> None:
    """Emit a prep-grid registration stub, modeled on gdtk's sg-minimal example.

    External block faces are pre-listed in each ``bcTags`` table using the
    topology's boundary tag where one exists, else an auto-assigned
    ``egg-untagged-N`` marker (faces on the same geometry share one; faces with
    no geometry get one per block edge) that the user maps in their sim bcDict.
    When ``instructions`` is set, a terse "how to run this in lmr" comment block
    is prepended.
    """
    tag_map = _face_tag_map(grid.topology)
    face_tags, _groups = _untagged_assignment(grid, d)
    lines = ["-- grid.lua (generated by egg)\n"]
    if instructions:
        lines += _grid_lua_instruction_lines(_groups)
    lines += [
        "-- prep-grid registers the blocks below and detects the interfaces.\n",
        f"config.dimensions = {d}\n",
        "\n",
    ]
    for bi, (name, fname) in enumerate(zip(block_names, filenames)):
        bc_entries = []
        for axis, side in _external_faces(grid.topology, name, d):
            face = _FACE_NAMES[(axis, side)]
            key = (name, axis, side)
            value = tag_map.get(key) or face_tags.get(key, f"{_UNTAGGED_PREFIX}0")
            bc_entries.append(f'{face}="{value}"')
        bc_str = ", ".join(bc_entries)
        lines.append(
            f"grid{bi} = registerFluidGrid{{\n"
            f'   grid=StructuredGrid:new{{filename="{fname}", fmt="{fmt}"}},\n'
            f'   fsTag="initial",\n'
            f"   bcTags={{{bc_str}}}\n"
            f"}}\n"
        )
    lines.append("identifyGridConnections()\n")
    with open(path, "w") as f:
        f.write("".join(lines))
