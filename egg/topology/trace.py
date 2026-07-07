# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Wireframe -> block topology, reported rather than raised.

A planar straight-line graph (welded corner nodes + edges) is traced into
quad block faces and emitted as a :class:`~egg.topology.builder.TopologyBuilder`.
Anything that stops a valid topology from forming — a non-quad face, no
edges, no blocks — is returned as a :class:`Diagnostic` instead of raising,
so a caller can render the valid blocks and flag the rest, or treat any
diagnostic as an error (as :func:`egg.geometry.svg_topology` does).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from egg.geometry.frontend2d import Vector3

from .builder import TopologyBuilder

__all__ = ["Diagnostic", "trace_topology"]


@dataclass
class Diagnostic:
    """A reason the wireframe does not (yet) flatten to a valid topology.

    ``where`` names the offending node ids and ``xy`` a representative world
    position, so a UI can localise the problem; ``msg`` is front-end neutral.
    """

    kind: str  # no_edges | non_quad_face | no_blocks | ...
    msg: str
    where: tuple[int, ...] = ()
    xy: tuple[float, float] | None = None


# Face edge k -> (axis, side): S, E, N, W of a CCW [sw, se, ne, nw] quad.
_EDGE_FACE = [(1, 0), (0, 1), (1, 1), (0, 0)]


def _seg_has(p, a, b, tol: float = 1e-7) -> bool:
    """Is point ``p`` on the segment ``a``–``b`` (within ``tol``)?"""
    ab = np.asarray(b) - np.asarray(a)
    ap = np.asarray(p) - np.asarray(a)
    lab2 = float(ab.dot(ab))
    if lab2 < 1e-18:
        return bool(np.linalg.norm(ap) < tol)
    t = float(ap.dot(ab) / lab2)
    if t < -tol or t > 1 + tol:
        return False
    return bool(np.linalg.norm(ap - t * ab) < tol)


def _propagate_loop_res(b, edge_res, user_res, default, n2i) -> None:
    """Make resolution consistent around every structured loop: opposite faces of
    a block share a resolution, and a shared face links two blocks — so a per-edge
    resolution (``user_res``, set explicitly) or a base resolution drives its whole
    loop. Union-find over the blocks, then re-assign each block's (n0, n1)."""
    from dataclasses import replace

    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, c):
        parent[find(a)] = find(c)

    def ek(names):
        idx = [n2i[nm] for nm in names if nm in n2i]
        return frozenset(idx) if len(idx) == 2 else None

    faces: dict = {}  # block name -> (south, north, west, east) edge keys
    for name, spec in b._block_specs.items():
        s, n = ek(spec.face_corner_names(1, 0, 2)), ek(spec.face_corner_names(1, 1, 2))
        w, e = ek(spec.face_corner_names(0, 0, 2)), ek(spec.face_corner_names(0, 1, 2))
        faces[name] = (s, n, w, e)
        if s and n:
            union(s, n)  # both span axis-0 -> share n0
        if w and e:
            union(w, e)  # both span axis-1 -> share n1
    cu: dict = {}  # class root -> user resolution (explicit, wins)
    cb: dict = {}  # class root -> base / inherited resolution
    for s, n, w, e in faces.values():
        for k in (s, n, w, e):
            if k is None:
                continue
            r = find(k)
            if k in user_res:
                cu[r] = max(cu.get(r, 0), user_res[k])
            if k in edge_res:
                cb[r] = max(cb.get(r, 0), edge_res[k])

    def res_of(k):
        if k is None:
            return default
        r = find(k)
        return cu.get(r) or cb.get(r) or default

    for name, (s, _n, w, _e) in faces.items():
        b._block_specs[name] = replace(
            b._block_specs[name], resolutions=(res_of(s), res_of(w))
        )


def _point_in_poly(pt, poly) -> bool:
    """Ray-cast point-in-polygon for a small ring of (x, y) vertices."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xc = x0 + (y - y0) / (y1 - y0) * (x1 - x0)
            if x < xc:
                inside = not inside
    return inside


def _loop_signed_area(loop: Sequence[int], pos: Sequence[np.ndarray]) -> float:
    s = 0.0
    n = len(loop)
    for k in range(n):
        x1, y1 = pos[loop[k]]
        x2, y2 = pos[loop[(k + 1) % n]]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def trace_topology(
    nodes: Sequence[np.ndarray],
    edges: Iterable[tuple[int, int]],
    geometry: Sequence[tuple[str, Any, bool]] = (),
    *,
    res: int = 10,
    snap_tol: float | None = None,
    declared_curves: dict[int, set[str]] | None = None,
    node_names: Sequence[str] | None = None,
    seed_builder: TopologyBuilder | None = None,
    base_edges: set[frozenset[int]] | None = None,
    fixed_nodes: set[int] | None = None,
    edge_res: dict[frozenset[int], int] | None = None,
    block_regions: list[tuple[str, list]] | None = None,
    face_markers: list[tuple[int, int, str]] | None = None,
    user_res: dict[frozenset[int], int] | None = None,
) -> tuple[TopologyBuilder, list[Diagnostic]]:
    """Trace welded nodes+edges into quad blocks; report problems, don't raise.

    Parameters
    ----------
    nodes
        Welded corner positions, indexed ``0..len(nodes)-1``.
    edges
        Unordered ``(i, j)`` node-index pairs (self-loops ignored).
    geometry
        ``(label, entity, closed)`` per boundary curve. An edge whose *both*
        ends lie within ``snap_tol`` of a curve is associated and tagged; a
        corner on two or more curves is pinned; the outer face and any
        closed-curve interior are dropped.
    res
        Uniform cells-per-axis for every block (keeps interfaces conforming).
    snap_tol
        Curve-proximity distance; defaults to ``1e-2`` of the node bbox
        diagonal.
    declared_curves
        Node index -> curve labels the node is *declared* to follow, merged
        with the proximity result. Lets a schematic (non-coincident) corner
        bind to a curve it is not drawn on top of.
    node_names
        Name per node index; base corners keep their names so new blocks
        connect to them. Defaults to ``n{index}``.
    seed_builder
        Start from this builder (a base copy) instead of a fresh one; its
        corners/blocks are kept and new blocks get non-clashing names.
    base_edges
        Edges (unordered index pairs) owned by the seed. A face bounded
        *entirely* by them is the base's own — a block or an intentional hole —
        and is left to the seed; only faces that use at least one new edge are
        emitted.

    Returns
    -------
    (TopologyBuilder, list[Diagnostic])
        The populated builder — partial when diagnostics are non-empty — and
        the diagnostics. An empty list means a fully valid topology.
    """
    pos = [np.asarray(p, dtype=float)[:2] for p in nodes]
    edge_set = {(min(i, j), max(i, j)) for i, j in edges if i != j}
    diags: list[Diagnostic] = []
    b = seed_builder if seed_builder is not None else TopologyBuilder(d=2)
    if not edge_set:
        diags.append(Diagnostic("no_edges", "wireframe has no non-degenerate edges"))
        return b, diags

    if snap_tol is None:
        allp = np.array(pos)
        diag = float(np.linalg.norm(allp.max(0) - allp.min(0))) or 1.0
        snap = 1e-2 * diag
    else:
        snap = snap_tol

    # --- angular adjacency for planar-face tracing -----------------------
    adj: dict[int, list[int]] = {i: [] for i in range(len(pos))}
    for i, j in edge_set:
        adj[i].append(j)
        adj[j].append(i)
    order = {
        i: sorted(
            nb, key=lambda j: math.atan2(pos[j][1] - pos[i][1], pos[j][0] - pos[i][0])
        )
        for i, nb in adj.items()
    }

    def next_dart(u: int, v: int) -> tuple[int, int]:
        # Arriving at v from u, turn to the neighbour clockwise-adjacent to
        # the way back (predecessor of u in v's CCW order): traces one face.
        ov = order[v]
        w = ov[(ov.index(u) - 1) % len(ov)]
        return v, w

    visited: set[tuple[int, int]] = set()
    faces: list[list[int]] = []
    for i, j in edge_set:
        for u, v in ((i, j), (j, i)):
            if (u, v) in visited:
                continue
            loop: list[int] = []
            a, c = u, v
            while (a, c) not in visited:
                visited.add((a, c))
                loop.append(a)
                a, c = next_dart(a, c)
            faces.append(loop)

    # --- geometry association helpers ------------------------------------
    def curves_on(p: np.ndarray) -> set[str]:
        hits = set()
        for lbl, ent, _ in geometry:
            q = np.asarray(ent.project(p), dtype=float)[:2]
            if np.linalg.norm(p - q) <= snap:
                hits.add(lbl)
        return hits

    node_curves = {i: curves_on(pos[i]) for i in range(len(pos))}
    if declared_curves:
        for i, labels in declared_curves.items():
            if 0 <= i < len(pos):
                node_curves[i] |= set(labels)
    ent_by_label = {lbl: ent for lbl, ent, _ in geometry}
    closed_by_label = {lbl: cl for lbl, _ent, cl in geometry}
    # carry the label onto the entity so the built topology's `entities` map
    # (and, downstream, rendering/export) sees the same names the blocking binds by
    for lbl, ent in ent_by_label.items():
        if getattr(ent, "name", None) is None:
            try:
                ent.name = lbl
            except AttributeError:
                pass

    # Advisory: a node drawn strictly inside a closed curve. A closed curve is a
    # hole the mesh wraps around, so a node inside it seeds a tangled start (block
    # interiors interpolate into the hole and the untangler may not recover — even
    # a boundary node that will project back onto the curve is drawn from a bad
    # position). Nodes within `snap` of the curve slide on it and are fine.
    for lbl, ent, closed in geometry:
        if not closed:
            continue
        try:
            poly = [
                np.asarray(ent.eval_frac(t), dtype=float)[:2]
                for t in np.linspace(0.0, 1.0, 128)
            ]
        except Exception:
            continue
        for i, p in enumerate(pos):
            q = np.asarray(ent.project(p), dtype=float)[:2]
            if float(np.linalg.norm(p - q)) <= snap:
                continue  # on the curve — slides on it, not trapped inside
            if _point_in_poly(p, poly):
                nm = node_names[i] if node_names is not None else f"n{i}"
                diags.append(
                    Diagnostic(
                        "warn_inside_closed_curve",
                        f"node {nm!r} lies inside the closed curve {lbl!r}; the mesh "
                        "wraps around it, so this seeds a tangled start",
                        where=(i,),
                        xy=(float(p[0]), float(p[1])),
                    )
                )

    def edge_curve(i: int, j: int) -> str | None:
        # A block edge follows a curve iff both its ends lie on that curve.
        common = node_curves[i] & node_curves[j]
        if not common:
            return None
        if len(common) == 1:
            return next(iter(common))
        mid = 0.5 * (pos[i] + pos[j])
        return min(
            common,
            key=lambda lbl: np.linalg.norm(
                mid - np.asarray(ent_by_label[lbl].project(mid), dtype=float)[:2]
            ),
        )

    # --- classify faces: drop the outer face and closed-curve interiors --
    outer = max(range(len(faces)), key=lambda k: abs(_loop_signed_area(faces[k], pos)))
    owned = base_edges or set()

    def _base_owned(loop: list[int]) -> bool:
        # every edge belongs to the seed -> its face (block or hole) is the
        # base's, not the blocking's; leave it be (checked before the quad rule
        # so a base hole with != 4 sides is not flagged).
        return bool(owned) and all(
            frozenset((loop[m], loop[(m + 1) % len(loop)])) in owned
            for m in range(len(loop))
        )

    blocks: list[list[int]] = []
    for k, loop in enumerate(faces):
        if k == outer:
            continue
        if _base_owned(loop):
            continue
        labels = [
            edge_curve(loop[m], loop[(m + 1) % len(loop)]) for m in range(len(loop))
        ]
        if (
            labels
            and all(x is not None and x == labels[0] for x in labels)
            and (closed_by_label.get(labels[0], False))
        ):
            continue  # a region we mesh around (e.g. the egg interior)
        if len(loop) != 4:
            pts = ", ".join(f"({pos[n][0]:.3g},{pos[n][1]:.3g})" for n in loop)
            cx = float(np.mean([pos[n][0] for n in loop]))
            cy = float(np.mean([pos[n][1] for n in loop]))
            diags.append(
                Diagnostic(
                    "non_quad_face",
                    f"face with {len(loop)} sides is not a quad and not a "
                    f"closed-curve interior — corners {pts}. Split it into quad "
                    f"blocks, or check for a missing/extra edge.",
                    where=tuple(loop),
                    xy=(cx, cy),
                )
            )
            continue
        blocks.append(loop)

    # With a seed, the base owns the blocks; only a from-scratch trace that
    # finds nothing is empty.
    if not blocks and seed_builder is None:
        diags.append(Diagnostic("no_blocks", "no block faces found in the wireframe"))
        return b, diags

    # --- emit corners then blocks (base blocks stay in the seed) ----------
    new_blocks = blocks

    def corner_name(n: int) -> str:
        return node_names[n] if node_names is not None else f"n{n}"

    used = sorted({n for loop in new_blocks for n in loop})
    for n in used:
        nm = corner_name(n)
        if nm not in b._corners:  # base corners are already in the seed
            # >=2 distinct curves meet here -> a domain corner: pin it.
            fixed = len(node_curves[n]) >= 2 or bool(fixed_nodes and n in fixed_nodes)
            b.add_corner(nm, Vector3(pos[n][0], pos[n][1]), fixed=fixed)

    counter = 0
    for loop in new_blocks:
        if _loop_signed_area(loop, pos) < 0:
            loop = loop[::-1]
        sw, se, ne, nw = (corner_name(n) for n in loop)
        # a block re-traced inside a retired base block inherits its name with a
        # suffix (…_0, …_1); otherwise a fresh blkN
        name = None
        if block_regions:
            cx = sum(pos[n][0] for n in loop) / len(loop)
            cy = sum(pos[n][1] for n in loop) / len(loop)
            for rname, quad in block_regions:
                if _point_in_poly((cx, cy), quad):
                    k = sum(1 for nm in b._block_specs if nm.startswith(rname + "_"))
                    name = f"{rname}_{k}"
                    break
        if name is None:
            while f"blk{counter}" in b._block_specs:
                counter += 1
            name = f"blk{counter}"
            counter += 1
        # per-axis cell counts from the edge-res map when known (so a re-blocked
        # base block keeps its resolution), else the uniform default. axis-0
        # rides the south/north edges, axis-1 the west/east edges.
        er = edge_res or {}
        n0 = er.get(frozenset((loop[0], loop[1]))) or er.get(
            frozenset((loop[2], loop[3]))
        )
        n1 = er.get(frozenset((loop[3], loop[0]))) or er.get(
            frozenset((loop[1], loop[2]))
        )
        b.add_block(name, sw=sw, se=se, ne=ne, nw=nw, res=(n0 or res, n1 or res))
        for m in range(4):
            u, v = loop[m], loop[(m + 1) % 4]
            lbl = edge_curve(u, v)
            # a sub-face lying on a retired block's marked face inherits that
            # block's marker (which need not match a geometry label), so
            # re-blocking never silently drops a programmatic boundary marker
            marker = None
            for ma, mb, mk in face_markers or ():
                if _seg_has(pos[u], pos[ma], pos[mb]) and _seg_has(
                    pos[v], pos[ma], pos[mb]
                ):
                    marker = mk
                    break
            axis, side = _EDGE_FACE[m]
            if lbl is not None:
                b.associate(name, axis, side, ent_by_label[lbl])
                # the marker auto-derives from the associated entity's tag
                # (which defaults to its name) in build(); a retired base
                # block's own marker still overrides it here
                if marker is not None:
                    b.tag_boundary(marker, name, axis, side)
            elif marker is not None:
                b.tag_boundary(marker, name, axis, side)

    # once all blocks exist, make each resolution loop consistent (and honour any
    # explicitly-set per-edge resolution) — a no-op for a plain uniform-res trace
    if (edge_res or user_res) and node_names is not None:
        _propagate_loop_res(
            b,
            edge_res or {},
            user_res or {},
            res,
            {nm: i for i, nm in enumerate(node_names)},
        )

    return b, diags
