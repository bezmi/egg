# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""3D wireframe to hex blocks by cube enumeration (the 3D lift of trace.py).

The 2D tracer walks a planar graph's rotation system into quad faces. In 3D a
node's neighbours have no canonical cyclic order, so instead of tracing faces we
enumerate hexes directly: every set of 8 nodes whose induced edges form the cube
graph Q3 is a block (the GridPro-style "draw the wireframe, infer the blocks").
This handles singular fans, where face-tracing has no well-defined next step.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product

import numpy as np

from .trace import Diagnostic

__all__ = ["infer_blocks", "block_edges"]


def block_edges(corners):
    """The 12 edges (as name pairs) of a hex given 8 corners in product order."""
    idx = list(product((0, 1), repeat=3))
    pos: dict[tuple[int, ...], int] = {ci: i for i, ci in enumerate(idx)}
    edges = []
    for a, ci in enumerate(idx):
        for axis in range(3):
            nb = [int(x) for x in ci]
            nb[axis] ^= 1
            b = pos[tuple(nb)]
            if a < b:
                edges.append((corners[a], corners[b]))
    return edges


def _contains_centroid(block_pos, p) -> bool:
    """True when point ``p`` lies *strictly* inside the convex hex given by 8
    corner positions (product order): strictly behind all six outward-oriented
    face planes. A point on a shared face is not contained, so sibling sub-hexes
    do not each claim the parent's centroid."""
    idx = list(product((0, 1), repeat=3))
    hc = np.mean(block_pos, axis=0)
    for axis in range(3):
        for side in (0, 1):
            fc = np.array(
                [block_pos[i] for i, ii in enumerate(idx) if ii[axis] == side]
            )
            fcc = fc.mean(0)
            outn = fcc - hc
            if float(np.dot(p - fcc, outn)) > -1e-9 * (
                float(np.linalg.norm(outn)) + 1.0
            ):
                return False
    return True


def _drop_nested(blocks, pos):
    """Drop any block that contains another block's centroid.

    A conforming decomposition has no block nested in another except a
    subdivision parent enclosing its children; dropping the parent keeps the
    finer (drawn) blocking. A no-op when nothing is nested.
    """
    cent = [np.mean([np.asarray(pos[c], float) for c in blk], axis=0) for blk in blocks]
    kept = []
    for i, blk in enumerate(blocks):
        bp = [np.asarray(pos[c], float) for c in blk]
        if not any(
            i != j and _contains_centroid(bp, cent[j]) for j in range(len(blocks))
        ):
            kept.append(blk)
    return kept


def _all_faces_bounded(block, node_on) -> bool:
    """True when every face's 4 corners share a bound surface (a void cell).

    A real block has interior faces shared with neighbours, whose corners span
    more than one surface; a cube enclosing a cavity (e.g. the sphere corners of
    an O-shell) has all six faces on boundary surfaces.
    """
    idx = list(product((0, 1), repeat=3))
    for axis in range(3):
        for side in (0, 1):
            fc = [block[i] for i, ii in enumerate(idx) if ii[axis] == side]
            common = set.intersection(*(node_on.get(c, set()) for c in fc))
            if not common:
                return False
    return True


def infer_blocks(pos, edges, node_on=None):
    """Enumerate hex blocks (Q3 cubes) in a node+edge wireframe.

    ``pos`` maps node name to a length-3 position; ``edges`` is an iterable of
    ``(name, name)`` pairs. ``node_on`` (optional) maps a node to the set of
    surfaces it lies on; when given, a candidate cube whose every face is
    surface-bound is dropped as a void (an edge wireframe cannot otherwise tell a
    meshed block from a surface-enclosed cavity). Returns ``(blocks,
    diagnostics)`` where each block is a list of 8 corner names in
    ``product((0,1), repeat=3)`` order, oriented right-handed (positive Jacobian).
    """
    node_on = node_on or {}
    diags: list[Diagnostic] = []
    adj: dict = defaultdict(set)
    eset: set = set()
    for i, j in edges:
        if i == j:
            continue
        adj[i].add(j)
        adj[j].add(i)
        eset.add(frozenset((i, j)))

    def has(i, j) -> bool:
        return frozenset((i, j)) in eset

    seen: set = set()
    blocks: list = []
    pruned = 0
    for v0 in adj:
        for a, b, c in combinations(sorted(adj[v0]), 3):
            for nab in (adj[a] & adj[b]) - {v0}:
                for nac in (adj[a] & adj[c]) - {v0}:
                    for nbc in (adj[b] & adj[c]) - {v0}:
                        if len({v0, a, b, c, nab, nac, nbc}) != 7:
                            continue
                        for nabc in (adj[nab] & adj[nac] & adj[nbc]) - {a, b, c}:
                            cset = frozenset({v0, a, b, c, nab, nac, nbc, nabc})
                            if len(cset) != 8 or cset in seen:
                                continue
                            need = [
                                (v0, a),
                                (v0, b),
                                (v0, c),
                                (a, nab),
                                (a, nac),
                                (b, nab),
                                (b, nbc),
                                (c, nac),
                                (c, nbc),
                                (nab, nabc),
                                (nac, nabc),
                                (nbc, nabc),
                            ]
                            if not all(has(x, y) for x, y in need):
                                continue
                            seen.add(cset)
                            block = _ordered(pos, v0, a, b, c, nab, nac, nbc, nabc)
                            if node_on and _all_faces_bounded(block, node_on):
                                pruned += 1
                                continue
                            blocks.append(block)

    if pruned:
        diags.append(
            Diagnostic(
                "warn_pruned_void",
                f"dropped {pruned} surface-enclosed cube(s) as void cavities",
            )
        )
    # Resolve subdivision: a cut base hex and its sub-hexes are both Q3 cubes;
    # keep the finer ones by dropping any hex enclosing another's centroid.
    blocks = _drop_nested(blocks, pos)
    if not blocks:
        diags.append(Diagnostic("no_blocks", "no hex blocks found in the wireframe"))
    return blocks, diags


def _ordered(pos, v0, a, b, c, nab, nac, nbc, nabc):
    """Corners in product order, with the two tangential axes swapped when the
    (a, b, c) frame is left-handed so the block Jacobian is positive."""
    e0 = np.asarray(pos[a], float) - np.asarray(pos[v0], float)
    e1 = np.asarray(pos[b], float) - np.asarray(pos[v0], float)
    e2 = np.asarray(pos[c], float) - np.asarray(pos[v0], float)
    if float(np.dot(np.cross(e0, e1), e2)) < 0.0:
        return [v0, b, c, nbc, a, nab, nac, nabc]
    return [v0, c, b, nbc, a, nac, nab, nabc]
