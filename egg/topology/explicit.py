# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Explicit-connectivity topology: geometry-by-name plus a drawable blocking.

An :class:`ExplicitTopology` defines a topology from an explicit ``connectivity``
blocking structure (nodes/edges), rather than the builder's programmatic
inference. It also
carries geometry (keyed by name) and an optional frozen base topology.
:meth:`ExplicitTopology.flatten` compiles the merged graph into a
:class:`~egg.topology.block_topology.BlockTopology` through the shared wireframe
tracer, returning problems as diagnostics rather than raising;
:meth:`ExplicitTopology.build` raises on the first diagnostic.

Editing is orthogonal: wrap the ``connectivity`` value in :func:`editable` to opt
the literal into the web UI's topology editor. Unwrapped, it is read-only, and
the class behaves identically headless either way.

The blocking is a plain dict so it round-trips as a source literal::

    {
        "nodes": {"a": {"xy": [0, 0]}, "b": {"xy": [1, 0], "on": ["wall"]}},
        "edges": [{"a": "a", "b": "b", "bind": "wall"}],
        "res": 10,
    }

``editable(...)`` wraps a literal to opt it into UI editing; at runtime it is
the identity, so a wrapped script builds the same headless.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .block_topology import ParallelWalkWarning
from .builder import TopologyBuilder
from .trace import Diagnostic, trace_topology

__all__ = ["ExplicitTopology", "editable"]


def editable(value, *, choices=None, label=None):
    """Opt a literal into interactive editing in the egg web UI.

    Identity at runtime (returns ``value`` unchanged); the web UI detects the
    call in the source and surfaces the wrapped literal — a scalar in the
    run-parameters panel, or a blocking structure in the topology edit view.
    A no-op headless, so a wrapped script builds identically under the CLI.
    """
    return value


def _lit(v) -> str:
    """One-line Python literal for a printed blocking (dicts/lists inline,
    booleans as ``True``/``False`` — json.dumps would emit ``true``)."""
    import json

    if v is True:
        return "True"
    if v is False:
        return "False"
    if v is None:
        return "None"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_lit(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_lit(str(k))}: {_lit(x)}" for k, x in v.items()) + "}"
    return repr(v)


def _is_closed(entity) -> bool:
    for attr in ("closed", "periodic", "is_closed"):
        if getattr(entity, attr, None) is True:
            return True
    # geometric fallback: a curve whose ends coincide bounds an interior
    try:
        a = np.asarray(entity.eval_frac(0.0), dtype=float)[:2]
        b = np.asarray(entity.eval_frac(1.0), dtype=float)[:2]
        return bool(np.linalg.norm(a - b) < 1e-9)
    except Exception:
        return False


def _point_in_quad(pt, poly) -> bool:
    """Ray-cast point-in-polygon for a small ring of (x, y) vertices."""
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = float(poly[i][0]), float(poly[i][1])
        x1, y1 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) / (y1 - y0) * (x1 - x0):
            inside = not inside
    return inside


def _base_parts(base):
    """Corners, block specs, and a fresh seed builder copied from a
    TopologyBuilder or BlockTopology (the original is never mutated)."""
    if isinstance(base, TopologyBuilder):
        corners, specs = base._corners, base._block_specs
        assoc, conns, tags = base._associations, base._connections, base._boundary_tags
        bl, bl_ents = base._boundary_layer_specs, base._bl_entities
        cids, cobjs = base._corner_ids, base._corner_objs
    else:  # BlockTopology
        corners, specs = base.corners, base.block_specs
        assoc, conns, tags = (
            base.associations,
            base.interface_connections,
            base.boundary_tags,
        )
        bl, bl_ents, cids, cobjs = base.boundary_layer_specs, {}, {}, {}
    tb = TopologyBuilder(d=2)
    tb._corners = dict(corners)
    tb._block_specs = dict(specs)
    tb._associations = list(assoc)
    tb._connections = list(conns)
    tb._boundary_tags = {k: list(v) for k, v in tags.items()}
    tb._boundary_layer_specs = dict(bl)
    tb._bl_entities = dict(bl_ents)
    tb._corner_ids = dict(cids)
    tb._corner_objs = dict(cobjs)
    return corners, specs, tb


class ExplicitTopology:
    """A base topology plus a drawable ``connectivity`` overlay.

    Parameters
    ----------
    base : TopologyBuilder or BlockTopology, optional
        Programmatic topology the overlay extends. Kept verbatim (blocks,
        per-axis ``res``, associations, tags); the blocking only appends new
        faces. A blocking edge may name a base corner to attach to it.
    geometry : dict, optional
        ``{name: entity}`` the blocking binds to by name.
    connectivity : dict, optional
        The blocking structure; wrap it with :func:`editable` to opt into UI
        editing. In 2D: ``nodes`` / ``edges`` / ``res`` (traced into quads). In
        3D: ``nodes`` (``{xyz, on}``) + ``blocks`` (each ``{corners: [8 ids in
        product order], res}``); a boundary face whose four corners share a
        bound surface (via each node's ``on``) rides it.
    relax_orthogonality : sequence, optional
        Domain boundaries whose orthogonality requirements are relaxed —
        geometry *labels* (or entities/Edges), passed to
        :meth:`~egg.topology.builder.TopologyBuilder.relax_orthogonality`
        on flatten and carried on the built topology.
    fan_frames : dict, optional
        Frames for singular fan nodes:
        ``{node: {"through": [a, b], "normal": c}}`` with legs named by
        their far node. The same section may live inside ``connectivity``
        instead (where the web UI round-trips it); declaring one fan in
        both places is an error. Passed to
        :meth:`~egg.topology.builder.TopologyBuilder.fan_frame` on flatten
        and carried resolved on the built topology.
    parallel : sequence, optional
        Chains of block edges held loosely parallel to a boundary:
        ``[{"to": label, "chain": [c0, c1, ...], "weight"?, "taper"?}, ...]``
        with ``to`` a geometry label (or entity/Edge) and the chain an
        ordered node list whose consecutive pairs are block edges. The same
        section may live inside ``connectivity`` (where the web UI
        round-trips it); both are applied. Passed to
        :meth:`~egg.topology.builder.TopologyBuilder.parallel_to` on flatten
        and carried resolved on the built topology.
    d : int
        Spatial dimension (2 or 3).
    """

    def __init__(
        self,
        *,
        base: Any = None,
        geometry: dict[str, Any] | None = None,
        connectivity: dict | None = None,
        relax_orthogonality: tuple = (),
        fan_frames: dict | None = None,
        parallel: tuple | list | None = None,
        d: int = 2,
    ):
        if d not in (2, 3):
            raise ValueError("ExplicitTopology supports d=2 or d=3")
        self.base = base
        self.geometry = dict(geometry or {})
        self.connectivity = dict(connectivity or {})
        self.relax_orthogonality = tuple(relax_orthogonality)
        self.fan_frames = dict(fan_frames or {})
        self.parallel = list(parallel or [])
        self.d = d

    def _gents(self) -> list[tuple[str, Any, bool]]:
        return [(lbl, ent, _is_closed(ent)) for lbl, ent in self.geometry.items()]

    def _relax_entries(self) -> list:
        """relax_orthogonality entries with geometry *labels* resolved to their
        entities; anything else passes through to the builder's resolution
        (entity names, Edges, raw entities)."""
        return [
            self.geometry.get(o, o) if isinstance(o, str) else o
            for o in self.relax_orthogonality
        ]

    def _fan_frame_decls(self, diags: list[Diagnostic]) -> list[dict]:
        """The kwarg and connectivity fan-frame declarations merged into
        builder call kwargs; malformed or doubly-declared entries become
        diagnostics (blocking the flatten) instead of raising."""
        section = self.connectivity.get("fan_frames") or {}
        both = {str(k) for k in section} & {str(k) for k in self.fan_frames}
        if both:
            diags.append(
                Diagnostic(
                    "fan_frame_conflict",
                    f"fan_frames for {sorted(both)} declared both as a "
                    "kwarg and in the connectivity; declare each fan once",
                )
            )
        out: list[dict] = []
        for src in (self.fan_frames, section):
            for node, spec in src.items():
                spec = spec or {}
                thr = spec.get("through")
                nrm = spec.get("normal")
                if not isinstance(thr, (list, tuple)) or len(thr) != 2 or not nrm:
                    diags.append(
                        Diagnostic(
                            "bad_fan_frame",
                            f"fan_frame {node!r} needs 'through': [a, b] "
                            "and 'normal': c",
                        )
                    )
                    continue
                out.append(
                    dict(
                        corner=str(node),
                        through=(str(thr[0]), str(thr[1])),
                        normal=str(nrm),
                    )
                )
        return out

    def _parallel_decls(self, diags: list[Diagnostic]) -> list[dict]:
        """The kwarg and connectivity parallel-chain declarations merged into
        builder call kwargs; malformed entries become diagnostics (blocking
        the flatten) instead of raising. Geometry labels resolve to their
        entities; other names pass through to the builder's resolution."""
        out: list[dict] = []
        for entry in list(self.parallel) + list(
            self.connectivity.get("parallel") or []
        ):
            entry = entry or {}
            to = entry.get("to")
            chain = entry.get("chain")
            if not to or not isinstance(chain, (list, tuple)) or len(chain) < 2:
                diags.append(
                    Diagnostic(
                        "bad_parallel",
                        "parallel entry needs 'to': <boundary> and "
                        "'chain': [c0, c1, ...]",
                    )
                )
                continue
            out.append(
                dict(
                    to=self.geometry.get(to, to) if isinstance(to, str) else to,
                    chain=tuple(str(c) for c in chain),
                    weight=float(entry.get("weight", 1.0)),
                    taper=(
                        None if entry.get("taper") is None else float(entry["taper"])
                    ),
                )
            )
        return out

    @staticmethod
    def _build_with_warn_diags(build, diags: list[Diagnostic]):
        """Run ``build()`` converting :class:`ParallelWalkWarning` into
        advisory ``warn_parallel_projection`` diagnostics; anything else
        re-emits unchanged."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            topo = build()
        for w in caught:
            if issubclass(w.category, ParallelWalkWarning):
                diags.append(Diagnostic("warn_parallel_projection", str(w.message)))
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        return topo

    def base_graph(self) -> dict:
        """Base corners (name -> ``{xy, fixed, on}``) and block-face edges
        (``[a, b, curve?]``), so the edit view can snap to / move / split against
        the programmatic base and report what each is bound to. Empty when there
        is no base."""
        if self.base is None:
            return {"nodes": {}, "edges": []}
        corners, specs, seed = _base_parts(self.base)
        # geometry a base face projects onto: association entity -> our label
        label_of = {id(ent): lbl for lbl, ent in self.geometry.items()}
        corner_on: dict = {name: set() for name in corners}
        edge_on: dict = {}
        for assoc in getattr(seed, "_associations", []):
            lbl = label_of.get(id(assoc.entity))
            spec = specs.get(assoc.face.block_name)
            if lbl is None or spec is None:
                continue
            fc = spec.face_corner_names(assoc.face.axis, assoc.face.side, 2)
            edge_on[frozenset(fc)] = lbl
            for c in fc:
                corner_on[c].add(lbl)
        nodes = {
            name: {
                "xy": [float(c.position[0]), float(c.position[1])],
                "fixed": bool(getattr(c, "fixed", False)),
                "on": sorted(corner_on.get(name, ())),
            }
            for name, c in corners.items()
        }
        edges, seen = [], set()
        for spec in specs.values():
            for axis in (0, 1):
                for side in (0, 1):
                    a, b = spec.face_corner_names(axis, side, 2)
                    k = tuple(sorted((a, b)))
                    if a != b and k not in seen:
                        seen.add(k)
                        edges.append([a, b, edge_on.get(frozenset((a, b)))])
        return {"nodes": nodes, "edges": edges}

    def _res(self) -> int:
        res = self.connectivity.get("res", 10)
        if isinstance(res, dict):
            return int(res.get("default", 10))
        return res if isinstance(res, int) else 10

    def _merge_graph(self, diags: list[Diagnostic]):
        """Base corners/edges (preserved) + blocking, as tracer inputs.

        Returns ``(pos, names, edges, declared, base_edges, seed)``. The seed's
        own edges go into ``base_edges`` (a face built only from them is left to
        the seed); base corner names carry through, so a blocking edge can name a
        base corner.
        """
        conn = self.connectivity
        nodes = conn.get("nodes", {}) or {}
        edges = conn.get("edges", []) or []

        seed = None
        base_corners: dict = {}
        base_specs: dict = {}
        if self.base is not None:
            base_corners, base_specs, seed = _base_parts(self.base)

        # Base corner -> the geometry labels the base associated to faces touching
        # it. A base corner carries these as *declared* bindings, so a base
        # boundary node keeps its curve (and thus projection + marker) even when
        # the overlay moves it off that curve — otherwise a nudged sliding node
        # silently drops its wall association with no diagnostic.
        base_corner_curves: dict[str, set[str]] = {}
        if seed is not None:
            _label_of = {id(ent): lbl for lbl, ent in self.geometry.items()}
            for assoc in getattr(seed, "_associations", []):
                lbl = _label_of.get(id(assoc.entity))
                spec = base_specs.get(assoc.face.block_name)
                if lbl is None or spec is None:
                    continue
                for c in spec.face_corner_names(assoc.face.axis, assoc.face.side, 2):
                    base_corner_curves.setdefault(c, set()).add(lbl)

        pos: list[np.ndarray] = []
        names: list[str] = []
        index: dict = {}
        declared: dict[int, set[str]] = {}
        edge_binds: dict[frozenset, str] = {}

        def bind_on(i: int, labels):
            for lbl in labels or ():
                if lbl in self.geometry:
                    declared[i].add(lbl)
                else:
                    diags.append(
                        Diagnostic(
                            "unknown_geometry", f"unknown geometry {lbl!r} in binding"
                        )
                    )

        def add(nid, xy, on=()):
            i = len(pos)
            index[nid] = i
            pos.append(np.asarray(xy, dtype=float)[:2])
            names.append(str(nid))
            declared[i] = set()
            bind_on(i, on)
            return i

        for cname, corner in base_corners.items():
            add(cname, corner.position[:2], on=base_corner_curves.get(cname, ()))

        # weld tolerance from the combined extent (base corners + blocking xy)
        fxy = [
            np.asarray(s["xy"], dtype=float)[:2]
            for s in nodes.values()
            if isinstance(s, dict) and s.get("xy") is not None
        ]
        allxy = list(pos) + fxy
        span = np.array(allxy) if allxy else np.zeros((1, 2))
        weld = 1e-6 * (float(np.linalg.norm(span.max(0) - span.min(0))) or 1.0)

        def onto_base(xy):
            p = np.asarray(xy, dtype=float)[:2]
            for cname in base_corners:
                if np.linalg.norm(p - pos[index[cname]]) <= weld:
                    return cname
            return None

        def split_ends(spec):
            sp = list(spec.get("split") or ())
            return tuple(sp[:2]) if len(sp) >= 2 else (None, None)

        pending: dict = {}
        for nid, spec in nodes.items():
            spec = spec or {}
            on = spec.get("on", ())
            if nid in index:  # blocking id names a base corner
                i = index[nid]
                xy = spec.get("xy")
                fx = spec.get("fixed")
                if (xy is not None or fx is not None) and nid in base_corners:
                    # a moved / re-pinned base corner: override it with a fresh
                    # Corner so the caller's base object is never mutated
                    from .block_topology import Corner

                    old = seed._corners.get(nid) if seed is not None else None
                    p = np.asarray(xy, dtype=float)[:2] if xy is not None else pos[i]
                    pos[i] = p
                    fixed = bool(fx) if fx is not None else getattr(old, "fixed", True)
                    if old is not None:
                        seed._corners[nid] = Corner(name=nid, position=p, fixed=fixed)
                bind_on(i, on)
                continue
            if "split" in spec:
                pending[nid] = spec
                continue
            xy = spec.get("xy")
            if xy is None:
                diags.append(Diagnostic("bad_node", f"node {nid!r} has no 'xy'"))
                continue
            w = onto_base(xy)
            if w is not None:
                index[nid] = index[w]
                bind_on(index[w], on)
            else:
                add(nid, xy, on)

        # Split children sit on their parent edge at a relative t (so they ride
        # a parent move) and inherit the parent edge's curves, so a binding
        # distributes to both halves. Fixpoint resolves chained splits.
        # split_of maps a parent edge (index pair) -> its split node index.
        split_of: dict = {}
        while pending:
            ready = [
                n for n, s in pending.items() if all(e in index for e in split_ends(s))
            ]
            if not ready:
                break
            for nid in ready:
                spec = pending.pop(nid)
                pa, pb = split_ends(spec)
                ia, ib = index[pa], index[pb]
                t = float(spec.get("t", 0.5))
                sxy = spec.get("xy")  # a moved bifurcation point rides free of t
                p = (
                    np.asarray(sxy, dtype=float)[:2]
                    if sxy is not None
                    else (1.0 - t) * pos[ia] + t * pos[ib]
                )
                i = add(nid, p, spec.get("on", ()))
                declared[i] |= declared[ia] & declared[ib]
                split_of[(min(ia, ib), max(ia, ib))] = i
        for nid, spec in pending.items():
            diags.append(
                Diagnostic(
                    "stale_ref",
                    f"split node {nid!r} parent {split_ends(spec)!r} missing",
                )
            )

        eidx: list[tuple[int, int]] = []
        base_edges: set = set()
        seen: set = set()

        def edge(ia: int, ib: int):
            e = (min(ia, ib), max(ia, ib))
            if e not in seen:
                seen.add(e)
                eidx.append(e)

        # Base edges the blocking has subdivided into editable sub-edges: each
        # sub-edge carries a ``base`` tag naming the base edge it came from, so
        # any number of them (recursively) can share one base edge.
        cut_edges: set = set()
        for e in edges:
            ba = e.get("base")
            if ba and len(ba) >= 2 and ba[0] in index and ba[1] in index:
                cut_edges.add(frozenset((index[ba[0]], index[ba[1]])))

        # Base blocks: an unsplit block stays verbatim (its edges are base_edges,
        # skipped by the tracer). A block with a split or cut edge is retired and
        # its region re-traced from the blocking's sub-edges + cut edges — that is
        # how a base line bifurcates into conforming sub-blocks.
        affected: set = set()
        block_regions: list = []  # (retired name, quad) so sub-blocks keep the name
        for bname, spec in base_specs.items():
            faces, fss = [], []
            for axis in (0, 1):
                for side in (0, 1):
                    a, b = spec.face_corner_names(axis, side, 2)
                    key = (min(index[a], index[b]), max(index[a], index[b]))
                    faces.append(key)
                    fss.append(frozenset(key))
            if any(k in split_of for k in faces) or any(f in cut_edges for f in fss):
                affected.add(bname)
                cn = spec.corner_names  # product order: sw, nw, se, ne
                ring = [cn[0], cn[2], cn[3], cn[1]]  # -> sw, se, ne, nw
                block_regions.append((bname, [tuple(pos[index[c]]) for c in ring]))
                for key, f in zip(faces, fss):
                    if key in split_of:
                        edge(key[0], split_of[key])
                        edge(split_of[key], key[1])
                    elif f in cut_edges:
                        pass  # the blocking's sub-edges already supply this face
                    else:
                        edge(*key)
            else:
                for key in faces:
                    edge(*key)
                    base_edges.add(frozenset(key))

        # Per-edge resolution from the base blocks, so a re-traced (re-blocked)
        # region keeps the parent's cell counts. A face on `axis` spans the other
        # axis, so its edge resolution is resolutions[1 - axis]. Each half of a
        # split base edge inherits the whole edge's resolution (conforming; the
        # cut edge then picks up the preserved perpendicular resolution).
        edge_res: dict = {}
        for spec in base_specs.values():
            for axis in (0, 1):
                for side in (0, 1):
                    a, b = spec.face_corner_names(axis, side, 2)
                    edge_res[frozenset((index[a], index[b]))] = spec.resolutions[
                        1 - axis
                    ]
        for key, m in split_of.items():
            r = edge_res.get(frozenset(key))
            if r is not None:
                edge_res[frozenset((key[0], m))] = r
                edge_res[frozenset((m, key[1]))] = r
        # cut sub-edges inherit their base edge's resolution (via the base tag)
        for e in edges:
            ba, a, b = e.get("base"), e.get("a"), e.get("b")
            if ba and len(ba) >= 2 and a in index and b in index and ba[0] in index:
                r = edge_res.get(frozenset((index[ba[0]], index[ba[1]])))
                if r is not None:
                    edge_res[frozenset((index[a], index[b]))] = r
        # A cut edge (blocking edge with no base tag) inside a retired block takes
        # that block's resolution for the axis it runs parallel to. Otherwise a
        # strip touching a base edge inherits its resolution while the adjacent
        # strip defaults — a mismatch on the shared cut edge that crashes build.
        for bname in affected:
            spec = base_specs[bname]
            cn = spec.corner_names  # product order: sw, nw, se, ne
            sw, nw, se = index[cn[0]], index[cn[1]], index[cn[2]]
            ne = index[cn[3]]
            quad = [pos[sw], pos[se], pos[ne], pos[nw]]
            d0 = pos[se] - pos[sw]  # axis-0 direction (south/north edges)
            d1 = pos[nw] - pos[sw]  # axis-1 direction (west/east edges)
            l0 = float(np.linalg.norm(d0)) or 1.0
            l1 = float(np.linalg.norm(d1)) or 1.0
            n0, n1 = spec.resolutions
            for e in edges:
                if e.get("base"):
                    continue
                a, b = e.get("a"), e.get("b")
                if a not in index or b not in index:
                    continue
                key = frozenset((index[a], index[b]))
                if key in edge_res:
                    continue
                mid = 0.5 * (pos[index[a]] + pos[index[b]])
                if not _point_in_quad(mid, quad):
                    continue
                ed = pos[index[b]] - pos[index[a]]
                edge_res[key] = (
                    n0 if abs(float(ed @ d0)) / l0 >= abs(float(ed @ d1)) / l1 else n1
                )

        # Markers on a retired block's faces, keyed by the face's corner indices,
        # so the tracer can re-tag the sub-faces that lie on them — a re-block
        # never silently drops a programmatic boundary marker.
        face_markers: list = []
        if seed is not None:  # retire re-blocked base blocks + their dangling refs
            for bname in affected:
                spec = base_specs[bname]
                for axis in (0, 1):
                    for side in (0, 1):
                        fc = spec.face_corner_names(axis, side, 2)
                        ia, ib = index[fc[0]], index[fc[1]]
                        for marker, fs in seed._boundary_tags.items():
                            if any(
                                f.block_name == bname
                                and f.axis == axis
                                and f.side == side
                                for f in fs
                            ):
                                face_markers.append((ia, ib, marker))
                seed._block_specs.pop(bname, None)
                seed._associations = [
                    a for a in seed._associations if a.face.block_name != bname
                ]
                seed._connections = [
                    c
                    for c in seed._connections
                    if bname not in (c.face_a.block_name, c.face_b.block_name)
                ]
                for tag, fs in list(seed._boundary_tags.items()):
                    seed._boundary_tags[tag] = [f for f in fs if f.block_name != bname]

        for e in edges:
            a, b = e.get("a"), e.get("b")
            if a not in index or b not in index:
                diags.append(
                    Diagnostic("stale_ref", f"edge references unknown node {(a, b)!r}")
                )
                continue
            edge(index[a], index[b])
            if e.get("bind") is not None:
                bind_on(index[a], (e["bind"],))
                bind_on(index[b], (e["bind"],))
                if e["bind"] in self.geometry:
                    edge_binds[frozenset((index[a], index[b]))] = e["bind"]

        # new (non-base) blocking nodes explicitly pinned via a fixed flag
        fixed_nodes = {
            index[nid]
            for nid, spec in nodes.items()
            if (spec or {}).get("fixed") and nid in index and nid not in base_corners
        }
        # explicit per-edge resolution the user set — drives its whole loop
        user_res: dict = {}
        for e in edges:
            r, a, b = e.get("res"), e.get("a"), e.get("b")
            if r and a in index and b in index:
                user_res[frozenset((index[a], index[b]))] = int(r)
        return (
            pos,
            names,
            eidx,
            declared,
            base_edges,
            seed,
            fixed_nodes,
            edge_res,
            block_regions,
            face_markers,
            user_res,
            edge_binds,
        )

    def _base_wireframe_3d(self):
        """The base topology as (positions, hex edges, node->surfaces, fixed,
        entities), so a drawn overlay can subdivide/extend it."""
        from collections import defaultdict
        from itertools import product as _product

        base = self.base
        if isinstance(base, TopologyBuilder):
            corners, specs, assocs = (
                base._corners,
                base._block_specs,
                base._associations,
            )
        else:
            corners, specs, assocs = base.corners, base.block_specs, base.associations
        idx = list(_product((0, 1), repeat=3))
        pmap = {ci: i for i, ci in enumerate(idx)}
        pos = {name: np.asarray(c.position, float) for name, c in corners.items()}
        fixed = {name: bool(getattr(c, "fixed", True)) for name, c in corners.items()}
        edges: set = set()
        for spec in specs.values():
            cn = spec.corner_names
            for a, ci in enumerate(idx):
                for axis in range(3):
                    nb = list(ci)
                    nb[axis] ^= 1
                    b = pmap[tuple(nb)]
                    if a < b:
                        edges.add(frozenset((cn[a], cn[b])))
        on: dict = defaultdict(set)
        entities: dict = {}
        for assoc in assocs:
            nm = getattr(assoc.entity, "name", None)
            if nm is None:
                continue
            entities[nm] = assoc.entity
            spec = specs.get(assoc.face.block_name)
            if spec is not None:
                for c in spec.face_corner_names(assoc.face.axis, assoc.face.side, 3):
                    on[c].add(nm)
        return pos, edges, on, fixed, entities

    def _flatten_3d(self):
        """Assemble the 3D blocking into a BlockTopology.

        Corners come from ``nodes`` (``xyz`` + optional ``on`` surface bindings)
        and, with a ``base``, its corners. Blocks are an explicit list (each 8
        ids in ``product((0,1), repeat=3)`` order) or, when only ``edges`` (or a
        base) are given, inferred by cube enumeration over the combined
        wireframe. A boundary face whose four corners share a bound surface is
        associated with it; a corner on two or more surfaces (or flagged
        ``fixed``) is pinned; shared interface faces are left for connection
        inference.
        """
        from collections import defaultdict

        diags: list[Diagnostic] = []
        conn = self.connectivity
        nodes = conn.get("nodes", {}) or {}
        blocks = conn.get("blocks", []) or []
        edges_in = conn.get("edges", []) or []
        default_res = conn.get("res", 10)
        default_res = default_res if isinstance(default_res, int) else 10

        geometry = dict(self.geometry)
        tb = TopologyBuilder(d=3)
        node_on: dict[str, set[str]] = defaultdict(set)
        base_edges: set = set()

        # Base (imperative + interactive): its corners and hex edges join the
        # overlay wireframe, which subdivides/extends it; the merged wireframe is
        # re-inferred (subdivision parents dropped by containment).
        if self.base is not None:
            bpos, base_edges, bon, bfixed, bents = self._base_wireframe_3d()
            geometry.update(bents)
            for name, p in bpos.items():
                tb.add_corner(name, p, fixed=bfixed.get(name, True))
                node_on[name] = set(bon.get(name, ()))

        for nid, spec in nodes.items():
            spec = spec or {}
            on = list(spec.get("on") or [])
            for lbl in on:
                if lbl not in geometry:
                    diags.append(
                        Diagnostic("unknown_geometry", f"unknown geometry {lbl!r}")
                    )
            known = [lbl for lbl in on if lbl in geometry]
            if str(nid) in tb._corners:  # names an existing (base) corner
                node_on[str(nid)].update(known)
                continue
            xyz = spec.get("xyz")
            if xyz is None:
                diags.append(Diagnostic("bad_node", f"node {nid!r} has no 'xyz'"))
                continue
            fixed = bool(spec.get("fixed")) or len(known) >= 2
            p = np.asarray(xyz, dtype=float)[:3]
            # Snap a bound corner onto its surface(s): drawn coordinates are
            # only approximate, and downstream sampling/constraints assume a
            # bound corner lies ON the geometry. Two or more surfaces -> the
            # shared junction, by alternating projection. An explicitly
            # pinned node keeps its drawn position.
            if known and not bool(spec.get("fixed")):
                ents = [geometry[lbl] for lbl in sorted(set(known))]
                if len(ents) == 1:
                    p = np.asarray(ents[0].project(p), dtype=float)[:3]
                else:
                    for _ in range(24):
                        p_prev = p
                        for e in ents:
                            p = np.asarray(e.project(p), dtype=float)[:3]
                        if float(np.linalg.norm(p - p_prev)) <= 1e-14:
                            break
            tb.add_corner(str(nid), p, fixed=fixed)
            node_on[str(nid)] = set(known)

        # Blocks: an explicit list, or inferred (cube enumeration) from the
        # combined base + overlay edge wireframe.
        block_lists: list = []
        if blocks and self.base is None:
            for i, b in enumerate(blocks):
                corners = b.get("corners")
                if not corners or len(corners) != 8:
                    diags.append(
                        Diagnostic(
                            "bad_block", f"block {i} needs 8 corners (product order)"
                        )
                    )
                    continue
                res = b.get("res", default_res)
                res = tuple(res) if isinstance(res, (list, tuple)) else (res, res, res)
                block_lists.append(
                    (b.get("name", f"blk{i}"), [str(c) for c in corners], res)
                )
        else:
            from .trace3d import infer_blocks

            combined = set(base_edges)
            for e in edges_in:
                a, b = (e.get("a"), e.get("b")) if isinstance(e, dict) else (e[0], e[1])
                a, b = str(a), str(b)
                if a in tb._corners and b in tb._corners:
                    combined.add(frozenset((a, b)))
                else:
                    diags.append(
                        Diagnostic(
                            "stale_ref", f"edge references unknown node {(a, b)!r}"
                        )
                    )
            if not combined:
                diags.append(
                    Diagnostic("no_blocks", "3D connectivity needs 'blocks' or 'edges'")
                )
                return None, diags
            pos = {name: c.position for name, c in tb._corners.items()}
            inferred, idiags = infer_blocks(
                pos, [tuple(e) for e in combined], node_on=node_on
            )
            diags += idiags
            block_lists = [
                (f"blk{i}", corners, (default_res,) * 3)
                for i, corners in enumerate(inferred)
            ]

        for name, corners, res in block_lists:
            try:
                tb.add_block(name, corners=corners, resolutions=res)
            except (ValueError, TypeError) as exc:
                diags.append(Diagnostic("bad_block", str(exc)))

        # Associate only boundary faces: an interface face (shared by two blocks)
        # is left for connection inference and never carries a surface.
        face_count: dict = defaultdict(int)
        for spec in tb._block_specs.values():
            for axis in range(3):
                for side in (0, 1):
                    face_count[frozenset(spec.face_corner_names(axis, side, 3))] += 1
        for name, spec in list(tb._block_specs.items()):
            for axis in range(3):
                for side in (0, 1):
                    fc = spec.face_corner_names(axis, side, 3)
                    key = frozenset(fc)
                    if len(key) != len(fc) or face_count[key] != 1:
                        continue  # degenerate or shared interface
                    common = set.intersection(*(node_on.get(c, set()) for c in fc))
                    for ent_name in sorted(common):
                        tb.associate(name, axis, side, geometry[ent_name])

        fan_decls = self._fan_frame_decls(diags)
        par_decls = self._parallel_decls(diags)
        errors = [d for d in diags if not d.kind.startswith("warn")]
        if errors:
            return None, diags
        try:
            if self.relax_orthogonality:
                tb.relax_orthogonality(*self._relax_entries())
            for f in fan_decls:
                tb.fan_frame(f["corner"], through=f["through"], normal=f["normal"])
            for p in par_decls:
                tb.parallel_to(
                    p["to"], chain=p["chain"], weight=p["weight"], taper=p["taper"]
                )
            return self._build_with_warn_diags(tb.build, diags), diags
        except Exception as exc:  # a non-conforming assembly still fails to build
            return None, [Diagnostic("build_error", str(exc))]

    def flatten(self):
        """Compile base + blocking into a BlockTopology, or ``(None, diagnostics)``.

        An empty diagnostic list means fully valid (committable); otherwise the
        topology is ``None`` and each :class:`~egg.topology.trace.Diagnostic`
        names what to fix.
        """
        if self.d == 3:
            return self._flatten_3d()
        diags: list[Diagnostic] = []
        (
            pos,
            names,
            eidx,
            declared,
            base_edges,
            seed,
            fixed_nodes,
            edge_res,
            block_regions,
            face_markers,
            user_res,
            edge_binds,
        ) = self._merge_graph(diags)
        builder, tdiags = trace_topology(
            pos,
            eidx,
            self._gents(),
            res=self._res(),
            declared_curves=declared,
            declared_edge_curves=edge_binds,
            node_names=names,
            seed_builder=seed,
            base_edges=base_edges,
            fixed_nodes=fixed_nodes,
            block_regions=block_regions,
            edge_res=edge_res,
            face_markers=face_markers,
            user_res=user_res,
        )
        diags += tdiags
        fan_decls = self._fan_frame_decls(diags)
        par_decls = self._parallel_decls(diags)
        # ``warn_*`` diagnostics are advisory (e.g. a dropped boundary marker):
        # they are reported but do not block a build/commit.
        errors = [d for d in diags if not d.kind.startswith("warn")]
        if errors:
            return None, diags
        try:
            if self.relax_orthogonality:
                builder.relax_orthogonality(*self._relax_entries())
            for f in fan_decls:
                builder.fan_frame(f["corner"], through=f["through"], normal=f["normal"])
            for p in par_decls:
                builder.parallel_to(
                    p["to"], chain=p["chain"], weight=p["weight"], taper=p["taper"]
                )
            topo = self._build_with_warn_diags(builder.build, diags)
        except Exception as exc:  # a non-conforming merge still fails to build
            return None, [Diagnostic("build_error", str(exc))]
        return topo, diags  # topo valid; diags holds only advisory warnings

    def diagnostics(self) -> list[Diagnostic]:
        """The flatten diagnostics; empty means green/committable."""
        return self.flatten()[1]

    def build(self):
        """The BlockTopology, raising ValueError on the first (non-warning)
        diagnostic."""
        topo, diags = self.flatten()
        errors = [d for d in diags if not d.kind.startswith("warn")]
        if errors:
            raise ValueError(f"ExplicitTopology: {errors[0].msg}")
        # No error diagnostics ⟹ flatten produced a topology. Assert it so the
        # (un-annotated) return type narrows to BlockTopology instead of
        # `... | None`, which otherwise makes `build().initialize_grid()` and
        # every `topo = builder.build()` look like an attribute-of-None error.
        assert topo is not None
        return topo

    def initialize_grid(self):
        """TFI-initialise the flattened topology (raises if not valid)."""
        return self.build().initialize_grid()

    # -- export: the flattened topology as a standalone blocking ----------

    def to_connectivity(self) -> dict:
        """The flattened topology (base included) as a standalone connectivity
        dict — the blocking that replicates it with ``base=None`` on the same
        geometry. See :meth:`print_topology` for the copy-pasteable block."""
        conn, _notes = self._export_connectivity()
        return conn

    def print_topology(self, *, geometry_var: str = "geometry", file=None) -> str:
        """Print (and return) a ready-to-paste ``ExplicitTopology(...)`` block
        replicating the full flattened topology — base blocks and drawn overlay
        alike — as a standalone blocking on the same geometry.

        Everything expressible in a blocking round-trips: node positions and
        pins, per-edge resolutions, and face→curve bindings (so the free /
        sliding / fixed classification and tag-derived boundary markers are
        preserved). What cannot ride a blocking is reported as ``# ...``
        comment lines above the block: boundary-layer specs (re-declare via
        ``curve.clustered(...)`` on the geometry), bindings to unnamed
        entities, and curve-following edge seeding of corners placed on an
        unassociated curve. 2D block names are re-derived by the tracer
        (``blk0``, ``blk1``, ...); 3D blocks keep their names.

        ``geometry_var`` is the variable name the printed block references for
        its ``geometry=`` argument.
        """
        import sys

        conn, notes = self._export_connectivity()
        out: list[str] = [f"# {n}" for n in notes]
        out += [
            "ExplicitTopology(",
            "    base=None,",
            f"    geometry={geometry_var},",
        ]
        if self.relax_orthogonality:
            labels = ", ".join(
                _lit(o if isinstance(o, str) else getattr(o, "name", None) or "?")
                for o in self.relax_orthogonality
            )
            out.append(f"    relax_orthogonality=({labels},),")
        out += [
            "    connectivity=editable({",
            '        "nodes": {',
        ]
        out += [f"            {_lit(k)}: {_lit(v)}," for k, v in conn["nodes"].items()]
        out.append("        },")
        if self.d == 3:
            out.append('        "blocks": [')
            out += [f"            {_lit(b)}," for b in conn["blocks"]]
            out.append("        ],")
        else:
            out.append('        "edges": [')
            out += [f"            {_lit(e)}," for e in conn["edges"]]
            out.append("        ],")
        if conn.get("fan_frames"):
            out.append('        "fan_frames": {')
            out += [
                f"            {_lit(k)}: {_lit(v)},"
                for k, v in conn["fan_frames"].items()
            ]
            out.append("        },")
        if conn.get("parallel"):
            out.append('        "parallel": [')
            out += [f"            {_lit(p)}," for p in conn["parallel"]]
            out.append("        ],")
        out += [f'        "res": {_lit(conn["res"])},', "    }),", ")"]
        text = "\n".join(out)
        print(text, file=file if file is not None else sys.stdout)
        return text

    def _export_connectivity(self) -> tuple[dict, list[str]]:
        """Build the standalone connectivity dict plus human-readable notes on
        anything the blocking cannot express."""
        from collections import Counter

        topo = self.build()
        notes: list[str] = []

        # entity id -> the label the printed block will reference. This
        # object's geometry labels win; association-derived names fill in for
        # base entities never passed in `geometry`.
        label_of = {id(e): lbl for lbl, e in self.geometry.items()}
        ent_of = dict(self.geometry)
        for lbl, e in topo.entities.items():
            label_of.setdefault(id(e), lbl)
            ent_of.setdefault(lbl, e)

        def rnd(v: float) -> float:
            return float(f"{float(v):.10g}")

        assoc_of: dict[tuple, list] = {}
        for a in topo.associations:
            key = (a.face.block_name, a.face.axis, a.face.side)
            assoc_of.setdefault(key, []).append(a.entity)

        nodes: dict[str, dict] = {}
        for name, c in topo.corners.items():
            spec: dict = {
                ("xyz" if self.d == 3 else "xy"): [
                    rnd(c.position[k]) for k in range(self.d)
                ]
            }
            if getattr(c, "fixed", False):
                spec["fixed"] = True
            nodes[name] = spec

        # Corners placed on a curve that has no association seed their edges
        # along it at init; a blocking has no free-yet-curve-seeded notion.
        associated_ids = {id(a.entity) for a in topo.associations}
        seeded: dict[str, list[str]] = {}
        for name, c in topo.corners.items():
            edge = getattr(getattr(c, "source", None), "edge", None)
            ent = getattr(edge, "entity", None)
            if ent is not None and id(ent) not in associated_ids:
                lbl = label_of.get(id(ent)) or getattr(ent, "name", None) or "?"
                seeded.setdefault(lbl, []).append(name)
        for lbl, names in seeded.items():
            notes.append(
                f"corners {', '.join(names)} were placed on {lbl!r}, which has no "
                "association: exported as free nodes (curve-following edge "
                "seeding at init is not replicated)"
            )

        if getattr(topo, "boundary_layer_specs", {}):
            notes.append(
                "boundary-layer spec(s) are not part of a blocking; re-declare "
                "them on the geometry via <curve>.clustered(...)"
            )

        if self.d == 3:
            blocks = [
                {
                    "name": bname,
                    "corners": list(spec.corner_names),
                    "res": [int(r) for r in spec.resolutions],
                }
                for bname, spec in topo.block_specs.items()
            ]
            # Bindings ride the corners in 3D: a boundary face whose corners
            # all share a surface re-associates with it on flatten.
            for (bname, axis, side), ents in assoc_of.items():
                spec = topo.block_specs[bname]
                for ent in ents:
                    lbl = label_of.get(id(ent))
                    if lbl is None:
                        notes.append(
                            f"face {bname}[{axis},{side}] is associated to an "
                            "unnamed entity; binding dropped"
                        )
                        continue
                    for cn in spec.face_corner_names(axis, side, 3):
                        on = nodes[cn].setdefault("on", [])
                        if lbl not in on:
                            on.append(lbl)
            for spec in nodes.values():
                if "on" in spec:
                    spec["on"] = sorted(spec["on"])
            out3 = {"nodes": nodes, "blocks": blocks, "res": 10}
            if topo.fan_frames:
                out3["fan_frames"] = self._fan_frames_export(topo)
            if getattr(topo, "parallel_chains", ()):
                out3["parallel"] = self._parallel_export(topo, label_of, notes)
            return out3, notes

        entries: dict[frozenset, dict] = {}
        order: list[frozenset] = []
        face_label: dict[tuple, str] = {}
        for bname, spec in topo.block_specs.items():
            for axis in (0, 1):
                for side in (0, 1):
                    a_, b_ = spec.face_corner_names(axis, side, 2)
                    if a_ == b_:
                        notes.append(f"degenerate face {bname}[{axis},{side}] skipped")
                        continue
                    key = frozenset((a_, b_))
                    r = int(spec.resolutions[1 - axis])
                    e = entries.get(key)
                    if e is None:
                        e = {"a": a_, "b": b_, "res": r}
                        entries[key] = e
                        order.append(key)
                    elif e["res"] != r:
                        notes.append(
                            f"edge {e['a']}-{e['b']}: conflicting resolutions "
                            f"{e['res']} vs {r}; kept {max(e['res'], r)}"
                        )
                        e["res"] = max(e["res"], r)
                    for ent in assoc_of.get((bname, axis, side), ()):
                        lbl = label_of.get(id(ent))
                        if lbl is None:
                            notes.append(
                                f"face {bname}[{axis},{side}] is associated to an "
                                "unnamed entity; binding dropped"
                            )
                        elif "bind" not in e:
                            e["bind"] = lbl
                            face_label[(bname, axis, side)] = lbl
                        elif e["bind"] != lbl:
                            notes.append(
                                f"edge {e['a']}-{e['b']} is associated to both "
                                f"{e['bind']!r} and {lbl!r}; kept {e['bind']!r}"
                            )
                        else:
                            face_label[(bname, axis, side)] = lbl

        # A marker that does not derive from its bound curve's tag is lost.
        for tag, faces in getattr(topo, "boundary_tags", {}).items():
            for f in faces:
                lbl = face_label.get((f.block_name, f.axis, f.side))
                ent = ent_of.get(lbl) if lbl else None
                if ent is None or getattr(ent, "tag", None) != tag:
                    notes.append(
                        f"boundary marker {tag!r} on {f.block_name}"
                        f"[{f.axis},{f.side}] does not derive from a bound "
                        "curve's tag and will be lost"
                    )

        # Modal resolution becomes the default; edges at it drop their
        # explicit "res" (a conforming loop shares one resolution, so a loop
        # is either fully explicit or fully default — never mixed).
        res_default = Counter(e["res"] for e in entries.values()).most_common(1)
        res_default = int(res_default[0][0]) if res_default else self._res()
        edges_out = []
        for key in order:
            e = entries[key]
            if e["res"] == res_default:
                del e["res"]
            edges_out.append(e)
        out2 = {"nodes": nodes, "edges": edges_out, "res": res_default}
        if topo.fan_frames:
            out2["fan_frames"] = self._fan_frames_export(topo)
        if getattr(topo, "parallel_chains", ()):
            out2["parallel"] = self._parallel_export(topo, label_of, notes)
        return out2, notes

    @staticmethod
    def _fan_frames_export(topo) -> dict:
        """The resolved fan frames back in declaration form (corner names)."""
        return {
            f.corner: {"through": [f.through[0], f.through[1]], "normal": f.normal}
            for f in topo.fan_frames
        }

    @staticmethod
    def _parallel_export(topo, label_of: dict, notes: list[str]) -> list:
        """The resolved parallel chains back in declaration form; a chain to
        an entity no label reaches cannot ride the blocking and is dropped
        with a note."""
        out = []
        for p in topo.parallel_chains:
            lbl = label_of.get(id(p.to)) or getattr(p.to, "name", None)
            if lbl is None:
                notes.append(
                    f"parallel chain {'-'.join(p.chain)} targets an unnamed "
                    "entity; entry dropped"
                )
                continue
            e: dict = {"to": lbl, "chain": list(p.chain)}
            if p.weight != 1.0:
                e["weight"] = p.weight
            if p.taper is not None:
                e["taper"] = p.taper
            out.append(e)
        return out
