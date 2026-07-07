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

from typing import Any

import numpy as np

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
        The blocking structure (``nodes`` / ``edges`` / ``res``); wrap it with
        :func:`editable` to opt into UI editing.
    d : int
        Spatial dimension (2 only).
    """

    def __init__(
        self,
        *,
        base: Any = None,
        geometry: dict[str, Any] | None = None,
        connectivity: dict | None = None,
        d: int = 2,
    ):
        if d != 2:
            raise ValueError("ExplicitTopology is 2D only")
        self.base = base
        self.geometry = dict(geometry or {})
        self.connectivity = dict(connectivity or {})
        self.d = d

    def _gents(self) -> list[tuple[str, Any, bool]]:
        return [(lbl, ent, _is_closed(ent)) for lbl, ent in self.geometry.items()]

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
        )

    def flatten(self):
        """Compile base + blocking into a BlockTopology, or ``(None, diagnostics)``.

        An empty diagnostic list means fully valid (committable); otherwise the
        topology is ``None`` and each :class:`~egg.topology.trace.Diagnostic`
        names what to fix.
        """
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
        ) = self._merge_graph(diags)
        builder, tdiags = trace_topology(
            pos,
            eidx,
            self._gents(),
            res=self._res(),
            declared_curves=declared,
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
        # ``warn_*`` diagnostics are advisory (e.g. a dropped boundary marker):
        # they are reported but do not block a build/commit.
        errors = [d for d in diags if not d.kind.startswith("warn")]
        if errors:
            return None, diags
        try:
            topo = builder.build()
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
        return topo

    def initialize_grid(self):
        """TFI-initialise the flattened topology (raises if not valid)."""
        return self.build().initialize_grid()
