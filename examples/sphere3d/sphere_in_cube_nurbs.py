"""NURBS sphere inside a cube — a heavy-entity variant of sphere_in_cube.

Identical to ``sphere_in_cube.py`` except the spherical cavity is an exact
*NURBS* surface (degree-2 surface of revolution, rational weights) instead of
the analytic ``Sphere``. Geometrically it is the same sphere (eval radius error
~1e-16), but every cavity node now slides via the heavy
``BSplineSurfaceParam`` device project (coarse-grid-seeded Newton on the
nearest-foot stationarity) — the workload that decides whether the per-colour
sweep kernel still wins once the constrained entity is expensive.

Mirrors the 2D circle example's topology one dimension up: a 6-block
"cubed-sphere" **O-shell** wraps the spherical cavity of radius ``r0`` out to
an *inset* cube of half-width ``cw`` (the 3D o-ring), and the gap between the
inset cube and the outer cube ``[-1, 1]^3`` is filled by the 26 axis-aligned
**H-grid** blocks of its 3x3x3 decomposition (6 face + 12 edge + 8 corner
blocks) — 32 blocks total. Shared block faces/edges are merged into single
global nodes by position.

Constrained DOFs exercise the whole 3D entity set:

- sphere-surface nodes  -> ``TAG_SPHERE``  (slide on the cavity),
- cube-face interiors   -> ``TAG_PLANE``   (slide in the face plane),
- cube-edge nodes       -> ``TAG_LINE3``   (slide along the edge),
- cube corners          -> fixed; everything else free.

The Python topology/TFI front-end is still 2D, so the flattened sweep context
(the wire format ``flatten_context`` produces) is assembled by hand and the
C++ core is driven directly at ``dim=3``. The live/grid plots show the grid's
planar sections in the XY (z=0) and YZ (x=0) planes — each is the familiar 2D
O-ring picture.

Usage::

    uv run sphere_in_cube.py [--n N] [--m M] [--mh M] [--r0 R] [--cw C]
        [--sweeps N] [--chunk N] [--device cpu|gpu|auto] [--report-every N]
        [--plot-live] [--plot-grid] [--plot-energy] [--plot-3d] [--plot-topology]
"""

import argparse
from itertools import product

import numpy as np

from egg._cpp import cpp_core
from egg.geometry.analytic3d import Line3
from egg.geometry.entity_encoding import TAG_FREE, TAG_PLANE, TAG_SPHERE
from egg.geometry.entity_soa import TAG_LINE3, group_entities_by_type
from egg.geometry.surfaces3d import BSplineSurface
from egg.smoothing.batch import make_chain_J_nd

# The six cube-face frames (e, t1, t2), each right-handed: t1 x t2 = e, so the
# local block axes (t1, t2, radial ~ e) give positively oriented hex cells.
_FACES = [
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
    ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
]


def build_grid(n, m, mh, r0, cw):
    """O-shell + H-grid block lattices, merged into global nodes by position.

    Mirrors the 2D circle example's topology one dimension up:

    - **O-shell** — 6 cubed-sphere blocks between the sphere surface and the
      *inset* cube of half-width ``cw`` (the 3D "o-ring"), ``n x n x m`` each;
    - **H-grid** — the 26 axis-aligned boxes of the 3x3x3 decomposition of the
      gap between the inset cube and the outer cube ``[-1, 1]^3`` (6 face + 12
      edge + 8 corner blocks), ``mh`` nodes across each thin direction.

    Returns (X (N,3), blocks [list of (na,nb,nc) id arrays]).
    """
    key_of = {}
    X = []

    def nid(p):
        key = tuple(np.round(p, 9))
        if key not in key_of:
            key_of[key] = len(X)
            X.append(np.asarray(p, dtype=float))
        return key_of[key]

    blocks = []
    # O-shell: sphere -> inset cube, radially interpolated.
    ab = np.linspace(-1.0, 1.0, n)
    for e, t1, t2 in _FACES:
        e, t1, t2 = map(np.asarray, (e, t1, t2))
        ids = np.empty((n, n, m), dtype=np.int64)
        for i, a in enumerate(ab):
            for j, b in enumerate(ab):
                p_in = cw * (e + a * t1 + b * t2)  # on the inset cube
                p_sph = r0 * p_in / np.linalg.norm(p_in)
                for k in range(m):
                    t = k / (m - 1)
                    ids[i, j, k] = nid(p_sph + t * (p_in - p_sph))
        blocks.append(ids)

    # H-grid: the 3x3x3 boxes around the inset cube (centre box omitted).
    # The middle segment reuses the O-shell face node count/spacing so shared
    # faces merge node-for-node.
    segs = [np.linspace(-1.0, -cw, mh), np.linspace(-cw, cw, n),
            np.linspace(cw, 1.0, mh)]
    for si, sj, sk in product(range(3), repeat=3):
        if si == sj == sk == 1:
            continue  # the inset cube interior is the O-shell + cavity
        xs, ys, zs = segs[si], segs[sj], segs[sk]
        ids = np.empty((len(xs), len(ys), len(zs)), dtype=np.int64)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                for k, z in enumerate(zs):
                    ids[i, j, k] = nid((x, y, z))
        blocks.append(ids)
    return np.asarray(X), blocks


def nurbs_sphere(r0, center=(0.0, 0.0, 0.0)):
    """Exact NURBS sphere of radius ``r0`` as a surface of revolution.

    A degree-2 nine-point NURBS full circle (u, around z) revolves a degree-2
    five-point NURBS meridian half-circle (v, north->south); the combined
    weights are the product, and the pole rows collapse to a point (the
    standard degenerate-pole rational sphere). Bit-for-bit the analytic sphere
    geometrically (eval radius error ~1e-16), but routed through the heavy
    ``BSplineSurfaceParam`` device project — the point of this benchmark.
    """
    s2 = np.sqrt(2.0) / 2.0
    circ = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0),
                     (-1, -1), (0, -1), (1, -1), (1, 0)], dtype=float)
    wc = np.array([1, s2, 1, s2, 1, s2, 1, s2, 1], dtype=float)
    knots_u = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4], dtype=float)
    merid = np.array([(0, 1), (1, 1), (1, 0), (1, -1), (0, -1)], dtype=float)  # (radius, height)
    wm = np.array([1, s2, 1, s2, 1], dtype=float)
    knots_v = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=float)
    nu, nv = 9, 5
    ctrl = np.zeros((nu, nv, 3))
    weights = np.zeros((nu, nv))
    c = np.asarray(center, float)
    for j in range(nu):
        for i in range(nv):
            x_i, z_i = merid[i]
            ctrl[j, i] = c + r0 * np.array([x_i * circ[j, 0], x_i * circ[j, 1], z_i])
            weights[j, i] = wm[i] * wc[j]
    return BSplineSurface(2, 2, knots_u, knots_v, ctrl, weights)


def nurbs_plane(o, a1, a2, half=1.0):
    """A flat cube face as a degree-1 (bilinear) NURBS surface patch.

    Spans ``o + s*a1 + t*a2`` for ``s, t in [-half, half]`` with the four
    corners as control points — geometrically the analytic ``Plane`` restricted
    to the face, but routed through the heavy ``BSplineSurfaceParam`` device
    project so the cube faces exercise the NURBS path too (not just the sphere).
    """
    o = np.asarray(o, float)
    a1 = np.asarray(a1, float)
    a2 = np.asarray(a2, float)
    ctrl = np.zeros((2, 2, 3))
    for i, s in enumerate((-half, half)):
        for j, t in enumerate((-half, half)):
            ctrl[i, j] = o + s * a1 + t * a2
    knots = np.array([0, 0, 1, 1], dtype=float)
    return BSplineSurface(1, 1, knots, knots, ctrl)


def classify(X, r0, tol=1e-9):
    """Per-node entity objects from position: sphere / plane / edge / corner.

    Returns (dof_entities, tags, fixed) where dof_entities maps nid → entity
    object (or None for free), tags is the int tag array (for stats/plots),
    and fixed marks the 8 cube corners. Both the sphere cavity AND the cube
    faces are NURBS surfaces (heavy device project); only the cube edges
    (Line3) and interior stay cheap.
    """
    N = X.shape[0]
    tags = np.zeros(N, dtype=np.int32)
    dof_entities: dict[int, object] = {}
    fixed = np.zeros(N, dtype=bool)
    sphere = nurbs_sphere(r0)  # one shared NURBS sphere for every cavity node
    # Six shared NURBS cube-face patches, keyed by (axis, sign).
    faces = {}
    for fax in range(3):
        for sgn in (-1.0, 1.0):
            fo = np.zeros(3)
            fo[fax] = sgn
            fa1, fa2 = np.eye(3)[(fax + 1) % 3], np.eye(3)[(fax + 2) % 3]
            faces[(fax, sgn)] = nurbs_plane(fo, fa1, fa2)
    for nid in range(N):
        p = X[nid]
        if abs(np.linalg.norm(p) - r0) < tol:
            tags[nid] = TAG_SPHERE
            dof_entities[nid] = sphere
            continue
        on = [ax for ax in range(3) if abs(abs(p[ax]) - 1.0) < tol]
        if len(on) == 3:
            fixed[nid] = True
        elif len(on) == 2:
            # Cube edge: a Line3 along the remaining free axis.
            free_ax = ({0, 1, 2} - set(on)).pop()
            p0, p1 = p.copy(), p.copy()
            p0[free_ax], p1[free_ax] = -1.0, 1.0
            tags[nid] = TAG_LINE3
            dof_entities[nid] = Line3(p0, p1, 0.0, 1.0)
        elif len(on) == 1:
            # Cube face: a NURBS planar patch for the face it lies on.
            ax = on[0]
            tags[nid] = TAG_PLANE
            dof_entities[nid] = faces[(ax, float(np.sign(p[ax])))]
        else:
            tags[nid] = TAG_FREE
            dof_entities[nid] = None
    return dof_entities, tags, fixed


def build_context(X, blocks, dof_entities, tags, fixed):
    """Hand-assembled flattened sweep context (identity target, W_inv = I)."""
    N = X.shape[0]

    samples = []  # (gc, (gn0, gn1, gn2), (s0, s1, s2))
    node_samples = [[] for _ in range(N)]
    adj = [set() for _ in range(N)]
    for ids in blocks:
        na, nb, nc = ids.shape
        for ci, cj, ck in product(range(na - 1), range(nb - 1), range(nc - 1)):
            cell = ids[ci:ci + 2, cj:cj + 2, ck:ck + 2]
            cell_ids = [int(v) for v in cell.ravel()]
            for u in cell_ids:
                for v in cell_ids:
                    if u != v:
                        adj[u].add(v)
            for o in product((0, 1), repeat=3):
                corner = int(cell[o])
                gn, s = [], []
                for ax in range(3):
                    nb = list(o)
                    nb[ax] = 1 - nb[ax]
                    gn.append(int(cell[tuple(nb)]))
                    s.append(1.0 if o[ax] == 0 else -1.0)
                si = len(samples)
                samples.append((corner, tuple(gn), tuple(s)))
                for u in cell_ids:
                    node_samples[u].append(si)

    moving = [nid for nid in range(N) if not fixed[nid]]

    # Welsh-Powell greedy colouring of the share-a-cell graph (mirrors
    # egg.smoothing.solver._greedy_colour).
    order = sorted(range(N), key=lambda v: -len(adj[v]))
    colours = [-1] * N
    for v in order:
        used = {colours[u] for u in adj[v] if colours[u] != -1}
        c = 0
        while c in used:
            c += 1
        colours[v] = c
    n_colours = max(colours) + 1

    groups = []
    for c in range(n_colours):
        dofs = [nid for nid in moving if colours[nid] == c]
        if not dofs:
            continue
        gc, gn, ss, role, dof_idx, P_of = [], [[], [], []], [[], [], []], [], [], []
        for nid in dofs:
            sids = node_samples[nid]
            P_of.append(len(sids))
            dof_idx.append(nid)
            for si in sids:
                sc, sgn, s = samples[si]
                gc.append(sc)
                for ax in range(3):
                    gn[ax].append(sgn[ax])
                    ss[ax].append(s[ax])
                if sc == nid:
                    role.append(0)
                elif nid in sgn:
                    role.append(1 + sgn.index(nid))
                else:
                    role.append(-1)
        P = len(gc)
        S = np.stack([np.asarray(ss[0]), np.asarray(ss[1]), np.asarray(ss[2])],
                     axis=1)
        W_inv = np.broadcast_to(np.eye(3), (P, 3, 3))
        J = make_chain_J_nd(S, W_inv)
        groups.append({
            "D": len(dofs),
            "gc": np.asarray(gc, dtype=np.int32),
            "gn0": np.asarray(gn[0], dtype=np.int32),
            "gn1": np.asarray(gn[1], dtype=np.int32),
            "gn2": np.asarray(gn[2], dtype=np.int32),
            "s0": np.asarray(ss[0], dtype=np.float64),
            "s1": np.asarray(ss[1], dtype=np.float64),
            "s2": np.asarray(ss[2], dtype=np.float64),
            "W_inv": np.ascontiguousarray(W_inv.reshape(P, 9)),
            "role": np.asarray(role, dtype=np.int32),
            "J": np.ascontiguousarray(J.reshape(P, 9 * 12)),
            "dof_idx": np.asarray(dof_idx, dtype=np.int32),
            "entities": group_entities_by_type(dof_idx, dof_entities, d=3),
            "P_of": np.asarray(P_of, dtype=np.int32),
        })

    ns = len(samples)
    energy_stencil = {
        "num_samples": ns,
        "gc": np.asarray([s[0] for s in samples], dtype=np.int32),
        "gn0": np.asarray([s[1][0] for s in samples], dtype=np.int32),
        "gn1": np.asarray([s[1][1] for s in samples], dtype=np.int32),
        "gn2": np.asarray([s[1][2] for s in samples], dtype=np.int32),
        "s0": np.asarray([s[2][0] for s in samples], dtype=np.float64),
        "s1": np.asarray([s[2][1] for s in samples], dtype=np.float64),
        "s2": np.asarray([s[2][2] for s in samples], dtype=np.float64),
        "W_inv": np.ascontiguousarray(
            np.broadcast_to(np.eye(3), (ns, 3, 3)).reshape(ns, 9)),
    }
    return {"groups": groups, "energy_stencil": energy_stencil}


def grid_edges(blocks):
    """Unique node-id edges of all hex cells (for section plots)."""
    edges = set()
    for ids in blocks:
        for axis in range(3):
            lo = ids[tuple(slice(None, -1) if ax == axis else slice(None)
                           for ax in range(3))]
            hi = ids[tuple(slice(1, None) if ax == axis else slice(None)
                           for ax in range(3))]
            for u, v in zip(lo.ravel(), hi.ravel()):
                edges.add((min(int(u), int(v)), max(int(u), int(v))))
    return sorted(edges)


_SECTIONS = [(2, (0, 1), "XY (z=0)"), (0, (1, 2), "YZ (x=0)")]


def section_edges(X0, edges, tol=1e-6):
    """Per-section edge lists, selected ONCE from the initial lattice.

    Nodes sit exactly on the z=0 / x=0 symmetry planes only in the initial
    grid; smoothing moves them off by round-off, so membership must be frozen
    here and the live plot just re-reads their current coordinates.
    """
    out = []
    for plane_ax, _keep, _name in _SECTIONS:
        out.append([(u, v) for u, v in edges
                    if abs(X0[u][plane_ax]) < tol and abs(X0[v][plane_ax]) < tol])
    return out


def plot_topology(X, blocks, r0):
    """PyVista view of the declared block topology, one colour per block.

    "Rough" on purpose: each block is drawn as the 12 *straight* edges between
    its 8 corner nodes (the declared topology), not projected onto the curved
    sphere surface. The cavity sphere is shaded for reference.
    """
    import pyvista as pv
    from matplotlib import colormaps

    cmap = colormaps["tab20"]
    pl = pv.Plotter()
    for bi, ids in enumerate(blocks):
        corners = {(i, j, k): X[ids[-(i == 1), -(j == 1), -(k == 1)]]
                   for i, j, k in product((0, 1), repeat=3)}
        pts, cells = [], []
        for a, b in [((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)),
                     ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
                     ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)),
                     ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
                     ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)),
                     ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1))]:
            cells.extend([2, len(pts), len(pts) + 1])
            pts.extend([corners[a], corners[b]])
        mesh = pv.PolyData(np.asarray(pts), lines=np.asarray(cells))
        pl.add_mesh(mesh, color=cmap(bi % 20), line_width=3)
    pl.add_mesh(pv.Sphere(radius=r0), color="lightgray", opacity=0.4)
    pl.add_text(f"{len(blocks)} blocks (6 O-shell + {len(blocks) - 6} H)",
                font_size=10)
    pl.show()


class GridPlots:
    """PyVista panes: the XY/YZ section wireframes (+ an optional 3D pane).

    The section panes render the (frozen) section edges as 3D line meshes
    viewed along the plane normal with parallel projection; live updates just
    swap the shared point array in place, so redraws stay cheap.
    """

    def __init__(self, X, sections, edges, plot3d, off_screen=False):
        import pyvista as pv

        self._pv = pv
        n_panes = 3 if plot3d else 2
        self.plotter = pv.Plotter(shape=(1, n_panes), off_screen=off_screen,
                                  window_size=(520 * n_panes, 520))
        self.meshes = []
        for i, (sel, (_plane_ax, _keep, name), view) in enumerate(
                zip(sections, _SECTIONS, ("xy", "yz"))):
            self.plotter.subplot(0, i)
            mesh = self._lines(X, sel)
            self.plotter.add_mesh(mesh, color="blue", line_width=1)
            self.plotter.add_text(name, font_size=10)
            getattr(self.plotter, f"view_{view}")()
            self.plotter.enable_parallel_projection()
            self.meshes.append(mesh)
        if plot3d:
            self.plotter.subplot(0, n_panes - 1)
            mesh = self._lines(X, edges)
            self.plotter.add_mesh(mesh, color="blue", line_width=1)
            self.plotter.add_text("3D", font_size=10)
            self.meshes.append(mesh)

    def _lines(self, X, edge_list):
        e = np.asarray(edge_list, dtype=np.int64)
        cells = np.column_stack([np.full(e.shape[0], 2, dtype=np.int64), e])
        return self._pv.PolyData(np.asarray(X, dtype=float).copy(),
                                 lines=cells.ravel())

    def open_live(self):
        self.plotter.show(interactive_update=True, auto_close=False)

    def update(self, X):
        for mesh in self.meshes:
            mesh.points = np.asarray(X, dtype=float)
        if self.plotter.off_screen:
            self.plotter.render()
        else:
            self.plotter.update()

    def show(self):
        self.plotter.show()


def _plot_energy(energies, mindets):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(energies, "-o", ms=3)
    ax[0].set(title="TMOP energy", xlabel="sweep", ylabel="F")
    ax[1].axhline(0, color="r", lw=0.8)
    ax[1].plot(mindets, "-o", ms=2)
    ax[1].set(title="min det A", xlabel="sweep")
    plt.tight_layout()
    plt.show()


def main():
    p = argparse.ArgumentParser(
        description="3D sphere-in-cube O-grid smoothing demo.")
    p.add_argument("--n", type=int, default=9,
                   help="nodes per inset-cube face edge (odd keeps nodes on "
                        "the section planes)")
    p.add_argument("--m", type=int, default=4,
                   help="radial layers in the O-shell")
    p.add_argument("--mh", type=int, default=3,
                   help="nodes across each H-grid layer")
    p.add_argument("--r0", type=float, default=0.5, help="sphere radius")
    p.add_argument("--cw", type=float, default=0.7,
                   help="inset-cube half-width (O-shell outer boundary)")
    p.add_argument("--sweeps", type=int, default=40)
    p.add_argument("--chunk", type=int, default=10,
                   help="sweeps per device-resident chunk")
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--structured", action="store_true",
                   help="run the halo-padded structured path (coalesced stencil) "
                        "instead of the global colored-GS session")
    p.add_argument("--smoother", choices=["colored-gs", "block-jacobi"],
                   default="colored-gs",
                   help="structured-path smoother (requires --structured): the "
                        "in-place per-colour chain or one merged block-Jacobi launch")
    p.add_argument("--omega", type=float, default=1.0,
                   help="block-Jacobi SOR/damping weight (1.0 = undamped)")
    p.add_argument("--report-every", type=int, default=0,
                   help="report (energy, min_det) every N sweeps: 0 (default) "
                        "reports once per chunk (chunk-end, minimal reduction "
                        "launches — the small-n GPU regime); 1 reports every "
                        "sweep (legacy full per-sweep history for --plot-energy); "
                        "k > 1 reports every k-th sweep plus the final one.")
    p.add_argument("--plot-live", action="store_true",
                   help="PyVista animated XY/YZ sections (one frame per chunk)")
    p.add_argument("--plot-grid", action="store_true",
                   help="PyVista final XY/YZ section plots")
    p.add_argument("--plot-energy", action="store_true",
                   help="matplotlib energy + min-det convergence curves")
    p.add_argument("--plot-3d", action="store_true",
                   help="add a 3D wireframe pane to the live/final plots "
                        "(dome-example style)")
    p.add_argument("--plot-topology", action="store_true",
                   help="PyVista view of the declared block topology only — "
                        "no pipeline run")
    a = p.parse_args()

    print("=" * 56)
    print("d=3: NURBS sphere in a cube (6-block O-grid) → TMOP smooth")
    print("=" * 56)

    X, blocks = build_grid(a.n, a.m, a.mh, a.r0, a.cw)

    if a.plot_topology:
        plot_topology(X, blocks, a.r0)
        return

    dof_entities, tags, fixed = classify(X, a.r0)
    ctx = build_context(X, blocks, dof_entities, tags, fixed)
    edges = grid_edges(blocks)
    sections = section_edges(X, edges)
    n_sphere = int((tags == TAG_SPHERE).sum())
    print(f"nodes={X.shape[0]} sphere={n_sphere} "
          f"plane={(tags == TAG_PLANE).sum()} edge={(tags == TAG_LINE3).sum()} "
          f"fixed={fixed.sum()} free={(tags == TAG_FREE).sum() - fixed.sum()}")

    if a.structured:
        # Re-home the hand-built block lattice onto the halo-padded structured
        # store (owner/broadcast from the per-block global-DOF arrays; cross-block
        # stencil neighbours mirrored by the C++ singular-fan fallback). Enables
        # the merged block-Jacobi launch as well as the structured colored-GS.
        from egg.smoothing.cpp_backend import (
            build_structured_context_from_block_maps, structured_arrays)
        bsc = build_structured_context_from_block_maps(3, blocks, X.shape[0])
        structured = structured_arrays(bsc)
        session = cpp_core.CppStructuredSweepSession(
            ctx, structured, X.ravel(), device=a.device, dim=3)
        print(f"structured: blocks={bsc.num_blocks} shared-node copies={bsc.num_share_entries} "
              f"smoother={a.smoother} omega={a.omega}")
        run_kwargs = {"smoother": a.smoother, "omega": a.omega}
    else:
        if a.smoother != "colored-gs":
            p.error("--smoother requires --structured")
        session = cpp_core.CppSweepSession(ctx, X.ravel(), device=a.device, dim=3)
        run_kwargs = {}
    energies, mindets = [], []

    live = None
    if a.plot_live:
        live = GridPlots(X, sections, edges, a.plot_3d)
        live.open_live()

    done = 0
    while done < a.sweeps:
        step = min(a.chunk, a.sweeps - done)
        e, m = session.run(step, phase="barrier", delta=0.0,
                       report_every=a.report_every, **run_kwargs)
        energies.extend(np.asarray(e))
        mindets.extend(np.asarray(m))
        done += step
        print(f"  sweeps={done:4d} energy={energies[-1]:.4e} "
              f"min_det={mindets[-1]:.4e}")
        if live is not None:
            live.update(session.get_X().reshape(-1, 3))

    if live is not None:
        live.show()

    X_out = session.get_X().reshape(-1, 3)

    print(f"energy : {energies[0]:.4e} -> {energies[-1]:.4e}")
    print(f"min det: {mindets[0]:.4e} -> {mindets[-1]:.4e}")

    # Constraint checks: every constrained node is still on its entity.
    sph = tags == TAG_SPHERE
    sph_dev = np.abs(np.linalg.norm(X_out[sph], axis=1) - a.r0).max()
    pl = tags == TAG_PLANE
    # Face nodes stay on the cube surface: their max-abs coordinate is 1.
    pl_dev = max(abs(np.max(np.abs(X_out[i])) - 1.0) for i in np.flatnonzero(pl))
    print(f"sphere |x|-r0 max dev: {sph_dev:.2e}; plane dev: {pl_dev:.2e}")
    assert mindets[-1] > 0.0
    assert sph_dev < 1e-9 and pl_dev < 1e-9

    if a.plot_grid:
        GridPlots(X_out, sections, edges, a.plot_3d).show()
    if a.plot_energy:
        _plot_energy(energies, mindets)

    print("Done.")


if __name__ == "__main__":
    main()
