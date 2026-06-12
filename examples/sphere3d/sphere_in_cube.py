"""Sphere inside a cube — the 3D analogue of the 2D circle-in-rectangle O-grid.

A 6-block "cubed-sphere" shell fills the region between a spherical cavity of
radius ``r0`` and the cube ``[-1, 1]^3``: each block maps one cube face
(``n x n`` nodes) radially inward to the matching spherical cap (``m`` layers).
Shared block faces/edges are merged into single global nodes by position.

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

    uv run sphere_in_cube.py [--n N] [--m M] [--r0 R] [--sweeps N] [--chunk N]
        [--device cpu|gpu|auto] [--plot-live] [--plot-grid] [--plot-energy]
"""

import argparse
from itertools import product

import numpy as np

from egg._cpp import cpp_core
from egg.geometry.entity_encoding import (
    PARAM_PAD_SIZE,
    TAG_FREE,
    TAG_PLANE,
    TAG_SPHERE,
)
from egg.smoothing.batch import make_chain_J_nd

TAG_LINE3 = 13  # 3D edge curve (C++ tag; no 2D Python entity class)

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


def build_grid(n, m, r0):
    """Cubed-sphere shell nodes + per-block id lattices (merged by position).

    Returns (X (N,3), blocks [(n,n,m) id arrays], frames per block).
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
    ab = np.linspace(-1.0, 1.0, n)
    for e, t1, t2 in _FACES:
        e, t1, t2 = map(np.asarray, (e, t1, t2))
        ids = np.empty((n, n, m), dtype=np.int64)
        for i, a in enumerate(ab):
            for j, b in enumerate(ab):
                p_cube = e + a * t1 + b * t2
                p_sph = r0 * p_cube / np.linalg.norm(p_cube)
                for k in range(m):
                    t = k / (m - 1)
                    ids[i, j, k] = nid(p_sph + t * (p_cube - p_sph))
        blocks.append(ids)
    return np.asarray(X), blocks


def classify(X, r0, tol=1e-9):
    """Per-node (tag, params) from position: sphere / plane / edge / corner."""
    N = X.shape[0]
    tags = np.zeros(N, dtype=np.int32)
    params = np.zeros((N, PARAM_PAD_SIZE))
    fixed = np.zeros(N, dtype=bool)
    for nid in range(N):
        p = X[nid]
        if abs(np.linalg.norm(p) - r0) < tol:
            tags[nid] = TAG_SPHERE
            # Blob: [c(3), r, ax(3), ay(3)].
            params[nid, 3] = r0
            params[nid, 4:7] = (1.0, 0.0, 0.0)
            params[nid, 7:10] = (0.0, 1.0, 0.0)
            continue
        on = [ax for ax in range(3) if abs(abs(p[ax]) - 1.0) < tol]
        if len(on) == 3:
            fixed[nid] = True
        elif len(on) == 2:
            # Cube edge: a TAG_LINE3 along the remaining free axis.
            free_ax = ({0, 1, 2} - set(on)).pop()
            p0, p1 = p.copy(), p.copy()
            p0[free_ax], p1[free_ax] = -1.0, 1.0
            tags[nid] = TAG_LINE3
            # Blob: [p0(3), p1(3), t0, t1].
            params[nid, :3] = p0
            params[nid, 3:6] = p1
            params[nid, 6:8] = (0.0, 1.0)
        elif len(on) == 1:
            # Cube face: a TAG_PLANE through the face spanned by the others.
            ax = on[0]
            o = np.zeros(3)
            o[ax] = np.sign(p[ax])
            a1, a2 = np.eye(3)[(ax + 1) % 3], np.eye(3)[(ax + 2) % 3]
            tags[nid] = TAG_PLANE
            # Blob: [o(3), ax(3), ay(3)].
            params[nid, :3] = o
            params[nid, 3:6] = a1
            params[nid, 6:9] = a2
        else:
            tags[nid] = TAG_FREE
    return tags, params, fixed


def build_context(X, blocks, tags, params, fixed):
    """Hand-assembled flattened sweep context (identity target, W_inv = I)."""
    N = X.shape[0]

    samples = []  # (gc, (gn0, gn1, gn2), (s0, s1, s2))
    node_samples = [[] for _ in range(N)]
    adj = [set() for _ in range(N)]
    for ids in blocks:
        n, _, m = ids.shape
        for ci, cj, ck in product(range(n - 1), range(n - 1), range(m - 1)):
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
            "tag": tags[dof_idx],
            "P_of": np.asarray(P_of, dtype=np.int32),
            "params": np.ascontiguousarray(params[dof_idx]),
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
                   help="nodes per cube-face edge (odd keeps nodes on the "
                        "section planes)")
    p.add_argument("--m", type=int, default=5, help="radial layers")
    p.add_argument("--r0", type=float, default=0.5, help="sphere radius")
    p.add_argument("--sweeps", type=int, default=40)
    p.add_argument("--chunk", type=int, default=10,
                   help="sweeps per device-resident chunk")
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--plot-live", action="store_true",
                   help="PyVista animated XY/YZ sections (one frame per chunk)")
    p.add_argument("--plot-grid", action="store_true",
                   help="PyVista final XY/YZ section plots")
    p.add_argument("--plot-energy", action="store_true",
                   help="matplotlib energy + min-det convergence curves")
    p.add_argument("--plot-3d", action="store_true",
                   help="add a 3D wireframe pane to the live/final plots "
                        "(dome-example style)")
    a = p.parse_args()

    print("=" * 56)
    print("d=3: sphere in a cube (6-block O-grid) → TMOP smooth")
    print("=" * 56)

    X, blocks = build_grid(a.n, a.m, a.r0)
    tags, params, fixed = classify(X, a.r0)
    ctx = build_context(X, blocks, tags, params, fixed)
    edges = grid_edges(blocks)
    sections = section_edges(X, edges)
    n_sphere = int((tags == TAG_SPHERE).sum())
    print(f"nodes={X.shape[0]} sphere={n_sphere} "
          f"plane={(tags == TAG_PLANE).sum()} edge={(tags == TAG_LINE3).sum()} "
          f"fixed={fixed.sum()} free={(tags == TAG_FREE).sum() - fixed.sum()}")

    session = cpp_core.CppSweepSession(ctx, X.ravel(), device=a.device, dim=3)
    energies, mindets = [], []

    live = None
    if a.plot_live:
        live = GridPlots(X, sections, edges, a.plot_3d)
        live.open_live()

    done = 0
    while done < a.sweeps:
        step = min(a.chunk, a.sweeps - done)
        e, m = session.run(step, phase="barrier", delta=0.0)
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
    pl_dev = max(
        abs(abs(X_out[i][np.argmax(np.abs(params[i, :3]))]) - 1.0)
        for i in np.flatnonzero(pl))
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
