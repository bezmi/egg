# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Driver for the 3D examples. ``dome.py``, ``sphere_in_cube.py`` and
``sphere_in_cube_nurbs.py`` hold the grid and context setup."""

import argparse
from itertools import product

import numpy as np

from egg._cpp import cpp_core
from egg.geometry.entity_encoding import TAG_FREE, TAG_PLANE, TAG_SPHERE
from egg.geometry.entity_soa import TAG_LINE3


def _add_run_args(p, *, sweeps):
    p.add_argument(
        "--sweeps",
        type=int,
        default=sweeps,
        help="block-Jacobi sweeps (or FAS V-cycles with --smoother fas)",
    )
    p.add_argument(
        "--chunk", type=int, default=10, help="sweeps per device-resident chunk"
    )
    p.add_argument(
        "--smoother",
        choices=["jacobi", "fas"],
        default="jacobi",
        help="jacobi = plain block-Jacobi sweeps; fas = FAS (nonlinear geometric "
        "multigrid) V-cycles — --sweeps/--chunk then count cycles (each is 4 fine "
        "sweeps plus the coarse-grid work). Falls back to Jacobi when the block "
        "shapes admit no coarse level.",
    )
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument(
        "--omega",
        type=float,
        default=0.8,
        help="block-Jacobi SOR/damping weight (1.0 = undamped)",
    )
    p.add_argument(
        "--report-every",
        type=int,
        default=0,
        help="report (energy, min_det) every N sweeps: 0 (default) "
        "reports once per chunk (chunk-end, minimal reduction "
        "launches — the small-n GPU regime); 1 reports every "
        "sweep (legacy full per-sweep history for --plot-energy); "
        "k > 1 reports every k-th sweep plus the final one.",
    )


def _drive(session, a, on_chunk=None):
    """Chunked device-resident drive loop; returns (energies, mindets).

    The context/X are uploaded once and the sweeps run in chunks so live
    plotting only pays one download per chunk.
    """
    energies, mindets = [], []
    unit = "cycles" if getattr(a, "smoother", "jacobi") == "fas" else "sweeps"
    done = 0
    while done < a.sweeps:
        step = min(a.chunk, a.sweeps - done)
        if unit == "cycles":
            e, m = session.run_fas(step, omega=a.omega)
        else:
            e, m = session.run(
                step,
                phase="barrier",
                delta=0.0,
                report_every=a.report_every,
                omega=a.omega,
            )
        energies.extend(np.asarray(e))
        mindets.extend(np.asarray(m))
        done += step
        print(f"  {unit}={done:4d} energy={energies[-1]:.4e} min_det={mindets[-1]:.4e}")
        if on_chunk is not None:
            on_chunk(done)
    return energies, mindets


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


# --- dome: matplotlib 3D wireframe -----------------------------------------


def _grid_polylines(X, n):
    """The structured grid's wireframe as a list of (n, 3) polylines."""
    P = X.reshape(n, n, n, 3)
    lines = []
    for a in range(n):
        for b in range(n):
            lines.append(P[a, b, :, :])  # k-lines
            lines.append(P[a, :, b, :])  # j-lines
            lines.append(P[:, a, b, :])  # i-lines
    return lines


def _draw_wireframe(ax, X, n, title=""):
    ax.cla()
    for line in _grid_polylines(X, n):
        ax.plot(line[:, 0], line[:, 1], line[:, 2], "b-", lw=0.4)
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1.1))


# --- sphere-in-cube: PyVista section/topology plots -------------------------

_SECTIONS = [(2, (0, 1), "XY (z=0)"), (0, (1, 2), "YZ (x=0)")]


def grid_edges(blocks):
    """Unique node-id edges of all hex cells (for section plots)."""
    edges = set()
    for ids in blocks:
        for axis in range(3):
            lo = ids[
                tuple(slice(None, -1) if ax == axis else slice(None) for ax in range(3))
            ]
            hi = ids[
                tuple(slice(1, None) if ax == axis else slice(None) for ax in range(3))
            ]
            for u, v in zip(lo.ravel(), hi.ravel()):
                edges.add((min(int(u), int(v)), max(int(u), int(v))))
    return sorted(edges)


def section_edges(X0, edges, tol=1e-6):
    """Per-section edge lists, selected ONCE from the initial lattice.

    Nodes sit exactly on the z=0 / x=0 symmetry planes only in the initial
    grid; smoothing moves them off by round-off, so membership must be frozen
    here and the live plot just re-reads their current coordinates.
    """
    out = []
    for plane_ax, _keep, _name in _SECTIONS:
        out.append(
            [
                (u, v)
                for u, v in edges
                if abs(X0[u][plane_ax]) < tol and abs(X0[v][plane_ax]) < tol
            ]
        )
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
        corners = {
            (i, j, k): X[ids[-(i == 1), -(j == 1), -(k == 1)]]
            for i, j, k in product((0, 1), repeat=3)
        }
        pts, cells = [], []
        for a, b in [
            ((0, 0, 0), (1, 0, 0)),
            ((0, 1, 0), (1, 1, 0)),
            ((0, 0, 1), (1, 0, 1)),
            ((0, 1, 1), (1, 1, 1)),
            ((0, 0, 0), (0, 1, 0)),
            ((1, 0, 0), (1, 1, 0)),
            ((0, 0, 1), (0, 1, 1)),
            ((1, 0, 1), (1, 1, 1)),
            ((0, 0, 0), (0, 0, 1)),
            ((1, 0, 0), (1, 0, 1)),
            ((0, 1, 0), (0, 1, 1)),
            ((1, 1, 0), (1, 1, 1)),
        ]:
            cells.extend([2, len(pts), len(pts) + 1])
            pts.extend([corners[a], corners[b]])
        mesh = pv.PolyData(np.asarray(pts), lines=np.asarray(cells))
        pl.add_mesh(mesh, color=cmap(bi % 20), line_width=3)
    pl.add_mesh(pv.Sphere(radius=r0), color="lightgray", opacity=0.4)
    pl.add_text(f"{len(blocks)} blocks (6 O-shell + {len(blocks) - 6} H)", font_size=10)
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
        self.plotter = pv.Plotter(
            shape=(1, n_panes), off_screen=off_screen, window_size=(520 * n_panes, 520)
        )
        self.meshes = []
        for i, (sel, (_plane_ax, _keep, name), view) in enumerate(
            zip(sections, _SECTIONS, ("xy", "yz"))
        ):
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
        return self._pv.PolyData(np.asarray(X, dtype=float).copy(), lines=cells.ravel())

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


# --- entry points ------------------------------------------------------------


def main_dome():
    import dome

    p = argparse.ArgumentParser(description="3D dome-on-a-cube smoothing demo.")
    p.add_argument("--n", type=int, default=7, help="nodes per edge")
    _add_run_args(p, sweeps=30)
    p.add_argument(
        "--plot-live",
        action="store_true",
        help="matplotlib animated 3D wireframe (one frame per chunk)",
    )
    p.add_argument(
        "--plot-grid", action="store_true", help="matplotlib final 3D wireframe grid"
    )
    p.add_argument(
        "--plot-energy",
        action="store_true",
        help="matplotlib energy + min-det convergence curves",
    )
    a = p.parse_args()

    print("=" * 56)
    print("d=3: spherical-cap face on a unit cube → TMOP smooth")
    print("=" * 56)
    for k, v in vars(a).items():
        print(f"  {k} = {v}")
    print("=" * 56)

    n = a.n
    # Sphere centred below the top face; the cap bulges to z ~ 1.08 mid-face.
    c = np.array([0.5, 0.5, -1.0])
    r = 2.08
    ctx = dome.build_context(n, c, r)
    X, top = dome.initial_lattice(n, c, r)

    # The dome lattice is a single n^3 block (C-order node ids), so the
    # structured store has an empty halo; the sweep relaxes block-Jacobi.
    from egg.smoothing.cpp_backend import (
        build_structured_context_from_block_maps,
        structured_arrays,
    )

    blocks = [np.arange(n**3).reshape(n, n, n)]
    bsc = build_structured_context_from_block_maps(3, blocks)
    session = cpp_core.CppStructuredSweepSession(
        ctx, structured_arrays(bsc), X.ravel(), device=a.device, dim=3
    )
    print(f"structured: blocks={bsc.num_blocks} omega={a.omega}")

    live = None
    if a.plot_live:
        import matplotlib.pyplot as plt

        plt.ion()
        fig = plt.figure(figsize=(7, 7))
        live = fig.add_subplot(projection="3d")
        _draw_wireframe(live, X.ravel(), n, "sweep 0")
        plt.pause(0.01)

    def on_chunk(done):
        if live is not None:
            import matplotlib.pyplot as plt

            _draw_wireframe(live, session.get_X(), n, f"sweep {done}")
            plt.pause(0.01)

    energies, mindets = _drive(session, a, on_chunk)

    if live is not None:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()

    X_out = session.get_X().reshape(-1, 3)

    print(f"energy : {energies[0]:.4e} -> {energies[-1]:.4e}")
    print(f"min det: {mindets[0]:.4e} -> {mindets[-1]:.4e}")

    # The constrained top-face nodes must still sit on the sphere.
    radii = np.linalg.norm(X_out[top] - c, axis=1)
    print(
        f"top-face |x - c|: max deviation from r = "
        f"{np.abs(radii - r).max():.2e} (r = {r})"
    )
    assert mindets[-1] > 0.0
    assert np.abs(radii - r).max() < 1e-9

    if a.plot_grid:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(projection="3d")
        _draw_wireframe(ax, X_out.ravel(), n, "final grid")
        plt.show()
    if a.plot_energy:
        _plot_energy(energies, mindets)

    print("Done.")


def main_sphere_in_cube(setup, banner):
    """Shared driver for the analytic and NURBS sphere-in-cube variants.

    ``setup`` is the example module providing ``build_grid`` / ``classify`` /
    ``build_context``.
    """
    p = argparse.ArgumentParser(description="3D sphere-in-cube O-grid smoothing demo.")
    p.add_argument(
        "--n",
        type=int,
        default=9,
        help="nodes per inset-cube face edge (odd keeps nodes on the section planes)",
    )
    p.add_argument("--m", type=int, default=4, help="radial layers in the O-shell")
    p.add_argument("--mh", type=int, default=3, help="nodes across each H-grid layer")
    p.add_argument("--r0", type=float, default=0.5, help="sphere radius")
    p.add_argument(
        "--cw",
        type=float,
        default=0.7,
        help="inset-cube half-width (O-shell outer boundary)",
    )
    _add_run_args(p, sweeps=40)
    p.add_argument(
        "--plot-live",
        action="store_true",
        help="PyVista animated XY/YZ sections (one frame per chunk)",
    )
    p.add_argument(
        "--plot-grid", action="store_true", help="PyVista final XY/YZ section plots"
    )
    p.add_argument(
        "--plot-energy",
        action="store_true",
        help="matplotlib energy + min-det convergence curves",
    )
    p.add_argument(
        "--plot-3d",
        action="store_true",
        help="add a 3D wireframe pane to the live/final plots (dome-example style)",
    )
    p.add_argument(
        "--plot-topology",
        action="store_true",
        help="PyVista view of the declared block topology only — no pipeline run",
    )
    a = p.parse_args()

    print("=" * 56)
    print(banner)
    print("=" * 56)
    for k, v in vars(a).items():
        print(f"  {k} = {v}")
    print("=" * 56)

    X, blocks = setup.build_grid(a.n, a.m, a.mh, a.r0, a.cw)

    if a.plot_topology:
        plot_topology(X, blocks, a.r0)
        return

    dof_entities, tags, fixed = setup.classify(X, a.r0)
    ctx = setup.build_context(X, blocks, dof_entities, tags, fixed)
    edges = grid_edges(blocks)
    sections = section_edges(X, edges)
    n_sphere = int((tags == TAG_SPHERE).sum())
    print(
        f"nodes={X.shape[0]} sphere={n_sphere} "
        f"plane={(tags == TAG_PLANE).sum()} edge={(tags == TAG_LINE3).sum()} "
        f"fixed={fixed.sum()} free={(tags == TAG_FREE).sum() - fixed.sum()}"
    )

    # Re-home the hand-built block lattice onto the halo-padded structured store
    # (owner/broadcast from the per-block global-DOF arrays; cross-block stencil
    # neighbours mirrored by the C++ singular-fan fallback), then relax block-Jacobi.
    from egg.smoothing.cpp_backend import (
        build_structured_context_from_block_maps,
        structured_arrays,
    )

    bsc = build_structured_context_from_block_maps(3, blocks)
    session = cpp_core.CppStructuredSweepSession(
        ctx, structured_arrays(bsc), X.ravel(), device=a.device, dim=3
    )
    print(f"structured: blocks={bsc.num_blocks} omega={a.omega}")

    live = None
    if a.plot_live:
        live = GridPlots(X, sections, edges, a.plot_3d)
        live.open_live()

    def on_chunk(_done):
        if live is not None:
            live.update(session.get_X().reshape(-1, 3))

    energies, mindets = _drive(session, a, on_chunk)

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
    # The deviation floor is ~1e-15 in double but ~1e-7 in fp32, so this is
    # reported (above) rather than asserted, keeping the fp32 run's plots viewable.

    if a.plot_grid:
        GridPlots(X_out, sections, edges, a.plot_3d).show()
    if a.plot_energy:
        _plot_energy(energies, mindets)

    print("Done.")
