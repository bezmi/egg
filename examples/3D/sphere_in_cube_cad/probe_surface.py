# MIT License
#
# Copyright (c) 2026 Shahzeb Imran and the Egg contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Inspect the sphere-surface grid at init (projection only, no TMOP solve).

Builds the ``sphere_in_cube_cad`` topology, runs ``initialize_grid`` (corners ->
curve-aware edge spacing -> per-face TFI + projection onto the sphere -> volume
TFI), and shows ONLY the sphere-surface quads, so the projection is visible
without the smoother touching it. ``--sphere`` swaps the imported CAD sphere for
the hand-authored NURBS sphere or the analytic sphere to compare projection
behaviour (e.g. the NURBS pole/seam degeneracy vs the closed-form analytic one).

Prints per-cell area spread and the radial deviation so the pole degeneracy is
quantified, not just eyeballed.

Run: ``uv run --no-sync python probe_surface.py --n 11 --sphere cad``.
Needs the ``cad`` group for ``--sphere cad``; plotting needs ``pyvista``.
"""

from __future__ import annotations

import argparse

import numpy as np

import sphere_in_cube_cad as ex  # sibling example, this directory
from driver import surface_faces  # local sibling module


def _sphere_entity(kind, r0):
    """The constraint sphere for ``--sphere``; None means the imported CAD one."""
    if kind == "cad":
        return None
    if kind == "nurbs":
        from sphere_in_cube_nurbs import nurbs_sphere

        return nurbs_sphere(r0)
    from egg.geometry.analytic3d import Sphere

    return Sphere((0.0, 0.0, 0.0), r0, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def build_init(n, kind, m=4, n_h=3, r0=0.5, cw=0.7):
    """Topology + initialize_grid only (no solve), with the chosen sphere."""
    override = _sphere_entity(kind, r0)
    if override is None:
        return ex.sphere_in_cube_cad(n, m, n_h, r0=r0, cw=cw).build().initialize_grid()
    orig = ex.cad_geometry

    def patched(radius):  # keep the CAD cube faces, swap only the sphere
        _, cube_face = orig(radius)
        return override, cube_face

    ex.cad_geometry = patched
    try:
        return ex.sphere_in_cube_cad(n, m, n_h, r0=r0, cw=cw).build().initialize_grid()
    finally:
        ex.cad_geometry = orig


def surface_quality(X, quads, r0):
    """Radial deviation and per-quad area spread of the sphere-surface grid."""
    r = np.linalg.norm(X[np.unique(quads)], axis=1)
    P = X[quads]  # (nq, 4, 3)
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 3] - P[:, 0]), axis=1)
    area += 0.5 * np.linalg.norm(np.cross(P[:, 3] - P[:, 2], P[:, 1] - P[:, 2]), axis=1)
    print(f"  surface cells    : {len(quads)}")
    print(f"  radius dev |r-r0|: max {np.abs(r - r0).max():.3e}")
    print(
        f"  cell area        : min {area.min():.3e}  max {area.max():.3e}  "
        f"max/min {area.max() / max(area.min(), 1e-30):.1f}"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="probe the init sphere-surface grid")
    p.add_argument("--n", type=int, default=9, help="tangential nodes per block face")
    p.add_argument("--m", type=int, default=4, help="radial nodes across the O-shell")
    p.add_argument("--nh", type=int, default=3, help="nodes across each H-grid layer")
    p.add_argument("--r0", type=float, default=0.5, help="sphere radius")
    p.add_argument("--cw", type=float, default=0.7, help="inset-cube half-width")
    p.add_argument(
        "--sphere",
        choices=["cad", "nurbs", "analytic"],
        default="cad",
        help="cad = imported build123d; nurbs = hand-authored; analytic = closed-form",
    )
    p.add_argument("--no-plot", action="store_true", help="print diagnostics only")
    a = p.parse_args(argv)

    grid = build_init(a.n, a.sphere, m=a.m, n_h=a.nh, r0=a.r0, cw=a.cw)
    X = grid.global_nodes
    on_sphere = np.abs(np.linalg.norm(X, axis=1) - a.r0) < 1e-6
    quads = surface_faces(ex._block_id_arrays(grid), on_sphere)
    print(
        f"sphere={a.sphere} n={a.n} (cells={a.n - 1})  nodes on sphere={int(on_sphere.sum())}"
    )
    surface_quality(X, quads, a.r0)

    if a.no_plot:
        return
    import pyvista as pv

    cells = np.hstack([np.full((len(quads), 1), 4, np.int64), quads]).ravel()
    mesh = pv.PolyData(np.asarray(X, dtype=float), faces=cells)
    pl = pv.Plotter()
    pl.add_mesh(
        mesh, color="lightgray", show_edges=True, edge_color="blue", line_width=1
    )
    pl.add_text(f"{a.sphere} sphere, init grid (n={a.n})", font_size=10)
    pl.show()


if __name__ == "__main__":
    main()
