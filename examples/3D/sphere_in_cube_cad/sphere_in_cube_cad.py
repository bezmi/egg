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

"""Sphere-in-cube O-shell + H-grid with geometry imported from build123d (CAD).

Same 32-block topology as ``sphere_in_cube_nurbs`` (6 cubed-sphere O-shell blocks
wrapping the cavity out to an inset cube of half-width ``cw``, plus the 26
axis-aligned H-grid blocks of the 3x3x3 decomposition filling the gap out to the
outer cube ``[-1, 1]^3`` -- 6 face + 12 edge + 8 corner blocks), but authored
through ``TopologyBuilder(d=3)`` and smoothed by the standard ``generate_steps``
pipeline, and the constraint surfaces come from a build123d model rather than
analytic primitives: a ``Sphere`` and a ``Box`` are built with build123d, each
face is extracted to an egg ``BSplineSurface`` (with UV trim) by
:mod:`egg.io.cad`. The O-shell inner faces bind to the imported sphere and the
H-grid outer faces to the six imported cube-face patches; the solver only ever
sees egg's own device entities, so OCCT never runs at solve time.

Cube edges (nodes shared by two cube faces) are pinned by the builder's
multi-entity rule rather than sliding on a ``Line3`` as ``sphere_in_cube_nurbs``
does -- the block topology is identical, this is only a boundary-constraint detail.

Plotting reuses the shared ``sphere3d/driver.py`` panes: the XY (z=0) / YZ (x=0)
section wireframes, the mesh on the sphere surface, and the energy / min-det
curves. Only the drive loop is this example's own (the pipeline, not the
hand-built structured session the sphere3d drivers use).

Needs the ``cad`` group: ``uv sync --group cad``; plotting needs ``pyvista``.
Run: ``uv run --no-sync python sphere_in_cube_cad.py --device cpu --plot-grid``.
"""

from __future__ import annotations

import argparse
import os
from itertools import product

import numpy as np

from egg.io import cad
from egg.pipeline import (
    ControlPointSmoother,
    FasSmoother,
    JacobiSmoother,
    Presmooth,
    Refit,
    Save,
    Untangle,
    generate_steps,
)
from egg.topology.builder import TopologyBuilder

# sign of (e_i x e_j) . e_k for the two axes other than k, in ascending order.
_SIGN_IJ = {0: 1, 1: -1, 2: 1}


def cad_geometry(r0: float):
    """Build a sphere + the outer unit cube in build123d and extract egg surfaces.

    Returns ``(sphere_surface, {(axis, side): cube_face_surface})`` for the outer
    cube ``[-1, 1]^3`` (its faces sit at ``|coord| == 1``).
    """
    from build123d import Box, Sphere

    sphere = cad.face_to_surface(Sphere(r0).faces()[0], name="sphere", tag="sphere")
    cube_face = {}
    for face in Box(2.0, 2.0, 2.0).faces():
        c = np.asarray(tuple(face.center()), dtype=float)
        axis = int(np.argmax(np.abs(c)))
        side = 1 if c[axis] > 0 else 0
        cube_face[(axis, side)] = cad.face_to_surface(
            face, name="cube_%d_%d" % (axis, side), tag="wall"
        )
    return sphere, cube_face


def sphere_in_cube_cad(n=9, m=4, n_h=3, r0=0.5, cw=0.7):
    """The 32-block O-shell + H-grid topology, bound to build123d surfaces.

    Node counts matching ``sphere_in_cube_nurbs``: ``n`` tangential nodes per
    O-shell / H-middle face, ``m`` radial nodes across the O-shell, ``n_h`` nodes
    across each thin H-grid layer; ``r0`` sphere radius, ``cw`` inset-cube
    half-width. Keep ``n`` odd (even cell count) so a node lands on the z=0 / x=0
    section planes and the octant fans untangle.
    """
    c_tan, c_rad, c_h = n - 1, m - 1, n_h - 1  # TopologyBuilder resolutions are cells
    sphere, cube_face = cad_geometry(r0)
    signs = (-1, 1)
    tb = TopologyBuilder(d=3)

    # sphere corners (cavity) and the 4x4x4 cube lattice (values -1, -cw, cw, 1).
    for sx, sy, sz in product(signs, repeat=3):
        d = np.array([sx, sy, sz], float)
        tb.add_corner("S_%+d%+d%+d" % (sx, sy, sz), r0 * d / np.linalg.norm(d))
    vals = (-1.0, -cw, cw, 1.0)
    for i, j, k in product(range(4), repeat=3):
        outer = all(t in (0, 3) for t in (i, j, k))  # outer cube corner -> fixed
        tb.add_corner("L_%d%d%d" % (i, j, k), (vals[i], vals[j], vals[k]), fixed=outer)

    def inset(s):  # sign +-1 -> lattice index of +-cw
        return 1 if s < 0 else 2

    # 6 O-shell blocks: sphere -> inset cube. Inner face on the sphere; the outer
    # (inset) face is an interface with the surrounding H-grid, so it is unbound.
    for k in (0, 1, 2):
        i, j = (a for a in (0, 1, 2) if a != k)
        for s in signs:
            a1, a2 = (i, j) if s == _SIGN_IJ[k] else (j, i)

            def sph(t1, t2, _k=k, _s=s, _a1=a1, _a2=a2):
                d = [0, 0, 0]
                d[_k], d[_a1], d[_a2] = _s, t1, t2
                return "S_%+d%+d%+d" % (d[0], d[1], d[2])

            def ins(t1, t2, _k=k, _s=s, _a1=a1, _a2=a2):
                d = [0, 0, 0]
                d[_k], d[_a1], d[_a2] = _s, t1, t2
                return "L_%d%d%d" % (inset(d[0]), inset(d[1]), inset(d[2]))

            corners = tuple(
                (sph if i0 == 0 else ins)(signs[i1], signs[i2])
                for i0 in (0, 1)
                for i1 in (0, 1)
                for i2 in (0, 1)
            )
            name = "O_%d%+d" % (k, s)
            tb.add_block(name, corners=corners, resolutions=(c_rad, c_tan, c_tan))
            tb.associate(name, "west", sphere)  # inner face on the imported sphere

    # 26 H-grid blocks: the 3x3x3 boxes of [-1,1]^3 minus the O-shell centre box.
    def rc(b):  # middle segment reuses the O-shell tangential count; else thin
        return c_tan if b == 1 else c_h

    for bi, bj, bk in product(range(3), repeat=3):
        if (bi, bj, bk) == (1, 1, 1):
            continue
        box = (bi, bj, bk)
        corners = tuple(
            "L_%d%d%d" % (bi + da, bj + db, bk + dc)
            for da in (0, 1)
            for db in (0, 1)
            for dc in (0, 1)
        )
        name = "H_%d%d%d" % (bi, bj, bk)
        tb.add_block(name, corners=corners, resolutions=(rc(bi), rc(bj), rc(bk)))
        for axis in range(3):
            if box[axis] == 0:  # low segment touches the -1 cube face
                tb.associate(name, axis, 0, cube_face[(axis, 0)])
            elif box[axis] == 2:  # high segment touches the +1 cube face
                tb.associate(name, axis, 1, cube_face[(axis, 1)])
    return tb


def _block_id_arrays(grid):
    """Per-block global-node id lattices (for the driver's edge/section helpers)."""
    return [
        dm.reshape(block.logical_shape)
        for dm, block in zip(grid.block_dof_maps, grid.blocks)
    ]


def _sphere_mask(grid):
    """Boolean per-node mask of DOFs bound to the imported sphere.

    Read from the grid's constraint map by entity tag, so it is the topology's
    own record of the sphere surface (not a geometric ``|x| == r0`` guess).
    """
    m = np.zeros(grid.global_node_count, dtype=bool)
    for dof, ent in grid.dof_constraints.items():
        if getattr(ent, "tag", None) == "sphere":
            m[dof] = True
    return m


def main(argv=None):
    from driver import (  # local sibling module
        _plot_energy,
        grid_edges,
        plot_topology,
        section_edges,
        surface_edges,
        surface_faces,
    )

    p = argparse.ArgumentParser(
        description="sphere-in-cube O-shell + H-grid from build123d CAD"
    )
    # node counts matching sphere_in_cube_nurbs (n=9, m=4, mh=3) -> the same
    # 3012-node grid. Keep n odd so a node lands on the z=0 / x=0 section planes.
    p.add_argument("--n", type=int, default=9, help="tangential nodes per block face")
    p.add_argument("--m", type=int, default=4, help="radial nodes across the O-shell")
    p.add_argument("--nh", type=int, default=3, help="nodes across each H-grid layer")
    p.add_argument("--r0", type=float, default=0.5, help="sphere radius")
    p.add_argument("--cw", type=float, default=0.7, help="inset-cube half-width")
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--sweeps", type=int, default=200)
    p.add_argument("--chunk", type=int, default=40)
    p.add_argument(
        "--smoother",
        choices=["jacobi", "fas", "control_point"],
        default="jacobi",
        help="TMOP-phase smoother: block-Jacobi sweeps, FAS V-cycles, or the "
        "control-net reduced Gauss-Newton solver (sliding sphere/plane "
        "walls, exact C1 seams, fan fallback)",
    )
    p.add_argument(
        "--plot-live",
        action="store_true",
        help="PyVista animated XY/YZ sections (one frame per chunk)",
    )
    p.add_argument(
        "--plot-grid", action="store_true", help="PyVista final XY/YZ section plots"
    )
    p.add_argument(
        "--plot-3d",
        action="store_true",
        help="add the sphere-surface mesh pane to the live/final plots",
    )
    p.add_argument(
        "--plot-energy",
        action="store_true",
        help="matplotlib energy + min-det convergence curves",
    )
    p.add_argument(
        "--plot-topology",
        action="store_true",
        help="PyVista view of the declared block topology only, no solve",
    )
    p.add_argument("--su2", metavar="PATH", help="write the solved grid to an SU2 mesh")
    p.add_argument(
        "--export-eggy",
        metavar="PATH",
        default=None,
        help="after the run, pack this example folder (script + STEP asset + "
        "net cache) into a .eggy archive",
    )
    a = p.parse_args(argv)

    topo = sphere_in_cube_cad(a.n, a.m, a.nh, r0=a.r0, cw=a.cw).build()  # node counts
    grid = topo.initialize_grid()
    X0 = grid.global_nodes
    blocks = _block_id_arrays(grid)
    print(
        f"blocks={len(topo.block_specs)} nodes={grid.global_node_count} "
        f"singular fans={len(topo.singularities)}"
    )
    on_sphere = _sphere_mask(grid)
    r = np.linalg.norm(X0[on_sphere], axis=1)
    print(
        f"{int(on_sphere.sum())} nodes bound to imported sphere: "
        f"r in [{r.min():.4f}, {r.max():.4f}] (r0={a.r0})"
    )

    if a.plot_topology:
        plot_topology(X0, blocks, a.r0)
        return

    edges = grid_edges(blocks)
    sections = section_edges(X0, edges)
    surf = surface_edges(edges, on_sphere)  # fallback line pane
    surf_faces = surface_faces(blocks, on_sphere)  # opaque sphere-surface 3D pane

    live = None
    if a.plot_live:
        from driver import GridPlots

        live = GridPlots(X0, sections, surf, a.plot_3d, surf_faces=surf_faces)
        live.open_live()

    if a.smoother == "fas":
        smoothing = [FasSmoother(sweeps=a.sweeps, chunk=a.chunk)]
    elif a.smoother == "control_point":
        # A nodal pre-pass gives the net fit a smooth, valid start.
        smoothing = [
            Presmooth(JacobiSmoother(sweeps=100, chunk=100)),
            ControlPointSmoother(chunk=a.chunk),
        ]
    else:
        smoothing = [JacobiSmoother(sweeps=a.sweeps, chunk=a.chunk)]
    # Refit + Save leave a control net beside the script so the exported .eggy
    # carries one (a regrid can resample from it); control mode keeps its solved net.
    net_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "net.npz")
    stages = [
        Untangle(name="untangle folds"),
        *smoothing,
        Refit(name="refit net to grid"),
        Save(net_path, name="save net"),
    ]
    energies, mindets = [], []
    for phase, info in generate_steps(grid, stages=stages, device=a.device):
        if "energy" in info:
            energies.append(info["energy"])
        if "min_det" in info:
            mindets.append(info["min_det"])
        if live is not None:
            live.update(grid.global_nodes)
        bits = " ".join(
            f"{k}={v:.4e}" if isinstance(v, float) else f"{k}={v}"
            for k, v in info.items()
        )
        print(f"  {phase}: {bits}")
    print("Done.")

    if a.su2:
        from egg.io.su2 import export_su2

        export_su2(grid, a.su2)
        print(f"wrote {a.su2}")

    if a.export_eggy:
        from egg.io import eggy

        eggy.pack(a.export_eggy, os.path.dirname(os.path.abspath(__file__)))
        print(f"Exported .eggy archive to {a.export_eggy}")

    if live is not None:
        live.show()
    if a.plot_grid:
        from driver import GridPlots

        GridPlots(
            grid.global_nodes, sections, surf, a.plot_3d, surf_faces=surf_faces
        ).show()
    if a.plot_energy:
        _plot_energy(energies, mindets)


if __name__ == "__main__":
    main()
