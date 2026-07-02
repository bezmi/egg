"""Phoebus capsule grid, ported from the Eilmer example (capsule-phoebus).

A 2x4 block array between the capsule wall and the inflow boundary,
matching the Lua ``ControlPointPatch`` + ``registerFluidGridArray{nib=2,
njb=4}``: the wall is an arc-line-arc-line ``Polyline`` (nose arc, cone
flank, shoulder arc, aft wall) and the inflow boundary is a cubic
``Bezier``, both arc-length parameterized. Sub-block corners are placed
parametrically along the bounding edges (TFI for the interior ones);
block-to-block connectivity is inferred from the shared corner objects.

The Lua wall clustering (GeometricFunction, h_wall=1e-4, r=1.2) is applied
directly: every wall-normal grid line is resampled with the same geometric
distribution at initialization, the TMOP pass smooths against a matching
boundary-layer target, and a final resample restores the exact wall
spacing. Where the Lua example massages the interior with manual control
points, the TMOP smoothing pass does that job.

Reference: D. Bianchi et al., Int. J. Heat Mass Transfer 177 (2021) 121430.

Usage::

    uv run phoebus.py [--plot-grid] [--plot-topology] [--grid-level N]
        [--tmop-sweeps N] [--device cpu|gpu|auto]
"""

import argparse
import math

import numpy as np

from egg.geometry import Arc, Bezier, Edge, Line, Polyline, Vector3
from egg.pipeline import drain, generate_steps
from egg.smoothing.targets import build_boundary_layer_target
from egg.topology.builder import TopologyBuilder

H_WALL = 1.0e-4  # first cell height at the wall [m] (h_wall in the Lua)
BL_GROWTH = 1.2  # near-wall geometric growth ratio (r in the Lua)


def _split(n: int, k: int) -> list[int]:
    """Split n cells into k contiguous per-block counts."""
    return [round(n * (t + 1) / k) - round(n * t / k) for t in range(k)]


def _tfi(u, v, south, north, west, east, p00, p10, p01, p11) -> Vector3:
    """Bilinear TFI point of the four bounding edges at (u, v)."""
    s, n = south.point_at(u), north.point_at(u)
    w, e = west.point_at(v), east.point_at(v)
    x = (
        (1 - v) * s.x
        + v * n.x
        + (1 - u) * w.x
        + u * e.x
        - (
            (1 - u) * (1 - v) * p00.x
            + u * (1 - v) * p10.x
            + (1 - u) * v * p01.x
            + u * v * p11.x
        )
    )
    y = (
        (1 - v) * s.y
        + v * n.y
        + (1 - u) * w.y
        + u * e.y
        - (
            (1 - u) * (1 - v) * p00.y
            + u * (1 - v) * p10.y
            + (1 - u) * v * p01.y
            + u * v * p11.y
        )
    )
    return Vector3(x, y)


def _geometric_fractions(n_nodes: int, a: float, r: float,
                         reverse: bool = False) -> np.ndarray:
    """Node fractions of Eilmer's GeometricFunction cluster.

    Spacings grow geometrically from ``a`` (a fraction of the edge) by ratio
    ``r`` until they reach the uniform spacing of the remaining interval,
    then stay uniform. ``reverse=True`` clusters at t=1 instead of t=0.
    """
    spacings = []
    total = 0.0
    n_cells = n_nodes - 1
    for k in range(n_cells):
        d = a * r ** k
        rest = (1.0 - total) / (n_cells - k)
        if d >= rest:
            spacings.extend([rest] * (n_cells - k))
            break
        spacings.append(d)
        total += d
    t = np.concatenate([[0.0], np.cumsum(spacings)])
    t /= t[-1]
    if reverse:
        t = 1.0 - t[::-1]
    return t


def _cluster_wall_normal(grid, topo, h_wall: float, r: float) -> None:
    """Redistribute nodes along each inflow->wall grid line.

    Applies the Lua GeometricFunction clustering at initialization: every
    wall-normal line is resampled by arc length with first spacing ``h_wall``
    at the wall growing by ``r``. Works on the ``b{i}{j}`` block-array
    layout (axis0 = wall-normal).
    """
    names = list(topo.block_specs.keys())
    idx = {n: k for k, n in enumerate(names)}
    nib = 1 + max(int(n[1]) for n in names)
    njb = 1 + max(int(n[2]) for n in names)
    for jb in range(njb):
        stack = [grid.blocks[idx[f"b{ib}{jb}"]].nodes for ib in range(nib)]
        for j in range(stack[0].shape[1]):
            line = np.vstack([stack[0][:, j]] + [s[1:, j] for s in stack[1:]])
            seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            if s[-1] <= 0.0:
                continue
            t = _geometric_fractions(line.shape[0], h_wall / s[-1], r,
                                     reverse=True) * s[-1]
            new = np.stack([np.interp(t, s, line[:, 0]),
                            np.interp(t, s, line[:, 1])], axis=1)
            ofs = 0
            for blk in stack:
                blk[:, j] = new[ofs:ofs + blk.shape[0]]
                ofs += blk.shape[0] - 1
    # Sync the redistributed nodes into the global DOF array.
    for bi in range(len(names)):
        dof = grid.block_dof_maps[bi]
        nodes = grid.blocks[bi].nodes
        for lidx in np.ndindex(nodes.shape[:-1]):
            grid.global_nodes[dof[lidx]] = nodes[lidx]


def build_phoebus(grid_level: int = 1):
    # -- geometric parameters (verbatim from grid.lua) -----------------------
    Rn = 20.0e-3  # nose radius [m]
    beta = math.radians(45.0)  # cone angle [rad]
    Rs = 1.57e-3  # shoulder radius [m]
    Rb = 20.0e-3  # base radius [m]
    L_aft = Rs  # artificial wall length downstream of shoulder [m]
    Ri = Rn * 1.2  # initial radius of inflow boundary [m]
    Rf = Rn * 1.5  # final radius of inflow boundary [m]

    # -- points ---------------------------------------------------------------
    A = Vector3(x=0.0, y=0.0)
    B = Vector3(x=Rn * (1.0 - math.cos(beta)), y=Rn * math.sin(beta))
    Cy = Rb + Rs * (math.cos(beta) - 1.0)
    C = Vector3(x=B.x + (Cy - B.y) / math.tan(beta), y=Cy)
    D = Vector3(x=C.x + Rs * math.sin(beta), y=Rb)
    E = Vector3(x=D.x + L_aft, y=D.y, fixed=True)
    theta = math.atan(E.y / (Rn - E.x))  # angle of outflow boundary
    F = Vector3(x=Rn - Rf * math.cos(theta), y=Rf * math.sin(theta), fixed=True)
    G = Vector3(x=Rn - Ri, y=0.0, fixed=True)
    centre_n = Vector3(x=Rn, y=0.0)
    centre_s = Vector3(x=D.x, y=D.y - Rs)

    # -- paths -> grid edges (axis0 = inflow->wall, axis1 = along the wall) ---
    wall = Edge(
        Polyline(
            [
                Arc(p0=A, p1=B, centre=centre_n),
                Line(p0=B, p1=C),
                Arc(p0=C, p1=D, centre=centre_s),
                Line(p0=D, p1=E),
            ]
        ),
        arc_length=True,
    )
    inflow = Edge(
        Bezier(points=[G, Vector3(x=G.x, y=B.y), Vector3(x=B.x, y=0.8 * F.y), F]),
        arc_length=True,
    )
    symm = Edge(Line(p0=G, p1=A))  # south: symmetry axis
    outflow = Edge(Line(p0=F, p1=E))  # north: outflow boundary

    # -- 2x4 block array (nib x njb of registerFluidGridArray) ----------------
    n_refine = 2 ** (grid_level / 2)
    n_wall_normal = math.ceil(20 * n_refine)  # cells inflow->wall
    n_along_wall = math.ceil(100 * n_refine)  # cells along the wall
    nib, njb = 2, 4
    nx, ny = _split(n_wall_normal, nib), _split(n_along_wall, njb)

    # Sub-block corners: edge nodes on the boundary, TFI points inside.
    corner = {}
    for i in range(nib + 1):
        u = i / nib
        for j in range(njb + 1):
            v = j / njb
            if j == 0:
                corner[i, j] = symm.place_node(u, fixed=(i in (0, nib)))
            elif j == njb:
                corner[i, j] = outflow.place_node(u, fixed=(i in (0, nib)))
            elif i == 0:
                corner[i, j] = inflow.place_node(v)
            elif i == nib:
                corner[i, j] = wall.place_node(v)
            else:
                corner[i, j] = _tfi(u, v, symm, outflow, inflow, wall, G, A, F, E)

    bld = TopologyBuilder(d=2)
    for i in range(nib):
        for j in range(njb):
            bld.add_block(
                f"b{i}{j}",
                sw=corner[i, j],
                se=corner[i + 1, j],
                nw=corner[i, j + 1],
                ne=corner[i + 1, j + 1],
                res=(nx[i], ny[j]),
            )
    # Connectivity is inferred from shared corner objects; boundary faces
    # are associated explicitly (patch corners sit on two edges each).
    for i in range(nib):
        bld.associate(f"b{i}0", 1, 0, symm)
        bld.associate(f"b{i}{njb - 1}", 1, 1, outflow)
    for j in range(njb):
        bld.associate(f"b0{j}", 0, 0, inflow)
        bld.associate(f"b{nib - 1}{j}", 0, 1, wall)

    bld.set_boundary_layer(wall, first_height=H_WALL, growth=BL_GROWTH)

    topology = bld.build()
    entities = {
        "wall": wall.entity,
        "inflow": inflow.entity,
        "symm": symm.entity,
        "outflow": outflow.entity,
    }
    return topology, entities


def main():
    p = argparse.ArgumentParser(description="Phoebus capsule grid")
    p.add_argument(
        "--plot-grid", action="store_true", help="matplotlib final wireframe grid"
    )
    p.add_argument(
        "--plot-topology",
        action="store_true",
        help="Plot the declared topology only — no pipeline run",
    )
    p.add_argument(
        "--grid-level",
        type=int,
        default=1,
        help="grid refinement factor (as in the Lua example)",
    )
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--tmop-sweeps", type=int, default=40)
    a = p.parse_args()

    print("=" * 56)
    print("Phoebus capsule → TMOP smooth")
    print("=" * 56)

    topo, ents = build_phoebus(grid_level=a.grid_level)

    if a.plot_topology:
        from egg.io.visualize import plot_topology

        plot_topology(topo, highlight_singularities=True, show=True)
        return

    grid = topo.initialize_grid()
    _cluster_wall_normal(grid, topo, h_wall=H_WALL, r=BL_GROWTH)
    target = build_boundary_layer_target(topo, interior_spacing=2.0e-4)
    print(f"Boundary layer: first_height={H_WALL} growth={BL_GROWTH} on wall")

    steps = generate_steps(
        grid,
        target,
        tmop_sweeps=a.tmop_sweeps,
        device=a.device,
        untangle_direct=True,
    )
    mindet_history, energy_history = [], []
    drain(steps, mindet_history=mindet_history, energy_history=energy_history)
    # TMOP trades some wall spacing for interior quality; restore the exact
    # geometric wall distribution as a post-pass (stays valid).
    _cluster_wall_normal(grid, topo, h_wall=H_WALL, r=BL_GROWTH)
    print(f"\nFinal min det A: {mindet_history[-1]:.4e}")

    if a.plot_grid:
        from egg.io.visualize import plot_grid

        plot_grid(grid)

    print("Done.")


if __name__ == "__main__":
    main()
