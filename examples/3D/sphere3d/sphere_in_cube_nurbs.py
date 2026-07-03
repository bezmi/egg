"""NURBS sphere inside a cube — a heavy-entity variant of sphere_in_cube.

Identical to ``sphere_in_cube.py`` except the spherical cavity is an exact
*NURBS* surface (degree-2 surface of revolution, rational weights) instead of
the analytic ``Sphere``. Geometrically it is the same sphere (eval radius error
~1e-16), but every cavity node now slides via the heavy
``BSplineSurfaceParam`` device project (coarse-grid-seeded Newton on the
nearest-foot stationarity) — the workload that stresses the boundary
sweep kernel once the constrained entity is expensive.

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
(the wire format ``build_flat_context`` produces) is assembled by hand and the
C++ core is driven directly at ``dim=3``. The live/grid plots show the grid's
planar sections in the XY (z=0) and YZ (x=0) planes — each is the familiar 2D
O-ring picture.

The command-line surface lives in ``driver.py``; run
``uv run sphere_in_cube_nurbs.py --help`` for options.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import numpy as np

from egg.geometry.analytic3d import Line3
from egg.geometry.entity_encoding import TAG_FREE, TAG_PLANE, TAG_SPHERE
from egg.geometry.entity_soa import TAG_LINE3
from egg.geometry.surfaces3d import BSplineSurface
from egg.smoothing.flat_context import build_flat_context
from sphere_in_cube import build_grid  # noqa: F401  (same lattice; re-exported for the driver)


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
    circ = np.array(
        [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)],
        dtype=float,
    )
    wc = np.array([1, s2, 1, s2, 1, s2, 1, s2, 1], dtype=float)
    knots_u = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4], dtype=float)
    merid = np.array(
        [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1)], dtype=float
    )  # (radius, height)
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
    """Flattened sweep context (identity target, W_inv = I).

    The structured-grid assembly (cell stencil, node->sample membership, roles,
    the single free-DOF group) lives in the dimension-generic
    :func:`egg.smoothing.flat_context.build_flat_context`; only the geometry
    (block lattice, entity classification) is example-specific.
    """
    return build_flat_context(blocks, ~fixed, dof_entities, 3, w_inv=np.eye(3))


if __name__ == "__main__":
    from driver import main_sphere_in_cube

    main_sphere_in_cube(
        sys.modules[__name__],
        banner="d=3: NURBS sphere in a cube (6-block O-grid) → TMOP smooth",
    )
