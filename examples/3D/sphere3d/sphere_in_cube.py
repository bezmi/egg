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

"""Sphere inside a cube — the 3D analogue of the 2D circle-in-rectangle O-grid.

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
``uv run sphere_in_cube.py --help`` for options.
"""

from itertools import product

import numpy as np

from egg.geometry.analytic3d import Line3, Plane, Sphere
from egg.geometry.entity_encoding import TAG_FREE, TAG_PLANE, TAG_SPHERE
from egg.geometry.entity_soa import TAG_LINE3
from egg.smoothing.flat_context import build_flat_context

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
    segs = [
        np.linspace(-1.0, -cw, mh),
        np.linspace(-cw, cw, n),
        np.linspace(cw, 1.0, mh),
    ]
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


def classify(X, r0, tol=1e-9):
    """Per-node entity objects from position: sphere / plane / edge / corner.

    Returns (dof_entities, tags, fixed) where dof_entities maps nid → entity
    object (or None for free), tags is the int tag array (for stats/plots),
    and fixed marks the 8 cube corners.
    """
    N = X.shape[0]
    tags = np.zeros(N, dtype=np.int32)
    dof_entities: dict[int, object] = {}
    fixed = np.zeros(N, dtype=bool)
    for nid in range(N):
        p = X[nid]
        if abs(np.linalg.norm(p) - r0) < tol:
            tags[nid] = TAG_SPHERE
            dof_entities[nid] = Sphere(
                (0.0, 0.0, 0.0), r0, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
            )
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
            # Cube face: a Plane through the face spanned by the others.
            ax = on[0]
            o = np.zeros(3)
            o[ax] = np.sign(p[ax])
            a1, a2 = np.eye(3)[(ax + 1) % 3], np.eye(3)[(ax + 2) % 3]
            tags[nid] = TAG_PLANE
            dof_entities[nid] = Plane(o, a1, a2)
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
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from driver import main_sphere_in_cube

    main_sphere_in_cube(
        sys.modules[__name__],
        banner="d=3: sphere in a cube (6-block O-grid) → TMOP smooth",
    )
