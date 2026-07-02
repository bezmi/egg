"""Single-block hex grid with a spherical-cap face — the first d=3 example.

The Python topology/TFI front-end is still 2D, so this example assembles the
flattened sweep context by hand (the same wire format ``build_flat_context``
produces) and drives the C++ core directly at ``dim=3``:

- an ``n x n x n`` unit-cube hex grid;
- the interior of the top face is constrained to a sphere (TAG_SPHERE) whose
  cap bulges above the cube — the sweep's constrained projection pulls those
  nodes onto the dome and slides them tangentially;
- cube edges/corners and the other five faces stay fixed;
- interior nodes are free and smoothed by the 3D condition-number TMOP barrier.

The command-line surface lives in ``driver.py``; run ``uv run dome.py --help``
for options.
"""

from itertools import product

import numpy as np

from egg.geometry.analytic3d import Sphere
from egg.smoothing.flat_context import build_flat_context

D = 3


def node_id(i, j, k, n):
    return (i * n + j) * n + k


def build_context(n, sphere_c, sphere_r):
    """Flattened sweep context for the single-block cube grid.

    The structured-grid assembly (cell stencil, node->sample membership, roles,
    the single free-DOF group) lives in the dimension-generic
    :func:`egg.smoothing.flat_context.build_flat_context`; here we set up the
    geometry: the n^3 lattice and the moving-DOF classification.
    """
    h = 1.0 / (n - 1)
    ids = np.arange(n ** 3).reshape(n, n, n)     # ids[i,j,k] == node_id(i,j,k,n)
    ii, jj, kk = np.indices((n, n, n))

    # Node classification. Moving DOFs = free interior + sphere-constrained
    # top-face interior; everything else (5 faces, all edges/corners) is fixed.
    on_bnd = ((ii == 0) | (ii == n - 1) | (jj == 0) | (jj == n - 1)
              | (kk == 0) | (kk == n - 1))
    top_int = (kk == n - 1) & (ii > 0) & (ii < n - 1) & (jj > 0) & (jj < n - 1)
    moving_mask = (top_int | ~on_bnd).reshape(-1)

    # One shared sphere for every top-interior node; free interior nodes carry no
    # entity. (top_int is disjoint from the free interior, so keys never clash.)
    sphere = Sphere(sphere_c, sphere_r, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    dof_entities: dict[int, object] = {int(nid): sphere for nid in ids[top_int]}
    for nid in ids[~on_bnd]:
        dof_entities[int(nid)] = None

    w_inv_sample = np.eye(3) / h                 # W = h I per sample

    return build_flat_context([ids], moving_mask, dof_entities, 3,
                              w_inv=w_inv_sample)


def initial_lattice(n, sphere_c, sphere_r):
    """Initial grid: the unit-cube lattice, gently perturbed, top snapped.

    The lattice is valid (min det A = 1 in W units); the interior is
    perturbed so the smoother has work to do, and the constrained top-face
    interior nodes are projected onto the sphere (as the pipeline's init
    boundary snap does) — the sweep then keeps them on it.

    Returns ``(X (n^3, 3), top)`` with ``top`` the top-face interior ids.
    """
    rng = np.random.default_rng(7)
    X = np.zeros((n ** 3, 3))
    for i, j, k in product(range(n), repeat=3):
        X[node_id(i, j, k, n)] = (i, j, k)
    X /= (n - 1)
    interior = np.array([
        node_id(i, j, k, n)
        for i, j, k in product(range(1, n - 1), repeat=3)
    ])
    X[interior] += rng.uniform(-0.15, 0.15, size=(interior.size, 3)) / (n - 1)

    top = np.array([
        node_id(i, j, n - 1, n)
        for i, j in product(range(1, n - 1), repeat=2)
    ])
    d = X[top] - sphere_c
    X[top] = sphere_c + sphere_r * d / np.linalg.norm(d, axis=1)[:, None]
    return X, top


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from driver import main_dome

    main_dome()
