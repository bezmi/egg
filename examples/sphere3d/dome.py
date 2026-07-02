"""Single-block hex grid with a spherical-cap face — the first d=3 example.

The Python topology/TFI front-end is still 2D, so this example assembles the
flattened sweep context by hand (the same wire format ``flatten_context``
produces) and drives the C++ core directly at ``dim=3``:

- an ``n x n x n`` unit-cube hex grid;
- the interior of the top face is constrained to a sphere (TAG_SPHERE) whose
  cap bulges above the cube — the sweep's constrained projection pulls those
  nodes onto the dome and slides them tangentially;
- cube edges/corners and the other five faces stay fixed;
- interior nodes are free and smoothed by the 3D condition-number TMOP barrier.

Usage::

    uv run dome.py [--n N] [--sweeps N] [--chunk N] [--device cpu|gpu|auto]
        [--report-every N] [--plot-live] [--plot-grid] [--plot-energy]
"""

import argparse
from itertools import product

import numpy as np

from egg._cpp import cpp_core
from egg.geometry.analytic3d import Sphere
from egg.smoothing.flat_context import build_flat_context

D = 3


def node_id(i, j, k, n):
    return (i * n + j) * n + k


def build_context(n, sphere_c, sphere_r):
    """Flattened sweep context for the single-block cube grid.

    The structured-grid assembly (cell stencil, node->sample membership, roles,
    ragged per-colour groups) lives in the dimension-generic
    :func:`egg.smoothing.flat_context.build_flat_context`; here we set up the
    geometry: the n^3 lattice, the moving-DOF classification, and the analytic
    parity colouring.
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

    # 8-colouring by index parity: same-parity nodes never share a cell.
    colours = ((ii % 2) * 4 + (jj % 2) * 2 + (kk % 2)).reshape(-1)
    w_inv_sample = np.eye(3) / h                 # W = h I per sample

    return build_flat_context([ids], moving_mask, dof_entities, 3,
                              w_inv=w_inv_sample, colours=colours)


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
    p = argparse.ArgumentParser(description="3D dome-on-a-cube smoothing demo.")
    p.add_argument("--n", type=int, default=7, help="nodes per edge")
    p.add_argument("--sweeps", type=int, default=30)
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
                   help="matplotlib animated 3D wireframe (one frame per chunk)")
    p.add_argument("--plot-grid", action="store_true",
                   help="matplotlib final 3D wireframe grid")
    p.add_argument("--plot-energy", action="store_true",
                   help="matplotlib energy + min-det convergence curves")
    a = p.parse_args()

    print("=" * 56)
    print("d=3: spherical-cap face on a unit cube → TMOP smooth")
    print("=" * 56)

    n = a.n
    # Sphere centred below the top face; the cap bulges to z ~ 1.08 mid-face.
    c = np.array([0.5, 0.5, -1.0])
    r = 2.08
    ctx = build_context(n, c, r)

    # Initial grid: the unit cube lattice (valid; min det A = 1 in W units),
    # interior gently perturbed so the smoother has work to do.
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

    # Boundary snap (as the pipeline's init does): project the constrained
    # top-face nodes onto the sphere; the sweep then keeps them on it.
    top = np.array([
        node_id(i, j, n - 1, n)
        for i, j in product(range(1, n - 1), repeat=2)
    ])
    d = X[top] - c
    X[top] = c + r * d / np.linalg.norm(d, axis=1)[:, None]

    # Chunked, device-resident driving: the context/X are uploaded once and the
    # sweeps run in chunks so live plotting only pays one download per chunk.
    if a.structured:
        # The dome lattice is a single n^3 block (C-order node ids), so the
        # structured store has an empty halo: structured colored-GS is bit-for-bit
        # the global path, and block-Jacobi is the merged single-launch smoother.
        from egg.smoothing.cpp_backend import (
            build_structured_context_from_block_maps, structured_arrays)
        blocks = [np.arange(n ** 3).reshape(n, n, n)]
        bsc = build_structured_context_from_block_maps(3, blocks, X.shape[0])
        structured = structured_arrays(bsc)
        session = cpp_core.CppStructuredSweepSession(
            ctx, structured, X.ravel(), device=a.device, dim=3)
        print(f"structured: blocks={bsc.num_blocks} smoother={a.smoother} omega={a.omega}")
        run_kwargs = {"smoother": a.smoother, "omega": a.omega}
    else:
        if a.smoother != "colored-gs":
            p.error("--smoother requires --structured")
        session = cpp_core.CppSweepSession(ctx, X.ravel(), device=a.device, dim=3)
        run_kwargs = {}
    energies, mindets = [], []

    live = None
    if a.plot_live:
        import matplotlib.pyplot as plt

        plt.ion()
        fig = plt.figure(figsize=(7, 7))
        live = fig.add_subplot(projection="3d")
        _draw_wireframe(live, X.ravel(), n, "sweep 0")
        plt.pause(0.01)

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
            import matplotlib.pyplot as plt

            _draw_wireframe(live, session.get_X(), n, f"sweep {done}")
            plt.pause(0.01)

    if live is not None:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()

    X_out = session.get_X().reshape(-1, 3)

    print(f"energy : {energies[0]:.4e} -> {energies[-1]:.4e}")
    print(f"min det: {mindets[0]:.4e} -> {mindets[-1]:.4e}")

    # The constrained top-face nodes must still sit on the sphere.
    radii = np.linalg.norm(X_out[top] - c, axis=1)
    print(f"top-face |x - c|: max deviation from r = "
          f"{np.abs(radii - r).max():.2e} (r = {r})")
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


if __name__ == "__main__":
    main()
