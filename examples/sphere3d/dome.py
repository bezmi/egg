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
        [--plot-live] [--plot-grid] [--plot-energy]
"""

import argparse
from itertools import product

import numpy as np

from egg._cpp import cpp_core
from egg.geometry.entity_encoding import PARAM_PAD_SIZE, TAG_FREE, TAG_SPHERE
from egg.smoothing.batch import make_chain_J_nd

D = 3


def node_id(i, j, k, n):
    return (i * n + j) * n + k


def build_context(n, sphere_c, sphere_r):
    """Hand-assembled flattened sweep context for the cube grid."""
    h = 1.0 / (n - 1)
    N = n ** 3

    # Node classification.
    def on_boundary(idx):
        return any(v in (0, n - 1) for v in idx)

    def on_top_interior(idx):
        i, j, k = idx
        return k == n - 1 and 0 < i < n - 1 and 0 < j < n - 1

    # Per-(cell, corner) samples: corner + the d axis neighbours within the cell.
    # Sign s_k = +1 if the corner sits at the low end of axis k, else -1.
    samples = []  # (gc, (gn0, gn1, gn2), (s0, s1, s2))
    node_samples = [[] for _ in range(N)]  # sample indices whose energy involves the node
    for ci, cj, ck in product(range(n - 1), repeat=3):
        cell = [(ci + a, cj + b, ck + c) for a, b, c in product((0, 1), repeat=3)]
        cell_ids = [node_id(*v, n) for v in cell]
        for o in product((0, 1), repeat=3):
            corner = (ci + o[0], cj + o[1], ck + o[2])
            gn, s = [], []
            for ax in range(3):
                nb = list(corner)
                nb[ax] += 1 if o[ax] == 0 else -1
                gn.append(node_id(*nb, n))
                s.append(1.0 if o[ax] == 0 else -1.0)
            si = len(samples)
            samples.append((node_id(*corner, n), tuple(gn), tuple(s)))
            for nid in cell_ids:
                node_samples[nid].append(si)

    # Moving DOFs: free interior + sphere-constrained top-face interior.
    tags = np.zeros(N, dtype=np.int32)
    params = np.zeros((N, PARAM_PAD_SIZE))
    moving = []
    for idx in product(range(n), repeat=3):
        nid = node_id(*idx, n)
        if on_top_interior(idx):
            tags[nid] = TAG_SPHERE
            # Blob: [c(3), r, ax(3), ay(3)].
            params[nid, :4] = (*sphere_c, sphere_r)
            params[nid, 4:7] = (1.0, 0.0, 0.0)
            params[nid, 7:10] = (0.0, 1.0, 0.0)
            moving.append(nid)
        elif not on_boundary(idx):
            tags[nid] = TAG_FREE
            moving.append(nid)

    # 8-colouring by index parity: same-parity nodes never share a cell.
    def colour(nid):
        i, j, k = nid // (n * n), (nid // n) % n, nid % n
        return (i % 2) * 4 + (j % 2) * 2 + (k % 2)

    w_inv_sample = np.eye(3) / h  # W = h I per sample

    groups = []
    for c in range(8):
        dofs = [nid for nid in moving if colour(nid) == c]
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
        S = np.stack([np.asarray(ss[0]), np.asarray(ss[1]), np.asarray(ss[2])], axis=1)
        W_inv = np.broadcast_to(w_inv_sample, (P, 3, 3))
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
            np.broadcast_to(w_inv_sample, (ns, 3, 3)).reshape(ns, 9)),
    }
    return {"groups": groups, "energy_stencil": energy_stencil}


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
    session = cpp_core.CppSweepSession(ctx, X.ravel(), device=a.device, dim=3)
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
        e, m = session.run(step, phase="barrier", delta=0.0)
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
