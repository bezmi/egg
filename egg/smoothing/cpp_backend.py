"""C++ backend bridge for the colored Gauss-Seidel sweep.

Provides :func:`cpp_sweep` — a drop-in replacement for
``build_fused_multisweep.run`` that calls the device-resident C++ sweep via
``egg._cpp.cpp_core``. Also provides :func:`flatten_context` to pack a
:class:`~egg.smoothing.solver.SweepContext` into the dict format the
binding expects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from egg.smoothing.solver import SweepContext

__all__ = ["CppSweepSession", "cpp_sweep", "cpp_untangle", "flatten_context"]


def _grid_mindet(X: np.ndarray, es: dict) -> float:
    """Raw min det A over the energy stencil (mirrors solver._grid_mindet)."""
    gc, gn0, gn1 = es["gc"], es["gn0"], es["gn1"]
    s0, s1 = es["s0"], es["s1"]
    Xc = X[gc]
    col0 = s0[:, None] * (X[gn0] - Xc)
    col1 = s1[:, None] * (X[gn1] - Xc)
    det = col0[:, 0] * col1[:, 1] - col1[:, 0] * col0[:, 1]
    return float(det.min())


def _ensure_group_batches(ctx: SweepContext) -> None:
    """Build jax_group_batches if not already populated (host-side NumPy path).

    The JAX path does this lazily on first sweep. We replicate it here with
    pure NumPy so the C++ backend does not require JAX.
    """
    if ctx.jax_group_batches is not None:
        return

    from .batch import make_chain_J

    ctx.jax_group_batches = {}
    for (colour, P), dof_indices in ctx.colour_P_groups.items():
        dof_indices = list(dof_indices)
        D = len(dof_indices)
        if D == 0:
            continue

        patches = [ctx.dof_patches[d] for d in dof_indices]

        gc = np.stack([p["gc"] for p in patches])       # (D, P)
        gn0 = np.stack([p["gn0"] for p in patches])
        gn1 = np.stack([p["gn1"] for p in patches])
        s0 = np.stack([p["s0"] for p in patches])
        s1 = np.stack([p["s1"] for p in patches])
        W_inv = np.stack([p["W_inv"] for p in patches])  # (D, P, 2, 2)
        role = np.stack([p["role"] for p in patches])

        J = make_chain_J(s0[0], s1[0], W_inv[0])[None].repeat(D, axis=0)
        # Recompute per-DOF J (signs/W_inv may vary per DOF)
        J = np.stack([make_chain_J(s0[i], s1[i], W_inv[i]) for i in range(D)])

        ctx.jax_group_batches[(colour, P)] = {
            "gc": gc.astype(np.int32),
            "gn0": gn0.astype(np.int32),
            "gn1": gn1.astype(np.int32),
            "s0": s0.astype(np.float64),
            "s1": s1.astype(np.float64),
            "W_inv": W_inv.astype(np.float64),
            "role": role.astype(np.int32),
            "J": J.astype(np.float64),
            "dof_idx": np.asarray(dof_indices, dtype=np.int32),
            "tag": ctx.dof_constraint_tags[dof_indices].astype(np.int32),
            "params": ctx.dof_constraint_params[dof_indices].astype(np.float64),
        }


def flatten_context(ctx: SweepContext) -> dict:
    """Pack a SweepContext into the ctx_arrays dict the C++ binding expects.

    Returns a dict with keys ``"groups"`` (list of group dicts) and
    ``"energy_stencil"`` (dict of stencil arrays). Each group dict contains
    the batched arrays the C++ ``Executor`` consumes.

    Parameters
    ----------
    ctx : SweepContext
        The sweep context from :func:`build_sweep_context`.

    Returns
    -------
    dict
        Flattened context ready for ``cpp_core.cpp_sweep``.
    """
    _ensure_group_batches(ctx)

    # One ragged group per colour: merge the colour's (colour, P) batches into a
    # single launch (#1 — see cpp_backend_plan.md §5). Per-sample arrays are
    # concatenated DOF-major over the colour's DOFs (each DOF contributes its P
    # samples contiguously); the C++ side derives sample_offset from P_of. This
    # cuts the per-sweep launch count from ~(colours × distinct-P) to ~colours.
    groups = []
    for colour in range(ctx.num_colours):
        cg = ctx.get_colour_P_groups(colour)  # {P: [dof_indices]}
        if not cg:
            continue

        gc, gn0, gn1, s0, s1, role = [], [], [], [], [], []
        W_inv, J = [], []
        dof_idx, tag, params, P_of = [], [], [], []

        for P, _dofs in cg.items():
            b = ctx.jax_group_batches[(colour, P)]
            D = int(b["dof_idx"].shape[0])
            # (D, P, ...) -> flat (D*P, ...) is DOF-major: DOF d's P samples are
            # contiguous, matching the ragged sample_offset layout.
            gc.append(b["gc"].reshape(D * P))
            gn0.append(b["gn0"].reshape(D * P))
            gn1.append(b["gn1"].reshape(D * P))
            s0.append(b["s0"].reshape(D * P))
            s1.append(b["s1"].reshape(D * P))
            role.append(b["role"].reshape(D * P))
            W_inv.append(b["W_inv"].reshape(D * P, 4))     # (D,P,2,2) -> (D*P,4)
            J.append(b["J"].reshape(D * P, 24))            # (D,P,4,6) -> (D*P,24)
            dof_idx.append(b["dof_idx"])
            tag.append(b["tag"])
            params.append(b["params"].reshape(D, 12))
            P_of.append(np.full(D, P, dtype=np.int32))

        groups.append({
            "D": int(np.concatenate(dof_idx).shape[0]),
            "gc": np.ascontiguousarray(np.concatenate(gc), dtype=np.int32),
            "gn0": np.ascontiguousarray(np.concatenate(gn0), dtype=np.int32),
            "gn1": np.ascontiguousarray(np.concatenate(gn1), dtype=np.int32),
            "s0": np.ascontiguousarray(np.concatenate(s0), dtype=np.float64),
            "s1": np.ascontiguousarray(np.concatenate(s1), dtype=np.float64),
            "W_inv": np.ascontiguousarray(np.concatenate(W_inv), dtype=np.float64),
            "role": np.ascontiguousarray(np.concatenate(role), dtype=np.int32),
            "J": np.ascontiguousarray(np.concatenate(J), dtype=np.float64),
            "dof_idx": np.ascontiguousarray(np.concatenate(dof_idx), dtype=np.int32),
            "tag": np.ascontiguousarray(np.concatenate(tag), dtype=np.int32),
            "P_of": np.ascontiguousarray(np.concatenate(P_of), dtype=np.int32),
            "params": np.ascontiguousarray(np.concatenate(params), dtype=np.float64),
        })

    es = ctx.energy_stencil
    n = int(es["gc"].shape[0])
    energy_stencil = {
        "num_samples": n,
        "gc": np.ascontiguousarray(es["gc"], dtype=np.int32),
        "gn0": np.ascontiguousarray(es["gn0"], dtype=np.int32),
        "gn1": np.ascontiguousarray(es["gn1"], dtype=np.int32),
        "s0": np.ascontiguousarray(es["s0"], dtype=np.float64),
        "s1": np.ascontiguousarray(es["s1"], dtype=np.float64),
        "W_inv": np.ascontiguousarray(es["W_inv"], dtype=np.float64),
    }

    return {
        "groups": groups,
        "energy_stencil": energy_stencil,
    }


class CppSweepSession:
    """Persistent device-resident smoothing session.

    The flattened context is uploaded once at construction and ``X`` is kept
    device-resident across :meth:`run` calls (no per-call upload/download or
    re-JIT), so chunked driving and warm steady-state timing avoid re-staging.

    Parameters
    ----------
    ctx : SweepContext
        The sweep context from :func:`build_sweep_context`.
    X : ndarray, shape (N, 2)
        Initial node positions (uploaded once).
    device : str, optional
        ``"auto"`` (default), ``"cpu"``, or ``"gpu"``.
    """

    def __init__(
        self,
        ctx: SweepContext,
        X: np.ndarray,
        *,
        device: str = "auto",
    ) -> None:
        from egg._cpp import cpp_core

        self._shape = X.shape
        ctx_arrays = flatten_context(ctx)
        X_flat = np.ascontiguousarray(X, dtype=np.float64).ravel()
        self._session = cpp_core.CppSweepSession(ctx_arrays, X_flat, device=device)

    def run(
        self,
        n_sweeps: int,
        *,
        phase: str = "barrier",
        delta: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run ``n_sweeps`` on the resident X; returns ``(energies, mindets)``."""
        return self._session.run(n_sweeps, phase=phase, delta=delta)

    def get_X(self) -> np.ndarray:
        """Return a host copy of the resident X, reshaped to ``(N, 2)``."""
        return self._session.get_X().reshape(self._shape)

    def set_X(self, X: np.ndarray) -> None:
        """Re-upload X to the device."""
        self._session.set_X(np.ascontiguousarray(X, dtype=np.float64).ravel())


def cpp_sweep(
    ctx: SweepContext,
    X: np.ndarray,
    n_sweeps: int,
    *,
    device: str = "auto",
    phase: str = "barrier",
    delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``n_sweeps`` of colored GS barrier smoothing via the C++ backend.

    Drop-in replacement for ``build_fused_multisweep(ctx).run(X, n_sweeps)``.

    Parameters
    ----------
    ctx : SweepContext
        The sweep context from :func:`build_sweep_context`.
    X : ndarray, shape (N, 2)
        Initial node positions.
    n_sweeps : int
        Number of sweeps to run.
    device : str, optional
        ``"auto"`` (default), ``"cpu"``, or ``"gpu"``.

    Returns
    -------
    X_out : ndarray, shape (N, 2)
        Final node positions.
    energies : ndarray, shape (n_sweeps,)
        Per-sweep total energy.
    mindets : ndarray, shape (n_sweeps,)
        Per-sweep min det A.
    """
    from egg._cpp import cpp_core

    ctx_arrays = flatten_context(ctx)
    X_flat = np.ascontiguousarray(X, dtype=np.float64).ravel()

    X_out_flat, energies, mindets = cpp_core.cpp_sweep(
        ctx_arrays, X_flat, n_sweeps, device=device, phase=phase, delta=delta,
    )

    return X_out_flat.reshape(X.shape), energies, mindets


def cpp_untangle(
    ctx: SweepContext,
    X: np.ndarray,
    *,
    device: str = "auto",
    sweeps_per_delta: int = 20,
    delta0_factor: float = 2.0,
    max_outer: int = 60,
    margin: float = 1e-9,
) -> tuple[np.ndarray, float]:
    """δ-continuation untangle via the C++ backend.

    Same δ-continuation as the stepped loop in
    ``egg.pipeline.generate_steps`` (``untangle_direct=False``), but the whole
    schedule runs in one call here: a
    persistent device-resident session runs ``sweeps_per_delta`` untangle sweeps
    per δ on a geometric schedule ``δ_k = δ_0 · 0.5^k`` (starting from
    ``delta0_factor · |min det A|``) until ``min det A > margin``.

    Parameters
    ----------
    ctx : SweepContext
        The (isotropic) sweep context from :func:`build_sweep_context`.
    X : ndarray, shape (N, 2)
        Initial (possibly folded) node positions.
    device : str, optional
        ``"auto"`` (default), ``"cpu"``, or ``"gpu"``.
    sweeps_per_delta, delta0_factor, max_outer, margin : optional
        Continuation schedule controls (defaults match the JAX driver).

    Returns
    -------
    X_out : ndarray, shape (N, 2)
        Untangled node positions (or the best found if the schedule stalls).
    mindet : float
        Final raw min det A.
    """
    es = ctx.energy_stencil
    md = _grid_mindet(X, es)
    if md > margin:
        return X, md

    session = CppSweepSession(ctx, X, device=device)
    delta = delta0_factor * max(abs(md), 1e-12)
    for _ in range(max_outer):
        _e, mds = session.run(sweeps_per_delta, phase="untangle", delta=delta)
        md = float(np.asarray(mds)[-1])
        if md > margin:
            break
        delta *= 0.5

    return session.get_X(), md
