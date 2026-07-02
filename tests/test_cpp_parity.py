"""Numerical parity gate for the C++ backend binding (Phase 1 gate).

This is the proper test of the *binding* layer: it drives the whole
``cpp_backend.cpp_sweep`` path — ``flatten_context`` (pack the SweepContext into
the dict the module expects) → ``cpp_core.cpp_sweep`` (host→USM staging, the
device sweep, host download) → reshape — and asserts the result matches the
NumPy oracle ``local_relaxation_sweep``. If the plumbing drops, transposes,
or mis-strides any array, the energies / final positions diverge here.

The exhaustive sweep-math coverage lives in the C++ golden test
(``tests/cpp/test_sweep_device.cpp``) and the Python end-to-end suite; this file
deliberately stays focused on the binding contract.

Skipped unless the extension is built (cmake build).
"""

from __future__ import annotations

import numpy as np
import pytest


def _has_cpp() -> bool:
    try:
        from egg._cpp import cpp_core  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_cpp(),
    reason="egg._cpp.cpp_core not built (requires cmake build)",
)

# C++ vs NumPy parity tolerances (same sweep ordering, same algorithm).
# Energy is a sum over ~10^4 samples, so the C++ device reduction and the NumPy
# oracle accumulate in different orders; 1e-9 keeps the gate meaningful while
# allowing that float64 reduction-order drift (abs diff ~1e-6 on energies ~6e3).
_RTOL_E = 1e-9
_RTOL_X = 1e-9
_ATOL = 1e-12


def _mini_context(perturb: bool = True):
    """A 4x4 IdentityTarget grid (matches the C++ golden), optionally jittered."""
    from egg.smoothing.solver import build_sweep_context
    from egg.smoothing.targets import IdentityTarget
    from egg.topology.builder import TopologyBuilder

    builder = TopologyBuilder(d=2)
    for name, pos in [("A", (0.0, 0.0)), ("B", (4.0, 0.0)),
                      ("C", (4.0, 4.0)), ("D", (0.0, 4.0))]:
        builder.add_corner(name, pos, fixed=True)
    builder.add_block("main", ("A", "D", "B", "C"), (4, 4))
    topo = builder.build()
    topo.initialize_grid()
    grid = topo.grid
    ctx = build_sweep_context(grid, IdentityTarget(d=2))

    X0 = grid.global_nodes.copy()
    if perturb:
        rng = np.random.default_rng(20240615)
        X0 = X0 + 0.05 * rng.standard_normal(X0.shape)
    return ctx, X0, topo


def _constrained_context(perturb: bool = True):
    """A circle-in-rectangle O-grid whose arc/edge boundaries yield constrained
    DOFs (``tag != 0``), exercising the kernel's project/tangent/solve1x1 branch.
    """
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "circles"))
    from topologies import build_circle_in_rectangle  # noqa: E402

    from egg.smoothing.solver import build_sweep_context
    from egg.smoothing.targets import IdentityTarget

    topo, _ents = build_circle_in_rectangle(rough=False, R=3)
    grid = topo.initialize_grid()
    ctx = build_sweep_context(grid, IdentityTarget(d=topo.d))

    X0 = grid.global_nodes.copy()
    if perturb:
        rng = np.random.default_rng(20240615)
        X0 = X0 + 0.02 * rng.standard_normal(X0.shape)
    return ctx, X0, topo


_CONTEXTS = {"free_4x4": _mini_context, "circle_in_rect": _constrained_context}


def _numpy_reference(X0: np.ndarray, topo, n_sweeps: int):
    """(X_final, per-sweep energies) using the same (colour,P)-grouped GS order
    as the C++ backend. Reproduces the JAX ``local_relaxation_sweep_jax_seq``
    logic in pure NumPy: batched grad/hess from the same X snapshot per colour,
    then sequential Newton+backtrack per DOF within each (colour,P) group."""
    from egg.smoothing.batch import energy_and_mindet, patch_eval
    from egg.smoothing.cpp_backend import _ensure_group_batches
    from egg.smoothing.solver import (
        build_sweep_context, _newton_backtrack_dof,
    )
    from egg.smoothing.targets import IdentityTarget

    grid = topo.initialize_grid()
    grid.global_nodes = X0.copy()
    ctx_np = build_sweep_context(grid, IdentityTarget(d=topo.d))
    _ensure_group_batches(ctx_np)              # populates ctx_np.jax_group_batches
    es = ctx_np.energy_stencil

    energies = []
    for _ in range(n_sweeps):
        for colour in range(ctx_np.num_colours):
            # Snapshot X for this colour — all groups of this colour read the
            # same X so they compute independent Gauss-Seidel moves.
            X_snap = grid.global_nodes.copy()
            for P, dofs in ctx_np.get_colour_P_groups(colour).items():
                b = ctx_np.jax_group_batches[(colour, P)]
                for i, dof_idx in enumerate(dofs):
                    g, H, e0, _ = patch_eval(
                        X_snap,
                        b["gc"][i], b["gn0"][i], b["gn1"][i],
                        b["s0"][i], b["s1"][i], b["W_inv"][i],
                        b["role"][i], b["J"][i],
                    )
                    _newton_backtrack_dof(
                        grid, int(dof_idx), g, H, float(e0), ctx_np,
                        quadratic_filter=False,
                    )
        e_val, _ = energy_and_mindet(
            grid.global_nodes,
            es["gc"], es["gn0"], es["gn1"],
            es["s0"], es["s1"], es["W_inv"],
        )
        energies.append(e_val)

    return grid.global_nodes.copy(), np.array(energies)


def _gpu_available() -> bool:
    """Probe for a usable SYCL GPU by attempting a 1-sweep run on it."""
    from egg.smoothing.cpp_backend import cpp_sweep
    ctx, X0, _ = _mini_context(perturb=False)
    try:
        cpp_sweep(ctx, X0, 1, device="gpu")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Binding plumbing: the flattened dict the module consumes.
# ---------------------------------------------------------------------------

def test_flatten_context_structure():
    """flatten_context emits the contiguous, correctly-typed ragged per-colour
    arrays the binding indexes by ``.size()`` — a transpose/dtype slip here
    corrupts the upload. One group per colour; per-sample arrays are flat over the
    concatenated DOFs (Σ P_of samples), per-DOF arrays are length D (#1 layout)."""
    from egg.smoothing.cpp_backend import flatten_context

    ctx, _, _ = _mini_context()
    fc = flatten_context(ctx)

    assert set(fc) == {"groups", "energy_stencil"}
    assert len(fc["groups"]) > 0
    assert len(fc["groups"]) <= ctx.num_colours

    for g in fc["groups"]:
        D = g["D"]
        assert isinstance(D, int)
        assert g["P_of"].shape == (D,)
        total_samples = int(g["P_of"].sum())
        for key, dtype in [("gc", np.int32), ("gn0", np.int32),
                           ("gn1", np.int32), ("role", np.int32),
                           ("s0", np.float64), ("s1", np.float64)]:
            arr = g[key]
            assert arr.dtype == dtype, f"{key} dtype {arr.dtype}"
            assert arr.flags["C_CONTIGUOUS"], f"{key} not contiguous"
            assert arr.shape == (total_samples,), f"{key} shape {arr.shape}"
        assert g["W_inv"].size == total_samples * 4
        assert g["J"].size == total_samples * 24
        assert g["dof_idx"].shape == (D,)
        # Entity data is carried solely by the typed SoA sub-dict (Phase 4
        # retired the positional tag/params/arena blob); every DOF is covered
        # by exactly one per-type entity group via its group-local dof_local.
        assert "tag" not in g and "params" not in g and "arena" not in g
        entities = g["entities"]
        assert "__blob__" not in entities
        covered = sorted(
            int(i) for e in entities.values() for i in e["dof_local"]
        )
        assert covered == list(range(D))

    es = fc["energy_stencil"]
    n = es["num_samples"]
    assert es["gc"].shape == (n,)
    assert es["W_inv"].size == n * 4


# ---------------------------------------------------------------------------
# Parity gate vs the NumPy oracle (the whole binding round-trip).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ctx_name", list(_CONTEXTS))
@pytest.mark.parametrize("n_sweeps", [1, 3])
def test_cpp_session_per_sweep_matches_numpy(n_sweeps, ctx_name):
    """CppSweepSession (1 sweep at a time) == NumPy local_relaxation_sweep.

    Both sides run individual sweeps — no internal batching — so per-sweep
    energies and final positions are directly comparable.
    """
    from egg.smoothing.cpp_backend import CppSweepSession

    ctx, X0, topo = _CONTEXTS[ctx_name]()
    if ctx_name == "circle_in_rect":
        assert np.any(ctx.dof_constraint_tags != 0), "expected constrained DOFs"

    X_ref, e_ref = _numpy_reference(X0, topo, n_sweeps)

    sess = CppSweepSession(ctx, X0.copy(), device="cpu")
    e_parts = []
    for _ in range(n_sweeps):
        e, _ = sess.run(1)
        e_parts.append(e)
    e_sess = np.concatenate(e_parts)
    X_sess = sess.get_X()

    assert e_sess.shape == (n_sweeps,)
    np.testing.assert_allclose(e_sess, e_ref, rtol=_RTOL_E, atol=_ATOL,
                               err_msg="per-sweep energy mismatch vs NumPy")
    np.testing.assert_allclose(X_sess, X_ref, rtol=_RTOL_X, atol=_ATOL,
                               err_msg="final position mismatch vs NumPy")


def test_cpp_sweep_mindet_matches_numpy():
    """The reduction's final min det A matches a NumPy recomputation at X_final."""
    from egg.smoothing.batch import energy_and_mindet
    from egg.smoothing.cpp_backend import cpp_sweep

    ctx, X0, _ = _mini_context()
    n_sweeps = 3
    X_cpp, _e, m_cpp = cpp_sweep(ctx, X0.copy(), n_sweeps, device="cpu")

    es = ctx.energy_stencil
    _e_ref, m_ref = energy_and_mindet(
        X_cpp, es["gc"], es["gn0"], es["gn1"],
        es["s0"], es["s1"], es["W_inv"])
    np.testing.assert_allclose(m_cpp[-1], m_ref, rtol=_RTOL_E, atol=_ATOL,
                               err_msg="final min det A mismatch vs NumPy")


# ---------------------------------------------------------------------------
# Session self-consistency gates (no oracle needed — purely internal).
# ---------------------------------------------------------------------------

def test_cpp_session_chunking_lossless():
    """A persistent session keeps X resident: one .run(N) == N .run(1) chunks.

    Proves the §0 gate — X stays device-resident across calls and chunked
    driving is lossless (identical per-sweep energies/min-dets and final X).
    """
    from egg.smoothing.cpp_backend import CppSweepSession

    ctx, X0, _ = _mini_context()
    n_sweeps = 10

    one = CppSweepSession(ctx, X0.copy(), device="cpu")
    e_one, m_one = one.run(n_sweeps)
    X_one = one.get_X()

    chunked = CppSweepSession(ctx, X0.copy(), device="cpu")
    e_parts, m_parts = [], []
    for _ in range(n_sweeps):
        e, m = chunked.run(1)
        e_parts.append(e)
        m_parts.append(m)
    e_ten = np.concatenate(e_parts)
    m_ten = np.concatenate(m_parts)
    X_ten = chunked.get_X()

    np.testing.assert_array_equal(e_one, e_ten)
    np.testing.assert_array_equal(m_one, m_ten)
    np.testing.assert_array_equal(X_one, X_ten)


def test_cpp_session_matches_one_shot():
    """Session .run matches the one-shot cpp_sweep (same staging, resident X)."""
    from egg.smoothing.cpp_backend import CppSweepSession, cpp_sweep

    ctx, X0, _ = _mini_context()
    n_sweeps = 5

    X_os, e_os, m_os = cpp_sweep(ctx, X0.copy(), n_sweeps, device="cpu")

    sess = CppSweepSession(ctx, X0.copy(), device="cpu")
    e_s, m_s = sess.run(n_sweeps)

    np.testing.assert_array_equal(e_s, e_os)
    np.testing.assert_array_equal(m_s, m_os)
    np.testing.assert_array_equal(sess.get_X(), X_os)


@pytest.mark.skipif(not _has_cpp() or not _gpu_available(),
                    reason="No usable SYCL GPU")
def test_cpp_sweep_matches_numpy_gpu():
    """cpp_sweep (GPU) == NumPy local_relaxation_sweep — same gate on the device path."""
    from egg.smoothing.cpp_backend import cpp_sweep

    ctx, X0, topo = _mini_context()
    n_sweeps = 5
    X_ref, e_ref = _numpy_reference(X0, topo, n_sweeps)

    X_gpu, e_gpu, _m = cpp_sweep(ctx, X0.copy(), n_sweeps, device="gpu")
    np.testing.assert_allclose(e_gpu, e_ref, rtol=_RTOL_E, atol=_ATOL,
                               err_msg="GPU per-sweep energy mismatch vs NumPy")
    np.testing.assert_allclose(X_gpu, X_ref, rtol=_RTOL_X, atol=_ATOL,
                               err_msg="GPU final position mismatch vs NumPy")
