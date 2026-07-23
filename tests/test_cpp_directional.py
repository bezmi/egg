# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""End-to-end: the directional soft-energy terms run through the C++ core —
oracle parity against the NumPy reference in 2D and 3D, the composed energy
report, and the structured sweep actually aligning a marked chain (free DOFs
via the interior kernels, constrained DOFs via the boundary kernels)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("egg._cpp.cpp_core", reason="egg._cpp.cpp_core not built")

from egg._cpp import cpp_core
from egg.geometry import Circle, LineSegment
from egg.smoothing.cpp_backend import (
    CppStructuredSweepSession,
    build_block_structured_context,
)
from egg.smoothing.directional import (
    KIND_LINE,
    KIND_PARALLEL,
    KIND_STEM,
    DirectionalSamples,
    build_directional_samples,
    directional_energy_grad,
)
from egg.smoothing.solver import build_sweep_context
from egg.smoothing.targets import IdentityTarget
from egg.topology.builder import TopologyBuilder
from tests.real_tol import real_tol


def _random_samples(d: int, rng) -> tuple[np.ndarray, DirectionalSamples]:
    X = rng.normal(size=(12, d))
    kind = np.array(
        [KIND_PARALLEL, KIND_PARALLEL, KIND_LINE, KIND_STEM, KIND_LINE], np.int32
    )
    nodes = np.full((kind.size, 4), -1, dtype=np.int32)
    for i, k in enumerate(kind):
        arity = 3 if k == KIND_LINE else 2
        nodes[i, :arity] = rng.choice(X.shape[0], size=arity, replace=False)
    ref = rng.normal(size=(kind.size, d))
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    ref[kind == KIND_LINE] = 0.0
    weight = rng.uniform(0.5, 2.0, size=kind.size)
    s = DirectionalSamples(
        nodes=nodes, kind=kind, ref=ref, axis=np.zeros((kind.size, d)), weight=weight
    )
    return X, s


@pytest.mark.parametrize("d", [2, 3])
def test_oracle_matches_numpy_reference(d):
    rng = np.random.default_rng(11 + d)
    X, s = _random_samples(d, rng)
    e_py, g_py = directional_energy_grad(X, s)
    e_cpp, g_cpp = cpp_core.directional_eval(X, s.nodes, s.kind, s.ref, s.weight, s.eps)
    scale = max(1.0, abs(e_py))
    assert abs(e_cpp - e_py) <= real_tol(1e-12) * scale
    assert np.abs(g_cpp - g_py).max() <= real_tol(1e-12) * max(1.0, np.abs(g_py).max())


def _strip(top_entity=None, res=(8, 4)):
    """Two blocks stacked over a y=0 wall. The marked chain runs along the mid
    row (free interface DOFs) by default; with ``top_entity`` the top edge is
    constrained to that curve and the chain marked there instead (boundary
    kernels)."""
    wall = LineSegment((0.0, 0.0), (4.0, 0.0)).named("wall")
    b = TopologyBuilder(d=2)
    for name, pos in [
        ("w0", (0.0, 0.0)),
        ("w1", (4.0, 0.0)),
        ("m0", (0.0, 1.0)),
        ("m1", (4.0, 1.0)),
        ("t0", (0.0, 2.0)),
        ("t1", (4.0, 2.0)),
    ]:
        b.add_corner(name, pos, fixed=True)
    b.add_block("lo", ("w0", "m0", "w1", "m1"), res)
    b.add_block("hi", ("m0", "t0", "m1", "t1"), res)
    b.associate("lo", 1, 0, wall)
    if top_entity is not None:
        b.associate("hi", 1, 1, top_entity)
        b.parallel_to("wall", chain=("t0", "t1"), weight=2.0)
    else:
        b.parallel_to("wall", chain=("m0", "m1"), weight=2.0)
    topo = b.build()
    topo.initialize_grid()
    return topo.grid


def _sessions(grid, X0, samples):
    ctx_plain = build_sweep_context(grid, IdentityTarget(d=2))
    ctx_dir = build_sweep_context(grid, IdentityTarget(d=2), directional=samples)
    bsc = build_block_structured_context(grid)
    return (
        CppStructuredSweepSession(ctx_plain, bsc, X0.copy(), device="cpu"),
        CppStructuredSweepSession(ctx_dir, bsc, X0.copy(), device="cpu"),
    )


def _bump(grid, amp=0.15):
    """Interior-only sinusoidal displacement (boundary rows untouched)."""
    X = grid.global_nodes.copy()
    lo = X.min(axis=0)
    span = X.max(axis=0) - lo
    s = np.sin(np.pi * (X - lo) / span)
    X[:, 1] += amp * s[:, 0] * s[:, 1]
    return X


def test_reported_energy_is_the_composed_objective():
    """With a zero step (omega=0) nothing moves, so the two reports differ by
    exactly the directional total at the initial state."""
    grid = _strip()
    samples = build_directional_samples(grid)
    assert samples is not None and samples.n > 0
    X0 = _bump(grid)
    e_dir0, _ = directional_energy_grad(X0, samples)
    assert e_dir0 > 0.0

    plain, composed = _sessions(grid, X0, samples)
    e_p, _ = plain.run(1, omega=0.0)
    e_c, _ = composed.run(1, omega=0.0)
    assert abs((e_c[-1] - e_p[-1]) - e_dir0) <= real_tol(1e-9) * max(1.0, e_dir0)


def test_sweep_aligns_a_free_chain_to_the_frozen_wall_normal():
    """The marked mid-row chain (free DOFs, interior kernels) ends up markedly
    more wall-parallel than the plain TMOP minimiser leaves it."""
    grid = _strip()
    samples = build_directional_samples(grid, energy_scale=50.0)
    X0 = _bump(grid)
    plain, composed = _sessions(grid, X0, samples)

    e_p, m_p = plain.run(200, omega=0.8)
    e_c, m_c = composed.run(200, omega=0.8)
    assert np.all(np.isfinite(e_c)) and m_c[-1] > 0.0

    def chain_misalignment(X_flat):
        X = X_flat.reshape(-1, 2)
        seg = X[samples.nodes[:, 1]] - X[samples.nodes[:, 0]]
        t = seg / np.linalg.norm(seg, axis=1, keepdims=True)
        return np.abs(np.einsum("ij,ij->i", t, samples.ref)).max()

    mis_plain = chain_misalignment(plain.get_X())
    mis_dir = chain_misalignment(composed.get_X())
    assert mis_dir < 0.5 * mis_plain, (mis_dir, mis_plain)
    # the composed run's own objective decreased
    e_dir_final, _ = directional_energy_grad(composed.get_X().reshape(-1, 2), samples)
    e_dir_0, _ = directional_energy_grad(X0, samples)
    assert e_dir_final < e_dir_0


def test_boundary_kernels_respect_constraints_under_the_term():
    """Chain DOFs constrained to a circular arc relax through the boundary
    kernels (cold projection + warm param-space line search) with the
    directional term folded in: the run stays finite, holds the barrier,
    keeps the chain exactly on its curve, and lands somewhere else than the
    plain run (the term reached the constrained line search)."""
    # circle through (0, 2) and (4, 2), crown at (2, 2·sqrt(2)): sliding along
    # it changes segment directions, so the term is active on the boundary.
    top = Circle((2.0, 0.0), float(np.sqrt(8.0))).named("top")
    grid = _strip(top_entity=top)
    samples = build_directional_samples(grid, energy_scale=50.0)
    X0 = _bump(grid, amp=0.08)
    plain, composed = _sessions(grid, X0, samples)

    e_p, _ = plain.run(120, omega=0.8)
    e_c, m_c = composed.run(120, omega=0.8)
    assert np.all(np.isfinite(e_c)) and m_c[-1] > 0.0

    X_c = composed.get_X().reshape(-1, 2)
    chain = grid.topology.parallel_chains[0]
    rows = np.asarray(chain.dofs)
    radii = np.linalg.norm(X_c[rows] - np.array([2.0, 0.0]), axis=1)
    assert np.abs(radii - np.sqrt(8.0)).max() <= real_tol(1e-9)
    X_p = plain.get_X().reshape(-1, 2)
    assert not np.allclose(X_c[rows], X_p[rows], atol=real_tol(1e-9))
