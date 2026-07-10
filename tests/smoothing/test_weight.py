# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Per-sample weight scales mu (energy/grad/Hessian) but not the det barrier."""

import numpy as np
import pytest

from egg.smoothing import batch


def _synthetic(seed=0, P=12):
    """P well-formed (non-degenerate) corner samples on jittered unit cells."""
    rng = np.random.default_rng(seed)
    gc = np.arange(P) * 3
    gn0 = gc + 1
    gn1 = gc + 2
    corner = rng.normal(size=(P, 2))
    X = np.empty((3 * P, 2))
    X[gc] = corner
    X[gn0] = corner + np.array([1.0, 0.0]) + 0.15 * rng.normal(size=(P, 2))
    X[gn1] = corner + np.array([0.0, 1.0]) + 0.15 * rng.normal(size=(P, 2))
    s0 = np.ones(P)
    s1 = np.ones(P)
    W_inv = np.broadcast_to(np.eye(2), (P, 2, 2)).copy()
    role = rng.integers(-1, 3, size=P)
    return X, gc, gn0, gn1, s0, s1, W_inv, role


@pytest.mark.parametrize("metric", ["shape_2d", "shape_size"])
def test_unit_weight_is_identity(metric):
    X, gc, gn0, gn1, s0, s1, W_inv, role = _synthetic()
    ones = np.ones(gc.shape[0])
    args = (X, gc, gn0, gn1, s0, s1, W_inv)

    e0, m0 = batch.energy_and_mindet(*args, metric=metric)
    e1, m1 = batch.energy_and_mindet(*args, metric=metric, weight=ones)
    assert e0 == pytest.approx(e1)
    assert m0 == pytest.approx(m1)

    g0, h0 = batch.dof_grad_hess(*args, role, metric=metric)
    g1, h1 = batch.dof_grad_hess(*args, role, metric=metric, weight=ones)
    assert np.allclose(g0, g1)
    assert np.allclose(h0, h1)


@pytest.mark.parametrize("metric", ["shape_2d", "shape_size"])
def test_scalar_weight_scales_energy_grad_hess(metric):
    X, gc, gn0, gn1, s0, s1, W_inv, role = _synthetic(seed=3)
    args = (X, gc, gn0, gn1, s0, s1, W_inv)
    w = np.full(gc.shape[0], 2.5)

    e0, m0 = batch.energy_and_mindet(*args, metric=metric)
    ew, mw = batch.energy_and_mindet(*args, metric=metric, weight=w)
    assert ew == pytest.approx(2.5 * e0)
    assert mw == pytest.approx(m0)  # barrier unaffected by weight

    g0, h0 = batch.dof_grad_hess(*args, role, metric=metric)
    gw, hw = batch.dof_grad_hess(*args, role, metric=metric, weight=w)
    assert np.allclose(gw, 2.5 * g0)
    assert np.allclose(hw, 2.5 * h0)


def test_patch_eval_matches_split_with_weight():
    X, gc, gn0, gn1, s0, s1, W_inv, role = _synthetic(seed=7)
    args = (X, gc, gn0, gn1, s0, s1, W_inv)
    w = np.linspace(0.2, 1.8, gc.shape[0])

    g, h, e, mdet = batch.patch_eval(*args, role, metric="shape_2d", weight=w)
    g_ref, h_ref = batch.dof_grad_hess(*args, role, metric="shape_2d", weight=w)
    e_ref, m_ref = batch.energy_and_mindet(*args, metric="shape_2d", weight=w)
    assert np.allclose(g, g_ref)
    assert np.allclose(h, h_ref)
    assert e == pytest.approx(e_ref)
    assert mdet == pytest.approx(m_ref)
