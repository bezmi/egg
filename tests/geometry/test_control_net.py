# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Structural invariants of the control-net map (bases, prolong/gather, TFI)."""

from itertools import product

import numpy as np
import pytest

from egg.geometry.control_net import (
    bspline_basis,
    clamped_knots,
    facets,
    greville,
    tensor_map,
    tfi_operator,
)
from egg.init.tfi import tfi_fill_interior

# --------------------------------------------------------------------------- #
# Facet enumeration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("d", [2, 3])
def test_facet_count_and_codim(d):
    fs = facets(d)
    assert len(fs) == 3**d - 1
    # unique offsets, none all-zero
    offs = {f.offset for f in fs}
    assert len(offs) == len(fs)
    assert all(any(o != 0 for o in f.offset) for f in fs)
    for f in fs:
        assert f.codim == sum(1 for o in f.offset if o != 0)
        assert f.idim == d - f.codim
        assert all(o in (-1, 0, 1) for o in f.offset)
    # ordered low codim first
    assert [f.codim for f in fs] == sorted(f.codim for f in fs)


def test_facet_dimension_tallies():
    # 2D: 4 edges (codim1), 4 corners (codim2).
    fs2 = facets(2)
    assert sum(f.codim == 1 for f in fs2) == 4
    assert sum(f.codim == 2 for f in fs2) == 4
    # 3D: 6 faces, 12 edges, 8 corners.
    fs3 = facets(3)
    assert sum(f.codim == 1 for f in fs3) == 6
    assert sum(f.codim == 2 for f in fs3) == 12
    assert sum(f.codim == 3 for f in fs3) == 8


# --------------------------------------------------------------------------- #
# 1D B-spline basis
# --------------------------------------------------------------------------- #


def test_basis_partition_of_unity_and_endpoints():
    params = np.linspace(0.0, 1.0, 25)
    B = bspline_basis(6, 3, params, n_deriv=0)[0][0]
    assert np.allclose(B.sum(axis=1), 1.0)
    # clamped: first/last node interpolate the end control points exactly.
    assert np.allclose(B[0], [1, 0, 0, 0, 0, 0])
    assert np.allclose(B[-1], [0, 0, 0, 0, 0, 1])


def test_basis_compact_support():
    # A cubic basis function is nonzero over at most degree+1 spans.
    params = np.linspace(0.0, 1.0, 200)
    B = bspline_basis(7, 3, params, n_deriv=0)[0][0]
    for j in range(7):
        support = params[B[:, j] > 1e-9]
        assert support.size > 0
        width = support.max() - support.min()
        # 4 interior knot spans of width 1/4 -> support <= (degree+1)*span.
        assert width <= (3 + 1) * (1.0 / 4) + 1e-6


@pytest.mark.parametrize("degree", [2, 3])
def test_basis_continuity_order(degree):
    # Degree p basis is C^{p-1}: derivative p-1 is continuous across an interior
    # knot, derivative p jumps. Probe just left/right of an interior knot.
    n_ctrl = degree + 3  # >= 2 interior knots
    U = clamped_knots(n_ctrl, degree)
    knot = U[degree + 1]  # first interior knot
    eps = 1e-6
    B = bspline_basis(n_ctrl, degree, [knot - eps, knot + eps], n_deriv=degree)[0]
    # highest continuous derivative matches across the knot
    left, right = B[degree - 1, 0], B[degree - 1, 1]
    assert np.allclose(left, right, atol=1e-3)


def test_cubic_second_derivative_continuous():
    # The whole point of cubic: C2 across every interior knot.
    n_ctrl = 8
    U = clamped_knots(n_ctrl, 3)
    eps = 1e-7
    for knot in U[4:-4]:
        B = bspline_basis(n_ctrl, 3, [knot - eps, knot + eps], n_deriv=2)[0]
        assert np.allclose(B[2, 0], B[2, 1], atol=1e-2)


def test_basis_deriv_above_degree_is_zero():
    # Requesting more derivatives than the degree zero-pads instead of erroring.
    B = bspline_basis(4, 2, np.linspace(0, 1, 9), n_deriv=3)[0]
    assert B.shape == (4, 9, 4)
    assert np.allclose(B[3], 0.0)
    assert not np.allclose(B[2], 0.0)


def test_greville_endpoint_exact_and_monotone():
    gv = greville(7, 3)
    assert gv[0] == pytest.approx(0.0)
    assert gv[-1] == pytest.approx(1.0)
    assert np.all(np.diff(gv) > 0)


# --------------------------------------------------------------------------- #
# Tensor-product map: prolong / gather
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("d", [2, 3])
def test_prolong_partition_of_unity(d):
    node_shape = (5,) * d
    ctrl_shape = (4,) * d
    m = tensor_map(node_shape, ctrl_shape, degree=3)
    # A constant control net maps to constant nodes (rows of M sum to 1).
    C = np.ones(ctrl_shape + (d,)) * np.array([1.5, -2.0, 0.7])[:d]
    X = m.prolong(C)
    assert np.allclose(X, np.array([1.5, -2.0, 0.7])[:d])


@pytest.mark.parametrize("d", [2, 3])
def test_prolong_linear_precision(d):
    # Control points at the Greville abscissae under an affine map reproduce that
    # affine map at the node parameters (B-splines have linear precision).
    node_shape = (7,) * d
    ctrl_shape = (5,) * d
    degree = 3
    m = tensor_map(node_shape, ctrl_shape, degree=degree)
    gvl = [greville(ctrl_shape[k], degree) for k in range(d)]
    rng = np.random.default_rng(0)
    A = rng.standard_normal((d, d))
    t = rng.standard_normal(d)
    # control net: affine image of the Greville lattice
    C = np.empty(ctrl_shape + (d,))
    for J in product(*(range(n) for n in ctrl_shape)):
        p = np.array([gvl[k][J[k]] for k in range(d)])
        C[J] = A @ p + t
    X = m.prolong(C)
    # node params (uniform)
    axis_params = [np.linspace(0.0, 1.0, node_shape[k]) for k in range(d)]
    R = m.node_params(axis_params)
    expected = R @ A.T + t
    assert np.allclose(X, expected, atol=1e-10)


@pytest.mark.parametrize("d", [2, 3])
def test_prolong_gather_transpose(d):
    node_shape = (6,) * d
    ctrl_shape = (4,) * d
    m = tensor_map(node_shape, ctrl_shape, degree=3)
    rng = np.random.default_rng(1)
    C = rng.standard_normal(ctrl_shape + (d,))
    gX = rng.standard_normal(node_shape + (d,))
    lhs = float(np.sum(gX * m.prolong(C)))
    rhs = float(np.sum(m.gather(gX) * C))
    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


def test_clustered_spacing_same_net_new_nodes():
    # Refine/cluster = resample with the same control net: values at shared
    # parameters are identical.
    ctrl_shape = (5, 4)
    rng = np.random.default_rng(3)
    C = rng.standard_normal(ctrl_shape + (2,))
    coarse = tensor_map((9, 7), ctrl_shape, degree=3)
    fine = tensor_map(
        (17, 13),
        ctrl_shape,
        degree=3,
        spacing=[np.linspace(0.0, 1.0, 17), np.linspace(0.0, 1.0, 13)],
    )
    Xc = coarse.prolong(C)
    Xf = fine.prolong(C)
    assert np.allclose(Xc, Xf[::2, ::2], atol=1e-12)


# --------------------------------------------------------------------------- #
# Linearized Boolean-sum (TFI) operator
# --------------------------------------------------------------------------- #


class _Blk:
    pass


@pytest.mark.parametrize("d", [2, 3])
def test_tfi_operator_matches_fill(d):
    node_shape = (5,) * d
    rng = np.random.default_rng(2)
    coords = rng.standard_normal(node_shape + (d,))

    # reference: fill interior from the given boundary
    blk = _Blk()
    blk.d = d
    blk.logical_shape = node_shape
    nodes = np.full(node_shape + (d,), np.nan)
    for idx in product(*(range(s) for s in node_shape)):
        if any(idx[a] in (0, node_shape[a] - 1) for a in range(d)):
            nodes[idx] = coords[idx]
    blk.nodes = nodes
    tfi_fill_interior(blk)

    # operator form: M @ boundary
    M = tfi_operator(node_shape)
    n_total = int(np.prod(node_shape))
    bnd = np.zeros((n_total, d))
    for idx in product(*(range(s) for s in node_shape)):
        if any(idx[a] in (0, node_shape[a] - 1) for a in range(d)):
            flat = int(np.ravel_multi_index(idx, node_shape))
            bnd[flat] = coords[idx]
    got = (M @ bnd).reshape(node_shape + (d,))
    assert np.allclose(got, blk.nodes, atol=1e-12)


# --------------------------------------------------------------------------- #
# End-refined fit knots + knot insertion
# --------------------------------------------------------------------------- #


def test_refine_fit_knots_matches_fit_knots_without_refinement():
    from egg.geometry.control_net import fit_knots, refine_fit_knots

    rng = np.random.default_rng(5)
    params = np.sort(np.concatenate([[0.0], rng.random(24), [1.0]]))
    U, n = refine_fit_knots(8, 3, params, (0, 0))
    assert n == 8
    assert np.array_equal(U, fit_knots(8, 3, params))


def test_refine_fit_knots_end_extras_keep_sample_support():
    from egg.geometry.control_net import fit_knots, refine_fit_knots

    rng = np.random.default_rng(5)
    params = np.sort(np.concatenate([[0.0], rng.random(24), [1.0]]))
    U0 = fit_knots(8, 3, params)
    U2, n2 = refine_fit_knots(8, 3, params, (0, 2))
    assert n2 > 8
    ins = np.setdiff1d(U2, U0)
    # extras land beyond the last base interior knot (toward the high end)
    assert np.all(ins > U0[7])
    # every refined knot span still holds at least one fit sample
    kn = U2[3 : n2 + 1]
    for a, b in zip(kn[:-1], kn[1:]):
        assert ((params >= a) & (params <= b)).sum() >= 1


def test_refine_fit_knots_truncates_below_sample_resolution():
    from egg.geometry.control_net import refine_fit_knots

    # 7 samples, base already at ~2 samples/span: deep refinement must stop
    # instead of producing empty end spans.
    params = np.linspace(0.0, 1.0, 7)
    U, n = refine_fit_knots(6, 3, params, (4, 4))
    kn = U[3 : n + 1]
    for a, b in zip(kn[:-1], kn[1:]):
        assert ((params >= a) & (params <= b)).sum() >= 1


def test_insertion_matrix_is_shape_exact():
    from egg.geometry.control_net import (
        fit_knots,
        insertion_matrix,
        refine_fit_knots,
    )

    rng = np.random.default_rng(7)
    params = np.sort(np.concatenate([[0.0], rng.random(30), [1.0]]))
    U0 = fit_knots(9, 3, params)
    U1, n1 = refine_fit_knots(9, 3, params, (2, 1))
    A, Ufull = insertion_matrix(3, U0, np.setdiff1d(U1, U0))
    assert A.shape == (n1, 9)
    assert np.array_equal(Ufull, U1)
    t = np.linspace(0.0, 1.0, 157)
    B0 = bspline_basis(9, 3, t, knots=U0)[0][0]
    B1 = bspline_basis(n1, 3, t, knots=U1)[0][0]
    # identical spline for every control net: B1 @ A == B0
    assert np.abs(B1 @ A - B0).max() < 1e-12
