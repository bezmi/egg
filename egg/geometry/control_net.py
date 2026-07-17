# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Control-net grid map (tensor-product B-spline + Boolean-sum boundary).

Dimension-agnostic. A block's node positions are a fixed *linear* function of a
per-block control net::

    X = M @ C + b

``M`` is the tensor-product blend of the control net (compact-support
clamped-uniform B-spline basis; cubic -> C2 inside the block), ``b`` carries
the Boolean-sum correction that overrides the block boundary with prescribed
geometry paths (zero for an interior-only block, where the map is the pure
tensor blend). Both the sequential NumPy reference and the C++ upload consume
this single source of truth.

The map is separable over axes: a structured block samples each axis with its
own 1D parameter distribution, so ``M`` is the Kronecker product of the
per-axis B-spline basis matrices and prolong/gather are mode products (no
dense operator is ever formed). The per-axis basis is exposed with first and
second derivatives so downstream code can assemble tangent (C1,
orthogonality) and curvature (soft C2) conditions.

Cross-block continuity is *not* this module's job: seams are expressed by the
control-DOF topology (shared boundary controls for C0, the S/T first-interior
elimination for exact C1), which composes per-block maps built here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import NamedTuple

import numpy as np

__all__ = [
    "Facet",
    "facets",
    "clamped_knots",
    "fit_knots",
    "refine_fit_knots",
    "insertion_matrix",
    "bspline_basis",
    "axis_basis",
    "greville",
    "ControlNetMap",
    "tensor_map",
    "tfi_operator",
]


# --------------------------------------------------------------------------- #
# Facet enumeration (the D-general backbone: 3**D - 1 boundary facets)
# --------------------------------------------------------------------------- #


class Facet(NamedTuple):
    """A boundary facet of a D-cube.

    ``offset`` is a vector in ``{-1, 0, +1}**D`` (never all-zero); a nonzero
    component pins that axis to a face (``-1`` low, ``+1`` high), a zero
    component spans it. ``codim`` is the number of pinned axes, so the facet's
    intrinsic dimension is ``idim = D - codim`` (faces codim 1, edges codim 2,
    corners codim D).
    """

    offset: tuple[int, ...]
    codim: int
    idim: int


def facets(d: int) -> list[Facet]:
    """All ``3**d - 1`` boundary facets of a d-cube, low codim first.

    Ordered by increasing codimension (faces, then edges, then corners) so a
    consumer can fill or share facets from lower-dimensional ones inward.
    """
    out: list[Facet] = []
    for off in product((-1, 0, 1), repeat=d):
        codim = sum(1 for o in off if o != 0)
        if codim == 0:
            continue
        out.append(Facet(tuple(off), codim, d - codim))
    out.sort(key=lambda f: f.codim)
    return out


# --------------------------------------------------------------------------- #
# 1D clamped-uniform B-spline basis with derivatives (endpoint-exact)
# --------------------------------------------------------------------------- #


def clamped_knots(n_ctrl: int, degree: int) -> np.ndarray:
    """Clamped-uniform knot vector on ``[0, 1]`` for ``n_ctrl`` control points.

    ``degree + 1`` repeated knots at each end (so the curve interpolates the end
    control points) and evenly spaced interior knots.
    """
    if n_ctrl < degree + 1:
        raise ValueError(
            f"need n_ctrl >= degree+1, got n_ctrl={n_ctrl}, degree={degree}"
        )
    n_interior = n_ctrl - degree - 1
    if n_interior > 0:
        interior = np.arange(1, n_interior + 1, dtype=float) / (n_interior + 1)
    else:
        interior = np.empty(0)
    return np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])


def _find_span(n: int, p: int, u: float, U: np.ndarray) -> int:
    """Knot span index of ``u`` (NURBS Book A2.1), clamped at both ends."""
    if u >= U[n + 1]:
        return n
    if u <= U[p]:
        return p
    low, high = p, n + 1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def _ders_basis_funs(i: int, u: float, p: int, nd: int, U: np.ndarray) -> np.ndarray:
    """Nonzero basis funcs and derivatives at span ``i`` (NURBS Book A2.3).

    Returns ``ders`` shape ``(nd+1, p+1)``: ``ders[k, r]`` is the k-th derivative
    of the basis function ``N_{i-p+r}`` at ``u``. Endpoint-exact by construction
    (the span clamp handles ``u = 0`` and ``u = 1``). ``nd`` must be <= ``p``
    (the caller zero-pads higher orders, which vanish identically).
    """
    ndu = np.zeros((p + 1, p + 1))
    ders = np.zeros((nd + 1, p + 1))
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    ndu[0, 0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved
    for j in range(p + 1):
        ders[0, j] = ndu[j, p]
    a = np.zeros((2, p + 1))
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0, 0] = 1.0
        for k in range(1, nd + 1):
            d = 0.0
            rk, pk = r - k, p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1
    fac = p
    for k in range(1, nd + 1):
        ders[k, :] *= fac
        fac *= p - k
    return ders


def fit_knots(n_ctrl: int, degree: int, params) -> np.ndarray:
    """Clamped knot vector adapted to a fit-sample distribution.

    Interior knots at quantiles of ``params``, so every knot span holds an
    (approximately equal) share of the samples — the Schoenberg–Whitney
    support condition a least-squares fit needs. A clustered sampling (e.g.
    boundary-layer chord parameters) gets matching control-row clustering;
    exactly uniform ``params`` reproduce :func:`clamped_knots` exactly.
    Falls back to the uniform knots when quantiles degenerate (duplicate
    parameters from zero-length cells).
    """
    U = clamped_knots(n_ctrl, degree)
    n_interior = n_ctrl - degree - 1
    if n_interior <= 0:
        return U
    params = np.asarray(params, dtype=float)
    fracs = np.arange(1, n_interior + 1, dtype=float) / (n_interior + 1)
    interior = np.quantile(params, fracs)
    knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
    if np.any(np.diff(knots[degree : n_ctrl + 1]) <= 1e-9):
        return U
    return knots


def refine_fit_knots(
    n_ctrl: int, degree: int, params, refine: tuple[int, int]
) -> tuple[np.ndarray, int]:
    """:func:`fit_knots` with extra end refinement; returns ``(knots, n_ctrl)``.

    ``refine`` asks for up to that many extra interior knots at the (low,
    high) end. Extras are placed in *quantile-fraction* space, each halving
    the remaining fraction toward the end — so control rows cluster
    geometrically toward the end while every new knot span keeps its share of
    the fit samples (the Schoenberg–Whitney support a least-squares fit
    needs). An extra whose end span would drop below one sample is not added,
    and the returned control count reflects what was actually inserted. With
    ``refine == (0, 0)`` this is exactly :func:`fit_knots`.
    """
    params = np.asarray(params, dtype=float)
    m = n_ctrl - degree - 1
    base = [j / (m + 1) for j in range(1, m + 1)]

    # Each extra must leave at least one sample (one fine cell) in the end
    # span it creates — beyond that the new controls would live below the
    # grid's own resolution. The refined net must NOT be least-squares
    # fitted directly (too few samples per refined span for a stable fit);
    # fit at the base density and carry the fit over by knot insertion
    # (:func:`insertion_matrix`), which is shape-exact at any density.
    lo: list[float] = []
    f = base[0] if base else 1.0
    for _ in range(int(refine[0])):
        f = f / 2.0
        if f * (params.size - 1) < 1.0:
            break
        lo.append(f)
    hi: list[float] = []
    f = base[-1] if base else 0.0
    for _ in range(int(refine[1])):
        f = (f + 1.0) / 2.0
        if (1.0 - f) * (params.size - 1) < 1.0:
            break
        hi.append(f)

    fracs = np.array(sorted(set(lo) | set(base) | set(hi)))
    n_out = int(degree + 1 + fracs.size)
    if n_out == n_ctrl:
        return fit_knots(n_ctrl, degree, params), n_ctrl
    interior = np.quantile(params, fracs)
    knots = np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])
    if np.any(np.diff(knots[degree : n_out + 1]) <= 1e-9):
        # Degenerate refined quantiles (duplicate parameters): drop the
        # refinement rather than fall back to a uniform refined vector.
        return fit_knots(n_ctrl, degree, params), n_ctrl
    return knots, n_out


def insertion_matrix(
    degree: int, base_knots: np.ndarray, values
) -> tuple[np.ndarray, np.ndarray]:
    """Knot-insertion (Boehm) operator: returns ``(A, knots)``.

    ``A`` has shape ``(n_base + len(values), n_base)`` and maps control
    points on ``base_knots`` to the control points of the *identical* spline
    on the refined vector (``base_knots`` with ``values`` inserted). Shape
    exactness makes this the safe way to densify a fitted net: unlike a
    least-squares fit at the refined density, it cannot wiggle between
    samples.
    """
    U = list(np.asarray(base_knots, dtype=float))
    n = len(U) - degree - 1
    A = np.eye(n)
    for u in sorted(float(v) for v in values):
        k = int(np.searchsorted(np.asarray(U), u, side="right")) - 1
        n_cur = len(U) - degree - 1
        B = np.zeros((n_cur + 1, n_cur))
        for i in range(n_cur + 1):
            if i <= k - degree:
                B[i, i] = 1.0
            elif i <= k:
                alpha = (u - U[i]) / (U[i + degree] - U[i])
                B[i, i - 1] = 1.0 - alpha
                B[i, i] = alpha
            else:
                B[i, i - 1] = 1.0
        A = B @ A
        U.insert(k + 1, u)
    return A, np.asarray(U)


def bspline_basis(
    n_ctrl: int, degree: int, params, n_deriv: int = 0, knots: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Clamped B-spline basis matrix (with derivatives) at ``params``.

    Returns ``(B, knots)`` where ``B`` has shape ``(n_deriv+1, len(params),
    n_ctrl)``: ``B[0]`` is the value matrix (rows sum to 1), ``B[k]`` its k-th
    derivative. Derivative orders above ``degree`` are identically zero and
    returned as zero-padded planes. A degree-``p`` basis is ``C**(p-1)``.
    ``knots`` overrides the default clamped-uniform vector (see
    :func:`fit_knots`).
    """
    U = clamped_knots(n_ctrl, degree) if knots is None else np.asarray(knots, float)
    n = n_ctrl - 1
    nd = min(n_deriv, degree)
    params = np.asarray(params, dtype=float)
    B = np.zeros((n_deriv + 1, params.size, n_ctrl))
    for pi, u in enumerate(params):
        span = _find_span(n, degree, float(u), U)
        ders = _ders_basis_funs(span, float(u), degree, nd, U)
        B[: nd + 1, pi, span - degree : span + 1] = ders
    return B, U


def axis_basis(
    n_ctrl: int, degree: int, params, n_deriv: int = 0, knots: np.ndarray | None = None
) -> np.ndarray:
    """Just the basis tensor from :func:`bspline_basis` (drops the knot vector)."""
    return bspline_basis(n_ctrl, degree, params, n_deriv, knots=knots)[0]


def greville(n_ctrl: int, degree: int, knots: np.ndarray | None = None) -> np.ndarray:
    """Greville abscissae of a clamped basis (one per control point)."""
    U = clamped_knots(n_ctrl, degree) if knots is None else np.asarray(knots, float)
    return np.array([U[j + 1 : j + 1 + degree].mean() for j in range(n_ctrl)])


# --------------------------------------------------------------------------- #
# Tensor-product control-net map
# --------------------------------------------------------------------------- #


@dataclass
class ControlNetMap:
    """Linear map ``X = M C + b`` for one block, stored as per-axis factors.

    ``basis[k]`` is the axis-``k`` value matrix ``(node_shape[k], ctrl_shape[k])``,
    ``basis_d1[k]`` / ``basis_d2[k]`` its first / second derivative (same shape);
    ``M`` is their Kronecker product, never materialised. ``b`` (``node_shape +
    (d,)`` or None) is the Boolean-sum boundary offset, zero for an
    interior-only block.
    """

    d: int
    node_shape: tuple[int, ...]
    ctrl_shape: tuple[int, ...]
    degrees: tuple[int, ...]
    basis: list[np.ndarray]
    basis_d1: list[np.ndarray]
    basis_d2: list[np.ndarray]
    b: np.ndarray | None = None
    knots: list[np.ndarray] | None = None

    def prolong(self, C: np.ndarray) -> np.ndarray:
        """Control net ``C`` (``ctrl_shape + (d,)``) -> nodes (``node_shape + (d,)``)."""
        X = np.asarray(C, dtype=float)
        for k, B in enumerate(self.basis):
            X = np.moveaxis(np.tensordot(B, X, axes=([1], [k])), 0, k)
        if self.b is not None:
            X = X + self.b
        return X

    def gather(self, gX: np.ndarray) -> np.ndarray:
        """Adjoint: node gradient (``node_shape + (d,)``) -> control gradient ``M^T gX``."""
        gC = np.asarray(gX, dtype=float)
        for k, B in enumerate(self.basis):
            gC = np.moveaxis(np.tensordot(B.T, gC, axes=([1], [k])), 0, k)
        return gC

    def node_params(self, axis_params: list[np.ndarray]) -> np.ndarray:
        """Node parameter field ``node_shape + (d,)`` from the per-axis samples."""
        grids = np.meshgrid(*axis_params, indexing="ij")
        return np.stack(grids, axis=-1)


def _axis_params(n_node: int, spacing: np.ndarray | None) -> np.ndarray:
    """Per-axis node parameters in ``[0, 1]``: uniform, or a given clustering."""
    if spacing is None:
        return np.linspace(0.0, 1.0, n_node)
    s = np.asarray(spacing, dtype=float)
    if s.shape != (n_node,):
        raise ValueError(f"spacing must have shape ({n_node},), got {s.shape}")
    return s


def tensor_map(
    node_shape,
    ctrl_shape,
    *,
    degree: int | tuple[int, ...] = 3,
    spacing: list[np.ndarray | None] | None = None,
    knots: list[np.ndarray | None] | None = None,
) -> ControlNetMap:
    """Build the interior-only (b=0) tensor-product control-net map.

    ``node_shape`` / ``ctrl_shape`` are per-axis counts; ``degree`` is scalar or
    per-axis (cubic -> C2 inside the block). ``spacing`` optionally gives each
    axis a clustered node parameter distribution in ``[0, 1]`` (default
    uniform); the control net is invariant across a resampling, so
    refine/derefine/cluster is a new ``spacing`` with the same ``ctrl_shape``.
    ``knots`` optionally overrides each axis's clamped-uniform knot vector —
    a net is defined by its knots, so a resampling of an existing net must
    pass that net's own knots, not re-derive them from the new spacing.
    """
    node_shape = tuple(int(s) for s in node_shape)
    ctrl_shape = tuple(int(s) for s in ctrl_shape)
    d = len(node_shape)
    if len(ctrl_shape) != d:
        raise ValueError("node_shape and ctrl_shape must have the same rank")
    degrees = (degree,) * d if isinstance(degree, int) else tuple(degree)
    if len(degrees) != d:
        raise ValueError("degree must be scalar or one per axis")
    if spacing is None:
        spacing = [None] * d
    if knots is None:
        knots = [None] * d

    basis: list[np.ndarray] = []
    basis_d1: list[np.ndarray] = []
    basis_d2: list[np.ndarray] = []
    knots_out: list[np.ndarray] = []
    for k in range(d):
        params = _axis_params(node_shape[k], spacing[k])
        Bk, Uk = bspline_basis(
            ctrl_shape[k], degrees[k], params, n_deriv=2, knots=knots[k]
        )
        basis.append(Bk[0])
        basis_d1.append(Bk[1])
        basis_d2.append(Bk[2])
        knots_out.append(Uk)
    return ControlNetMap(
        d,
        node_shape,
        ctrl_shape,
        degrees,
        basis,
        basis_d1,
        basis_d2,
        knots=knots_out,
    )


# --------------------------------------------------------------------------- #
# Linearized Boolean-sum (transfinite) operator
# --------------------------------------------------------------------------- #


def tfi_operator(node_shape) -> np.ndarray:
    """Dense linear operator of the D-general Boolean-sum TFI fill.

    Reproduces ``egg.init.tfi.tfi_fill_interior``: given all block-boundary node
    positions, it returns every interior node as a fixed linear combination of
    them. Returned as a dense ``(n_total, n_total)`` matrix acting per coordinate
    (small, built once, used to form the ``b`` offset when a block boundary is
    pinned to geometry). Boundary rows are identity (boundary nodes pass through).
    """
    node_shape = tuple(int(s) for s in node_shape)
    d = len(node_shape)
    n_total = int(np.prod(node_shape))
    strides = np.array([int(np.prod(node_shape[k + 1 :])) for k in range(d)], dtype=int)

    def flat(idx: tuple[int, ...]) -> int:
        return int(np.dot(np.asarray(idx, dtype=int), strides))

    # Column basis vectors run through the value-form fill; column j is the
    # interior response to a unit boundary node j. Cheaper than a symbolic
    # expansion and exact for the linear map.
    from egg.init.tfi import tfi_fill_interior

    class _Blk:
        pass

    def is_boundary(idx: tuple[int, ...]) -> bool:
        return any(idx[a] in (0, node_shape[a] - 1) for a in range(d))

    boundary_flats = [
        flat(idx)
        for idx in product(*(range(s) for s in node_shape))
        if is_boundary(idx)
    ]

    M = np.zeros((n_total, n_total))
    # Interior response: seed one boundary node, fill, read interior.
    for j in boundary_flats:
        blk = _Blk()
        blk.d = d
        blk.logical_shape = node_shape
        nodes = np.full(node_shape + (1,), np.nan)
        # Set every boundary node to 0 except node j -> 1 (unit boundary input).
        for bf in boundary_flats:
            idx = np.unravel_index(bf, node_shape)
            nodes[idx + (0,)] = 1.0 if bf == j else 0.0
        blk.nodes = nodes
        tfi_fill_interior(blk)
        M[:, j] = nodes.reshape(n_total)
    # Boundary rows: identity (a boundary node maps to itself).
    for j in boundary_flats:
        M[j, :] = 0.0
        M[j, j] = 1.0
    return M
