"""Composite-path entity parity gate for the C++ backend (Phase 3/4).

Validates the composite SoA wire end-to-end — the Python encoder
(``egg.geometry.entity_soa._encode_composite`` → a self-contained per-composite
arena slice), the C++ device load (``EntitySoA<CompositePath>::load``), and the
per-segment ``dispatch_entity_type`` projection in ``CompositePath::project_frame``
— for every composite flavour the 2D front-end produces:

- ``"lines"``   : a Polyline of line segments (the common Polyline case);
- ``"arcs"``    : a Polyline of circular arcs (the capsule-fire body case);
- ``"beziers"`` : a closed Spline (CompositePath of cubic Béziers — the blob case);
- ``"bspline"`` : a composite containing a B-spline sub-segment (a degree-4
  Bezier → BSplineCurve nested in a Polyline — the NURBS-relevant case the
  positional blob path never carried).

The gate is a block-Jacobi sweep on the constrained grid: it projects each
constrained DOF onto its composite every backtracking trial, so a wrong device
reconstruction (bad offsets, a dropped B-spline sub-segment, the wrong nearest
segment) leaves the DOF off its geometry, which the on-curve check catches. A
pure-encode round-trip check guards the wire-format shape too.

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

# fp32 builds compute the device path in float against a double NumPy reference,
# so the tolerances floor at the fp32 parity level (identity in double builds).
from tests.real_tol import real_tol

def _make_composite(kind: str):
    """Build the composite entity for ``kind`` (see module docstring)."""
    from egg.geometry.frontend2d import Arc, Bezier, Line, Polyline, Spline, Vector3

    # Every composite hugs the left edge (x ~ 0) closely, so a slightly-jittered
    # constrained DOF projects back onto it with a small, accepted move (rather
    # than a large move that would invert the adjacent element and be rejected).
    if kind == "lines":
        return Polyline([Line(Vector3(0, 0), Vector3(0, 2)),
                         Line(Vector3(0, 2), Vector3(0, 4))])
    if kind == "arcs":
        # Two large-radius arcs: a shallow ~0.05 bulge into the domain, meeting
        # at (0,2). Centres sit far left so the angular span straddles 0 (not the
        # +/-pi branch cut the frontend Arc adapter rejects).
        return Polyline([Arc(Vector3(0, 0), Vector3(0, 2), Vector3(-10, 1)),
                         Arc(Vector3(0, 2), Vector3(0, 4), Vector3(-10, 3))])
    if kind == "beziers":
        # An open cubic spline (CompositePath of cubic Béziers) interpolating the
        # two sliding nodes (0,1),(0,3) with a tiny bulge between.
        pts = [Vector3(0, 0), Vector3(0, 1), Vector3(0.1, 2),
               Vector3(0, 3), Vector3(0, 4)]
        return Spline(pts, closed=False)
    if kind == "bspline":
        # A degree-4 Bezier (-> BSplineCurve) hugging the edge with a ~0.1 bulge,
        # joined to a return line: a B-spline nested inside a composite.
        arch = Bezier([Vector3(0, 0), Vector3(0.05, 1), Vector3(0.1, 2),
                       Vector3(0.05, 3), Vector3(0, 4)])
        return Polyline([arch, Line(Vector3(0, 4), Vector3(0, 0))])
    raise ValueError(kind)


def _build_constrained_grid(kind: str):
    """A fresh 4x4 grid with the (0,1) and (0,3) left-edge DOFs constrained to
    the ``kind`` composite. Returns ``(grid, topo, dof_ids)``."""
    from egg.topology.builder import TopologyBuilder

    builder = TopologyBuilder(d=2)
    for name, pos in [("A", (0.0, 0.0)), ("B", (4.0, 0.0)),
                      ("C", (4.0, 4.0)), ("D", (0.0, 4.0))]:
        builder.add_corner(name, pos, fixed=True)
    builder.add_block("main", ("A", "D", "B", "C"), (4, 4))
    topo = builder.build()
    topo.initialize_grid()
    grid = topo.grid

    comp = _make_composite(kind)

    def gidx_at(x, y):
        m = np.where((np.abs(grid.global_nodes[:, 0] - x) < 1e-9)
                     & (np.abs(grid.global_nodes[:, 1] - y) < 1e-9))[0]
        assert len(m) == 1, f"expected one node at ({x},{y})"
        return int(m[0])

    dof_ids = [gidx_at(0.0, 1.0), gidx_at(0.0, 3.0)]
    for d in dof_ids:
        grid.dof_constraints[d] = comp
    return grid, topo, dof_ids


@pytest.mark.parametrize("kind", ["lines", "arcs", "beziers", "bspline"])
def test_composite_device_relaxes_on_curve(kind):
    """Block-Jacobi projects each composite-constrained DOF onto its composite.

    A wrong device reconstruction of the composite SoA wire — bad offsets, a
    dropped B-spline sub-segment, the wrong nearest segment — would project a
    constrained DOF off its geometry, which the on-curve check below catches.
    """
    from egg.smoothing.cpp_backend import cpp_structured_sweep
    from egg.smoothing.solver import build_sweep_context
    from egg.smoothing.targets import IdentityTarget

    # Jitter everything and push the constrained DOFs ~0.2 into the domain (+x)
    # off their near-edge composite, so projecting them back reduces distortion
    # and the move is accepted (the projection actually fires).
    grid, _topo, dof_ids = _build_constrained_grid(kind)
    rng = np.random.default_rng(20240615)
    X0 = grid.global_nodes + 0.02 * rng.standard_normal(grid.global_nodes.shape)
    X0[dof_ids, 0] += 0.2

    ctx = build_sweep_context(grid, IdentityTarget(d=2))
    assert np.any(ctx.dof_constraint_tags == 11), "expected a composite-constrained DOF"

    X_cpp, _e, _m = cpp_structured_sweep(
        ctx, grid, X0.copy(), 5, device="cpu")

    assert np.all(np.isfinite(X_cpp)), "device produced non-finite positions"
    # Each constrained DOF ends ON its composite: it moved off its jittered start
    # and onto the correctly reconstructed geometry.
    comp = _make_composite(kind)
    for d in dof_ids:
        moved = not np.allclose(X_cpp[d], X0[d])
        on_curve = np.allclose(comp.project(X_cpp[d]), X_cpp[d], atol=real_tol(1e-7))
        assert moved and on_curve, (
            f"{kind}: DOF {d} not projected onto its composite "
            f"(moved={moved}, on_curve={on_curve}, X={X_cpp[d]})"
        )


@pytest.mark.parametrize("kind", ["lines", "arcs", "beziers", "bspline"])
def test_composite_soa_encode_shape(kind):
    """The SoA encoder emits a self-contained (n_segs, rec_off) + arena slice."""
    from egg.geometry.entity_soa import TAG_COMPOSITE, encode_entity_soa

    comp = _make_composite(kind)
    tag, k_fields, row, seg = encode_entity_soa(comp)
    assert tag == TAG_COMPOSITE
    assert k_fields == 2                      # (n_segs, rec_off)
    n_segs, rec_off = int(row[0]), int(row[1])
    assert n_segs == len(comp.segments)
    assert seg is not None and len(seg) == 1  # one CSR slot: the arena slice
    slice_len = len(seg[0])
    # Records occupy the tail; rec_off bytes of sub-segment data precede them.
    assert 0 <= rec_off <= slice_len
    assert (slice_len - rec_off) == n_segs * 13   # 1 tag + 12 params per record
    # Only a B-spline sub-segment contributes pre-record arena data.
    if kind == "bspline":
        assert rec_off > 0
    else:
        assert rec_off == 0
