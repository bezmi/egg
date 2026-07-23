# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Open-ended rails: a sliding node runs along the segment and releases —
becomes free — beyond either end, instead of clamping at the endpoint."""

import numpy as np
import pytest

from egg.geometry import Line, Vector3
from egg.geometry.analytic3d import Line3
from egg.topology.builder import TopologyBuilder

cpp_core = pytest.importorskip("egg._cpp.cpp_core")


def test_projection_releases_beyond_ends_2d():
    rail = Line(Vector3(0, 0), Vector3(1, 0)).open_ended()
    assert np.allclose(rail.project(np.array([0.5, 0.3])), [0.5, 0.0])
    assert np.allclose(rail.project(np.array([1.4, 0.3])), [1.4, 0.3])  # identity
    assert np.allclose(rail.project(np.array([-0.2, 0.1])), [-0.2, 0.1])
    clamped = Line(Vector3(0, 0), Vector3(1, 0))
    assert np.allclose(clamped.project(np.array([1.4, 0.3])), [1.0, 0.0])


def test_projection_releases_beyond_ends_3d():
    rail = Line3((0, 0, 0), (0, 0, 1)).open_ended()
    assert np.allclose(rail.project(np.array([0.2, 0.0, 0.5])), [0.0, 0.0, 0.5])
    assert np.allclose(rail.project(np.array([0.2, 0.1, 1.7])), [0.2, 0.1, 1.7])


def test_open_ended_unsupported_kind_raises():
    from egg.geometry import Circle
    from egg.geometry.entity_soa import encode_entity_soa

    with pytest.raises(NotImplementedError, match="open_ended"):
        encode_entity_soa(Circle(center=(0, 0), radius=1.0).open_ended())


def _smooth_with_rail(rail, d3=False, sweeps=300):
    """Single-block grid with one interior column constrained to `rail`;
    returns (final X, the constrained global ids sorted by transverse coord)."""
    from egg.smoothing.cpp_backend import (
        CppStructuredSweepSession,
        build_block_structured_context,
    )
    from egg.smoothing.solver import build_sweep_context
    from egg.smoothing.targets import IdentityTarget

    b = TopologyBuilder(d=3 if d3 else 2)
    if d3:
        for x in (0, 2):
            for y in (0, 1):
                for z in (0, 1):
                    b.add_corner(f"p{x}{y}{z}", (x, y, z), fixed=True)
        corners = [f"p{i * 2}{j}{k}" for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        b.add_block("H", corners=corners, resolutions=(10, 4, 4))
    else:
        b.add_block(
            "B",
            sw=Vector3(0, 0, fixed=True),
            se=Vector3(2, 0, fixed=True),
            nw=Vector3(0, 1, fixed=True),
            ne=Vector3(2, 1, fixed=True),
            res=(10, 6),
        )
    topo = b.build()
    grid = topo.initialize_grid()
    X0 = grid.global_nodes
    d = 3 if d3 else 2

    # constrain the interior nodes nearest the x=1.0 grid column
    ids = []
    for g in range(X0.shape[0]):
        if not grid.free_mask[g]:
            continue
        p = X0[g]
        if abs(p[0] - 1.0) < 1e-9 and 1e-9 < p[1] < 1 - 1e-9:
            if d3 and not (1e-9 < p[2] < 1 - 1e-9):
                continue
            grid.dof_constraints[g] = rail
            ids.append(g)
    assert len(ids) >= 3

    ctx = build_sweep_context(grid, IdentityTarget(d))
    bsc = build_block_structured_context(grid)
    sess = CppStructuredSweepSession(ctx, bsc, X0, device="cpu")
    sess.run(sweeps, phase="barrier", omega=0.8)
    X = sess.get_X().reshape(-1, d)
    key = 2 if d3 else 1
    ids.sort(key=lambda g: X0[g][key])
    return X, ids


def test_solver_release_2d():
    # rail at x=0.9 covering y in [0, 0.5]; the column's natural position is
    # x=1.0, so nodes below the end slide onto the rail and nodes above it
    # must release instead of clamping at (0.9, 0.5)
    rail = Line(Vector3(0.9, 0.0), Vector3(0.9, 0.5)).open_ended()
    X, ids = _smooth_with_rail(rail)
    on = [g for g in ids if X[g][1] <= 0.5 and abs(X[g][0] - 0.9) < 1e-9]
    released = [g for g in ids if X[g][1] > 0.55]
    assert len(on) >= 1  # lower nodes ride the rail exactly
    assert len(released) >= 1
    for g in released:
        # free, not clamped at the rail end: stayed near the natural column
        assert X[g][0] > 0.95
        assert np.linalg.norm(X[g] - [0.9, 0.5]) > 0.05

    # control: the CLAMPED segment piles the upper nodes onto its endpoint
    clamped = Line(Vector3(0.9, 0.0), Vector3(0.9, 0.5))
    Xc, idsc = _smooth_with_rail(clamped)
    end_hits = [g for g in idsc if np.linalg.norm(Xc[g] - [0.9, 0.5]) < 1e-6]
    assert len(end_hits) >= 1


def test_solver_release_3d():
    rail = Line3((0.9, 0.5, 0.0), (0.9, 0.5, 0.5)).open_ended()
    X, ids = _smooth_with_rail(rail, d3=True)
    on = [
        g
        for g in ids
        if X[g][2] <= 0.5 and abs(X[g][0] - 0.9) < 1e-9 and abs(X[g][1] - 0.5) < 1e-9
    ]
    released = [g for g in ids if X[g][2] > 0.55]
    assert len(on) >= 1
    assert len(released) >= 1
    for g in released:
        assert X[g][0] > 0.95  # relaxed toward the natural column, not clamped
        assert np.linalg.norm(X[g] - [0.9, 0.5, 0.5]) > 0.05
