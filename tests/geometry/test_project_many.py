# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Batch projection (`project_many`) agrees with per-point `project`.

The vectorized overrides run the same seeds and Newton/secant iterations
lane-wise, so they must reproduce the scalar path exactly — these tests
pin that contract for every curve kind (plus the GeometryEntity fallback
loop), since respace/project_nodes now route through the batch API.
"""

import numpy as np
import pytest

from egg.geometry import Spline, Vector3
from egg.geometry.analytic2d import LineSegment
from egg.geometry.curves2d import (
    BSplineCurve,
    CircleArc,
    CubicBezier,
    EllipseArc,
    QuadBezier,
)


def _egg_spline():
    theta = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
    ring = [
        Vector3(2.0 + (0.66 - 0.15 * np.sin(t)) * np.cos(t), 2.0 + 0.85 * np.sin(t))
        for t in theta
    ]
    return Spline(ring, closed=True)


def _bspline(weights=False):
    rng = np.random.default_rng(7)
    n_ctrl, deg = 7, 3
    knots = np.concatenate([[0.0] * (deg + 1), [0.3, 0.6, 0.8], [1.0] * (deg + 1)])
    w = rng.uniform(0.5, 2.0, n_ctrl) if weights else None
    return BSplineCurve(deg, knots, rng.uniform(0.0, 3.0, (n_ctrl, 2)), weights=w)


ENTITIES = {
    "circle_arc": CircleArc([1.0, 0.5], 1.3, -2.0, 1.1),
    "circle_arc_closed": CircleArc([1.0, 0.5], 1.3, -np.pi, np.pi, closed=True),
    "ellipse_arc": EllipseArc([0.5, 1.0], 1.5, 0.7, 0.4, 0.3, 5.1),
    "quad_bezier": QuadBezier([0, 0], [1, 2], [3, 1]),
    "cubic_bezier": CubicBezier([0, 0], [1, 2], [2, -1], [3, 1]),
    "bspline": _bspline(),
    "bspline_rational": _bspline(weights=True),
    "line_segment_fallback": LineSegment([0, 0], [3, 2]),
    "composite_closed_spline": _egg_spline(),
}


@pytest.mark.parametrize("name", sorted(ENTITIES))
def test_project_many_matches_scalar_project(name):
    entity = ENTITIES[name]
    rng = np.random.default_rng(42)
    Q = rng.uniform(-2.0, 4.0, size=(100, 2))
    scalar = np.stack([np.asarray(entity.project(q), dtype=float) for q in Q])
    batch = np.asarray(entity.project_many(Q), dtype=float)
    assert batch.shape == scalar.shape
    # The lane-wise seeds/iterations mirror the scalar path, but the vectorized
    # eval/deriv reductions reassociate the same float ops, so the seeded-Newton
    # foot can drift by the last ULP (BLAS/numpy dependent). Pin the feet to a
    # tight absolute tolerance rather than bitwise equality.
    np.testing.assert_allclose(batch, scalar, rtol=0, atol=1e-12)
