# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""End-to-end: the pipeline enables the directional terms automatically when
the topology declares them. On a five-block disk with a singular five-fan
centre, a declared fan frame pulls the through legs straight and holds the
normal leg perpendicular; the undeclared control keeps the five-fold TMOP
optimum (72-degree sectors, so the through pair sits ~36 degrees off
straight)."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

pytest.importorskip("egg._cpp.cpp_core", reason="egg._cpp.cpp_core not built")

from egg.geometry import Circle, Vector3
from egg.pipeline import PipelineConfig, run_pipeline
from egg.smoothing.solver import build_sweep_context
from egg.smoothing.targets import IdentityTarget
from egg.smoothing.untangle import grid_min_det
from egg.topology.builder import TopologyBuilder


def _five_fan(framed: bool, res: int = 7):
    rim = Circle((0.0, 0.0), 1.5).named("rim")
    C = Vector3(0.0, 0.0)
    R = [
        Vector3(math.cos(2 * math.pi * k / 5), math.sin(2 * math.pi * k / 5))
        for k in range(5)
    ]
    M = [
        Vector3(
            1.5 * math.cos(2 * math.pi * (k + 0.5) / 5),
            1.5 * math.sin(2 * math.pi * (k + 0.5) / 5),
        )
        for k in range(5)
    ]
    b = TopologyBuilder(d=2)
    for k in range(5):
        b.add_block(
            f"b{k}", corners=(C, R[(k + 1) % 5], R[k], M[k]), resolutions=(res, res)
        )
        b.associate(f"b{k}", 0, 1, rim)
        b.associate(f"b{k}", 1, 1, rim)
    if framed:
        b.fan_frame(C, through=(R[0], R[2]), normal=R[1])
    topo = b.build()
    topo.initialize_grid()
    return topo.grid


def _deviations(grid) -> tuple[float, float]:
    """(through, normal) deviations in degrees at the framed fan vertex:
    through = angle the two through legs miss a straight line by; normal =
    angle the normal leg misses the through-axis perpendicular by."""
    f = grid.topology.fan_frames[0]
    X = np.asarray(grid.global_nodes)
    a, c, b = f.through_rails[0][1], f.dof, f.through_rails[1][1]

    def unit(v):
        return v / np.linalg.norm(v)

    q1, q2 = unit(X[a] - X[c]), unit(X[b] - X[c])
    q3 = unit(X[f.normal_rail[1]] - X[c])
    through = math.degrees(math.acos(float(np.clip(np.dot(-q1, q2), -1.0, 1.0))))
    axis = unit(X[b] - X[a])
    normal = math.degrees(math.asin(abs(float(np.clip(np.dot(q3, axis), -1.0, 1.0)))))
    return through, normal


def _run(grid, **cfg_kw):
    energies: list = []
    cfg = dict(tmop_sweeps=300, tmop_chunk=50, tmop_metric="shape", device="cpu")
    cfg.update(cfg_kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        run_pipeline(
            grid,
            config=PipelineConfig(**cfg),
            energy_history=energies,
            verbose=False,
        )
    return energies


def test_declared_fan_frame_straightens_the_five_fan():
    grid = _five_fan(framed=True)
    energies = _run(grid)

    assert np.all(np.isfinite(energies))
    assert energies[-1] < energies[0]
    es = build_sweep_context(grid, IdentityTarget(2)).energy_stencil
    assert grid_min_det(grid.global_nodes, es) > 0.0

    through, normal = _deviations(grid)
    assert through < 5.0, through
    assert normal < 8.0, normal


def test_undeclared_control_keeps_the_symmetric_fan():
    grid = _five_fan(framed=False)
    _run(grid)

    # measure with a framed twin's rail indices on the control's coordinates
    framed = _five_fan(framed=True)
    framed.global_nodes[:] = grid.global_nodes
    through, _normal = _deviations(framed)
    assert through > 30.0, through


def test_fas_matches_the_jacobi_fan_angles():
    """res=8 gives every block a 7-node (odd) interior, so real V-cycles run;
    the directional term rides the fine level and its safeguard line search."""
    grid = _five_fan(framed=True, res=8)
    energies = _run(grid, tmop_smoother="fas", tmop_sweeps=60, tmop_chunk=20)
    assert np.all(np.isfinite(energies))

    through, normal = _deviations(grid)
    assert through < 5.0, through
    assert normal < 8.0, normal


def test_control_point_matches_the_jacobi_fan_angles():
    """The control solve's accept energies and reduced GN gradient both
    compose the directional term, so the framed fan straightens through the
    control net too."""
    grid = _five_fan(framed=True)
    _run(grid, tmop_smoother="control_point")

    es = build_sweep_context(grid, IdentityTarget(2)).energy_stencil
    assert grid_min_det(grid.global_nodes, es) > 0.0
    through, normal = _deviations(grid)
    assert through < 5.0, through
    assert normal < 8.0, normal
