# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Resample / Save stages: persist a net, reload it, regrid it.

``Save`` writes a net; ``Resample`` restores it (same resolution) or re-tabulates
it (a new resolution) and marks the run warm so the smoother polishes; the
exact residual layer reproduces the saved grid; ``Resample(cluster=True)``
re-clusters the loaded net algebraically in the same step.
"""

import os
import sys

import numpy as np
import pytest

from egg.pipeline import (
    ControlPointSmoother,
    JacobiSmoother,
    Presmooth,
    Refit,
    Resample,
    Save,
    Untangle,
    generate_steps,
    validate,
)

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "examples", "2D", "circles")
)


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


def _grid(R=1, bl=None):
    from topologies import build_circle_in_rectangle

    topo, _ents = build_circle_in_rectangle(R=R, bl=bl)
    return topo.initialize_grid()


def _min_det(grid):
    from egg.smoothing.solver import build_sweep_context
    from egg.smoothing.targets import IdentityTarget
    from egg.smoothing.untangle import grid_min_det

    es = build_sweep_context(grid, IdentityTarget(2)).energy_stencil
    return grid_min_det(grid.global_nodes, es)


def _phases(grid, stages):
    return [(p, info) for p, info in generate_steps(grid, stages=stages)]


_CONTROL = [
    Untangle(),
    Presmooth(JacobiSmoother(sweeps=100)),
    ControlPointSmoother(ratio=2),
]


def test_save_then_load_restores(tmp_path):
    path = str(tmp_path / "net.npz")
    _phases(_grid(), _CONTROL + [Save(path)])
    assert os.path.exists(path)

    grid = _grid()
    phases = dict(_phases(grid, [Resample(path), ControlPointSmoother(ratio=2)]))
    assert "resample" in phases
    assert phases["resample"]["exact_restore"] is True
    assert grid.control_net is not None
    assert _min_det(grid) > 0.0


def test_load_regrids_to_a_finer_resolution(tmp_path):
    path = str(tmp_path / "net.npz")
    _phases(_grid(R=1), _CONTROL + [Save(path)])

    fine = _grid(R=2)
    phases = dict(_phases(fine, [Resample(path), ControlPointSmoother(ratio=2)]))
    # A finer grid re-tabulates the net rather than restoring it exactly.
    assert phases["resample"]["exact_restore"] is False
    assert fine.control_net is not None
    assert _min_det(fine) > 0.0


def test_save_exact_residual_reproduces_the_saved_grid(tmp_path):
    path = str(tmp_path / "net.npz")
    # A nodal run + refit leaves a net that only APPROXIMATES the grid, so the
    # residual is what makes the reload exact.
    grid = _grid()
    _phases(
        grid,
        [
            Untangle(),
            JacobiSmoother(sweeps=40, chunk=20),
            Refit(),
            Save(path, exact=True),
        ],
    )
    if grid.control_net is None:
        pytest.skip("refit declined; nothing to reproduce")
    saved = np.array(grid.global_nodes)

    grid2 = _grid()
    phases = dict(_phases(grid2, [Resample(path)]))
    assert phases["resample"]["residual"] is True
    # Resample reproduces the saved (post-processed) grid, not just the net.
    np.testing.assert_allclose(grid2.global_nodes, saved, atol=1e-9)


def test_resample_with_cluster_reclusters_the_loaded_net(tmp_path):
    path = str(tmp_path / "net.npz")
    bl0 = dict(first_height=0.05, growth=1.3, n_fixed=0)
    _phases(_grid(bl=bl0), _CONTROL + [Save(path)])

    # Rebuild with a TIGHTER clustering; Resample(cluster=True) loads the net AND
    # re-evaluates it at the new boundary-layer spec, algebraically, in one step.
    grid = _grid(bl=dict(first_height=0.02, growth=1.3, n_fixed=0))
    phases = dict(_phases(grid, [Resample(path, cluster=True)]))
    assert phases["resample"]["cluster"] is True
    assert phases["resample"]["min_det"] > 0.0
    assert grid.control_net is not None


def test_warm_nodal_polish_stops_early(tmp_path):
    path = str(tmp_path / "net.npz")
    _phases(_grid(R=1), _CONTROL + [Save(path)])

    fine = _grid(R=2)
    stages = [Resample(path), JacobiSmoother(sweeps=200, chunk=10)]
    tmop = [info["sweeps"] for p, info in _phases(fine, stages) if p == "tmop"]
    # Seeded from the loaded net, the nodal polish converges before the
    # 200-sweep budget and stops early.
    assert tmop and tmop[-1] < 200


def test_load_errors_on_a_topology_mismatch(tmp_path):
    path = str(tmp_path / "net.npz")
    _phases(_grid(), _CONTROL + [Save(path)])

    from egg.topology.builder import TopologyBuilder

    b = TopologyBuilder(d=2)
    for name, pos in [
        ("A", (0.0, 0.0)),
        ("D", (0.0, 2.0)),
        ("B", (4.0, 0.0)),
        ("C", (4.0, 2.0)),
    ]:
        b.add_corner(name, pos, fixed=True)
    b.add_block("S", ("A", "D", "B", "C"), (9, 9))
    other = b.build().initialize_grid()
    with pytest.raises(ValueError, match="does not fit"):
        _phases(other, [Resample(path)])


def test_load_missing_file_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        _phases(_grid(), [Resample(str(tmp_path / "nope.npz"))])


def test_validate_save_requires_a_net(tmp_path):
    path = str(tmp_path / "net.npz")
    # No Refit / control smoother upstream, so there is no net to save; the
    # capability check rejects the composition before it runs.
    with pytest.raises(ValueError, match="needs"):
        validate([Untangle(), JacobiSmoother(sweeps=20, chunk=10), Save(path)])


def test_validate_pathless_resample_requires_a_net():
    # Resample() with no path re-samples an in-memory net; without one upstream
    # (a Resample(path=...) or a control smoother) it has nothing to sample.
    with pytest.raises(ValueError, match="needs"):
        validate([Resample()])
    # A control smoother produces a net, so it can feed a path-less resample.
    validate([Untangle(), ControlPointSmoother(), Resample(cluster=True)])
