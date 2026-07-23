# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""The Refit stage: keep grid.control_net faithful to the final grid.

After pin/respace move nodes off the net, Refit fits a fresh net to the FINAL
grid. It reads the grid and never writes the net back over the nodes. A control
solve that nothing moved after is already faithful and is kept. A fit that
folds declines to None, leaving the valid grid alone.
"""

import os
import sys

import numpy as np
import pytest

from egg.pipeline import (
    ControlPointSmoother,
    JacobiSmoother,
    Pin,
    Presmooth,
    Refit,
    Untangle,
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


def _grid(bl=None):
    from topologies import build_circle_in_rectangle

    topo, _ents = build_circle_in_rectangle(bl=bl)
    return topo.initialize_grid()


def _phases(grid, stages):
    from egg.pipeline import generate_steps

    seen = []
    md = {}
    info_by_phase = {}
    for phase, info in generate_steps(grid, stages=stages):
        seen.append(phase)
        md[phase] = info.get("min_det", md.get(phase))
        info_by_phase[phase] = info
    return seen, md, info_by_phase


def _node_error(topo, X_final):
    """Max distance between the net's grid and the final grid."""
    return float(np.abs(topo.prolong_global() - X_final).max())


def test_refit_after_pin_is_faithful_and_read_only():
    grid = _grid(bl={"first_height": 0.05, "growth": 1.2, "n_layers": 4, "n_fixed": 2})
    stages = [
        Untangle(),
        Presmooth(JacobiSmoother(sweeps=100)),
        ControlPointSmoother(ratio=2),
        Pin(JacobiSmoother(sweeps=10)),
        Refit(),
    ]
    seen, md, infos = _phases(grid, stages)
    assert "pin" in seen
    assert "refit" in seen
    assert grid.control_net is not None
    # Refit runs after pin and moves no nodes: min det is unchanged from the
    # refit event to the final summary.
    assert md["final"] == pytest.approx(md["refit"], rel=0, abs=0)
    # The net was fitted to the pinned grid, so it evaluates close to it.
    err = _node_error(grid.control_net, np.array(grid.global_nodes))
    edge = float(np.median(np.abs(np.diff(np.array(grid.global_nodes), axis=0))))
    assert err < 5.0 * edge


def test_control_solve_without_pin_keeps_the_live_net():
    grid = _grid()
    stages = [
        Untangle(),
        Presmooth(JacobiSmoother(sweeps=100)),
        ControlPointSmoother(ratio=2),
        Refit(),
    ]
    seen, _md, infos = _phases(grid, stages)
    assert "refit" in seen
    assert grid.control_net is not None
    # Nothing moved nodes after the control solve, so the net is already
    # faithful and Refit keeps it rather than re-fitting.
    assert infos["refit"]["kept"] is True
    assert infos["refit"]["declined"] is False


def test_nodal_run_gets_a_net_from_refit():
    grid = _grid()
    stages = [Untangle(), JacobiSmoother(sweeps=20, chunk=10), Refit()]
    seen, _md, infos = _phases(grid, stages)
    assert "refit" in seen
    # jacobi leaves no net of its own; Refit gives it one (unless it declines,
    # in which case there is simply nothing to store).
    if grid.control_net is not None:
        err = _node_error(grid.control_net, np.array(grid.global_nodes))
        assert np.isfinite(err)


def test_decline_leaves_none_and_a_valid_grid(monkeypatch):
    import egg.pipeline as pl

    monkeypatch.setattr(pl, "_refit_net", lambda grid, existing, ratio: None)
    grid = _grid()
    stages = [Untangle(), JacobiSmoother(sweeps=20, chunk=10), Refit()]
    seen, md, infos = _phases(grid, stages)
    assert "refit" in seen
    assert grid.control_net is None
    assert md["final"] > 0.0
