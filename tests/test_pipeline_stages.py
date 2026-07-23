# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Stage-composed pipeline (the new way) and the deprecated config shim.

The stage list is the canonical form; PipelineConfig translates to the same
stages and warns; validate() rejects an ill-ordered composition; and a run
driven by stages matches the same run driven by the old config.
"""

import os
import sys

import pytest

from egg.pipeline import (
    ControlPointSmoother,
    FasSmoother,
    JacobiSmoother,
    Pin,
    Presmooth,
    Refit,
    Resample,
    Respace,
    Save,
    Untangle,
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


# --------------------------------------------------------------------------- #
# validate() needs no grid or C++ backend.
# --------------------------------------------------------------------------- #


def test_validate_accepts_a_well_ordered_list():
    validate([Untangle(), JacobiSmoother(), Pin(JacobiSmoother(sweeps=10)), Respace()])


def test_validate_rejects_pin_before_a_smoother():
    with pytest.raises(ValueError, match="Pin"):
        validate([Untangle(), Pin(JacobiSmoother(sweeps=10))])


def test_validate_rejects_respace_before_a_smoother():
    with pytest.raises(ValueError, match="Respace"):
        validate([Respace()])


def test_validate_save_needs_a_net():
    # A nodal smoother produces no net, so a Save has nothing to serialise.
    with pytest.raises(ValueError, match="net"):
        validate([Untangle(), JacobiSmoother(), Save("net.npz")])
    # The control-point smoother produces a net, so the same Save is fine.
    validate([Untangle(), ControlPointSmoother(), Save("net.npz")])
    # A Refit stage after a nodal smoother also produces a net.
    validate([Untangle(), JacobiSmoother(), Refit(), Save("net.npz")])


def test_validate_pathless_resample_needs_a_net():
    # Resample() with no path re-samples an in-memory net; without one upstream
    # it has nothing to sample.
    with pytest.raises(ValueError, match="net"):
        validate([Resample()])
    # Resample(path=...) loads from disk (needs only a grid); a control smoother
    # then a path-less Resample(cluster=True) is also well ordered.
    validate([Resample("net.npz")])
    validate([Untangle(), ControlPointSmoother(), Resample(cluster=True)])


def test_fas_smoother_is_a_valid_smoother():
    validate([Untangle(), FasSmoother(), Pin(JacobiSmoother(sweeps=10))])


def test_validate_rejects_untangle_after_a_smoother():
    # A stage checks its own position against the others: untangle prepares a
    # valid grid for the smoother, so it may not come after one.
    with pytest.raises(ValueError, match="Untangle must come before"):
        validate([JacobiSmoother(), Untangle()])
    with pytest.raises(ValueError, match="Untangle must come before"):
        validate([ControlPointSmoother(), Untangle()])


def test_validate_accepts_untangle_first():
    validate([Untangle(), JacobiSmoother()])
    # Untangle is optional: a valid start needs none.
    validate([JacobiSmoother()])


def test_presmooth_only_before_the_control_smoother():
    # A pre-pass is for the control-point smoother's fit.
    validate([Untangle(), Presmooth(JacobiSmoother()), ControlPointSmoother()])
    # Before a nodal smoother it is a redundant second node smooth.
    with pytest.raises(ValueError, match="redundant"):
        validate([Untangle(), Presmooth(JacobiSmoother()), JacobiSmoother()])
    # With no smoother after it there is nothing to prepare for.
    with pytest.raises(ValueError, match="control-point"):
        validate([Untangle(), Presmooth(JacobiSmoother())])


def test_presmooth_and_pin_reject_a_control_smoother():
    # Both take a NODAL smoother; a control-point smoother is not one.
    with pytest.raises(TypeError):
        Presmooth(ControlPointSmoother())
    with pytest.raises(TypeError):
        Pin(ControlPointSmoother())


# --------------------------------------------------------------------------- #
# Running stages needs the C++ backend.
# --------------------------------------------------------------------------- #

cpp = pytest.mark.skipif(not _has_cpp(), reason="egg._cpp.cpp_core not built")


def _grid():
    from topologies import build_circle_in_rectangle

    topo, _ents = build_circle_in_rectangle()
    return topo.initialize_grid()


@cpp
def test_stage_list_runs_end_to_end():
    from egg.pipeline import generate_steps

    grid = _grid()
    phases = [
        p
        for p, _ in generate_steps(
            grid, stages=[Untangle(), JacobiSmoother(sweeps=20, chunk=10)]
        )
    ]
    assert phases[0] == "init"
    assert phases[-1] == "final"
    assert "tmop" in phases
    assert "untangle" not in phases  # valid start


@cpp
def test_stages_and_config_are_mutually_exclusive():
    from egg.pipeline import generate_steps

    grid = _grid()
    with pytest.raises(TypeError):
        list(generate_steps(grid, stages=[Untangle()], tmop_sweeps=10))


@cpp
def test_deprecated_config_path_warns():
    from egg.pipeline import generate_steps

    grid = _grid()
    with pytest.warns(DeprecationWarning):
        # The warning fires when the generator starts; the first event is
        # enough to trigger it.
        next(iter(generate_steps(grid, tmop_sweeps=10)))


@cpp
def test_stage_run_matches_the_deprecated_config_run():
    from egg.pipeline import generate_steps

    g1 = _grid()
    ev1 = list(generate_steps(g1, tmop_sweeps=20, tmop_chunk=10))  # config path
    g2 = _grid()
    ev2 = list(
        generate_steps(g2, stages=[Untangle(), JacobiSmoother(sweeps=20, chunk=10)])
    )
    assert [p for p, _ in ev1] == [p for p, _ in ev2]
    # Same grid, same stage params: final numbers agree up to reduction noise.
    assert ev1[-1][1]["min_det"] == pytest.approx(ev2[-1][1]["min_det"], rel=1e-3)
    assert ev1[-1][1]["energy"] == pytest.approx(ev2[-1][1]["energy"], rel=1e-3)
