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
from egg.smoothing.config_types import InterfaceC2, InterfaceOrtho

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


def test_validate_accepts_a_stamp_only_pin():
    # Pin() without a smoother stamps and freezes the layers, no re-smooth.
    validate([Untangle(), JacobiSmoother(), Pin()])
    validate([Untangle(), ControlPointSmoother(), Pin()])


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


# --------------------------------------------------------------------------- #
# Stamp-only pin.
# --------------------------------------------------------------------------- #

FIRST_H = 0.02
GROWTH = 1.2


def _clustered_strip():
    """One block over a clustered wall (uniform first height 0.02)."""
    from egg.geometry import LineSegment
    from egg.topology.builder import TopologyBuilder

    wall = (
        LineSegment((0.0, 0.0), (4.0, 0.0))
        .named("wall")
        .clustered(first_height=FIRST_H, growth=GROWTH, n_layers=4, n_fixed=4)
    )
    b = TopologyBuilder(d=2)
    for n, p in [("A", (0, 0)), ("B", (4, 0)), ("C", (0, 2)), ("D", (4, 2))]:
        b.add_corner(n, p, fixed=True)
    b.add_block("main", ("A", "C", "B", "D"), (12, 12))
    b.associate("main", 1, 0, wall)
    topo = b.build()
    topo.initialize_grid()
    return topo.grid


def _first_layer_heights(grid):
    import numpy as np

    dm = grid.block_dof_maps[0]
    X = np.asarray(grid.global_nodes)
    return np.linalg.norm(X[dm[:, 1]] - X[dm[:, 0]], axis=1)


@cpp
def test_stamp_only_pin_sets_exact_heights_without_resmoothing():
    import numpy as np

    from egg.pipeline import generate_steps

    grid = _clustered_strip()
    events = list(
        generate_steps(
            grid,
            stages=[
                Untangle(),
                JacobiSmoother(sweeps=100, chunk=50, metric="shape_size"),
                Pin(),
            ],
        )
    )
    phases = [p for p, _ in events]
    assert "pin" in phases
    # stamp-only: no smoothing events after the pin
    assert "tmop" not in phases[phases.index("pin") :]
    # the first n_fixed layer heights are stamped to the exact per-column
    # profile regardless of what the smoother left
    h0 = _first_layer_heights(grid)
    interior = slice(1, -1)  # corner columns meet the side boundaries
    assert np.allclose(h0[interior], FIRST_H, rtol=1e-4)
    assert events[-1][1]["min_det"] > 0.0


def _spy_sweep_contexts(monkeypatch):
    """Capture the kwargs of every build_sweep_context call during a run."""
    import egg.smoothing.solver as solver_mod

    calls = []
    orig = solver_mod.build_sweep_context

    def spy(grid, target, **kw):
        calls.append(kw)
        return orig(grid, target, **kw)

    monkeypatch.setattr(solver_mod, "build_sweep_context", spy)
    return calls


@cpp
def test_pin_applies_its_own_smoother_interface_terms(monkeypatch):
    """The pin re-smooth builds its context from ITS OWN smoother's composed
    terms — a term set on the Pin's smoother is applied even when the main
    smoother carried none."""
    from egg.pipeline import generate_steps

    calls = _spy_sweep_contexts(monkeypatch)
    grid = _clustered_strip()
    list(
        generate_steps(
            grid,
            stages=[
                Untangle(),
                JacobiSmoother(sweeps=40, chunk=40, metric="shape_size"),
                Pin(
                    JacobiSmoother(
                        sweeps=40,
                        chunk=40,
                        metric="shape_size",
                        interface_c2={"weight": 3.0},
                        interface_ortho={"weight": 2.0},
                    )
                ),
            ],
        )
    )
    # the pin context (the last one built) carries the Pin smoother's terms...
    assert calls[-1].get("interface_c2") == InterfaceC2(weight=3.0)
    assert calls[-1].get("interface_ortho") == InterfaceOrtho(weight=2.0)
    # ...and no earlier (main) context did
    assert all(c.get("interface_c2") is None for c in calls[:-1])


@cpp
def test_pin_bare_smoother_does_not_inherit_main_interface_terms(monkeypatch):
    """A term left unset on the Pin's smoother is off — no fall-back to what
    the main phase used."""
    from egg.pipeline import generate_steps

    calls = _spy_sweep_contexts(monkeypatch)
    grid = _clustered_strip()
    list(
        generate_steps(
            grid,
            stages=[
                Untangle(),
                JacobiSmoother(
                    sweeps=40,
                    chunk=40,
                    metric="shape_size",
                    interface_c2={"weight": 9.0},  # MAIN carries a c2 term
                ),
                Pin(JacobiSmoother(sweeps=40, chunk=40, metric="shape_size")),
            ],
        )
    )
    # the main context carried weight 9, but the pin context (last) carries none
    assert any(c.get("interface_c2") == InterfaceC2(weight=9.0) for c in calls[:-1])
    assert calls[-1].get("interface_c2") is None


def _sheared_clustered_strip():
    """A clustered wall under a sloped top, so the wall-normal columns are
    non-trivial and a kink at the pinned/free boundary can actually appear."""
    from egg.geometry import LineSegment
    from egg.topology.builder import TopologyBuilder

    wall = (
        LineSegment((0.0, 0.0), (4.0, 0.0))
        .named("wall")
        .clustered(first_height=FIRST_H, growth=GROWTH, n_layers=6, n_fixed=4)
    )
    b = TopologyBuilder(d=2)
    for n, p in [("A", (0, 0)), ("B", (4, 0)), ("C", (0.6, 2)), ("D", (4.6, 2))]:
        b.add_corner(n, p, fixed=True)
    b.add_block("main", ("A", "C", "B", "D"), (10, 14))
    b.associate("main", 1, 0, wall)
    topo = b.build()
    topo.initialize_grid()
    return topo.grid


def _boundary_kink(grid, n_fixed=4):
    import numpy as np

    nodes = np.asarray(grid.blocks[0].nodes)  # (n_tan, n_normal, 2)
    ks = []
    for i in range(nodes.shape[0]):
        col = nodes[i]
        a = col[n_fixed] - col[n_fixed - 1]
        c = col[n_fixed + 1] - col[n_fixed]
        a /= np.linalg.norm(a)
        c /= np.linalg.norm(c)
        ks.append(np.degrees(np.arccos(np.clip(np.dot(a, c), -1, 1))))
    return float(np.mean(ks))


@cpp
def test_frozen_boundary_c2_boost_reduces_the_pinned_free_kink():
    """End to end: the frozen-band-edge C2 boost (threaded Pin -> context ->
    C++) leaves the free grid continuing more smoothly out of the pinned band
    than a plain pin does."""
    from egg.pipeline import generate_steps

    def run(pin_c2):
        grid = _sheared_clustered_strip()
        list(
            generate_steps(
                grid,
                stages=[
                    Untangle(),
                    JacobiSmoother(sweeps=100, chunk=100, metric="shape_size"),
                    Pin(
                        JacobiSmoother(
                            sweeps=250,
                            chunk=100,
                            metric="shape_size",
                            interface_c2=pin_c2,
                        )
                    ),
                ],
            )
        )
        return _boundary_kink(grid)

    plain = run(None)
    boosted = run({"weight": 5.0, "iface_boost": 40.0})
    assert boosted < 0.75 * plain, (boosted, plain)
