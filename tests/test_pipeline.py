"""End-to-end pipeline tests (M7)."""

import numpy as np
import pytest

from egg.geometry.analytic2d import Circle, LineSegment
from egg.pipeline import generate, generate_steps, drain, PipelineConfig
from egg.smoothing.targets import build_boundary_layer_target
from egg.topology.builder import TopologyBuilder


def _has_cpp() -> bool:
    try:
        from egg._cpp import cpp_core  # noqa: F401
        return True
    except ImportError:
        return False


# The pipeline runs on the C++ backend (block-Jacobi sweep / cpp_untangle), so the
# whole module skips when the extension isn't built.
pytestmark = pytest.mark.skipif(
    not _has_cpp(), reason="egg._cpp.cpp_core not built")

# ---------------------------------------------------------------------------
# Inline topology builders (self-contained; no dependency on examples/).
# ---------------------------------------------------------------------------

_INNER_PROPER = [
    ("isw", (1.3, 1.3)),
    ("ise", (2.7, 1.3)),
    ("ine", (2.7, 2.7)),
    ("inw", (1.3, 2.7)),
]
_INNER_ROUGH = [
    ("isw", (2.5, 2.4)),
    ("ise", (1.6, 2.5)),
    ("ine", (1.5, 1.6)),
    ("inw", (2.4, 1.5)),
]


def _build_circle_in_rectangle(rough: bool = False, R: int = 1):
    """Single-circle O-grid topology (Phase-4). ``rough`` → folded TFI start."""
    circle = Circle(center=(2.0, 2.0), radius=0.8)
    bottom = LineSegment(start=(0.0, 0.0), end=(4.0, 0.0))
    right = LineSegment(start=(4.0, 0.0), end=(4.0, 4.0))
    top = LineSegment(start=(4.0, 4.0), end=(0.0, 4.0))
    left = LineSegment(start=(0.0, 0.0), end=(0.0, 4.0))

    b = TopologyBuilder(d=2)
    for n, p in [("sw", (0, 0)), ("se", (4, 0)), ("ne", (4, 4)), ("nw", (0, 4))]:
        b.add_corner(n, p, fixed=True)
    for n, p in [("msw", (1, 1)), ("mse", (3, 1)), ("mne", (3, 3)), ("mnw", (1, 3))]:
        b.add_corner(n, p, fixed=False)
    for n, p in _INNER_ROUGH if rough else _INNER_PROPER:
        b.add_corner(n, p, fixed=False)
    for n, p in [
        ("bsw", (1, 0)), ("bse", (3, 0)),
        ("rse", (4, 1)), ("rne", (4, 3)),
        ("tne", (3, 4)), ("tnw", (1, 4)),
        ("lnw", (0, 3)), ("lsw", (0, 1)),
    ]:
        b.add_corner(n, p, fixed=False)

    for nm, sw, nw, se, ne in [
        ("o_s", "msw", "isw", "mse", "ise"),
        ("o_e", "mse", "ise", "mne", "ine"),
        ("o_n", "mne", "ine", "mnw", "inw"),
        ("o_w", "mnw", "inw", "msw", "isw"),
    ]:
        b.add_block(nm, (sw, nw, se, ne), (10 * R, 4 * R))
    for nm, sw, nw, se, ne in [
        ("e_s", "bsw", "msw", "bse", "mse"),
        ("e_e", "rse", "mse", "rne", "mne"),
        ("e_n", "tne", "mne", "tnw", "mnw"),
        ("e_w", "lnw", "mnw", "lsw", "msw"),
    ]:
        b.add_block(nm, (sw, nw, se, ne), (10 * R, 5 * R))
    for nm, sw, nw, se, ne in [
        ("c_sw", "sw", "lsw", "bsw", "msw"),
        ("c_se", "se", "bse", "rse", "mse"),
        ("c_ne", "ne", "rne", "tne", "mne"),
        ("c_nw", "nw", "tnw", "lnw", "mnw"),
    ]:
        b.add_block(nm, (sw, nw, se, ne), (5 * R, 5 * R))

    for a, b_ in [("o_s", "o_e"), ("o_e", "o_n"), ("o_n", "o_w"), ("o_w", "o_s")]:
        b.connect(a, 0, 1, b_, 0, 0)
    for e, o in [("e_s", "o_s"), ("e_e", "o_e"), ("e_n", "o_n"), ("e_w", "o_w")]:
        b.connect(e, 1, 1, o, 1, 0)
    for cb, ca, cs, eb, ea, es in [
        ("c_sw", 0, 1, "e_s", 0, 0),
        ("c_sw", 1, 1, "e_w", 0, 1),
        ("c_se", 0, 1, "e_e", 0, 0),
        ("c_se", 1, 1, "e_s", 0, 1),
        ("c_ne", 0, 1, "e_n", 0, 0),
        ("c_ne", 1, 1, "e_e", 0, 1),
        ("c_nw", 0, 1, "e_w", 0, 0),
        ("c_nw", 1, 1, "e_n", 0, 1),
    ]:
        b.connect(cb, ca, cs, eb, ea, es)

    for blk in ("o_s", "o_e", "o_n", "o_w"):
        b.associate(blk, 1, 1, circle)
    for blk, ent in [("e_s", bottom), ("e_e", right), ("e_n", top), ("e_w", left)]:
        b.associate(blk, 1, 0, ent)
    for blk, a0, a1 in [
        ("c_sw", left, bottom),
        ("c_se", bottom, right),
        ("c_ne", right, top),
        ("c_nw", top, left),
    ]:
        b.associate(blk, 0, 0, a0)
        b.associate(blk, 1, 0, a1)

    topology = b.build()
    entities = {
        "circle": circle,
        "bottom": bottom,
        "right": right,
        "top": top,
        "left": left,
    }
    return topology, entities


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _max_boundary_dist(grid):
    mx = 0.0
    for gidx, ent in grid.dof_constraints.items():
        p = grid.global_nodes[gidx]
        q = np.asarray(ent.project(p))
        mx = max(mx, float(np.linalg.norm(p - q)))
    return mx


def test_generate_does_not_raise():
    topo, _ = _build_circle_in_rectangle(rough=False)
    grid = generate(topo, PipelineConfig(tmop_sweeps=10))
    assert hasattr(grid, "pipeline_report")


def test_circle_in_rectangle_valid_and_conformant():
    topo, _ = _build_circle_in_rectangle(rough=False)
    grid = generate(topo, PipelineConfig(tmop_sweeps=40))
    rep = grid.pipeline_report
    assert rep.final_min_det > 0
    assert _max_boundary_dist(grid) < 1e-8


def test_idempotent_on_rerun():
    topo, _ = _build_circle_in_rectangle(rough=False)
    grid = generate(topo, PipelineConfig(tmop_sweeps=40))
    before = grid.global_nodes.copy()
    # Re-run TMOP on the converged grid: nodes should barely move.
    grid2 = generate(topo, PipelineConfig(tmop_sweeps=40))
    assert np.linalg.norm(grid2.global_nodes - before) < 1e-3


def test_folded_start_recovers():
    topo, _ = _build_circle_in_rectangle(rough=True)
    grid = generate(topo, PipelineConfig(
        tmop_sweeps=40,
        untangle_sweeps_per_delta=40,
        untangle_max_outer=100,
    ))
    rep = grid.pipeline_report
    assert rep.untangled
    assert rep.untangle_converged
    assert rep.final_min_det > 0


def test_folded_with_boundary_layer():
    topo, ents = _build_circle_in_rectangle(rough=True)
    topo.boundary_layer_specs = {
        id(ents["circle"]): dict(first_height=0.03, growth=1.3, n_layers=4,
                                 max_height=None, tangential_spacing=None)
    }
    tgt = build_boundary_layer_target(topo, interior_spacing=0.2)
    grid = generate(topo, PipelineConfig(
        target_fn=tgt, tmop_sweeps=60, tmop_chunk=20,
        untangle_sweeps_per_delta=40,
        untangle_max_outer=100,
    ))
    assert grid.pipeline_report.final_min_det > 0


# ---------------------------------------------------------------------------
# generate_steps / drain (step-wise pipeline; direct vs stepped untangle)
# ---------------------------------------------------------------------------

def _phases(events):
    return [p for p, _ in events]


def _tmop(events):
    return [info for p, info in events if p == "tmop"]


def _untangle(events):
    return [info for p, info in events if p == "untangle"]


def test_generate_steps_phase_sequence_good_topology():
    """Valid start: init → tmop* → final, no untangle phase."""
    topo, _ = _build_circle_in_rectangle(rough=False)
    grid = topo.initialize_grid()
    events = list(generate_steps(grid, tmop_sweeps=20, tmop_chunk=10))
    phases = _phases(events)
    assert phases[0] == "init"
    assert phases[-1] == "final"
    assert "tmop" in phases
    assert "untangle" not in phases          # already valid → untangle skipped
    assert events[-1][1]["min_det"] > 0


def test_generate_steps_tmop_chunking():
    """TMOP yields one event per chunk; chunk == sweeps ⇒ a single event."""
    topo, _ = _build_circle_in_rectangle(rough=False)
    grid = topo.initialize_grid()
    one = _tmop(list(generate_steps(grid, tmop_sweeps=20, tmop_chunk=20)))
    assert len(one) == 1
    assert one[0]["sweeps"] == 20

    topo2, _ = _build_circle_in_rectangle(rough=False)
    grid2 = topo2.initialize_grid()
    many = _tmop(list(generate_steps(grid2, tmop_sweeps=20, tmop_chunk=5)))
    assert len(many) == 4
    assert [m["sweeps"] for m in many] == [5, 10, 15, 20]


def test_generate_steps_untangle_direct_single_event():
    """untangle_direct=True runs the whole continuation in one call → 1 event."""
    topo, _ = _build_circle_in_rectangle(rough=True)
    grid = topo.initialize_grid()
    events = list(generate_steps(
        grid, tmop_sweeps=20, tmop_chunk=10,
        sweeps_per_delta=40, max_outer=100, untangle_direct=True))
    unt = _untangle(events)
    assert len(unt) == 1
    assert unt[0].get("direct") is True
    assert unt[0]["converged"]
    assert events[-1][1]["min_det"] > 0


def test_generate_steps_untangle_stepped_multiple_events():
    """untangle_direct=False steps per δ → several events carrying the schedule."""
    topo, _ = _build_circle_in_rectangle(rough=True)
    grid = topo.initialize_grid()
    events = list(generate_steps(
        grid, tmop_sweeps=20, tmop_chunk=10,
        sweeps_per_delta=40, max_outer=100, untangle_direct=False))
    unt = _untangle(events)
    assert len(unt) >= 1
    assert all("delta" in i and "outer_iter" in i for i in unt)
    assert [i["outer_iter"] for i in unt] == list(range(1, len(unt) + 1))
    assert unt[-1]["converged"]
    assert events[-1][1]["min_det"] > 0


def test_generate_steps_direct_and_stepped_both_untangle_to_valid():
    """Both untangle modes recover a folded start to a valid mesh."""
    results = {}
    for direct in (True, False):
        topo, _ = _build_circle_in_rectangle(rough=True)
        grid = topo.initialize_grid()
        events = list(generate_steps(
            grid, tmop_sweeps=40, tmop_chunk=10,
            sweeps_per_delta=40, max_outer=100, untangle_direct=direct))
        assert _untangle(events), "expected an untangle phase on a folded start"
        results[direct] = events[-1][1]["min_det"]
    assert results[True] > 0
    assert results[False] > 0


def test_drain_collects_history():
    topo, _ = _build_circle_in_rectangle(rough=False)
    grid = topo.initialize_grid()
    md_hist, e_hist = [], []
    drain(generate_steps(grid, tmop_sweeps=20, tmop_chunk=5),
          mindet_history=md_hist, energy_history=e_hist, verbose=False)
    assert len(md_hist) >= 1
    assert len(e_hist) >= 1
    assert md_hist[-1] > 0


def test_generate_drains_steps_consistently():
    """generate() (which drains generate_steps with untangle_direct) agrees with
    a direct generate_steps run on the same folded topology."""
    topo, _ = _build_circle_in_rectangle(rough=True)
    grid = generate(topo, PipelineConfig(
        tmop_sweeps=40, untangle_sweeps_per_delta=40, untangle_max_outer=100))
    assert grid.pipeline_report.untangled
    assert grid.pipeline_report.untangle_converged
    assert grid.pipeline_report.final_min_det > 0

    topo2, _ = _build_circle_in_rectangle(rough=True)
    g2 = topo2.initialize_grid()
    events = list(generate_steps(
        g2, tmop_sweeps=40, tmop_chunk=10,
        sweeps_per_delta=40, max_outer=100, untangle_direct=True))
    assert events[-1][1]["min_det"] > 0
