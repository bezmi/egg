# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""The render worker: protocol roundtrip, error paths, coalescing, timeout.

Editor renders exec the script in a separate warm process; these tests
drive the parent-side ``RenderWorker`` handle end to end (real child
subprocess, real pickled SceneResults back).
"""

import pytest

from egg.webui.render_worker import RenderWorker

SCRIPT = """
from egg.geometry import Line, Vector3

p0 = Vector3(0.0, 0.0)
p1 = Vector3(1.0, 1.0)
ln = Line(p0, p1)
"""


@pytest.fixture(scope="module")
def worker():
    w = RenderWorker()
    yield w
    w.close()


def test_render_roundtrip(worker):
    r = worker.submit(SCRIPT).result(timeout=120)
    assert r.error is None
    assert r.svg.startswith("<svg")
    assert r.stats["curves"] == 1
    assert r.stats["points"] == 2


def test_script_error_comes_back_inside_the_result(worker):
    r = worker.submit("1/0").result(timeout=60)
    assert r.error is not None and "ZeroDivisionError" in r.error
    assert "<svg" in r.svg  # the view still gets a drawable (empty) scene


def test_burst_coalesces_to_newest_code(worker):
    # occupy the child so the burst queues up behind one in-flight render
    hold = worker.submit("import time; time.sleep(1.0)")
    futs = [worker.submit(f"x = {i}") for i in (1, 2, 3)]
    assert hold.result(timeout=60).error is None
    results = [f.result(timeout=60) for f in futs]
    # stale keystrokes resolve with the newest code's (single) result
    assert results[0] is results[1] is results[2]


def test_cancelled_future_is_skipped(worker):
    hold = worker.submit("import time; time.sleep(1.0)")
    gone = worker.submit("x = 1")
    assert gone.cancel()
    kept = worker.submit("y = 2")
    assert hold.result(timeout=60).error is None
    assert kept.result(timeout=60).error is None
    assert gone.cancelled()


LIBRARY_SCRIPT = """
from egg.geometry import Line, Vector3
from egg.topology.builder import TopologyBuilder


def build_channel():
    b = TopologyBuilder()
    for name, x, y in (("sw", 0, 0), ("se", 1, 0), ("ne", 1, 1), ("nw", 0, 1)):
        b.add_corner(name, (x, y))
    b.add_block(sw="sw", se="se", ne="ne", nw="nw", res=(5, 5))
    return b.build(), []
"""

GRID_SCRIPT = (
    LIBRARY_SCRIPT
    + """

topo, _ents = build_channel()
grid = topo.initialize_grid()
"""
)


def test_suggest_probe_runs_in_the_worker(worker):
    suggestion = worker.suggest(LIBRARY_SCRIPT).result(timeout=60)
    assert suggestion is not None and "egg_webui.run(" in suggestion
    # a script that draws something gets no suggestion
    assert worker.suggest(SCRIPT).result(timeout=60) is None


def test_su2_export_runs_in_the_worker(worker):
    text, why = worker.su2(GRID_SCRIPT).result(timeout=120)
    assert why == "" and text is not None
    assert "NDIME" in text and "NPOIN" in text
    # failure comes back as a reason, not an exception
    text, why = worker.su2("x = 1").result(timeout=60)
    assert text is None and "no grid" in why


def test_validate_blocking_runs_in_the_worker(worker):
    code = (
        "from egg.topology import ExplicitTopology, editable\n"
        "topo = ExplicitTopology(connectivity=editable({'nodes': {}, 'edges': []}))\n"
    )
    good = {
        "nodes": {
            "a": {"xy": [0, 0]},
            "b": {"xy": [1, 0]},
            "c": {"xy": [1, 1]},
            "d": {"xy": [0, 1]},
        },
        "edges": [
            {"a": "a", "b": "b"},
            {"a": "b", "b": "c"},
            {"a": "c", "b": "d"},
            {"a": "d", "b": "a"},
        ],
        "res": 3,
    }
    green = worker.validate(code, good).result(timeout=120)
    assert green["diagnostics"] == []
    assert green["edge_res"]  # loop-propagated per-edge resolutions ride along
    bad = {
        "nodes": {"a": {"xy": [0, 0]}, "b": {"xy": [1, 0]}, "c": {"xy": [0.5, 1]}},
        "edges": [{"a": "a", "b": "b"}, {"a": "b", "b": "c"}, {"a": "c", "b": "a"}],
    }
    red = worker.validate(code, bad).result(timeout=60)["diagnostics"]
    assert red and red[0]["kind"] in ("non_quad_face", "no_blocks")


def test_stuck_script_is_killed_and_the_worker_recovers(worker):
    # warm first (module fixture already is), then a short-timeout job
    r = worker.submit("while True:\n    pass", timeout=3.0).result(timeout=60)
    assert r.error is not None and "timed out" in r.error
    # the next render respawns the child and works
    r2 = worker.submit(SCRIPT).result(timeout=120)
    assert r2.error is None
    assert r2.stats["curves"] == 1
