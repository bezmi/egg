# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""``egg.webui_print``: the UI print bridge.

A ``print`` that reaches the web UI when one drives the script and is a
no-op headless. Covers the core sink contract, its formatting parity with
the builtin ``print``, the ``egg_webui.print`` alias, and the render
worker folding it into a render's captured stdout end to end.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "webui"))

import egg._webui_print as wp  # noqa: E402  (core bridge, sink lives here)
from egg import webui_print  # noqa: E402  (the re-exported public name)


@pytest.fixture(autouse=True)
def _clear_sink():
    """The sink is process-global; keep tests from leaking it into each other."""
    wp._set_sink(None)
    yield
    wp._set_sink(None)


def test_noop_without_a_sink(capsys):
    # headless (CLI, tests): silent, and it does not fall back to stdout/stderr
    webui_print("nothing here", 1, 2, sep="-")
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_formats_like_builtin_print():
    seen = []
    wp._set_sink(seen.append)
    webui_print("a", "b", 3, sep="-", end="!\n")
    webui_print("x")
    webui_print()  # a bare newline, like print()
    assert seen == ["a-b-3!\n", "x\n", "\n"]


def test_ignores_file_and_flush_kwargs():
    seen = []
    wp._set_sink(seen.append)
    # a script may pass file=/flush= out of habit; the destination is the UI
    webui_print("ignored stream", file=sys.stderr, flush=True)
    assert seen == ["ignored stream\n"]


def test_egg_webui_print_alias_shares_the_sink():
    import egg_webui

    seen = []
    wp._set_sink(seen.append)
    egg_webui.print("via", "alias")
    assert seen == ["via alias\n"]


def test_render_worker_folds_webui_print_into_stdout():
    """End to end: egg_webui.print during a render lands in the render's
    captured stdout (the view's stdout panel), the same text a run streams."""
    from render_worker import RenderWorker

    w = RenderWorker()
    try:
        code = (
            "from egg.geometry import Line, Vector3\n"
            "import egg_webui\n"
            "egg_webui.print('render log line', 42)\n"
            "ln = Line(Vector3(0.0, 0.0), Vector3(1.0, 1.0))\n"
        )
        r = w.submit(code).result(timeout=120)
    finally:
        w.close()
    assert r.error is None
    assert "render log line 42" in r.stdout
