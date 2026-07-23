# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""webui 'save net (npz)' route (/export/net).

The escape hatch for a run whose pipeline had no Save stage: export the last
run's control net straight from the resident grid, no re-run. 400 when there is
no run, or the run produced no net.
"""

import io
from types import SimpleNamespace

import pytest

pytest.importorskip("fasthtml")

from starlette.testclient import TestClient  # noqa: E402

import egg.webui.app as webui_app  # noqa: E402


def _rect_grid():
    """A clean single-block rectangle: its TFI grid fits a net without folding."""
    from egg.topology.builder import TopologyBuilder

    b = TopologyBuilder(d=2)
    for nm, pos in [
        ("A", (0.0, 0.0)),
        ("D", (0.0, 2.0)),
        ("B", (4.0, 0.0)),
        ("C", (4.0, 2.0)),
    ]:
        b.add_corner(nm, pos, fixed=True)
    b.add_block("S", ("A", "D", "B", "C"), (9, 9))
    return b.build().initialize_grid()


@pytest.fixture(autouse=True)
def _restore_last():
    saved = dict(webui_app._last)
    yield
    webui_app._last.clear()
    webui_app._last.update(saved)


def test_export_net_without_a_run_is_rejected():
    webui_app._last["harvest"] = None
    webui_app._last["code"] = None
    with TestClient(webui_app.app) as c:
        r = c.post("/export/net", data={"code": "x = 1", "path": ""})
    assert r.status_code == 400


def test_export_net_without_a_net_is_rejected():
    grid = _rect_grid()
    grid.control_net = None
    webui_app._last["harvest"] = SimpleNamespace(grid=grid)
    webui_app._last["code"] = "CODE"
    with TestClient(webui_app.app) as c:
        r = c.post("/export/net", data={"code": "CODE", "path": ""})
    assert r.status_code == 400
    assert "no control net" in r.text


def test_export_net_returns_a_loadable_npz():
    from egg.io import load_control_net
    from egg.smoothing.control_fit import fit_control_net

    grid = _rect_grid()
    grid.control_net = fit_control_net(grid, ratio=2, walls=False)
    webui_app._last["harvest"] = SimpleNamespace(grid=grid)
    webui_app._last["code"] = "CODE"
    with TestClient(webui_app.app) as c:
        r = c.post("/export/net", data={"code": "CODE", "path": ""})
    assert r.status_code == 200
    assert r.headers["content-disposition"] == 'attachment; filename="net.npz"'
    # The bytes are a real control-net .npz that loads back onto the same grid.
    topo = load_control_net(_rect_grid(), io.BytesIO(r.content))
    assert topo.q.shape == grid.control_net.q.shape
