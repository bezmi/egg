# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""webui .eggy Open/Save routes.

Open unpacks an archive and returns its script (nothing runs); Save writes the
editor script and packs its folder into a .eggy. The bare control-net routes
are gone.
"""

import io
import os
import zipfile

import pytest

pytest.importorskip("fasthtml")

from starlette.testclient import TestClient  # noqa: E402

import egg.webui.app as webui_app  # noqa: E402


def _eggy_bytes():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("script.py", 'print("hi")\n')
        z.writestr("net.npz", b"\x00")
        z.writestr("assets/shape.step", "STEP")
    return buf.getvalue()


def test_open_eggy_returns_the_unpacked_script():
    with TestClient(webui_app.app) as c:
        r = c.post(
            "/open/eggy",
            files={"file": ("case.eggy", _eggy_bytes(), "application/octet-stream")},
        )
    assert r.status_code == 200
    j = r.json()
    assert j["path"].endswith("script.py")
    assert j["code"] == 'print("hi")\n'
    # The script and its assets landed on disk beside it (so a later run finds
    # the net cache and assets by relative path).
    d = os.path.dirname(j["path"])
    assert os.path.exists(os.path.join(d, "net.npz"))
    assert os.path.exists(os.path.join(d, "assets", "shape.step"))


def test_open_rejects_a_non_archive():
    with TestClient(webui_app.app) as c:
        r = c.post(
            "/open/eggy",
            files={"file": ("x.eggy", b"not a zip", "application/octet-stream")},
        )
    assert r.status_code == 400


def test_save_eggy_packs_the_script_folder(tmp_path):
    script = str(tmp_path / "script.py")
    with TestClient(webui_app.app) as c:
        r = c.post("/save/eggy", data={"code": "print(1)\n", "path": script})
    assert r.status_code == 200
    assert zipfile.is_zipfile(io.BytesIO(r.content))
    # The editor content was written to disk before packing.
    with open(script) as f:
        assert f.read() == "print(1)\n"


def test_save_without_a_path_is_rejected():
    with TestClient(webui_app.app) as c:
        r = c.post("/save/eggy", data={"code": "x=1", "path": ""})
    assert r.status_code == 400


def test_net_import_route_is_gone_export_is_the_escape_hatch():
    paths = {getattr(rt, "path", "") for rt in webui_app.app.router.routes}
    # Bare net IMPORT is gone (a net is brought in by opening a .eggy case).
    assert "/import/net" not in paths
    # Net EXPORT stays as the escape hatch: save the last run's net when the
    # pipeline had no Save stage (see tests/webui/test_net_export_route.py).
    assert "/export/net" in paths
    assert "/open/eggy" in paths
    assert "/save/eggy" in paths
