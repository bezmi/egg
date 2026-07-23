# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Tests for the web UI security layer: token / Host / Origin enforcement, the
injected headers, and path canonicalisation. The middleware is driven directly
with crafted ASGI scopes, so nothing depends on the module-level app or on
import ordering."""

from __future__ import annotations

import asyncio

import pytest

from egg.webui.security import SecurityMiddleware, canonical_path

TOKEN = "secret-token-value"
BIND = "127.0.0.1:5001"


async def _ok_app(scope, receive, send):
    """A trivial downstream app that returns 200 with no headers."""
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def _run_http(mw, *, headers, query=b""):
    """Drive one HTTP request through ``mw`` and return (status, headers dict)."""
    events: list[dict] = []

    async def send(ev):
        events.append(ev)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": query,
        "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
    }
    asyncio.run(mw(scope, receive, send))
    start = next(e for e in events if e["type"] == "http.response.start")
    hdrs = {k.decode().lower(): v.decode() for k, v in start.get("headers", [])}
    return start["status"], hdrs


def _mw(monkeypatch, token=TOKEN, bind=BIND):
    monkeypatch.setenv("EGG_WEBUI_BIND", bind)
    if token is None:
        monkeypatch.delenv("EGG_WEBUI_TOKEN", raising=False)
    else:
        monkeypatch.setenv("EGG_WEBUI_TOKEN", token)
    return SecurityMiddleware(_ok_app)


def test_request_without_token_is_refused(monkeypatch):
    status, _ = _run_http(_mw(monkeypatch), headers={"host": BIND})
    assert status == 403


def test_token_in_query_passes_and_sets_cookie(monkeypatch):
    status, hdrs = _run_http(
        _mw(monkeypatch), headers={"host": BIND}, query=f"token={TOKEN}".encode()
    )
    assert status == 200
    assert "egg_token=" + TOKEN in hdrs.get("set-cookie", "")
    assert "httponly" in hdrs.get("set-cookie", "").lower()


def test_token_in_cookie_passes(monkeypatch):
    status, _ = _run_http(
        _mw(monkeypatch), headers={"host": BIND, "cookie": f"egg_token={TOKEN}"}
    )
    assert status == 200


def test_wrong_token_is_refused(monkeypatch):
    status, _ = _run_http(
        _mw(monkeypatch), headers={"host": BIND}, query=b"token=nope"
    )
    assert status == 403


def test_bad_host_is_refused(monkeypatch):
    status, _ = _run_http(
        _mw(monkeypatch), headers={"host": "evil.example"}, query=f"token={TOKEN}".encode()
    )
    assert status == 421


def test_cross_origin_is_refused(monkeypatch):
    status, _ = _run_http(
        _mw(monkeypatch),
        headers={"host": BIND, "origin": "http://evil.example", "cookie": f"egg_token={TOKEN}"},
    )
    assert status == 403


def test_security_headers_present_on_success(monkeypatch):
    _, hdrs = _run_http(
        _mw(monkeypatch), headers={"host": BIND}, query=f"token={TOKEN}".encode()
    )
    csp = hdrs.get("content-security-policy", "")
    assert "default-src 'self'" in csp
    assert "connect-src 'self'" in csp
    assert hdrs.get("x-frame-options") == "DENY"
    assert hdrs.get("x-content-type-options") == "nosniff"


def test_layer_is_inert_without_a_token(monkeypatch):
    # No token configured (tests / the developer opt-out): every check is off,
    # so even a foreign Host and no token pass straight through.
    mw = _mw(monkeypatch, token=None)
    status, hdrs = _run_http(mw, headers={"host": "testserver"})
    assert status == 200
    assert "content-security-policy" not in hdrs


def test_canonical_path_rejects_nul():
    with pytest.raises(ValueError):
        canonical_path("/tmp/evil\x00.py")


def test_canonical_path_resolves(tmp_path):
    (tmp_path / "real.py").write_text("x=1\n")
    # a ".." detour resolves to the same absolute file
    p = canonical_path(str(tmp_path / "sub" / ".." / "real.py"))
    assert p == (tmp_path / "real.py").resolve()
    assert p.is_absolute()
