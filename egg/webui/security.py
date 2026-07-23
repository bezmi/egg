# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Treat the web UI server as privileged local IPC and lock it down.

The server runs code and reads/writes files for whoever can reach it, so the
client/server link is a trust boundary, not a public API. This module adds:

- a per-launch auth token: every request must present it (query param on the
  first load, then an HttpOnly SameSite cookie the browser replays). Without the
  token a request is refused. The launcher makes the token and hands it to the
  browser or the desktop window.
- Host allowlist: the ``Host`` header must match the address the server bound
  to. This stops DNS-rebinding, where a hostile page resolves its own name to
  127.0.0.1 and talks to the server as if same-origin.
- Origin allowlist: a cross-origin ``Origin`` (fetch / XHR / WebSocket) is
  refused, so another site in the same browser cannot drive the server.
- a restrictive Content-Security-Policy and related headers on every response,
  so the page can only load its own (vendored) assets and can only talk back to
  its own origin. No external fetch, no framing, no plugins.
- path canonicalisation for the file endpoints (resolve ~, symlinks and ``..``
  to one absolute path, reject NUL), so a path means exactly one file.

The token is read from ``EGG_WEBUI_TOKEN`` and the bind address from
``EGG_WEBUI_BIND`` (``host:port``); the launcher sets both. With no token set
(a bare import, the test suite) the middleware is inert, so nothing else has to
know about auth.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from urllib.parse import parse_qs

# One policy for every response. Assets are vendored and served from our own
# origin, so 'self' covers them. 'unsafe-inline' is still needed because the page
# inlines its script/style and a few onclick handlers; even so this blocks every
# external origin, framing, plugins, and base-tag hijacking. connect-src 'self'
# is the important one: it stops a script (ours or injected) from exfiltrating to
# another host. Tightening script-src to nonces is a later step (it needs the
# inline handlers refactored out).
_CSP_BASE = (
    "default-src 'self'",
    "script-src 'self' 'unsafe-inline'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    "font-src 'self' data:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "frame-ancestors 'none'",
    "form-action 'self'",
)


def content_security_policy(desktop: bool = False) -> str:
    """The CSP for a response. In the native desktop app only, script-src also
    allows 'unsafe-eval': pywebview builds its ``window.pywebview`` bridge with
    ``new Function()`` (see webview/js/api.js), which the policy would otherwise
    block, killing the window controls and drag. The desktop is a local,
    token-authed context, so this narrow relaxation stays there."""
    directives: list[str] = list(_CSP_BASE)
    if desktop:
        directives[1] = "script-src 'self' 'unsafe-inline' 'unsafe-eval'"
    return "; ".join(directives)


# Kept for the browser (non-desktop) case and the tests.
CONTENT_SECURITY_POLICY = content_security_policy(False)


def _security_headers(desktop: bool) -> tuple[tuple[bytes, bytes], ...]:
    return (
        (b"content-security-policy", content_security_policy(desktop).encode()),
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"no-referrer"),
    )

_LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]", "0.0.0.0")


def configured_token() -> str | None:
    """The launch auth token, or ``None`` when auth is off (no env / bare import)."""
    return os.environ.get("EGG_WEBUI_TOKEN") or None


def prepare_auth(host: str, port: int, *, disabled: bool) -> str | None:
    """Set up the environment the server reads at import: record the bind
    address (for the Host/Origin allowlists) and settle the auth token.

    Returns the token the launcher must hand to the client, or ``None`` when
    auth is disabled. An existing ``EGG_WEBUI_TOKEN`` (the desktop launcher sets
    it before spawning the server child) is reused so parent and child agree.
    Called by the launchers before ``uvicorn.run`` / spawning the child, so the
    values are in ``os.environ`` when ``egg.webui.app`` imports.
    """
    os.environ["EGG_WEBUI_BIND"] = f"{host}:{port}"
    if disabled:
        os.environ.pop("EGG_WEBUI_TOKEN", None)
        return None
    token = os.environ.get("EGG_WEBUI_TOKEN")
    if not token:
        token = secrets.token_urlsafe(32)
        os.environ["EGG_WEBUI_TOKEN"] = token
    return token


def _bind() -> tuple[str, str]:
    """(host, port) the server bound to, from ``EGG_WEBUI_BIND`` (host:port)."""
    raw = os.environ.get("EGG_WEBUI_BIND", "127.0.0.1:5001")
    host, _, port = raw.rpartition(":")
    return (host or "127.0.0.1", port or "5001")


def allowed_hosts() -> set[str]:
    """Host header values that map to this server. A loopback bind also accepts
    the other loopback spellings a browser might use."""
    host, port = _bind()
    hosts = {f"{host}:{port}"}
    if host in _LOOPBACK:
        for h in ("127.0.0.1", "localhost", "[::1]"):
            hosts.add(f"{h}:{port}")
    return hosts


def allowed_origins() -> set[str]:
    """Origin header values allowed for cross-origin-sensitive requests."""
    return {f"http://{h}" for h in allowed_hosts()}


def canonical_path(raw: str | os.PathLike) -> Path:
    """One absolute path for ``raw``: expand ~, resolve symlinks and ``..``,
    reject an embedded NUL. So a client path names exactly one file and cannot
    smuggle traversal past a later check. It does not jail to a root: the file
    browser is meant to reach the user's whole tree, and the token is the real
    boundary."""
    s = os.fspath(raw)
    if "\x00" in s:
        raise ValueError("path contains NUL")
    p = Path(s).expanduser()
    try:
        return p.resolve()
    except OSError:
        return p.absolute()


class SecurityMiddleware:
    """Pure-ASGI gate for HTTP and WebSocket. Enforces the Host / Origin / token
    checks, then adds the security headers (and the auth cookie on first sight of
    a valid token) to the response. Inert when no token is configured."""

    def __init__(self, app) -> None:
        self.app = app
        self.token = configured_token()
        self.hosts = allowed_hosts()
        self.origins = allowed_origins()
        # A bind to every interface (dev "--host 0.0.0.0") can't know the Host
        # the client used, so skip the Host check there; Origin + token still run.
        self.relax_host = _bind()[0] == "0.0.0.0"
        # The desktop launcher sets EGG_DESKTOP so the CSP can allow the
        # pywebview bridge's use of new Function() (see content_security_policy).
        self.sec_headers = _security_headers(bool(os.environ.get("EGG_DESKTOP")))

    async def __call__(self, scope, receive, send):
        # No token configured means auth is off: a bare import / the test suite,
        # or the developer-only experimental opt-out. Then the whole layer is
        # inert (no Host/Origin/token checks, no injected headers), so nothing
        # else has to know about it.
        if self.token is None or scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        hdr = {k.decode("latin1").lower(): v.decode("latin1")
               for k, v in scope.get("headers", [])}

        host = hdr.get("host", "")
        if host and not self.relax_host and host not in self.hosts:
            return await self._reject(scope, receive, send, 421, "bad host")

        origin = hdr.get("origin")
        if origin and origin not in self.origins:
            return await self._reject(scope, receive, send, 403, "bad origin")

        presented, via = self._read_token(scope, hdr)
        if presented != self.token:
            return await self._reject(scope, receive, send, 403, "auth required")

        if scope["type"] == "http":
            set_cookie = via in ("query", "header")

            async def send_wrap(event):
                if event["type"] == "http.response.start":
                    event = dict(event)
                    headers = list(event.get("headers", []))
                    headers.extend(self.sec_headers)
                    if set_cookie:
                        cookie = (f"egg_token={self.token}; Path=/; HttpOnly; "
                                  "SameSite=Strict")
                        headers.append((b"set-cookie", cookie.encode()))
                    event["headers"] = headers
                await send(event)

            return await self.app(scope, receive, send_wrap)

        return await self.app(scope, receive, send)

    def _read_token(self, scope, hdr) -> tuple[str | None, str | None]:
        """Find the token on the request: query param, header, then cookie."""
        qs = scope.get("query_string", b"").decode("latin1")
        q = parse_qs(qs)
        if q.get("token"):
            return q["token"][0], "query"
        head = hdr.get("x-egg-token")
        if head:
            return head, "header"
        for part in hdr.get("cookie", "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == "egg_token":
                return value, "cookie"
        return None, None

    async def _reject(self, scope, receive, send, code: int, msg: str):
        if scope["type"] == "websocket":
            # Consume the connect, then refuse the handshake.
            try:
                await receive()
            except Exception:
                pass
            await send({"type": "websocket.close", "code": 1008})
            return
        body = msg.encode()
        await send({
            "type": "http.response.start",
            "status": code,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"content-length", str(len(body)).encode()),
                *self.sec_headers,
            ],
        })
        await send({"type": "http.response.body", "body": body})
