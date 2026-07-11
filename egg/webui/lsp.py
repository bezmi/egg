# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""based-pyright language server bridge for the web UI editor.

Runs ``basedpyright-langserver --stdio`` as a subprocess and shuttles LSP
JSON-RPC frames between it and the browser editor (over the ``/lsp``
websocket in :mod:`egg.webui.app`). The browser drives the standard LSP
flow — ``initialize`` / ``didOpen`` / ``didChange`` / completion / hover —
and renders diagnostics; this module is a transparent pipe EXCEPT for the
server→client requests that need host knowledge, which it answers itself:

- ``workspace/configuration`` → the interpreter path (this venv, so
  ``import egg`` / numpy / … resolve for real completions) and analysis
  settings;
- capability (un)registration and work-done progress creation → acked.

If the language server is not installed the bridge is simply unavailable
and the editor keeps working without language features (graceful
degradation); nothing here reaches the network.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable

# A scratch workspace so pyright treats the buffer as a real file with this
# venv's site-packages on the path (the editor opens an in-memory doc at
# DOC_URI; the file need not exist on disk).
_WORKSPACE = Path(tempfile.gettempdir()) / "egg-webui-lsp"
_WORKSPACE.mkdir(exist_ok=True)
ROOT_URI = _WORKSPACE.as_uri()
DOC_URI = (_WORKSPACE / "scratch.py").as_uri()

_VENV = Path(sys.executable).resolve().parent.parent


def langserver_cmd() -> list[str] | None:
    """The ``basedpyright-langserver --stdio`` command, or ``None`` if the
    language server is not installed alongside this interpreter."""
    local = Path(sys.executable).resolve().parent / "basedpyright-langserver"
    exe = str(local) if local.exists() else shutil.which("basedpyright-langserver")
    return [exe, "--stdio"] if exe else None


def lsp_available() -> bool:
    return langserver_cmd() is not None


def _analysis_settings() -> dict:
    return {
        "autoImportCompletions": True,
        "completeFunctionParens": True,  # callables complete as `f(…)` snippets
        "diagnosticMode": "openFilesOnly",
        "typeCheckingMode": "basic",
    }


def _config_for(section: str | None) -> dict:
    """Answer a ``workspace/configuration`` item for the given section. pyright
    pulls ``python`` (interpreter + analysis), ``basedpyright`` and
    ``basedpyright.analysis`` — each wants the value AT that section."""
    if section in (None, "", "python"):
        return {"pythonPath": sys.executable, "analysis": _analysis_settings()}
    if section in ("python.analysis", "basedpyright.analysis"):
        return _analysis_settings()
    if section == "basedpyright":
        return {"analysis": _analysis_settings()}
    return {}


def _frame(msg: dict) -> bytes:
    body = json.dumps(msg).encode()
    return b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body


def _read_message(stream) -> dict | None:
    """Read one Content-Length-framed JSON-RPC message, or None at EOF."""
    length = 0
    while True:
        line = stream.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        key, _, val = line.partition(b":")
        if key.strip().lower() == b"content-length":
            length = int(val.strip())
    body = stream.read(length)
    if len(body) < length:
        return None
    return json.loads(body)


class LspProcess:
    """A basedpyright-langserver subprocess speaking framed JSON-RPC on stdio."""

    def __init__(self, on_message: Callable[[dict | None], None]):
        cmd = langserver_cmd()
        if cmd is None:
            raise RuntimeError("basedpyright-langserver is not installed")
        env = {**os.environ, "VIRTUAL_ENV": str(_VENV)}
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=str(_WORKSPACE),
            env=env,
        )
        self._on_message = on_message
        self._wlock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        while True:
            msg = _read_message(self.proc.stdout)
            if msg is None:
                break
            try:
                self._on_message(msg)
            except Exception:  # a client callback must never kill the reader
                pass
        self._on_message(None)  # EOF: signal close

    def send(self, msg: dict) -> None:
        data = _frame(msg)
        with self._wlock:
            try:
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass

    def close(self) -> None:
        with self._wlock:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        try:
            self.proc.terminate()
        except Exception:
            pass


class LspBridge:
    """Bridges one browser LSP session to a basedpyright subprocess.

    ``send_to_client`` forwards a JSON-RPC message to the browser. Messages
    from the browser go straight to the server via :meth:`from_client`;
    messages from the server are forwarded to the browser EXCEPT the
    server→client requests this bridge answers itself.
    """

    def __init__(self, send_to_client: Callable[[dict], None]):
        self._send_client = send_to_client
        self.proc = LspProcess(self._from_server)

    def from_client(self, msg: dict) -> None:
        self.proc.send(msg)

    def _from_server(self, msg: dict | None) -> None:
        if msg is None:
            self._send_client({"jsonrpc": "2.0", "method": "egg/closed"})
            return
        method = msg.get("method")
        if "id" in msg and method:  # a request the server expects us to answer
            if method == "workspace/configuration":
                items = msg.get("params", {}).get("items", [])
                result = [_config_for(it.get("section")) for it in items]
                self.proc.send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
                return
            if method in (
                "client/registerCapability",
                "client/unregisterCapability",
                "window/workDoneProgress/create",
            ):
                self.proc.send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
                return
        # responses, diagnostics/other notifications, and server requests the
        # browser can handle all flow through.
        self._send_client(msg)

    def close(self) -> None:
        self.proc.close()
