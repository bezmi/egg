# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Render worker: editor preview renders in their own process (fasthtml-free).

The server keeps one ``python render_worker.py`` child warm (egg imported
once) and serializes every editor render through it, so script exec never
happens in the server process: a script stuck in a loop is killed with
its worker at the render timeout and the UI stays live, and a native
crash in a script costs one render plus a respawn instead of the server.
``RenderWorker`` is the parent-side handle; ``main()`` is the child.

The same child also serves the other on-demand paths that exec user
scripts — the open-dialog "library-style script" probe (``suggest``) and
the SU2 export (``su2``). The one remaining in-server exec is the run
pipeline's frame renderer, which needs the live grid object to scatter
streamed nodes into.

Requests and responses are pickles over the child's stdin / original
stdout. fd 1 is re-pointed at stderr before anything runs, so stray
C-level prints can't corrupt the channel (Python-level script prints are
captured per-exec by ``build_scene``). Messages:

- parent → child: ``(kind, req_id, code, mode, path)`` with kind one of
  ``render`` / ``suggest`` / ``su2``
- child → parent: ``("result", req_id, value)`` — a SceneResult, a
  suggestion ``str | None``, or a ``(text | None, reason)`` pair — or
  ``("error", req_id, traceback)`` for a harvest/render bug; script
  errors come back *inside* the value, like the old in-process paths.

stdin EOF (the parent died, e.g. a dev-server reload) exits the child,
so no orphan lingers.
"""

from __future__ import annotations

import json
import os
import pickle
import select
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import Future, InvalidStateError
from contextlib import suppress
from queue import Empty, SimpleQueue

RENDER_TIMEOUT = 30.0  # s; a script exec'ing longer than this is killed

_EMPTY_SVG: str | None = None


# scene (→ egg) imports stay out of module scope: the child must claim
# stdout before anything heavy imports, or a stray C-level print would
# corrupt the pickle channel.
def _error_result(msg: str) -> "SceneResult":  # noqa: F821
    from .scene import Scene, SceneResult, render_svg

    global _EMPTY_SVG
    if _EMPTY_SVG is None:
        _EMPTY_SVG = render_svg(Scene())
    return SceneResult(svg=_EMPTY_SVG, stats={}, warnings=[], stdout="", error=msg)


# What the caller gets when the child times out or crashes, per kind.
_FAIL = {
    "render": _error_result,
    "suggest": lambda msg: None,
    "su2": lambda msg: (None, msg),
    "validate": lambda msg: [
        {"kind": "worker_error", "msg": msg, "where": [], "xy": None}
    ],
}


class RenderWorker:
    """Parent-side handle: one warm child, renders serialized through it.

    Bursts coalesce: queued jobs with the same (mode, path) collapse to
    the newest code, and the stale requests resolve with its result — so
    a slow render followed by more keystrokes never renders stale code,
    and the timeout only ever fires on the *newest* code being stuck.
    """

    def __init__(self, timeout: float = RENDER_TIMEOUT):
        self.timeout = timeout
        self._q: SimpleQueue = SimpleQueue()
        self._proc: subprocess.Popen | None = None
        self._req = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(
        self, code: str, mode: str = "grid", path: str = "", timeout: float = None
    ) -> Future:
        """Queue one render. The Future always resolves to a SceneResult —
        timeouts and worker crashes are reported inside it. Cancelling the
        Future (the client gave up) skips the render."""
        return self._submit("render", code, mode, path, timeout)

    def suggest(self, code: str, path: str = "", timeout: float = None) -> Future:
        """Probe a script for the ``__egg_webui__`` block suggestion.
        Resolves to ``str | None`` (None on timeout/crash: no nagging)."""
        return self._submit("suggest", code, "", path, timeout)

    def su2(self, code: str, path: str = "", timeout: float = None) -> Future:
        """Exec + export the script's grid as SU2 text. Resolves to
        ``(text, "")`` or ``(None, reason)``."""
        return self._submit("su2", code, "", path, timeout)

    def validate(
        self, code: str, blocking: dict, path: str = "", timeout: float = None
    ) -> Future:
        """Flatten a candidate blocking against the script's base + geometry
        (the blocking rides the ``mode`` slot as JSON). Resolves to a list of
        diagnostic dicts — empty means green/committable."""
        return self._submit("validate", code, json.dumps(blocking), path, timeout)

    def _submit(
        self, kind: str, code: str, mode: str, path: str, timeout: float | None
    ) -> Future:
        fut: Future = Future()
        self._q.put((kind, code, mode, path, timeout or self.timeout, fut))
        return fut

    def close(self) -> None:
        """Kill the child (tests; the daemon thread just goes idle)."""
        self._kill()

    # --- worker thread ---

    def _ensure(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            # Run as a package module so the child's `from .scene import …`
            # resolves from the installed egg.webui package.
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "egg.webui.render_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,  # the response channel; stderr stays on the console
            )
        return self._proc

    def _kill(self) -> None:
        if self._proc is not None:
            with suppress(Exception):
                self._proc.kill()
                self._proc.wait()
            self._proc = None

    def _loop(self) -> None:
        self._ensure()  # warm (egg imports) ahead of the first render
        while True:
            jobs = [self._q.get()]
            with suppress(Empty):
                while True:
                    jobs.append(self._q.get_nowait())
            # newest job per (kind, mode, path); the rest are stale keystrokes
            newest: dict[tuple, tuple] = {}
            futures: dict[tuple, list[Future]] = {}
            for kind, code, mode, path, timeout, fut in jobs:
                key = (kind, mode, path)
                newest[key] = (code, timeout)
                futures.setdefault(key, []).append(fut)
            for key, (code, timeout) in newest.items():
                futs = [f for f in futures[key] if not f.cancelled()]
                if not futs:
                    continue
                result = self._request(code, *key, timeout)
                for f in futs:
                    with suppress(InvalidStateError):
                        f.set_result(result)

    def _request(self, code: str, kind: str, mode: str, path: str, timeout: float):
        self._req += 1
        rid = self._req
        try:
            proc = self._ensure()
            pickle.dump(
                (kind, rid, code, mode, path),
                proc.stdin,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            proc.stdin.flush()
            # Strict request/response — exactly one reply is in flight, so
            # selecting on the raw fd is safe (no read-ahead sits in the
            # BufferedReader between renders).
            deadline = time.monotonic() + timeout
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    self._kill()
                    return _FAIL[kind](
                        f"timed out after {timeout:.0f} s — the script was "
                        "killed (stuck in a loop?); the next request gets a "
                        "fresh worker"
                    )
                if not select.select([proc.stdout], [], [], left)[0]:
                    continue
                msg = pickle.load(proc.stdout)
                if msg[1] != rid:
                    continue
                if msg[0] == "result":
                    return msg[2]
                return _FAIL[kind](f"{kind} failed:\n{msg[2]}")
        except Exception:
            self._kill()
            return _FAIL[kind](
                "the render worker crashed (native fault in the script?) — "
                "it restarts on the next request"
            )


# --- the child ---


def _claim_stdout():
    """The response channel; anything else writing fd 1 goes to stderr."""
    chan = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    return chan


def main() -> None:
    chan = _claim_stdout()

    def send(msg: tuple) -> None:
        pickle.dump(msg, chan, protocol=pickle.HIGHEST_PROTOCOL)
        chan.flush()

    from .scene import (
        build_scene,
        su2_export_text,
        validate_blocking,
        webui_block_suggestion,
    )

    from egg._webui_print import _set_sink

    # egg.webui_print(...) during a render folds into that render's captured
    # stdout (build_scene redirects sys.stdout per exec), so it lands in the
    # view's "stdout" panel — the same text a run would stream live. Late-bound
    # so it follows the active redirect, not the original stream.
    _set_sink(lambda text: sys.stdout.write(text))

    handlers = {
        "render": lambda code, mode, path: build_scene(code, mode=mode, path=path),
        "suggest": lambda code, mode, path: webui_block_suggestion(code, path),
        "su2": lambda code, mode, path: su2_export_text(code, path),
        "validate": lambda code, mode, path: validate_blocking(
            code, json.loads(mode), path
        ),
    }
    stdin = sys.stdin.buffer
    while True:
        try:
            kind, rid, code, mode, path = pickle.load(stdin)
        except (EOFError, pickle.UnpicklingError):
            return
        try:
            send(("result", rid, handlers[kind](code, mode, path or None)))
        except Exception:
            send(("error", rid, traceback.format_exc()))


if __name__ == "__main__":
    main()
