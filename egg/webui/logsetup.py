# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Tee a launcher's stdout and stderr to a timestamped logfile.

The web UI / desktop launchers call :func:`start_file_logging` once at startup.
It redirects file descriptors 1 and 2 through pipes that a pair of reader
threads copy to both the original terminal streams *and* a logfile, so the
console still shows everything while a durable copy is written to disk. Because
the redirect is at the fd level, child processes that inherit these descriptors
(uvicorn's reloader, the desktop app's server subprocess, a solver worker's
stderr) are captured too.

The logfile lives in the configured directory (``logging.dir`` or
``<config_dir>/logs``), is named from an ISO-ish launch timestamp, and older
logs beyond ``logging.keep`` are pruned. Everything is best-effort and guarded:
a logging failure must never stop the app from starting, and the terminal
output is never lost.
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from pathlib import Path

from .config import load_config, logs_dir


def _prune(dirp: Path, prefix: str, keep: int) -> None:
    """Keep only the newest ``keep`` ``<prefix>-*.log`` files (by name, which is
    the timestamp, so lexicographic order is chronological)."""
    try:
        logs = sorted(dirp.glob(f"{prefix}-*.log"))
    except OSError:
        return
    drop = logs if keep <= 0 else logs[:-keep]
    for p in drop:
        try:
            p.unlink()
        except OSError:
            pass


def start_file_logging(name: str) -> Path | None:
    """Begin teeing this process's stdout/stderr to a logfile; return its path.

    ``name`` distinguishes the launcher (``"webui"`` / ``"desktop"``) in the
    filename and prune group. Returns ``None`` (leaving the streams untouched)
    when logging is disabled, suppressed by ``EGG_NO_LOGFILE`` (set for a child
    process whose parent already tees), or if setup fails for any reason.
    """
    if os.environ.get("EGG_NO_LOGFILE"):
        return None
    cfg = load_config()
    log_cfg = cfg.get("logging", {})
    if not log_cfg.get("enabled", True):
        return None
    try:
        keep = int(log_cfg.get("keep", 10) or 0)
    except (TypeError, ValueError):
        keep = 10
    d = logs_dir(cfg)
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    _prune(d, f"egg-{name}", keep)
    # Colons are invalid in filenames on Windows, so use dashes in the time.
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = d / f"egg-{name}-{stamp}.log"
    try:
        logf = open(path, "ab", buffering=0)
    except OSError:
        return None

    lock = threading.Lock()

    def _tee(real_fd: int):
        """Redirect one std fd through a pipe copied to ``real_fd`` + the log."""
        r, w = os.pipe()

        def pump():
            while True:
                try:
                    chunk = os.read(r, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                try:
                    os.write(real_fd, chunk)  # keep the terminal output
                except OSError:
                    pass
                with lock:
                    try:
                        logf.write(chunk)
                    except OSError:
                        pass

        threading.Thread(target=pump, daemon=True).start()
        return w

    try:
        # Flush the Python-level buffers before the fds move under them.
        sys.stdout.flush()
        sys.stderr.flush()
        real_out = os.dup(1)
        real_err = os.dup(2)
        w_out = _tee(real_out)
        w_err = _tee(real_err)
        os.dup2(w_out, 1)
        os.dup2(w_err, 2)
        os.close(w_out)
        os.close(w_err)
    except OSError:
        return None
    # A short header so a log's launch time and command are obvious.
    try:
        argv = " ".join(sys.argv)
        sys.stderr.write(f"# egg {name} log {stamp}  ({argv})\n")
        sys.stderr.flush()
    except OSError:
        pass
    return path
