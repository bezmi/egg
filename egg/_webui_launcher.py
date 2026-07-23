# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""``egg-webui`` console script: serve the web UI.

The web UI lives inside the package at ``egg/webui/`` and ships in the
wheel, so this launcher works from both an editable checkout and an
installed wheel. Usage::

    egg-webui                      # http://127.0.0.1:5001
    egg-webui my_geometry.py       # open a script
    egg-webui my_geometry.py --watch    # follow it on disk
    egg-webui --host 0.0.0.0 --reload   # dev server
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


def _reap_with_parent() -> None:
    """When egg-desktop launched us detached (our own session), make sure we and
    our worker subprocesses die if the desktop app dies, even abruptly (a crash
    that skips its own cleanup). Asks the kernel to signal us on parent death and,
    on that signal, takes down our whole process group (uvicorn + the render and
    solver workers, which inherit our group). No-op unless egg-desktop spawned us
    and we lead our own group, so a plain ``egg-webui`` in a shell is unaffected.
    """
    if not os.environ.get("EGG_DESKTOP"):
        return
    try:
        if os.getpid() != os.getpgrp():  # not our own group leader: leave it be
            return
    except OSError:
        return

    def _reap(signum=None, frame=None):
        try:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        except Exception:
            pass
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)  # take the workers down too
        except Exception:
            pass
        os._exit(0)

    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
        # If the parent already died before we got here, PDEATHSIG won't fire.
        if os.getppid() == 1:
            _reap()
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _reap)
    except Exception:
        pass


def _find_repo_root(start: Path) -> Path | None:
    """The source-checkout root (holds ``pyproject.toml``/``.git``) above the
    installed package, or ``None`` when running from a packaged wheel."""
    for p in (start, *start.parents):
        if (p / "pyproject.toml").is_file() or (p / ".git").exists():
            return p
    return None


def _ensure_assets(repo: Path | None, dev: bool) -> None:
    """Make sure the vendored browser assets are present before serving.

    In ``--dev`` (source checkout), run the out-of-tree vendoring tool to
    download them if missing. Otherwise, the assets must already be on disk
    (shipped in the wheel, or vendored earlier by ``--dev``) — if not, exit
    with an actionable message rather than reaching out to a CDN at runtime.
    """
    from egg.webui._assets import MISSING_MSG, vendor_ready

    if vendor_ready():
        return
    if dev and repo is not None:
        script = repo / "tools" / "vendor_webui.py"
        print("egg-webui: vendoring browser assets (needs internet)…", flush=True)
        r = subprocess.run([sys.executable, str(script)])
        if r.returncode != 0 or not vendor_ready():
            raise SystemExit(
                "egg-webui: vendoring failed — cannot start without the "
                "browser assets (are you online?)"
            )
        return
    raise SystemExit(MISSING_MSG)


def _build_docs(repo: Path) -> None:
    """Best-effort docs refresh: the UI serves ``docs/_build/html`` when it
    exists (help -> documentation). Missing sphinx or the doxygen binary just
    means the previously built docs, if any, keep being served."""
    try:
        import sphinx  # noqa: F401
    except ImportError:
        print(
            "egg-webui: sphinx not installed (uv sync), "
            "serving previously built docs, if any",
            file=sys.stderr,
        )
        return
    print("egg-webui: refreshing docs (sphinx incremental build)…", flush=True)
    r = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", "docs", "docs/_build/html"],
        cwd=repo,
    )
    if r.returncode != 0:
        print(
            "egg-webui: docs build failed — serving previously built docs, if any",
            file=sys.stderr,
        )


def main() -> None:
    # If egg-desktop spawned us, bind our lifetime to it so a desktop crash can
    # never leave the server (and its workers) holding the port.
    _reap_with_parent()

    egg_dir = Path(__file__).resolve().parent
    repo = _find_repo_root(egg_dir)

    p = argparse.ArgumentParser(prog="egg-webui", description="serve the egg web UI")
    p.add_argument("script", nargs="?", help="geometry script to open in the editor")
    p.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: %(default)s)"
    )
    p.add_argument("--port", type=int, default=5001, help="port (default: %(default)s)")
    p.add_argument(
        "--reload",
        action="store_true",
        help="dev mode: restart on edits to egg/ (drops websocket "
        "connections and any in-flight run)",
    )
    p.add_argument(
        "--dev",
        action="store_true",
        help="developer mode (source checkout only): vendor the browser "
        "assets if missing. Not available from an installed wheel, which "
        "already ships the assets. Combine with --reload for a live server.",
    )
    p.add_argument(
        "--no-docs",
        action="store_true",
        help="skip the docs refresh (help → documentation serves whatever "
        "docs/_build/html already holds)",
    )
    p.add_argument(
        "--watch",
        action="store_true",
        help="start in watch mode: no editor pane, the UI follows the given "
        "script on disk (requires a script argument)",
    )
    a = p.parse_args()

    if a.watch and not a.script:
        p.error("--watch needs a script to watch (pass its path)")

    # Tee stdout+stderr to a timestamped logfile (help > view logs opens the
    # directory). Best-effort; a logging failure never blocks the server.
    try:
        from egg.webui.logsetup import start_file_logging

        start_file_logging("webui")
    except Exception:
        pass

    # --dev is a source-checkout affordance: it invokes the out-of-tree
    # vendoring tool (which does not ship in the wheel). Reject it in a
    # packaged install rather than silently ignoring it. It does NOT imply
    # --reload — reloading re-imports egg, which reruns the editable rebuild
    # hook; keep that opt-in via --reload so plain `--dev` never rebuilds.
    if a.dev and repo is None:
        p.error(
            "--dev is only available from a source checkout; an installed "
            "wheel already ships the vendored browser assets"
        )

    _ensure_assets(repo, dev=a.dev)

    # Docs are a checkout-only feature (docs/ isn't shipped in the wheel).
    if repo is not None and not a.no_docs:
        _build_docs(repo)

    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "egg-webui needs the ui dependency group: uv sync --group ui"
        ) from None

    if a.script:
        os.environ["EGG_WEBUI_SCRIPT"] = str(Path(a.script).resolve())
    if a.watch:
        os.environ["EGG_WEBUI_WATCH"] = "1"

    # Auth: the server is privileged local IPC (it runs code and reads/writes
    # files), so require a per-launch token by default. The link below carries
    # it; a request without it is refused. The developer-only experimental
    # config flag can turn this off (open server, trusted machines only).
    from egg.webui.config import auth_disabled
    from egg.webui.security import prepare_auth

    token = prepare_auth(a.host, a.port, disabled=auth_disabled())
    url = f"http://{a.host}:{a.port}/"
    if token:
        print(f"egg-webui: open  {url}?token={token}", flush=True)
        print("egg-webui: that link carries the required token; keep it private.",
              flush=True)
    else:
        print(f"egg-webui: auth token disabled (experimental) — open {url}",
              flush=True)

    uvicorn.run(
        "egg.webui.app:app",
        host=a.host,
        port=a.port,
        reload=a.reload,
        reload_dirs=[str(egg_dir)] if a.reload else None,
        # Don't reload on the compiled extension (the editable rebuild hook
        # touches it on import → a reload loop) or on generated web assets.
        reload_excludes=["*.so", "static/vendor/*", "docs/*"] if a.reload else None,
    )


if __name__ == "__main__":
    main()
