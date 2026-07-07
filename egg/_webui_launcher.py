# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""``egg-webui`` console script: serve the web UI from a repo checkout.

The web UI prototype lives in ``webui/`` next to the ``egg`` package (it
is not shipped inside the wheel), so this launcher only works with the
editable/development install — which is the only supported way to run
the prototype anyway. Usage::

    uv run --no-sync egg-webui                      # http://127.0.0.1:5001
    uv run --no-sync egg-webui my_geometry.py       # open a script
    uv run --no-sync egg-webui my_geometry.py --watch    # follow it on disk
    uv run --no-sync egg-webui --host 0.0.0.0 --reload   # dev server
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _build_docs(repo: Path) -> None:
    """Best-effort docs refresh: the UI serves ``docs/_build/html`` when it
    exists (help → documentation). Missing sphinx (docs group) or doxygen
    just means the previously built docs — if any — keep being served."""
    try:
        import sphinx  # noqa: F401
    except ImportError:
        print(
            "egg-webui: sphinx not installed (uv sync --group docs) — "
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
    repo = Path(__file__).resolve().parents[1]
    webui = repo / "webui"
    if not (webui / "app.py").is_file():
        raise SystemExit(
            "egg-webui: webui/app.py not found — the web UI runs from a "
            "repo checkout (editable install), not from an installed wheel"
        )

    p = argparse.ArgumentParser(prog="egg-webui", description="serve the egg web UI")
    p.add_argument("script", nargs="?", help="geometry script to open in the editor")
    p.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: %(default)s)"
    )
    p.add_argument("--port", type=int, default=5001, help="port (default: %(default)s)")
    p.add_argument(
        "--reload",
        action="store_true",
        help="dev mode: restart on edits to webui/ or egg/ (drops websocket "
        "connections and any in-flight run)",
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

    if not a.no_docs:
        _build_docs(repo)

    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "egg-webui needs the webui dependency group: uv sync --group webui"
        ) from None

    if a.script:
        os.environ["EGG_WEBUI_SCRIPT"] = str(Path(a.script).resolve())
    if a.watch:
        os.environ["EGG_WEBUI_WATCH"] = "1"
    uvicorn.run(
        "app:app",
        app_dir=str(webui),
        host=a.host,
        port=a.port,
        reload=a.reload,
        reload_dirs=[str(webui), str(repo / "egg")] if a.reload else None,
    )


if __name__ == "__main__":
    main()
