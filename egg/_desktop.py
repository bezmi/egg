# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""``egg-desktop`` console script: run the web UI in a native window.

This starts the same server as ``egg-webui`` as a child process, so all of
that launcher's behaviour (asset vendoring, docs refresh, ``--watch``,
opening a script) is reused. It then points a *frameless* pywebview window
at ``/?desktop=1``. At that URL the FastHTML app renders a custom titlebar
(min / maximize / close plus a drag region) whose buttons call the window
controls this module exposes to JavaScript through ``js_api`` as
``window.pywebview.api``. Usage::

    egg-desktop                      # native window
    egg-desktop my_geometry.py       # open a script
    egg-desktop my_geometry.py --watch
    egg-desktop --dev                # source checkout: vendor assets if missing
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time


class WindowControls:
    """Exposed to the page as ``window.pywebview.api``.

    The frameless titlebar's buttons (see :func:`egg.webui.app._desktop_titlebar`)
    call these; pywebview surfaces each method as an async JS call. The
    window reference is filled in after the window is created. The methods
    are no-ops until then so a stray early call can't raise across the
    Python/JS bridge.
    """

    def __init__(self) -> None:
        self.window = None
        self._maximized = False

    def start_drag(self) -> None:
        """Begin a compositor-driven window move for the frameless window.

        pywebview's own drag region moves the window to an absolute
        coordinate, which Wayland compositors forbid (so it does nothing
        there). Qt's ``QWindow.startSystemMove()`` instead asks the compositor
        to move the window, which works on both Wayland and X11. This runs on
        the Qt GUI thread (pywebview dispatches ``js_api`` calls through a
        main-thread ``QObject`` slot), so the native window method can be
        called directly.
        """
        if self.window is None:
            return
        try:
            from webview.platforms.qt import BrowserView
        except Exception:
            return  # not the Qt backend; nothing to do
        view = BrowserView.instances.get(self.window.uid)
        handle = view.windowHandle() if view is not None else None
        if handle is not None:
            handle.startSystemMove()

    def open_url(self, url: str) -> None:
        """Open a URL in the user's default browser (help > report a problem).
        A frameless webview must not navigate away to an external site."""
        import webbrowser

        webbrowser.open(url)

    def minimize(self) -> None:
        if self.window is not None:
            self.window.minimize()

    def toggle_maximize(self) -> None:
        # pywebview has no reliable cross-platform "is maximized" query, so
        # track it ourselves. A frameless Linux window offers no native
        # maximize affordance to desync this.
        if self.window is None:
            return
        if self._maximized:
            self.window.restore()
        else:
            self.window.maximize()
        self._maximized = not self._maximized

    def close(self) -> None:
        if self.window is not None:
            self.window.destroy()


def _wait_until_serving(
    host: str, port: int, proc: subprocess.Popen, timeout: float = 60.0
) -> None:
    """Block until the child server accepts a TCP connection on host:port.

    Stops early (instead of waiting the full timeout) if the child exits
    first, for example on a bad port, a failed vendoring step, or a missing
    dependency group.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(
                f"egg-desktop: the server exited (code {proc.returncode}) "
                "before it began serving"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(0.1)
    raise SystemExit(
        f"egg-desktop: server did not come up on {host}:{port} within "
        f"{timeout:.0f}s"
    )


def _theme_qt_tooltips() -> None:
    """Recolor Qt's native tooltip to a neutral dark instead of the default
    pale yellow. QtWebEngine renders HTML ``title=`` tooltips with the Qt
    application palette / QToolTip style, so a stylesheet + palette override
    covers them. Best-effort and guarded: a failure just leaves the default.
    Runs from pywebview's startup hook once the GUI (and its QApplication) is up.
    """
    try:
        from qtpy.QtGui import QColor, QPalette
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            return
        pal = app.palette()
        pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#181825"))
        pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#cdd6f4"))
        app.setPalette(pal)
        app.setStyleSheet(
            (app.styleSheet() or "")
            + "\nQToolTip { background: #181825; color: #cdd6f4; "
            "border: 1px solid #45475a; }"
        )
    except Exception:
        pass


def main() -> None:
    p = argparse.ArgumentParser(
        prog="egg-desktop", description="run the egg web UI in a native window"
    )
    p.add_argument("script", nargs="?", help="geometry script to open in the editor")
    p.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: %(default)s)"
    )
    p.add_argument("--port", type=int, default=5001, help="port (default: %(default)s)")
    p.add_argument(
        "--watch",
        action="store_true",
        help="start in watch mode: no editor pane, follow the given script "
        "on disk (requires a script argument)",
    )
    p.add_argument(
        "--dev",
        action="store_true",
        help="developer mode (source checkout only): vendor the browser "
        "assets if missing",
    )
    p.add_argument(
        "--no-docs",
        action="store_true",
        help="skip the docs refresh at startup",
    )
    a = p.parse_args()

    # Tee stdout+stderr to a timestamped logfile. The child server inherits
    # these fds, so its output lands in the same log; EGG_NO_LOGFILE stops it
    # from opening a second logfile of its own.
    try:
        from egg.webui.logsetup import start_file_logging

        start_file_logging("desktop")
        os.environ["EGG_NO_LOGFILE"] = "1"
    except Exception:
        pass

    try:
        import webview
    except ImportError:
        raise SystemExit(
            "egg-desktop needs pywebview and a GUI backend: "
            "`uv sync --group desktop` (or `uv pip install 'pywebview[qt]'`)"
        ) from None

    # Reuse the egg-webui launcher verbatim, as a child process: it owns
    # asset vendoring, the docs refresh, --watch and opening a script. We
    # only add the native window on top.
    server_args = ["--host", a.host, "--port", str(a.port)]
    if a.script:
        server_args.append(a.script)
    if a.watch:
        server_args.append("--watch")
    if a.dev:
        server_args.append("--dev")
    if a.no_docs:
        server_args.append("--no-docs")

    proc = subprocess.Popen([sys.executable, "-m", "egg._webui_launcher", *server_args])
    try:
        _wait_until_serving(a.host, a.port, proc)

        controls = WindowControls()
        controls.window = webview.create_window(
            title="egg",
            url=f"http://{a.host}:{a.port}/?desktop=1",
            js_api=controls,
            width=1280,
            height=800,
            min_size=(800, 500),
            resizable=True,
            frameless=True,
            # Disable pywebview's whole-window / absolute-move dragging; the
            # titlebar spacer starts a compositor move via start_drag instead
            # (works on Wayland), so buttons and the editor keep their clicks.
            easy_drag=False,
            background_color="#181818",
        )
        # Blocks on the GUI event loop until the window is closed. The startup
        # hook recolors Qt's native tooltip (a garish yellow by default on some
        # Linux setups) to match the dark app chrome.
        webview.start(_theme_qt_tooltips)
    finally:
        # The window is gone (or we failed to open it): stop the server.
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
