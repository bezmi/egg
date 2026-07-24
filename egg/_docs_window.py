# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""egg-desktop's documentation window, run as a SEPARATE process.

A second pywebview/QtWebEngine window inside the main egg-desktop process is
unreliable on this Qt build: creating one from a ``js_api`` callback deadlocks
the GUI thread, and pre-creating one corrupts the main window with cross-thread
``QObject`` errors. So the docs get their own process with its own event loop,
which cannot touch the main window. It loads the egg-themed ``/docs-view`` shell
(a titlebar plus an iframe of the built docs); the titlebar's buttons drive this
process's own window through the same :class:`~egg._desktop.WindowControls`
``js_api`` the main window uses. Auth rides on the token already in the URL.

Spawned by :meth:`egg._desktop.WindowControls.open_docs`; not a user-facing
entry point.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading

from egg._desktop import WindowControls, _reveal_when_ready


def main() -> None:
    p = argparse.ArgumentParser(prog="egg-docs-window")
    p.add_argument("url", help="the /docs-view shell URL (carries the auth token)")
    a = p.parse_args()

    import webview

    controls = WindowControls()
    controls.window = webview.create_window(
        title="egg documentation",
        url=a.url,
        js_api=controls,
        width=1100,
        height=800,
        min_size=(640, 480),
        resizable=True,
        # Frameless with the egg titlebar the /docs-view shell renders; the
        # buttons call this process's WindowControls (start_drag / min / max /
        # close). Matches the main window's chrome.
        frameless=True,
        easy_drag=False,
        # Hidden until the shell's first frame paints (avoids the white flash).
        hidden=True,
        background_color="#181818",
    )
    threading.Thread(
        target=_reveal_when_ready, args=(controls.window,), daemon=True
    ).start()
    # Ephemeral profile: no persistent storage_path, which would clash with the
    # main window's locked QtWebEngine profile; the token in the URL authenticates.
    webview.start(private_mode=True)

    # QtWebEngine teardown can hang the process on some Qt/Wayland setups; once
    # the window is closed all work is done, so exit hard (mirrors egg._desktop).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
