# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""User configuration for the web UI / desktop app.

Deep, file-based configuration read from ``~/.config/egg/config.toml``
(honouring ``XDG_CONFIG_HOME``). This is the escape hatch for the settings that
are not everyday UI checkboxes: keybindings, the various interaction delays,
the auto-run policy, the log location / retention, and experimental feature
flags. Everything has a default, so the file is entirely optional and only the
keys the user overrides need to appear.

The browser reads the client-relevant subset (delays, keybinds, behaviour) as
``window.eggConfig``, injected into the page; the launchers read the logging
section. Values are merged over :data:`DEFAULTS`, so a partial file (or a bad
one) still yields a complete, valid config.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover  (older interpreters)
    tomllib = None  # type: ignore[assignment]


# The full schema with defaults. Nested tables map to TOML ``[section]`` /
# ``[section.subsection]``. Keybinds use the same "Mod+Key" spelling the editor
# understands ("Ctrl", "Meta"/"Cmd", "Alt", "Shift"; a bare key for the topology
# tools); delays are milliseconds.
DEFAULTS: dict[str, Any] = {
    "delays": {
        # hover before a diagnostic/hover tooltip appears
        "tooltip_hover_ms": 120,
        # typing pause before the autocomplete popup opens
        "autocomplete_ms": 250,
        # typing pause before an unsaved buffer auto-runs (see behavior.autorun)
        "autorun_ms": 2000,
    },
    "behavior": {
        # how the grid view updates from editor edits:
        #   "delay" - auto-run the buffer after delays.autorun_ms (syntax permitting)
        #   "save"  - only re-run when the file is saved
        #   "off"   - never auto-run; use the run button
        "autorun": "delay",
    },
    "ui": {
        # height the documentation popup opens at, as a percent of the window
        # (it can still be dragged taller/shorter for the current view). Fixed so
        # a long docs page does not open a taller pane than a short one.
        "doc_pane_pct": 25,
    },
    "fonts": {
        # Interface font: the whole UI chrome (menus, buttons, panels, labels).
        # Empty uses the built-in monospace stack. Set to any installed font
        # family, proportional or monospace, e.g. "Inter" or "JetBrains Mono".
        "interface": "",
        # Base interface font size in px. Every other UI size is a fixed factor
        # of this, so changing it scales the whole interface proportionally.
        "interface_size": 13,
        # Editor font: the code editor plus code and program output (stdout, run
        # log, error text). Empty uses the built-in monospace stack; set it to
        # your preferred programming font, e.g. "JetBrains Mono".
        "editor": "",
        # Base editor font size in px (code editor and code/output). The editor's
        # Ctrl+scroll zoom starts from this; other code/output sizes are fixed
        # factors of it.
        "editor_size": 13,
        # For the families: if the named font is not installed the built-in stack
        # is used, and only a plain family name is honored (letters, digits,
        # spaces, dots, hyphens). All four take effect on the next server start.
    },
    "keybinds": {
        "run": "Ctrl+Enter",  # also stops when a run is streaming
        "comment_toggle": "Ctrl+/",
        # topology edit-view node operations (bare keys, no modifier)
        "node_split": "s",
        "node_join": "j",
        "node_coincident": "c",
        "node_set_res": "r",
    },
    "logging": {
        # write a per-launch logfile (in addition to the terminal); set false
        # to keep output on the terminal only
        "enabled": True,
        # directory for launch logfiles; empty -> <config_dir>/logs
        "dir": "",
        # how many previous run logs to keep (older ones are pruned)
        "keep": 10,
    },
    "export": {
        # Prepend a short "how to run this in lmr" comment block to an exported
        # gdtk/Eilmer grid.lua (prep-grid/prep-gas/prep-sim steps, bcDict hint).
        # Set false to emit just the registration calls.
        "lmr_grid_lua_instructions": True,
    },
    "experimental": {
        # Developer-only: skip the per-launch auth token, so any local client
        # can reach the server without it. This removes the main protection on
        # what is otherwise privileged local IPC (it runs code and reads/writes
        # files). Never set this outside a trusted development machine.
        "disable_auth": False,
    },
}


def config_dir() -> Path:
    """``~/.config/egg`` (or ``$XDG_CONFIG_HOME/egg``)."""
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "egg"


def state_dir() -> Path:
    """``~/.local/state/egg`` (or ``$XDG_STATE_HOME/egg``).

    Persistent, non-config state for the desktop app lives here, notably the
    webview's storage (localStorage/cookies) so the picked theme and other UI
    preferences survive a relaunch.
    """
    base = os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state")
    return Path(base) / "egg"


def config_path() -> Path:
    """The config file, whether or not it exists."""
    return config_dir() / "config.toml"


def _deep_merge(base: dict, over: dict) -> dict:
    """Recursively overlay ``over`` onto a copy of ``base`` (dicts merge, other
    values replace). Unknown keys in ``over`` are kept, so a forward-compatible
    file does not lose settings a newer egg would understand."""
    out = deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    """Return the merged config (defaults overlaid with the user's file).

    A missing file, a parse error, or the absence of ``tomllib`` all fall back
    to the defaults rather than raising: the UI must always come up.
    """
    p = config_path()
    if tomllib is None or not p.is_file():
        return deepcopy(DEFAULTS)
    try:
        user = tomllib.loads(p.read_text())
    except (OSError, ValueError):
        return deepcopy(DEFAULTS)
    return _deep_merge(DEFAULTS, user if isinstance(user, dict) else {})


def logs_dir(cfg: dict | None = None) -> Path:
    """The directory launch logs are written to (``logging.dir`` or the default
    ``<config_dir>/logs``); the path is expanded but not created here."""
    cfg = cfg if cfg is not None else load_config()
    d = str(cfg.get("logging", {}).get("dir") or "").strip()
    return Path(d).expanduser() if d else config_dir() / "logs"


def auth_disabled(cfg: dict | None = None) -> bool:
    """Whether the developer-only experimental flag turns the auth token off."""
    cfg = cfg if cfg is not None else load_config()
    return bool(cfg.get("experimental", {}).get("disable_auth", False))


def client_config(cfg: dict | None = None) -> dict:
    """The subset the browser needs (delays, behaviour, keybinds, ui, fonts)."""
    cfg = cfg if cfg is not None else load_config()
    return {
        "delays": cfg.get("delays", {}),
        "behavior": cfg.get("behavior", {}),
        "keybinds": cfg.get("keybinds", {}),
        "ui": cfg.get("ui", {}),
        "fonts": cfg.get("fonts", {}),
    }
