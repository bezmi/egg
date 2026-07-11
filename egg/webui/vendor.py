# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Vendor the web UI's pinned browser dependencies for offline serving.

Mirrors the esm.sh module graph of the pinned CodeMirror packages (and
split.js) into ``webui/static/vendor/``, rewriting the absolute
``/pkg@ver/...`` import specifiers to relative local filenames. Each
distinct esm.sh path maps to exactly one local file, so the module-graph
deduplication that makes CodeMirror work (single @codemirror/state
instance) is preserved.

Run at deploy time (or let the app call :func:`ensure_vendor` on startup):

    uv run --no-sync python webui/vendor.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

ESM_BASE = "https://esm.sh"

# Keep in lockstep with EDITOR_JS in app.py. state/language/lezer-highlight
# (the editor theming entry points) are pinned to the exact versions the
# rest of the graph already resolves to — a second @codemirror/state
# instance would break CodeMirror's instanceof checks.
CM_ENTRIES = {
    "codemirror": "/codemirror@6.0.2",
    "view": "/@codemirror/view@6.43.4",
    "lang-python": "/@codemirror/lang-python@6.2.1",
    "autocomplete": "/@codemirror/autocomplete@6.20.3",
    "commands": "/@codemirror/commands@6.10.4",
    "state": "/@codemirror/state@6.7.0",
    "language": "/@codemirror/language@6.12.4",
    "lezer-highlight": "/@lezer/highlight@1.2.3",
}
SPLIT_URL = "https://cdn.jsdelivr.net/npm/split.js@1.6.5/dist/split.min.js"

VENDOR_DIR = Path(__file__).resolve().parent / "static" / "vendor"

# Only genuine module specifiers (import"/x", from"/x", import("/x")) —
# a bare quoted-string match also hits SVG data-URIs etc. in minified code.
_SPEC_RE = re.compile(r'\b(?:from|import)\s*\(?\s*"(/[^"]+)"')
_PATH_OK = re.compile(r"^/[\w@^~.\-/%?=&+]+$")


def _fname(path: str) -> str:
    """Deterministic local filename for an esm.sh path (query included)."""
    name = re.sub(r"[^A-Za-z0-9.-]+", "_", path.strip("/"))
    return name if name.endswith(".mjs") else name + ".mjs"


def _fetch(url: str) -> bytes:
    # semver-range characters in esm.sh paths must be percent-encoded
    url = url.replace("^", "%5E").replace("~", "%7E")
    req = urllib.request.Request(url, headers={"User-Agent": "egg-webui-vendor"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def vendor(verbose: bool = True) -> dict[str, str]:
    """Mirror everything; returns the manifest (logical name -> filename)."""
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    queue = list(CM_ENTRIES.values())
    seen: set[str] = set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        text = _fetch(ESM_BASE + path).decode()
        deps = {d for d in _SPEC_RE.findall(text) if _PATH_OK.match(d)}
        for dep in deps:
            text = text.replace(f'"{dep}"', f'"./{_fname(dep)}"')
            queue.append(dep)
        (VENDOR_DIR / _fname(path)).write_text(text)
        if verbose:
            print(f"  {path} -> {_fname(path)} ({len(deps)} deps)")

    (VENDOR_DIR / "split.min.js").write_bytes(_fetch(SPLIT_URL))
    if verbose:
        print("  split.js -> split.min.js")

    manifest = {name: _fname(path) for name, path in CM_ENTRIES.items()}
    (VENDOR_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if verbose:
        print(f"vendored {len(seen) + 1} files into {VENDOR_DIR}")
    return manifest


def ensure_vendor() -> bool:
    """Vendor if not already present (or the entry list grew since the
    last vendoring). Returns True when the vendor dir is usable; False
    (with a warning) when offline and nothing is cached."""
    mf = VENDOR_DIR / "manifest.json"
    if mf.is_file():
        try:
            if set(CM_ENTRIES) <= set(json.loads(mf.read_text())):
                return True
        except Exception:
            pass  # unreadable manifest: re-vendor
    try:
        vendor(verbose=True)
        return True
    except Exception as exc:
        print(
            f"webui: vendoring failed ({exc}); the client will fall back "
            f"to the CDN (needs internet)",
            file=sys.stderr,
        )
        return False


if __name__ == "__main__":
    vendor()
