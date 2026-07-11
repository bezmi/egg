# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Where the vendored browser assets live, and whether they are present.

This module deliberately contains NO download code — fetching the assets
is the job of the out-of-tree build tool ``tools/vendor_webui.py`` (run at
wheel-build time, or on demand by ``egg-webui --dev``). At runtime the web
UI only serves what is already on disk under :data:`VENDOR_DIR`; it never
reaches the network and there is no CDN fallback (a security requirement).
Both the app and the launcher import this light module to check readiness.
"""

from __future__ import annotations

import json
from pathlib import Path

# egg/webui/static/vendor/ — populated by tools/vendor_webui.py.
VENDOR_DIR = Path(__file__).resolve().parent / "static" / "vendor"
MANIFEST = VENDOR_DIR / "manifest.json"

MISSING_MSG = (
    "egg-webui: browser dependencies not installed in egg/webui/static/vendor. "
    "These should be included in the python package. If you are running "
    "directly from the git repo, append the --dev flag to install them."
)


def vendor_ready() -> bool:
    """True iff the manifest and every file it references are on disk."""
    if not MANIFEST.is_file():
        return False
    try:
        man = json.loads(MANIFEST.read_text())
    except Exception:
        return False
    files = [v for v in man.values() if isinstance(v, str)]
    files += [c for c in man.get("_css", []) if isinstance(c, str)]
    files.append("split.min.js")
    return all((VENDOR_DIR / f).is_file() for f in files)
