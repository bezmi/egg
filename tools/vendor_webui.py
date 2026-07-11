# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Build the web UI's shippable assets for offline, CDN-free serving.

This is a **build/dev tool**; it deliberately lives OUTSIDE the shipped
``egg`` package (the package must never contain code that reaches the
network). It runs in exactly two situations:

- at wheel-build time, from CMake, so the wheel ships the assets;
- from ``egg-webui --dev`` in a source checkout, to populate the local
  assets once.

At runtime the web UI only *serves* what this script produced; it never
downloads. There is no CDN fallback (a security requirement) — the UI
refuses to start if the browser assets are missing.

Two jobs:

- **vendor** the `prism-code-editor@5.2.0` ES-module graph (editor core +
  the enabled extensions + the Python grammar and language behavior), plus
  its CSS and split.js, from the package's published ESM **dist** on
  jsDelivr. The dist tree is *mirrored* under the output dir preserving
  structure, and EVERY import specifier (relative or absolute) is rewritten
  to a ``/vendor/...`` URL (the server mount). The dist files import shared,
  content-hashed chunks (e.g. the tokenizer's ``languages`` registry) with
  relative paths, so mirroring dedups them to one file each — the grammar
  registers into the same registry the editor tokenizes against, and there
  is a single editor core instance. (We use the dist, NOT esm.sh, because
  esm.sh serves each subpath as a standalone bundle with a *private*
  registry, which silently disables syntax highlighting.)
- **build the docs** (Sphinx) so an installed package serves them offline
  (help → documentation). Users of the wheel never build docs themselves.

Usage::

    python tools/vendor_webui.py [VENDOR_DEST] [--docs DOCS_DEST]

``VENDOR_DEST`` defaults to ``egg/webui/static/vendor/`` in this checkout.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

NPM_CDN = "https://cdn.jsdelivr.net/npm"
PRISM = "prism-code-editor@5.2.0"
PKG_BASE = f"{NPM_CDN}/{PRISM}"  # published-dist base for both JS and CSS

# Published-dist entry points (from the package's exports map), one per logical
# name the client imports. They import shared hashed chunks by relative path, so
# the mirror below collapses those to a single file each — one editor core, one
# tokenizer `languages` registry shared by the grammar.
JS_ENTRIES = {
    "core": "/dist/index.js",  # createEditor + core
    "commands": "/dist/extensions/commands/index.js",  # editorCommands, editHistory
    "match-brackets": "/dist/extensions/matchBrackets/index.js",
    "highlight-brackets": "/dist/extensions/matchBrackets/highlight.js",
    "cursor": "/dist/extensions/cursor/index.js",  # cursorPosition, customCursor
    "search": "/dist/extensions/search/index.js",  # searchWidget, selectionMatches
    "autocomplete": "/dist/extensions/autocomplete/index.js",
    "tooltips": "/dist/tooltips.js",  # addTooltip (hover)
    "utils": "/dist/utils/index.js",
    "prism-python": "/dist/prism/languages/python.js",  # token grammar
    "lang-python": "/dist/languages/python.js",  # comments + auto-indent behavior
}

# Raw CSS served directly from /vendor/<name> and linked in the page head.
CSS_FILES = [
    "layout.css",
    "search.css",
    "autocomplete.css",
    "autocomplete-icons.css",
    "cursor.css",
]

SPLIT_URL = f"{NPM_CDN}/split.js@1.6.5/dist/split.min.js"

# Any module specifier in an `import`/`export ... from "..."` (or `import("...")`).
_SPEC_RE = re.compile(r'(?:from|import|export)\s*\(?\s*"([^"]+)"')


def _default_vendor_dest() -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "egg" / "webui" / "static" / "vendor"


def _fetch(url: str) -> bytes:
    # semver-range characters in esm.sh paths must be percent-encoded
    url = url.replace("^", "%5E").replace("~", "%7E")
    req = urllib.request.Request(url, headers={"User-Agent": "egg-webui-vendor"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def _local_rel(path: str) -> str:
    """Vendor-relative filename for an absolute package path. Keep the file's
    own extension; only extensionless specifiers get ``.js``."""
    rel = path.lstrip("/")
    return rel if "." in posixpath.basename(rel) else rel + ".js"


def _vendor_js(dest: Path, verbose: bool) -> dict[str, str]:
    """Mirror the module graph under ``dest``, rewriting EVERY specifier
    (absolute and relative alike) to an absolute ``/vendor/<path>`` URL so
    nothing depends on relative resolution or serving extensionless files.
    Returns name -> vendor-relative path (client imports ``/vendor/<path>``)."""
    queue: list[str] = list(JS_ENTRIES.values())
    seen: set[str] = set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        text = _fetch(PKG_BASE + path).decode()
        base = posixpath.dirname(path)
        for spec in {m.group(1) for m in _SPEC_RE.finditer(text)}:
            if spec.startswith("/"):
                absu = spec
            elif spec.startswith("."):
                absu = posixpath.normpath(posixpath.join(base, spec))
            else:
                # bare specifier: an external npm dep. prism-code-editor has
                # none in this graph; surface it loudly rather than shipping a
                # module the browser can't resolve.
                raise RuntimeError(f"unexpected bare import {spec!r} in {path}")
            queue.append(absu)
            text = text.replace(f'"{spec}"', f'"/vendor/{_local_rel(absu)}"')
        out = dest / _local_rel(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        if verbose:
            print(f"  {path} -> {_local_rel(path)}")
    if verbose:
        print(f"  ({len(seen)} modules mirrored)")
    return {name: _local_rel(entry) for name, entry in JS_ENTRIES.items()}


def vendor(dest: Path | None = None, verbose: bool = True) -> dict[str, str]:
    """Vendor the browser assets into ``dest``; returns the manifest."""
    dest = Path(dest) if dest else _default_vendor_dest()
    dest.mkdir(parents=True, exist_ok=True)

    manifest = _vendor_js(dest, verbose)

    for name in CSS_FILES:
        (dest / name).write_bytes(_fetch(f"{PKG_BASE}/dist/{name}"))
        if verbose:
            print(f"  css -> {name}")

    (dest / "split.min.js").write_bytes(_fetch(SPLIT_URL))
    if verbose:
        print("  split.js -> split.min.js")

    manifest["_css"] = CSS_FILES  # informational; CSS is served by fixed name
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if verbose:
        print(f"vendored into {dest}")
    return manifest


def build_docs(out_dir: Path, repo: Path | None = None, verbose: bool = True) -> bool:
    """Build the Sphinx docs into ``out_dir`` so an installed package serves
    them offline. Soft-fails (returns False) if the docs toolchain is absent —
    docs are non-critical, unlike the browser assets."""
    repo = repo or Path(__file__).resolve().parent.parent
    src = repo / "docs"
    if not (src / "conf.py").is_file():
        print(f"docs: no source at {src}, skipping", file=sys.stderr)
        return False
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"building docs {src} -> {out_dir}")
    r = subprocess.run(
        [sys.executable, "-m", "sphinx", "-b", "html", str(src), str(out_dir)],
        cwd=str(repo),
    )
    if r.returncode != 0:
        print(
            "docs: sphinx build failed (missing docs group or doxygen?) — "
            "the packaged UI will fall back to no bundled docs",
            file=sys.stderr,
        )
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="build the web UI's shippable assets")
    p.add_argument(
        "vendor_dest",
        nargs="?",
        help="output dir for browser assets (default: egg/webui/static/vendor)",
    )
    p.add_argument(
        "--docs",
        metavar="DIR",
        help="also build the Sphinx docs into DIR (for offline serving)",
    )
    a = p.parse_args()
    vendor(Path(a.vendor_dest) if a.vendor_dest else None)
    if a.docs:
        build_docs(Path(a.docs))


if __name__ == "__main__":
    main()
