# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""``egg.file_import``: bring an out-of-tree file into a ``.eggy`` archive.

A ``.eggy`` already bundles everything under the case directory (assets the
script references by relative path). :func:`file_import` is the opt-in escape
hatch for a file that lives *outside* the case directory but should still
travel with the archive::

    wing = egg.file_import("/data/shared/wing.step")   # anywhere on disk
    shape = import_step(wing.path)                      # use .path, not a raw path

Using ``file_import`` *is* the request to bundle: on pack the referenced file
is copied into the archive under ``deps/`` and the literal path in the packed
script is rewritten to that archive-relative location. :attr:`FileImport.path`
resolves to the bundled copy beside the script when present (so an extracted
archive works), else the original path (so it works live, before archiving).
It is correct either way, whether or not the source literal was rewritten.

Discovery for the rewrite is static (AST): only string-literal paths are
rewritten; a computed ``file_import(var)`` still resolves correctly at runtime
but is not bundled (a pack-time warning). A runtime registry of every declared
import is kept in :data:`_REGISTRY` for tooling.
"""

from __future__ import annotations

import ast
import inspect
import shutil
import zipfile
from pathlib import Path

__all__ = ["file_import", "FileImport", "pack_case"]

# Where bundled out-of-tree files land inside the archive.
_DEPS_SUBDIR = "deps"

# Every file_import() declared this session (runtime introspection / tooling).
_REGISTRY: list[FileImport] = []


class FileImport:
    """A file dependency declared with :func:`file_import`. Use :attr:`path`."""

    __slots__ = ("raw", "base")

    def __init__(self, raw: str, base: Path) -> None:
        self.raw = str(raw)
        self.base = Path(base)

    @property
    def _dest(self) -> str:
        """Archive-relative location the bundled copy lands at."""
        return f"{_DEPS_SUBDIR}/{Path(self.raw).name}"

    @property
    def path(self) -> Path:
        """The usable filesystem path: the bundled copy beside the script if it
        exists (extracted archive), else the original (absolute, or resolved
        against the declaring script's directory)."""
        local = self.base / self._dest
        if local.exists():
            return local
        p = Path(self.raw).expanduser()
        return p if p.is_absolute() else (self.base / p)

    def __fspath__(self) -> str:  # usable directly anywhere a path is expected
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"FileImport({self.raw!r})"


def _caller_dir() -> Path:
    """Directory of the user script that called :func:`file_import` (so a
    relative import path resolves against the script, live and extracted)."""
    for fr in inspect.stack()[1:]:
        fn = fr.filename
        if fn and not fn.startswith("<") and not fn.endswith("deps.py"):
            return Path(fn).resolve().parent
    return Path.cwd()


def file_import(path: str | Path) -> FileImport:
    """Declare an out-of-tree file to bundle into the ``.eggy`` archive.

    Returns a :class:`FileImport`; read :attr:`FileImport.path` wherever the
    script needs the file (never hard-code the path). See the module docstring.
    """
    dep = FileImport(str(path), _caller_dir())
    _REGISTRY.append(dep)
    return dep


def _is_file_import(func: ast.expr) -> bool:
    """True for ``egg.file_import(...)`` or a bare ``file_import(...)``."""
    if isinstance(func, ast.Attribute):
        return func.attr == "file_import"
    return isinstance(func, ast.Name) and func.id == "file_import"


def _plan_imports(script_src: str, case_dir: Path):
    """Find ``file_import("literal")`` calls in ``script_src``. Returns the
    rewritten source (literals pointed at ``deps/…``) and the list of
    ``(absolute_source, archive_dest)`` files to copy in. Missing files and
    non-literal calls are skipped (left untouched)."""
    try:
        tree = ast.parse(script_src)
    except SyntaxError:
        return script_src, []
    data = bytearray(script_src.encode("utf-8"))
    line_starts = [0]
    for i, b in enumerate(data):
        if b == 0x0A:
            line_starts.append(i + 1)
    boff = lambda ln, col: line_starts[ln - 1] + col  # noqa: E731 (byte offset)

    deps: list[tuple[str, str]] = []
    edits: list[tuple[int, int, bytes]] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_file_import(node.func) and node.args):
            continue
        arg = node.args[0]
        if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
            continue  # computed path: resolves at runtime, but can't be rewritten
        src = Path(arg.value).expanduser()
        if not src.is_absolute():
            src = case_dir / src
        src = src.resolve()
        if not src.is_file():
            continue
        dest = f"{_DEPS_SUBDIR}/{src.name}"
        n = 1
        while dest in seen:  # de-dup basename collisions across imports
            dest = f"{_DEPS_SUBDIR}/{src.stem}-{n}{src.suffix}"
            n += 1
        seen.add(dest)
        deps.append((str(src), dest))
        edits.append(
            (
                boff(arg.lineno, arg.col_offset),
                boff(arg.end_lineno, arg.end_col_offset),
                repr(dest).encode("utf-8"),
            )
        )
    for s, e, txt in sorted(edits, reverse=True):
        data[s:e] = txt
    return data.decode("utf-8"), deps


def pack_case(out_path: str, case_dir: str, entry_script: str) -> str:
    """Pack ``case_dir`` into the ``.eggy`` at ``out_path``, additionally
    bundling the ``file_import`` dependencies of ``entry_script`` under
    ``deps/`` and rewriting their literal paths in the packed script.

    The user's working directory is left untouched: the rewritten script and
    the copied deps are written straight into the zip, never back to disk.
    """
    from egg.io import eggy

    case = Path(case_dir).resolve()
    entry = Path(entry_script).resolve()
    new_src, deps = _plan_imports(entry.read_text(), case)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for full, arc in sorted(eggy._iter_files(str(case)), key=lambda t: t[1]):
            if Path(full).resolve() == entry:
                zf.writestr(arc, new_src)  # the rewritten entry script
            else:
                zf.write(full, arc)
        for src_abs, dest_rel in deps:
            zf.write(src_abs, dest_rel)  # the bundled out-of-tree file
    return str(out_path)
