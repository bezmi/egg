# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""The ``.eggy`` case archive: a plain zip of a runnable case workspace.

An ``.eggy`` is just a zip whose root IS the case directory, so ``unzip`` gives
a directory you can run with no fixups: the case script, its input assets
(STEP / SVG / profiles), and the net cache ``net.npz`` sit exactly where the
script's relative paths expect them.

    egg.eggy
    |- script.py     the case script (entry point)
    |- net.npz       the warm-start cache (if the run wrote one)
    |- assets/...    input data, referenced by relative path

There is no manifest and no marker: the Python script is the source of truth,
and the archive shape is fixed (script + assets + net cache). Opening an
``.eggy`` unpacks it; it does NOT run anything. The script runs only when the
user runs it, which is the normal path (warm-started by the net cache).
"""

from __future__ import annotations

import os
import zipfile

__all__ = ["pack", "unpack", "is_eggy"]

# Directories / files that are build noise, not part of a case.
_SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
_SKIP_SUFFIXES = (".pyc", ".pyo")


def _iter_files(case_dir: str):
    """Yield ``(absolute_path, arcname)`` for every case file worth packing."""
    case_dir = os.path.abspath(case_dir)
    for root, dirs, files in os.walk(case_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            if name.endswith(_SKIP_SUFFIXES) or name.endswith(".eggy"):
                continue
            full = os.path.join(root, name)
            # arcname relative to the case root, so unzip reproduces the
            # workspace with the script at the top level.
            yield full, os.path.relpath(full, case_dir)


def pack(out_path: str, case_dir: str) -> str:
    """Zip a case workspace directory into an ``.eggy`` archive (verbatim).

    Every file under ``case_dir`` is stored relative to that directory, so the
    archive root is the runnable workspace. Build noise (``__pycache__``,
    ``*.pyc``) and any nested ``.eggy`` are skipped. Returns ``out_path``.
    """
    if not os.path.isdir(case_dir):
        raise NotADirectoryError(f"case_dir {case_dir!r} is not a directory")
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for full, arc in sorted(_iter_files(case_dir), key=lambda t: t[1]):
            zf.write(full, arc)
    return out_path


def unpack(eggy_path: str, into: str) -> str:
    """Extract an ``.eggy`` archive into ``into`` (nothing is executed).

    Creates ``into`` if needed and returns it. The result is a plain directory:
    run the script inside it when ready, the normal way.
    """
    os.makedirs(into, exist_ok=True)
    with zipfile.ZipFile(eggy_path, "r") as zf:
        zf.extractall(into)
    return into


def is_eggy(path: str) -> bool:
    """True when ``path`` looks like a case archive: a zip holding a ``.py``.

    Detection is structural (there is no marker): a zip whose members include a
    Python script. A ``net.npz`` beside it is expected but not required (a case
    that has never been solved has no cache yet).
    """
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path, "r") as zf:
        return any(n.endswith(".py") for n in zf.namelist())


def entry_script(case_dir: str) -> str:
    """The case's entry script: ``script.py`` if present, else the only ``.py``.

    Raises when the entry point is ambiguous (several scripts and no
    ``script.py``) so the caller names one explicitly.
    """
    script = os.path.join(case_dir, "script.py")
    if os.path.isfile(script):
        return script
    pys = [
        os.path.join(case_dir, n)
        for n in sorted(os.listdir(case_dir))
        if n.endswith(".py")
    ]
    if len(pys) == 1:
        return pys[0]
    raise ValueError(
        f"cannot pick an entry script in {case_dir!r}: expected script.py or a "
        f"single .py, found {[os.path.basename(p) for p in pys]}"
    )


def _main(argv=None) -> int:
    """``python -m egg.io.eggy {pack,unpack,run}`` (stdlib only)."""
    import argparse
    import subprocess
    import sys
    import tempfile

    p = argparse.ArgumentParser(prog="egg.io.eggy", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("pack", help="zip a case directory into a .eggy")
    q.add_argument("case_dir")
    q.add_argument("out")

    q = sub.add_parser("unpack", help="extract a .eggy into a directory")
    q.add_argument("eggy")
    q.add_argument("into")

    q = sub.add_parser("run", help="unpack a .eggy to a temp dir and run it")
    q.add_argument("eggy")
    q.add_argument(
        "args", nargs=argparse.REMAINDER, help="args after -- go to the script"
    )

    a = p.parse_args(argv)
    if a.cmd == "pack":
        print(pack(a.out, a.case_dir))
        return 0
    if a.cmd == "unpack":
        print(unpack(a.eggy, a.into))
        return 0
    if a.cmd == "run":
        work = tempfile.mkdtemp(prefix="eggy-")
        unpack(a.eggy, work)
        script = entry_script(work)
        forwarded = a.args[1:] if a.args and a.args[0] == "--" else a.args
        return subprocess.call(
            [sys.executable, os.path.basename(script), *forwarded], cwd=work
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
