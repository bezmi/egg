# MIT License
#
# Copyright (c) 2026 Shahzeb Imran and the Egg contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Regrid a ``.eggy`` archive at a new resolution.

This is a thin harness, not a second solver: it unpacks the archive and runs
the packed ``case.py`` with new grid-shape arguments. The cached net inside the
archive re-tabulates onto the new grid, so the run polishes instead of solving
cold. Regridding is just "re-run the packed script with new ``--n``".

    uv run regrid.py demo.eggy --n 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

from egg.io import eggy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("eggy", help="a .eggy archive made with case.py --export-eggy")
    p.add_argument("--n", type=int, default=4, help="new resolution scale")
    p.add_argument(
        "--smoother", default="control_point", choices=["control_point", "jacobi"]
    )
    a = p.parse_args()

    work = tempfile.mkdtemp(prefix="eggy-regrid-")
    eggy.unpack(a.eggy, work)
    script = eggy.entry_script(work)
    print(f"regridding {a.eggy} to n={a.n} in {work}")
    rc = subprocess.call(
        [
            sys.executable,
            os.path.basename(script),
            "--n",
            str(a.n),
            "--smoother",
            a.smoother,
        ],
        cwd=work,
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
