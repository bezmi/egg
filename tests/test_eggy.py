# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""The .eggy case archive: a verbatim zip of a runnable workspace.

Pack/unpack round-trips the case files, skips build noise, detects a real
archive structurally, and unzips to a directory that runs directly.
"""

import os
import zipfile

import pytest

from egg.io import eggy


def _make_case(root):
    os.makedirs(os.path.join(root, "assets"))
    with open(os.path.join(root, "script.py"), "w") as f:
        f.write("import sys\nprint('ran', sys.argv[1:])\n")
    with open(os.path.join(root, "net.npz"), "wb") as f:
        f.write(b"\x00net\x00")
    with open(os.path.join(root, "assets", "shape.step"), "w") as f:
        f.write("STEP DATA")
    # build noise that must not be packed
    os.makedirs(os.path.join(root, "__pycache__"))
    with open(os.path.join(root, "__pycache__", "x.pyc"), "wb") as f:
        f.write(b"junk")


def test_pack_unpack_roundtrip(tmp_path):
    case = tmp_path / "case"
    _make_case(str(case))
    out = str(tmp_path / "egg.eggy")
    eggy.pack(out, str(case))

    dest = str(tmp_path / "unpacked")
    eggy.unpack(out, dest)
    # The script and assets land at the archive root (runnable directly).
    assert os.path.isfile(os.path.join(dest, "script.py"))
    assert os.path.isfile(os.path.join(dest, "net.npz"))
    with open(os.path.join(dest, "assets", "shape.step")) as f:
        assert f.read() == "STEP DATA"


def test_pack_skips_build_noise(tmp_path):
    case = tmp_path / "case"
    _make_case(str(case))
    out = str(tmp_path / "egg.eggy")
    eggy.pack(out, str(case))
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)


def test_is_eggy_detects_structurally(tmp_path):
    case = tmp_path / "case"
    _make_case(str(case))
    out = str(tmp_path / "egg.eggy")
    eggy.pack(out, str(case))
    assert eggy.is_eggy(out)

    # A zip with no python script is not a case archive.
    plain = str(tmp_path / "plain.zip")
    with zipfile.ZipFile(plain, "w") as zf:
        zf.writestr("data.txt", "hi")
    assert not eggy.is_eggy(plain)

    # A non-zip file is not an archive.
    notzip = str(tmp_path / "notzip.bin")
    with open(notzip, "wb") as f:
        f.write(b"not a zip")
    assert not eggy.is_eggy(notzip)


def test_run_executes_the_entry_script(tmp_path):
    case = tmp_path / "case"
    _make_case(str(case))
    out = str(tmp_path / "egg.eggy")
    eggy.pack(out, str(case))
    # run unpacks to a temp dir and executes script.py; a clean exit is 0.
    assert eggy._main(["run", out, "--", "--n", "49"]) == 0


def test_entry_script_is_ambiguous_without_script_py(tmp_path):
    d = str(tmp_path / "multi")
    os.makedirs(d)
    for name in ("a.py", "b.py"):
        with open(os.path.join(d, name), "w") as f:
            f.write("pass\n")
    with pytest.raises(ValueError, match="entry script"):
        eggy.entry_script(d)
