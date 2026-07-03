#!/usr/bin/env python3
"""Build a standalone Egg wheel (generic SSCP, CPU+CUDA+ROCm) and attach a
PolyForm Countdown notice that converts it to GNU AGPL-3.0 after a period.

Intended to run **inside the dev container** built with both GPU backends
(``./.devcontainer/podman-run.sh all``), where acpp has the CUDA and ROCm
backends available to ``--acpp-deploy``. The application is compiled once with
``ACPP_TARGETS=generic`` (hardware-agnostic), and the runtime is bundled for
``core;cuda;hip`` so one wheel runs on CPU, NVIDIA, and AMD.

The build runs in three stages:

1. ``uv build`` compiles cpp_core and, via ``EGG_BUNDLE_RUNTIME``, has acpp
   ``--acpp-deploy`` vendor its runtime tree into ``egg/_cpp/acpp_runtime``.
   acpp only copies its *direct* dependencies, so libLLVM's own indirect deps
   (libedit, libxml2, libz3, ...) are still missing at this point.
2. ``auditwheel repair`` walks the full transitive closure of every ``.so`` in
   the wheel, copies the remaining indirect deps in, and ``patchelf``-es each
   relocated lib's RPATH to ``$ORIGIN`` so they resolve each other. It leaves
   the host-provided libs out: the manylinux whitelist (libc/libm/libstdc++/
   libgcc_s) plus the GPU driver stubs we exclude explicitly (libcuda,
   libdrm*), which must come from the target's driver. The result is retagged
   ``manylinux_2_XX`` (the glibc floor it actually needs).
3. The wheel's ``.dist-info`` gains a ``LICENSE-COUNTDOWN.md`` (the same notice
   ``make_source_release.py`` adds to source tarballs) and its ``RECORD`` is
   updated so the artifact stays installable.

auditwheel + patchelf are pulled on demand via ``uv tool run`` (nothing needs to
be pre-installed in the image).

Usage (from the repo root, inside the container)::

    scripts/make_wheel_release.py
    scripts/make_wheel_release.py --years 3 --components "core;cuda"
    scripts/make_wheel_release.py --skip-repair        # raw, non-portable wheel
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Reuse the Countdown/AGPL logic so the wheel and source tarball stay identical.
sys.path.insert(0, str(Path(__file__).parent))
from make_source_release import add_years, build_countdown_notice, git  # noqa: E402


def detect_acpp_dir(override: str | None) -> str | None:
    """Locate the AdaptiveCpp CMake config dir, or None to rely on CMAKE_PREFIX_PATH."""
    if override:
        return override
    if os.environ.get("AdaptiveCpp_DIR"):
        return os.environ["AdaptiveCpp_DIR"]
    # Dev-container default (Containerfile installs acpp here).
    devcontainer = Path("/opt/adaptivecpp/lib/cmake/AdaptiveCpp")
    if devcontainer.is_dir():
        return str(devcontainer)
    return None  # let find_package use CMAKE_PREFIX_PATH


def build_wheel(out_dir: Path, components: str, acpp_dir: str | None) -> Path:
    """Run `uv build --wheel` and return the path to the produced wheel."""
    before = set(out_dir.glob("*.whl")) if out_dir.is_dir() else set()

    defines = {
        "ACPP_TARGETS": "generic",
        "EGG_BUNDLE_RUNTIME": "ON",
        "EGG_DEPLOY_COMPONENTS": components,
    }
    if acpp_dir:
        defines["AdaptiveCpp_DIR"] = acpp_dir

    cmd = ["uv", "build", "--wheel", "--out-dir", str(out_dir)]
    for key, val in defines.items():
        cmd += ["-C", f"cmake.define.{key}={val}"]

    print(">> building wheel:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    after = set(out_dir.glob("*.whl"))
    new = after - before
    if not new:
        # Rebuild of an existing version overwrites in place; fall back to newest.
        new = after
    if not new:
        sys.exit("error: no wheel produced in " + str(out_dir))
    return max(new, key=lambda p: p.stat().st_mtime)


# GPU driver stubs that are bound to the target's kernel/driver and so must NOT
# be vendored — the host provides them at runtime. libcuda: NVIDIA driver;
# libdrm*: the DRM userspace behind libhsa-runtime64 (AMD). Everything else acpp
# pulls in (libamdhip64, libhsa-runtime64, libamd_comgr, libcudart, ...) IS
# bundled. auditwheel ignores an exclude for a lib the wheel doesn't reference,
# so this list is safe to pass for any component set (core / cuda / hip).
DEFAULT_EXCLUDES = ["libcuda.so.1", "libdrm.so.2", "libdrm_amdgpu.so.1"]


def repair_wheel(wheel: Path, out_dir: Path, excludes: list[str]) -> Path:
    """Vendor indirect deps + patch RPATHs via auditwheel; return the repaired,
    manylinux-tagged wheel.

    acpp's ``--acpp-deploy`` copies only direct dependencies, so the freshly
    built wheel is missing libLLVM's indirect deps (libedit, libxml2, ...). This
    runs ``auditwheel repair`` to complete the transitive closure and rewrite
    rpaths so the result is self-contained on any host with a new-enough glibc.

    auditwheel and its ``patchelf`` binary are run via ``uv tool run`` so neither
    has to be installed in the image. On success the original (unrepaired,
    ``linux_x86_64``-tagged) wheel is removed, leaving only the portable one.
    """
    before = set(out_dir.glob("*.whl"))
    cmd = [
        "uv",
        "tool",
        "run",
        "--with",
        "patchelf",
        "auditwheel",
        "repair",
        "--wheel-dir",
        str(out_dir),
    ]
    for lib in excludes:
        cmd += ["--exclude", lib]
    cmd.append(str(wheel))

    print(">> repairing wheel:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    new = set(out_dir.glob("*.whl")) - before
    if not new:
        sys.exit("error: auditwheel produced no wheel in " + str(out_dir))
    repaired = max(new, key=lambda p: p.stat().st_mtime)
    if repaired != wheel:
        wheel.unlink()  # drop the non-portable linux_x86_64 wheel
    return repaired


def _record_hash(data: bytes) -> str:
    """RECORD-format hash: ``sha256=<urlsafe-b64, no padding>``."""
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("=")


def inject_countdown(wheel: Path, notice: str) -> str:
    """Add LICENSE-COUNTDOWN.md to the wheel's .dist-info and fix RECORD.

    Returns the arcname of the added file.
    """
    data = notice.encode()
    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        distinfo = next(
            (n.split("/")[0] for n in names if n.endswith(".dist-info/RECORD")), None
        )
        if distinfo is None:
            sys.exit(f"error: no .dist-info/RECORD found in {wheel.name}")
        record_name = f"{distinfo}/RECORD"
        arcname = f"{distinfo}/LICENSE-COUNTDOWN.md"
        old_record = z.read(record_name).decode()

    # Rebuild RECORD: drop any stale entry for our file, insert a fresh line
    # right before RECORD's own (hashless) self-entry.
    new_line = f"{arcname},{_record_hash(data)},{len(data)}\n"
    out_lines, inserted = [], False
    for line in old_record.splitlines(keepends=True):
        if line.startswith(arcname + ","):
            continue  # remove stale entry (idempotent re-runs)
        if line.startswith(record_name + ","):
            out_lines.append(new_line)
            inserted = True
        out_lines.append(line)
    if not inserted:
        out_lines.append(new_line)
    new_record = "".join(out_lines)

    # Rewrite the archive: copy every entry except RECORD/our file, then append
    # our file and the updated RECORD (RECORD must come last by convention).
    tmp = wheel.with_name(wheel.name + ".tmp")
    with (
        zipfile.ZipFile(wheel) as zin,
        zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout,
    ):
        for item in zin.infolist():
            if item.filename in (record_name, arcname):
                continue
            zout.writestr(item, zin.read(item.filename))
        zi = zipfile.ZipInfo(arcname)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = 0o644 << 16
        zout.writestr(zi, data)
        zout.writestr(record_name, new_record)
    tmp.replace(wheel)
    return arcname


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years", type=int, default=4, help="years until AGPL conversion (default: 4)"
    )
    parser.add_argument(
        "--components",
        default="core;cuda;hip",
        help="acpp deploy components (default: core;cuda;hip)",
    )
    parser.add_argument(
        "--acpp-dir",
        default=None,
        help="AdaptiveCpp CMake config dir (default: autodetect)",
    )
    parser.add_argument(
        "--out-dir", default="dist/wheel", help="output directory (default: dist/wheel)"
    )
    parser.add_argument(
        "--skip-repair",
        action="store_true",
        help="skip the auditwheel repair step (produces a non-portable linux_x86_64 wheel)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        metavar="LIB",
        help=f"library to leave host-provided instead of bundling (repeatable; default: {' '.join(DEFAULT_EXCLUDES)})",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="git ref whose date sets the Countdown clock (default: HEAD)",
    )
    args = parser.parse_args()

    os.chdir(Path(git("rev-parse", "--show-toplevel")))

    # Countdown clock: release date = committer date of the ref; AGPL begins +years.
    epoch = int(git("show", "-s", "--format=%ct", args.ref))
    release_date = datetime.fromtimestamp(epoch, tz=timezone.utc).date()
    start_date = add_years(release_date, args.years)
    notice = build_countdown_notice(start_date)

    if (
        args.ref == "HEAD"
        and subprocess.run(["git", "diff", "--quiet", "HEAD"]).returncode
    ):
        print(
            "warning: working tree is dirty; the wheel is built from it but the "
            "Countdown date uses HEAD's commit date",
            file=sys.stderr,
        )

    out_dir = Path(args.out_dir)
    acpp_dir = detect_acpp_dir(args.acpp_dir)
    if acpp_dir is None:
        print(
            "note: AdaptiveCpp_DIR not set; relying on CMAKE_PREFIX_PATH",
            file=sys.stderr,
        )

    wheel = build_wheel(out_dir, args.components, acpp_dir)
    if not args.skip_repair:
        wheel = repair_wheel(wheel, out_dir, args.exclude or DEFAULT_EXCLUDES)
    arcname = inject_countdown(wheel, notice)

    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (out_dir / f"{wheel.name}.sha256").write_text(f"{digest}  {wheel.name}\n")

    print("Built wheel with Countdown notice:")
    print(f"  wheel:         {wheel}")
    print(f"  checksum:      {wheel}.sha256")
    print(f"  components:    {args.components}")
    print(
        f"  repaired:      {'no (--skip-repair)' if args.skip_repair else 'yes (auditwheel)'}"
    )
    print(f"  countdown:     {arcname}")
    print(f"  release date:  {release_date.isoformat()}")
    print(f"  AGPL-3.0 from: {start_date.isoformat()} (release date + {args.years}y)")


if __name__ == "__main__":
    main()
