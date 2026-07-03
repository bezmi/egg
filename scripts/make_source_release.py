#!/usr/bin/env python3
# Required Notice: Copyright (c) Shahzeb Imran and Egg contributors
#
# PolyForm Noncommercial License 2.0.0-pre.2
# https://github.com/bezmi/egg/blob/main/LICENSE.md
# Free to use and redistribute for personal and noncommercial purposes.
# See the license for details.
# For commercial licensing, contact s.imran@tuta.io

"""Bundle a source-only release of Egg with a PolyForm Countdown notice.

The bundle contains exactly the git-tracked files of the chosen ref (so
``build/``, ``_deps/`` and other untracked/ignored files are excluded), plus a
generated ``LICENSE-COUNTDOWN.md``. The Countdown "Start Date" -- the date the
AGPL-3.0 terms begin -- is set to the ref's commit date plus ``--years``.

Usage::

    scripts/make_source_release.py [REF] [--years N] [--out-dir DIR]

    REF   git tag, branch, or commit to release (default: HEAD).

Examples::

    scripts/make_source_release.py v0.1.0
    scripts/make_source_release.py --years 3
"""

from __future__ import annotations

import argparse
import hashlib
import io
import subprocess
import sys
import tarfile
from datetime import date, datetime, timezone
from pathlib import Path

# Verbatim text of the license that releases convert to (the Countdown target).
# This is NOT the project's current license -- that is PolyForm Noncommercial,
# in /LICENSE.md. This file is input data for the release tool only; its
# contents are embedded into each release's LICENSE-COUNTDOWN.md.
NEW_LICENSE = Path(__file__).parent / "conversion-target-AGPL-3.0.txt"
PLACEHOLDER = "{Copy the scheduled license terms here.}"

# PolyForm Countdown License Grant 1.0.0
# <https://polyformproject.org/licenses/countdown/1.0.0>
# {start date} is filled with the AGPL-3.0 start date; the placeholder line is
# replaced with the verbatim contents of NEW_LICENSE.
COUNTDOWN_TEMPLATE = """\
# PolyForm Countdown License Grant

Version 1.0.0

<https://polyformproject.org/licenses/countdown/1.0.0>

## Start Date

{start date} (ISO 8601-1:2019)

## License

Each contributor licenses this release to you on the new
license terms below, starting at 12:00 noon UTC on the
start date above.

## Scope

This license grant applies only to this release, not to
any other releases.  Other releases may come with their
own license grants.

## Reliability

No contributor can revoke the new license before it starts.
If the new license terms allow a contributor to revoke,
they can do so only after the new license starts.

## Legalities

Legally, this is a present grant of a license on the date of
release, not a contract promise to grant the license later.

## New License Terms

{Copy the scheduled license terms here.}
"""


def git(*args: str) -> str:
    """Run a git command and return its stripped stdout."""
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def add_years(d: date, years: int) -> date:
    """Return ``d`` shifted by ``years``, clamping Feb 29 to Feb 28."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 Feb -> non-leap year
        return d.replace(year=d.year + years, day=28)


def build_countdown_notice(start_date: date) -> str:
    """Fill the Countdown template: start date + embedded AGPL terms."""
    agpl = NEW_LICENSE.read_text()
    notice = COUNTDOWN_TEMPLATE.replace("{start date}", start_date.isoformat())
    return notice.replace(PLACEHOLDER, agpl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ref", nargs="?", default="HEAD", help="git ref to release")
    parser.add_argument(
        "--years", type=int, default=4, help="years until AGPL conversion (default: 4)"
    )
    parser.add_argument(
        "--out-dir", default="dist", help="output directory (default: dist)"
    )
    args = parser.parse_args()

    repo_root = Path(git("rev-parse", "--show-toplevel"))
    import os

    os.chdir(repo_root)

    if not NEW_LICENSE.is_file():
        sys.exit(f"error: required file not found: {NEW_LICENSE}")

    # Verify the ref resolves to a commit.
    try:
        git("rev-parse", "--verify", f"{args.ref}^{{commit}}")
    except subprocess.CalledProcessError:
        sys.exit(f"error: not a valid git ref: {args.ref}")

    try:
        version = git("describe", "--tags", "--always", args.ref)
    except subprocess.CalledProcessError:
        version = git("rev-parse", "--short", args.ref)

    # Release date = committer date of the ref, in UTC.
    epoch = int(git("show", "-s", "--format=%ct", args.ref))
    release_date = datetime.fromtimestamp(epoch, tz=timezone.utc).date()
    start_date = add_years(release_date, args.years)

    # Warn if releasing HEAD with uncommitted changes (git archive captures
    # only the committed tree, so local edits are silently excluded).
    if (
        args.ref == "HEAD"
        and subprocess.run(["git", "diff", "--quiet", "HEAD"]).returncode
    ):
        print(
            "warning: working tree has uncommitted changes; they will NOT be included",
            file=sys.stderr,
        )

    prefix = f"egg-{version}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tarball = out_dir / f"{prefix}-src.tar.gz"

    # Source tree of the ref as a tar stream, with the release prefix.
    base_tar = git_archive(args.ref, prefix)

    notice = build_countdown_notice(start_date).encode()
    mtime = datetime(
        release_date.year, release_date.month, release_date.day, tzinfo=timezone.utc
    ).timestamp()

    # Repack: copy tracked entries, append the generated Countdown notice.
    with tarfile.open(tarball, "w:gz") as out:
        with tarfile.open(fileobj=io.BytesIO(base_tar), mode="r:") as src:
            for member in src.getmembers():
                out.addfile(
                    member, src.extractfile(member) if member.isfile() else None
                )
        info = tarfile.TarInfo(f"{prefix}/LICENSE-COUNTDOWN.md")
        info.size = len(notice)
        info.mtime = int(mtime)
        info.mode = 0o644
        out.addfile(info, io.BytesIO(notice))

    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    (out_dir / f"{tarball.name}.sha256").write_text(f"{digest}  {tarball.name}\n")

    print("Created source release:")
    print(f"  artifact:      {tarball}")
    print(f"  checksum:      {tarball}.sha256")
    print(f"  ref:           {args.ref} ({version})")
    print(f"  release date:  {release_date.isoformat()}")
    print(f"  AGPL-3.0 from: {start_date.isoformat()} (release date + {args.years}y)")


def git_archive(ref: str, prefix: str) -> bytes:
    """Return a tar (uncompressed) of the tracked tree of ``ref``."""
    return subprocess.run(
        ["git", "archive", "--format=tar", f"--prefix={prefix}/", ref],
        check=True,
        capture_output=True,
    ).stdout


if __name__ == "__main__":
    main()
