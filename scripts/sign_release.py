#!/usr/bin/env python3
"""Sign release artifacts with keyless Sigstore (cosign), for manual releases.

Keyless signing uses a short-lived certificate bound to your OAuth identity
(no long-lived key to manage). Running this opens a browser to log in; pick the
provider whose **verified email is `s.imran@tuta.io`** (your GitHub account).
For each artifact it writes ``<artifact>.sig`` and ``<artifact>.pem`` next to it,
and records the signature in Sigstore's public transparency log.

Usage::

    scripts/sign_release.py dist/wheel/egg-*.whl dist/egg-*-src.tar.gz
    scripts/sign_release.py            # defaults to the standard release outputs

Recipients verify with the command this script prints (identity pinned below).
"""

from __future__ import annotations

import glob
import shutil
import subprocess
import sys
from pathlib import Path

# The identity an official Egg release is signed under. Recipients pin BOTH.
# IDENTITY must be the verified email on the OAuth account you sign in with.
IDENTITY = "s.imran@tuta.io"
OIDC_ISSUER = "https://github.com/login/oauth"  # GitHub login (where that email lives)

DEFAULT_GLOBS = ["dist/wheel/*.whl", "dist/*-src.tar.gz"]


def resolve_artifacts(argv: list[str]) -> list[Path]:
    if argv:
        paths = [Path(a) for a in argv]
    else:
        paths = [Path(p) for pat in DEFAULT_GLOBS for p in glob.glob(pat)]
    # Don't sign signatures/certs/checksums themselves.
    paths = [p for p in paths if p.suffix not in (".sig", ".pem", ".sha256")]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        sys.exit("error: not found: " + ", ".join(str(m) for m in missing))
    if not paths:
        sys.exit("error: no artifacts to sign (build them first, or pass paths)")
    return paths


def sign(artifact: Path) -> None:
    sig = artifact.with_name(artifact.name + ".sig")
    cert = artifact.with_name(artifact.name + ".pem")
    print(f">> signing {artifact}")
    subprocess.run(
        [
            "cosign",
            "sign-blob",
            "--yes",
            str(artifact),
            "--output-signature",
            str(sig),
            "--output-certificate",
            str(cert),
        ],
        check=True,
    )


def main() -> None:
    if shutil.which("cosign") is None:
        sys.exit(
            "error: cosign not found. Install it: https://docs.sigstore.dev/cosign/installation/"
        )

    artifacts = resolve_artifacts(sys.argv[1:])
    for a in artifacts:
        sign(a)

    print("\nSigned. Recipients verify each artifact with:\n")
    for a in artifacts:
        print(f"  cosign verify-blob {a.name} \\")
        print(f"    --signature {a.name}.sig --certificate {a.name}.pem \\")
        print(f"    --certificate-identity {IDENTITY} \\")
        print(f"    --certificate-oidc-issuer {OIDC_ISSUER}")
    print(
        "\nPublish the .sig and .pem next to each artifact (e.g. as GitHub Release assets)."
    )


if __name__ == "__main__":
    main()
