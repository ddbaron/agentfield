#!/usr/bin/env python3
"""Verify the OpenCode executable surface used by profile-managed runs.

This check intentionally probes the executable instead of trusting a package
metadata table. It performs no model request and never prints command output,
which keeps provider credentials and startup diagnostics out of CI logs.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence


MIN_SUPPORTED_VERSION = (1, 18)
REQUIRED_RUN_FLAGS = ("--agent", "--format", "--dir", "--model", "--variant")
VERSION_RE = re.compile(r"(?<!\d)v?(\d+)\.(\d+)(?:\.(\d+))?(?!\d)")


class CapabilityCheckError(RuntimeError):
    """The executable did not expose the required profile run surface."""


def _run(binary: str, arguments: Sequence[str]) -> tuple[str, str]:
    try:
        completed = subprocess.run(
            [binary, *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CapabilityCheckError("the executable could not be probed") from exc
    if completed.returncode != 0:
        raise CapabilityCheckError("the executable rejected a capability probe")
    return completed.stdout, completed.stderr


def verify(binary: str, expected_version: str | None = None) -> str:
    """Return the normalized executable version after validating its surface."""

    version_stdout, version_stderr = _run(binary, ("--version",))
    match = VERSION_RE.search(f"{version_stdout}\n{version_stderr}")
    if match is None:
        raise CapabilityCheckError("version output did not contain a version")

    version = ".".join(part for part in match.groups() if part is not None)
    major, minor = int(match.group(1)), int(match.group(2))
    if (major, minor) < MIN_SUPPORTED_VERSION or major != 1:
        raise CapabilityCheckError("the executable version is unsupported")
    if expected_version is not None and version != expected_version:
        raise CapabilityCheckError("the executable version does not match the pin")

    help_stdout, help_stderr = _run(binary, ("run", "--help"))
    help_text = f"{help_stdout}\n{help_stderr}"
    missing = [flag for flag in REQUIRED_RUN_FLAGS if flag not in help_text]
    if missing:
        raise CapabilityCheckError("the executable lacks required run flags")
    return version


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="opencode")
    parser.add_argument(
        "--expected-version",
        help="require an exact normalized version in addition to the minimum",
    )
    args = parser.parse_args(argv)

    try:
        version = verify(args.binary, args.expected_version)
    except CapabilityCheckError as exc:
        print(f"OpenCode capability check failed: {exc}", file=sys.stderr)
        return 1

    print(f"OpenCode {version}: run profile capability verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
