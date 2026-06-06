#!/usr/bin/env python3
"""Verify the showcase artifact contract.

This is the reproducibility gate that CI runs after the pipeline: it checks that every required
artifact exists and validates their contents against a manifest, so a broken or partial run fails
loudly instead of shipping silently. It implements no RL itself; it guards the evidence the other
runners produce. See docs/evaluation-and-governance.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from student_support_rl.reporting import artifact_validation_errors, missing_required_artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the artifact-verifier's command-line flags.

    Exposes ``--output-dir`` (project root holding the ``artifacts`` directory, default ``.``) and
    ``--manifest`` (an optional contract file; the built-in manifest is used when omitted).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Project root containing the artifacts directory.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest path. Falls back to the built-in contract when omitted.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Check the artifact contract and report any missing or invalid files.

    First lists required artifacts that are absent; if any are missing it prints them and returns
    ``1``. Otherwise it runs content validation and, if any check fails, prints the errors and
    returns ``1``. When everything is present and valid it prints a success line and returns
    ``0``. This script writes no artifacts; it only verifies them.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code: ``0`` when the contract holds, ``1`` on missing or invalid artifacts.

    RL concept:
        Reproducibility and the artifact contract. See docs/evaluation-and-governance.md.
    """
    args = parse_args(argv)
    missing = missing_required_artifacts(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    if missing:
        print("Missing required artifacts:")
        for relative_path in missing:
            print(f"- {relative_path}")
        return 1

    validation_errors = artifact_validation_errors(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
    )
    if validation_errors:
        print("Artifact validation errors:")
        for message in validation_errors:
            print(f"- {message}")
        return 1

    print("All required artifacts are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
