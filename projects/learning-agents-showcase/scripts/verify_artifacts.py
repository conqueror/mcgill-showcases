#!/usr/bin/env python3
"""Verify the showcase artifact contract (presence + shape)."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from learning_agents.reporting import artifact_validation_errors, missing_required_artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the artifact-verifier's command-line flags.

    What + why: this is the reproducibility gate that CI runs after the pipeline, so its flags only
    need to locate the evidence to check. It exposes ``--output-dir`` (project root holding the
    ``artifacts`` directory, default ``.``) and ``--manifest`` (the contract file to enforce,
    default ``artifacts/manifest.json``).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.

    RL concept:
        Reproducibility and the artifact contract -- the verifier guards evidence, it has no policy.
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
        default=Path("artifacts/manifest.json"),
        help="Manifest path relative to the project root or as an absolute path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Check the artifact contract and report any missing or invalid files.

    What + why: a broken or partial run must fail loudly instead of shipping silently, so this gate
    runs after the other runners produce evidence. It first lists required artifacts that are
    absent; if any are missing it prints them and returns ``1``. Otherwise it runs content
    validation and, if any check fails, prints the errors and returns ``1``. When everything is
    present and valid it prints a success line and returns ``0``. It writes no artifacts.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code: ``0`` when the contract holds, ``1`` on missing or invalid artifacts.

    RL concept:
        Reproducibility and the artifact contract -- completeness and shape gate the showcase.
    """
    args = parse_args(argv)
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = args.output_dir / manifest_path
    if not manifest_path.exists():
        print(f"Manifest file is missing: {manifest_path}")
        return 1

    missing = missing_required_artifacts(
        output_dir=args.output_dir,
        manifest_path=manifest_path,
    )
    if missing:
        print("Missing required artifacts:")
        for relative_path in missing:
            print(f"- {relative_path}")
        return 1

    validation_errors = artifact_validation_errors(
        output_dir=args.output_dir,
        manifest_path=manifest_path,
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
