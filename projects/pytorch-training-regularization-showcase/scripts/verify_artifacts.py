#!/usr/bin/env python3
"""Artifact verification entry point for the PyTorch showcase."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pytorch_training_regularization_showcase import config

DEFAULT_REQUIRED_ARTIFACTS = (
    "baseline_metrics.json",
    "training_history.csv",
    "optimizer_comparison.csv",
    "learning_rate_schedule_comparison.csv",
    "regularization_ablation.csv",
    "gradient_health_report.md",
    "error_analysis.csv",
    "summary.md",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for artifact verification."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.ARTIFACTS_DIR,
        help="Directory containing generated artifacts.",
    )
    return parser.parse_args(argv)


def required_artifact_files(output_dir: Path | None = None) -> list[str]:
    """Read the artifact manifest and return the required filenames."""

    manifest_path = (
        config.ARTIFACTS_DIR if output_dir is None else output_dir
    ) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["required_files"]
    if (
        not isinstance(files, list)
        or len(files) != len(DEFAULT_REQUIRED_ARTIFACTS)
        or set(files) != set(DEFAULT_REQUIRED_ARTIFACTS)
    ):
        raise ValueError("Manifest does not match the required artifact contract.")
    return files


def main(argv: list[str] | None = None) -> int:
    """Reject incomplete, empty, or changed outputs using the run manifest."""

    args = parse_args(argv)
    output_dir = args.output_dir
    try:
        required = required_artifact_files(output_dir)
        manifest = json.loads(
            (output_dir / "manifest.json").read_text(encoding="utf-8")
        )
        hashes = manifest["sha256"]
        if not isinstance(hashes, dict) or set(hashes) != set(required):
            raise ValueError("Manifest must hash every required output.")
        for name in required:
            content = (output_dir / name).read_bytes()
            if (
                not content.strip()
                or hashlib.sha256(content).hexdigest() != hashes[name]
            ):
                raise ValueError(f"Empty or corrupt artifact: {name}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Artifact verification failed: {error}")
        return 1

    print("All required artifacts match the run manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
