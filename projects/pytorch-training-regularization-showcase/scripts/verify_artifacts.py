#!/usr/bin/env python3
"""Artifact verification entry point for the PyTorch showcase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pytorch_training_regularization_showcase import config


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


def required_artifact_files() -> list[str]:
    """Read the artifact manifest and return the required filenames."""

    manifest_path = config.ARTIFACTS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [Path(path).name for path in manifest["required_files"]]


def main(argv: list[str] | None = None) -> int:
    """Verify that all required artifact files exist."""

    args = parse_args(argv)
    output_dir = args.output_dir
    missing = [
        artifact_name
        for artifact_name in required_artifact_files()
        if not (output_dir / artifact_name).exists()
    ]

    if missing:
        print("Missing required artifacts:")
        for artifact_name in missing:
            print(f"- {artifact_name}")
        return 1

    print("All required artifacts are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
