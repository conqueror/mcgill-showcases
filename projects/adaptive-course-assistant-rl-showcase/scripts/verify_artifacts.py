#!/usr/bin/env python3
"""Verify the adaptive course assistant RL artifact contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_course_assistant_rl.reporting import (
    artifact_validation_errors,
    missing_required_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Project root or artifact directory to validate.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--require-optional-drl",
        action="store_true",
        help="Require the DQN/PPO optional artifact bundle in addition to the core contract.",
    )
    return parser.parse_args(argv)


def _resolve_project_root(
    output_dir: Path,
    manifest_path: Path | None,
) -> tuple[Path, Path | None, str | None]:
    """Accept either a project root or a direct artifact directory."""
    if manifest_path is not None:
        resolved_manifest = manifest_path if manifest_path.is_absolute() else output_dir / manifest_path
        if not resolved_manifest.exists():
            return output_dir, resolved_manifest, f"Manifest file is missing: {resolved_manifest}"
        return output_dir, resolved_manifest, None

    if (output_dir / "manifest.json").exists():
        return output_dir.parent, output_dir / "manifest.json", None

    if (output_dir / "artifacts" / "manifest.json").exists():
        return output_dir, output_dir / "artifacts" / "manifest.json", None

    expected_manifest = output_dir / "artifacts" / "manifest.json"
    return output_dir, expected_manifest, f"Manifest file is missing: {expected_manifest}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root, manifest_path, manifest_error = _resolve_project_root(args.output_dir, args.manifest)
    if manifest_error is not None:
        print(manifest_error)
        return 1
    missing = missing_required_artifacts(output_dir=project_root, manifest_path=manifest_path)
    if missing:
        print("Missing required artifacts:")
        for relative_path in missing:
            print(f"- {relative_path}")
        return 1
    errors = artifact_validation_errors(
        output_dir=project_root,
        manifest_path=manifest_path,
        require_optional_drl=args.require_optional_drl,
    )
    if errors:
        print("Artifact validation errors:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("All required artifacts are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
