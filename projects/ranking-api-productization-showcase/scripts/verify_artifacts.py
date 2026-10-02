#!/usr/bin/env python
"""Validate required ranking API model artifacts for local serving."""

from __future__ import annotations

from pathlib import Path

from ranking_api_showcase.model.artifacts import load_artifacts

REQUIRED_FILES = [
    "artifacts/model.txt",
    "artifacts/feature_names.json",
    "artifacts/model_meta.json",
]


def main() -> None:
    """Fail with actionable guidance when required model files are missing."""

    root = Path(__file__).resolve().parents[1]
    missing = [rel for rel in REQUIRED_FILES if not (root / rel).exists()]
    if missing:
        raise SystemExit(
            f"Missing required model artifacts. Run `make train-demo` first. Missing: {missing}"
        )
    try:
        for name in REQUIRED_FILES:
            path = root / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Empty or invalid artifact: {name}")
        artifacts = load_artifacts(
            root / "artifacts/model.txt",
            root / "artifacts/feature_names.json",
            root / "artifacts/model_meta.json",
        )
        if not artifacts.meta:
            raise ValueError("Model metadata must be a nonempty JSON object")
    except Exception as exc:
        raise SystemExit(f"Invalid artifact contents: {exc}") from exc


if __name__ == "__main__":
    main()
