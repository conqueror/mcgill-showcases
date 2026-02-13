#!/usr/bin/env python
"""Validate required ranking API model artifacts for local serving."""

from __future__ import annotations

from pathlib import Path

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
            "Missing required model artifacts. Run `make train-demo` first. Missing: "
            f"{missing}"
        )


if __name__ == "__main__":
    main()
