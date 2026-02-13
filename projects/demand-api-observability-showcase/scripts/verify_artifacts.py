#!/usr/bin/env python
"""Validate required demand API artifacts and OpenAPI contract file."""

from __future__ import annotations

from pathlib import Path

REQUIRED_FILES = [
    "artifacts/model.joblib",
    "artifacts/metrics.json",
    "openapi.json",
]


def main() -> None:
    """Fail with missing-path details when required outputs are absent."""

    root = Path(__file__).resolve().parents[1]
    missing = [rel for rel in REQUIRED_FILES if not (root / rel).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")


if __name__ == "__main__":
    main()
