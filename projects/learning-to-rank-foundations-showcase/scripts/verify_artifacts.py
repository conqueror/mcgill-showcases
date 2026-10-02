#!/usr/bin/env python
"""Validate required artifacts declared in ``artifacts/manifest.json``."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    """Fail fast if any required ranking artifact is missing."""

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "artifacts/manifest.json").read_text(encoding="utf-8"))
    required = manifest.get("required_files", [])
    if (
        not isinstance(required, list)
        or not required
        or any(not isinstance(rel, str) or not rel.strip() for rel in required)
    ):
        raise SystemExit("Manifest must list nonempty required artifact paths.")
    missing = [
        rel for rel in required if not (root / rel).is_file() or (root / rel).stat().st_size == 0
    ]
    if missing:
        raise SystemExit(f"Missing or empty required artifacts: {missing}")
    for rel in required:
        path = root / rel
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not payload:
                raise SystemExit(f"Empty or invalid JSON artifact: {rel}")
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, strict=True))
            if (
                len(rows) < 2
                or not rows[0]
                or not any(rows[0])
                or not all(rows[0][1:])
                or any(len(row) != len(rows[0]) for row in rows[1:])
            ):
                raise SystemExit(f"Empty or invalid CSV artifact: {rel}")


if __name__ == "__main__":
    main()
