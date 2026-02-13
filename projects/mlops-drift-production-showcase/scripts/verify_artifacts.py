#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "artifacts/manifest.json"

    if not manifest_path.exists():
        raise SystemExit("Missing manifest: artifacts/manifest.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_files = manifest.get("required_files", [])
    if not isinstance(required_files, list):
        raise SystemExit("Manifest field 'required_files' must be a list")

    missing: list[str] = []
    for file_name in required_files:
        if not (root / file_name).exists():
            missing.append(file_name)

    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    print("All required artifacts exist.")


if __name__ == "__main__":
    main()
