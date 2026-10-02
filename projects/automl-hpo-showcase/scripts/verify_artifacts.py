#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "artifacts/manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Missing artifacts/manifest.json")

    required = json.loads(manifest_path.read_text(encoding="utf-8")).get("required_files")
    if not isinstance(required, list) or not required or not all(
        isinstance(path, str) and path for path in required
    ):
        raise SystemExit("required_files must be a nonempty list of paths")
    missing = [path for path in required if not (root / path).is_file()
               or (root / path).stat().st_size == 0]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    print("All required artifacts exist.")


if __name__ == "__main__":
    main()
