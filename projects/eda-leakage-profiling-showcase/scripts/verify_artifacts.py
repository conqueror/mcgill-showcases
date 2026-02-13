#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "artifacts/manifest.json").read_text(encoding="utf-8"))
    required = manifest.get("required_files", [])
    missing = [rel for rel in required if not (root / rel).exists()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")


if __name__ == "__main__":
    main()
