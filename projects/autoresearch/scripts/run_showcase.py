#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from autoresearch_showcase.reporting import build_showcase


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    written = build_showcase(project_root)
    print(f"Wrote {len(written)} artifact files.")


if __name__ == "__main__":
    main()
