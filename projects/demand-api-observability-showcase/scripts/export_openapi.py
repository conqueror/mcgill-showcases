#!/usr/bin/env python
"""Export or check demand API OpenAPI contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from demand_api_observability_showcase.api.app import create_app


def parse_args() -> argparse.Namespace:
    """Parse CLI options for OpenAPI export and drift checking."""

    parser = argparse.ArgumentParser(description="Export or verify OpenAPI contract")
    parser.add_argument("--output", type=Path, default=Path("openapi.json"))
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Write OpenAPI spec or validate existing file for drift."""

    args = parse_args()
    app = create_app()
    spec = app.openapi()

    if args.check:
        if not args.output.exists():
            raise SystemExit(f"{args.output} does not exist. Run: make export-openapi")
        current = json.loads(args.output.read_text(encoding="utf-8"))
        if current != spec:
            raise SystemExit("OpenAPI drift detected. Run: make export-openapi")
        print("OpenAPI is up to date.")
        return

    args.output.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
