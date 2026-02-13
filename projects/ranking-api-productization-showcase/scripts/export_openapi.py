#!/usr/bin/env python
"""Export ranking API OpenAPI schema for contract checks and docs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_api_showcase.api.app import create_app


def main() -> None:
    """Generate and write the ranking API OpenAPI JSON document."""

    parser = argparse.ArgumentParser(description="Export OpenAPI JSON")
    parser.add_argument("--out", type=Path, default=Path("openapi.json"))
    args = parser.parse_args()

    app = create_app(load_model=False)
    args.out.write_text(
        json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
