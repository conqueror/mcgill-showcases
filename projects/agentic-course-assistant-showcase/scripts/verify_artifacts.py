#!/usr/bin/env python
"""Validate the course assistant artifact bundle."""

import argparse
from pathlib import Path

from agentic_course_assistant.artifact_contract import verify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help=(
            "Bundle root to verify. Defaults to the project root, but opt-in live bundles can "
            "pass a separate namespace such as artifacts/live_openai."
        ),
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Verify only the base artifact contract and skip harness artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    errors = verify(root, require_harness=not args.base_only)
    if errors:
        raise SystemExit("; ".join(errors))

    print("Artifact contract verified.")


if __name__ == "__main__":
    main()
