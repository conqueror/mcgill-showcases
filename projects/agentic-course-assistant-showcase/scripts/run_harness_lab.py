#!/usr/bin/env python
"""Generate public-safe harness lab artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_course_assistant.harness_lab import DEFAULT_HARNESS_QUESTION, run_harness_lab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default=DEFAULT_HARNESS_QUESTION,
        help="Student question used to generate deterministic harness artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    summary = run_harness_lab(project_root, question=args.question)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["verification_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
