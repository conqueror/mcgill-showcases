#!/usr/bin/env python
"""Run the offline agentic course assistant demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_course_assistant.harness_lab import DEFAULT_HARNESS_QUESTION, write_showcase_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default=DEFAULT_HARNESS_QUESTION,
        help="Student question to route and answer.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        type=Path,
        help="Directory for generated showcase artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_showcase_artifacts(args.question, args.output_dir)
    for path in summary["written_files"]:
        print(path)


if __name__ == "__main__":
    main()
