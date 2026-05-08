#!/usr/bin/env python
"""Run the offline agentic course assistant demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_course_assistant import answer_question
from agentic_course_assistant.artifacts import write_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Student question to route and answer.")
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        type=Path,
        help="Directory for generated showcase artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = answer_question(args.question)
    written = write_artifacts(result, args.output_dir)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
