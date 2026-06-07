#!/usr/bin/env python
"""Run the hosted OpenAI teaching bundle for the course assistant showcase."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from agentic_course_assistant.openai_live_artifacts import run_live_openai_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Student question to route and answer.")
    parser.add_argument(
        "--output-root",
        default=Path("artifacts/live_openai"),
        type=Path,
        help="Bundle root for the hosted artifact set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = asyncio.run(run_live_openai_bundle(args.question, args.output_root))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["verification_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
