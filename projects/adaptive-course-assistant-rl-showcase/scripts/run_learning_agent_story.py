#!/usr/bin/env python3
"""Write one concrete learning-agent bridge artifact for students."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_course_assistant_rl.learning_agent_story import build_learning_agent_story
from adaptive_course_assistant_rl.reporting import write_text_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--scenario-id", type=int, default=1)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    story = build_learning_agent_story(
        quick=args.quick,
        scenario_id=args.scenario_id,
    )
    write_text_artifact(
        args.output_dir / "bridge" / "learning_agent_story.md",
        story.markdown,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
