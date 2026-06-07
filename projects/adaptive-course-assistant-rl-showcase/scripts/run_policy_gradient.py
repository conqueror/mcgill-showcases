#!/usr/bin/env python3
"""Train the small REINFORCE path."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_course_assistant_rl.config import DEFAULT_REINFORCE_EPISODES, QUICK_REINFORCE_EPISODES
from adaptive_course_assistant_rl.policy_gradient import train_reinforce
from adaptive_course_assistant_rl.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = train_reinforce(
        episodes=QUICK_REINFORCE_EPISODES if args.quick else DEFAULT_REINFORCE_EPISODES
    )
    write_csv_artifact(
        args.output_dir / "policy_gradient" / "training_curve.csv",
        result.training_curve,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
