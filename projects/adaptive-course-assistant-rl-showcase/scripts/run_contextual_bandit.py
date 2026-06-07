#!/usr/bin/env python3
"""Run the first-turn contextual bandit experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_course_assistant_rl.contextual_bandit import run_bandit_experiment
from adaptive_course_assistant_rl.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_bandit_experiment(steps=120 if args.quick else 400)
    write_csv_artifact(
        args.output_dir / "bandit" / "contextual_policy_metrics.csv",
        result.metrics_rows,
    )
    write_csv_artifact(args.output_dir / "bandit" / "regret_trace.csv", result.regret_rows)
    write_csv_artifact(args.output_dir / "bandit" / "action_breakdown.csv", result.action_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
