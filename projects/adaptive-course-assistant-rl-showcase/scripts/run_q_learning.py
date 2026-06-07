#!/usr/bin/env python3
"""Train tabular Q-learning and write its core artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_course_assistant_rl.config import DEFAULT_Q_EPISODES, QUICK_Q_EPISODES
from adaptive_course_assistant_rl.q_learning import q_table_rows, train_q_learning
from adaptive_course_assistant_rl.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = train_q_learning(episodes=QUICK_Q_EPISODES if args.quick else DEFAULT_Q_EPISODES)
    write_csv_artifact(args.output_dir / "q_learning" / "training_curve.csv", result.training_curve)
    write_csv_artifact(args.output_dir / "q_learning" / "q_table.csv", q_table_rows(result.q_table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
