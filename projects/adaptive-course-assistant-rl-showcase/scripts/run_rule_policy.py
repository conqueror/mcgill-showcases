#!/usr/bin/env python3
"""Evaluate the fixed rule-based baseline before any learning happens."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_course_assistant_rl.agent_bridge import intervention_decision_rows
from adaptive_course_assistant_rl.config import DEFAULT_SCENARIO_IDS, QUICK_EVAL_EPISODES
from adaptive_course_assistant_rl.evaluation import evaluate_policies
from adaptive_course_assistant_rl.policies import RandomPolicy, RuleBasedPolicy
from adaptive_course_assistant_rl.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary_rows, _ = evaluate_policies(
        policies=[RandomPolicy(seed=7), RuleBasedPolicy()],
        scenario_ids=DEFAULT_SCENARIO_IDS,
        episodes_per_scenario=QUICK_EVAL_EPISODES if args.quick else 4,
    )
    write_csv_artifact(args.output_dir / "policy" / "rule_policy_summary.csv", summary_rows)
    write_csv_artifact(
        args.output_dir / "policy" / "intervention_decisions.csv",
        intervention_decision_rows(summary_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
