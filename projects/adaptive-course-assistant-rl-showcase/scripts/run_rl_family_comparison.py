#!/usr/bin/env python3
"""Compare random, rule-based, and learned policies on the same scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_course_assistant_rl.config import (
    DEFAULT_EVAL_EPISODES,
    DEFAULT_Q_EPISODES,
    DEFAULT_REINFORCE_EPISODES,
    QUICK_EVAL_EPISODES,
    QUICK_Q_EPISODES,
    QUICK_REINFORCE_EPISODES,
)
from adaptive_course_assistant_rl.evaluation import evaluate_policies
from adaptive_course_assistant_rl.policies import RandomPolicy, RuleBasedPolicy
from adaptive_course_assistant_rl.policy_gradient import train_reinforce
from adaptive_course_assistant_rl.q_learning import train_q_learning
from adaptive_course_assistant_rl.reporting import (
    deployment_recommendation_markdown,
    safety_summary_markdown,
    write_csv_artifact,
    write_text_artifact,
)
from adaptive_course_assistant_rl.sarsa import train_sarsa

COMPARISON_TRAIN_SEED = 23
COMPARISON_EVAL_BASE_SEED = 123


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    q_result = train_q_learning(
        episodes=QUICK_Q_EPISODES if args.quick else DEFAULT_Q_EPISODES,
        seed=COMPARISON_TRAIN_SEED,
    )
    sarsa_result = train_sarsa(
        episodes=QUICK_Q_EPISODES if args.quick else DEFAULT_Q_EPISODES,
        seed=COMPARISON_TRAIN_SEED,
    )
    reinforce_result = train_reinforce(
        episodes=QUICK_REINFORCE_EPISODES if args.quick else DEFAULT_REINFORCE_EPISODES,
        seed=COMPARISON_TRAIN_SEED,
    )
    summary_rows, scenario_rows = evaluate_policies(
        policies=[
            RandomPolicy(seed=7),
            RuleBasedPolicy(),
            q_result.greedy_policy(),
            sarsa_result.greedy_policy(),
            reinforce_result.greedy_policy(),
        ],
        scenario_ids=(0, 1, 2, 3, 4),
        episodes_per_scenario=QUICK_EVAL_EPISODES if args.quick else DEFAULT_EVAL_EPISODES,
        base_seed=COMPARISON_EVAL_BASE_SEED,
    )
    write_csv_artifact(args.output_dir / "eval" / "offline_policy_eval.csv", summary_rows)
    write_csv_artifact(args.output_dir / "eval" / "scenario_rollups.csv", scenario_rows)
    write_text_artifact(args.output_dir / "eval" / "safety_summary.md", safety_summary_markdown(summary_rows))
    write_text_artifact(
        args.output_dir / "business" / "deployment_recommendation.md",
        deployment_recommendation_markdown(summary_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
