#!/usr/bin/env python3
"""Evaluate random, heuristic, and Q-learning policies on fixed scenarios.

This is the evaluation-and-governance rung: before any policy is trusted, it is scored offline
against held-out scenarios alongside baselines (random, heuristic) so a learned policy must beat
simple references to earn deployment. The script trains Q-learning, runs the three-way
comparison, and writes the comparison tables plus the governance artifacts. See
docs/evaluation-and-governance.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from student_support_rl.evaluation import evaluate_policies
from student_support_rl.policies import HeuristicPolicy, RandomPolicy
from student_support_rl.q_learning import train_q_learning
from student_support_rl.reporting import (
    governance_artifacts,
    write_csv_artifact,
    write_text_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the policy-evaluation runner's command-line flags.

    Exposes ``--output-dir``, ``--episodes`` (Q-learning training length; ``None`` defers to the
    default chosen in ``main``), and ``--quick`` (which shortens both training and the per-scenario
    evaluation count for CI).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the three-way offline policy comparison and write the evaluation artifacts.

    Trains Q-learning, then scores a random baseline, the heuristic policy, and the greedy
    Q-learning policy across fixed scenarios. Writes ``artifacts/eval/policy_comparison.csv``
    (per-policy summary) and ``artifacts/eval/scenario_results.csv`` (per-scenario detail), plus
    ``artifacts/governance/safety_controls.md`` and ``artifacts/governance/offline_eval_plan.md``.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Offline policy evaluation against baselines. See docs/evaluation-and-governance.md.
    """
    args = parse_args(argv)
    # --quick shortens both training and the held-out evaluation budget for CI.
    train_episodes = args.episodes if args.episodes is not None else (600 if args.quick else 5000)
    eval_episodes = 3 if args.quick else 12
    q_result = train_q_learning(episodes=train_episodes)
    summary_rows, scenario_rows = evaluate_policies(
        policies=[
            RandomPolicy(seed=7),
            HeuristicPolicy(),
            q_result.greedy_policy(),
        ],
        scenario_ids=(0, 1, 2, 3, 4),
        episodes_per_scenario=eval_episodes,
    )
    write_csv_artifact(args.output_dir / "eval" / "policy_comparison.csv", summary_rows)
    write_csv_artifact(args.output_dir / "eval" / "scenario_results.csv", scenario_rows)
    governance = governance_artifacts()
    write_text_artifact(
        args.output_dir / "governance" / "safety_controls.md",
        governance["safety_controls"],
    )
    write_text_artifact(
        args.output_dir / "governance" / "offline_eval_plan.md",
        governance["offline_eval_plan"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
