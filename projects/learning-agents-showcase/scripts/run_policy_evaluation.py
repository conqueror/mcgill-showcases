#!/usr/bin/env python3
"""Evaluate random, heuristic-router, and Q-learning policies on fixed agent scenarios."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for candidate in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from learning_agents.dynamic_programming import optimal_action_values
from learning_agents.evaluation import evaluate_policies
from learning_agents.offline_rl import collect_logged_dataset, fitted_q_iteration
from learning_agents.policies import HeuristicRouterPolicy, QTablePolicy, RandomPolicy
from learning_agents.q_learning import train_q_learning
from learning_agents.reporting import (
    governance_artifacts,
    write_csv_artifact,
    write_text_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the policy-evaluation runner's command-line flags.

    What + why: this is the evaluation-and-governance rung -- before a learned routing policy is
    trusted it must be scored offline against fixed scenarios alongside cheap baselines (random,
    heuristic router), so the agent only earns deployment by beating simple references. The flags
    expose ``--output-dir`` (where the eval/governance artifacts land), ``--episodes`` (Q-learning
    training length; ``None`` defers to the default chosen in ``main``), and ``--quick`` (which
    shortens both training and the per-scenario evaluation budget for CI).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace with ``output_dir``, ``episodes``, and ``quick``.

    RL concept:
        Offline policy evaluation against baselines on the value-based rung of the ladder.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train Q-learning, run the four-way offline comparison, and write the eval artifacts.

    What + why: trains a tabular Q-learning router, then re-simulates four policies across the fixed
    scenarios -- a uniform-random baseline, the hand-written heuristic router, the greedy learned
    policy, and the DP planning optimum (greedy over exact backward-induction Q*) -- so the learned
    agent must out-score the cheap references to justify its complexity, and its shortfall against
    the model-based ceiling is visible in the same table. The learned policy is deliberately wrapped
    as ``QTablePolicy(q_table=..., name="q_learning")`` (rather than ``q_result.greedy_policy()``,
    which is named ``"q_table"``) so the downstream deploy/shadow/reject rule can find a
    ``"q_learning"`` row beside ``"heuristic_router"``. Writes
    ``artifacts/eval/policy_comparison.csv`` (per-policy summary) and
    ``artifacts/eval/scenario_results.csv`` (per-scenario detail), plus the governance notes
    ``artifacts/governance/safety_controls.md`` and ``artifacts/governance/offline_eval_plan.md``.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Simulator-based offline policy evaluation against baselines (value-based control).
    """
    args = parse_args(argv)
    # --quick shortens both training and the held-out evaluation budget for CI.
    episodes = args.episodes if args.episodes is not None else (120 if args.quick else 400)
    eval_episodes = 3 if args.quick else 12
    offline_episodes = 200 if args.quick else 600
    q_result = train_q_learning(episodes=episodes)
    # CRITICAL: name MUST be "q_learning" so recommendation_from_summary pairs this learned row
    # with the "heuristic_router" baseline; greedy_policy() would name it "q_table" instead.
    learned = QTablePolicy(q_table=q_result.q_table, name="q_learning")
    # Offline RL policy: Fitted-Q Iteration on a fixed behaviour log (no new interaction). Including
    # it contrasts online vs offline value-based control in the same comparison.
    offline_dataset = collect_logged_dataset(episodes=offline_episodes, epsilon=0.6, seed=7)
    offline_fqi = QTablePolicy(
        q_table=fitted_q_iteration(offline_dataset, gamma=0.9).q_table,
        name="offline_fqi",
    )
    # Planning ceiling: the greedy policy from exact backward-induction Q* (model-based optimum).
    # Including it shows the full ladder -- random < online Q-learning < heuristic < offline FQI ~
    # DP optimum -- so the learned policies' standing against the planned best is clear at a glance.
    dp_optimal = QTablePolicy(q_table=optimal_action_values(), name="dp_optimal")
    summary_rows, scenario_rows = evaluate_policies(
        policies=[
            RandomPolicy(seed=7),
            HeuristicRouterPolicy(),
            learned,
            offline_fqi,
            dp_optimal,
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
