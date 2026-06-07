#!/usr/bin/env python3
"""Estimate several target policies' values from one behaviour log (OPE) and grade vs truth."""

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
from learning_agents.offline_rl import collect_logged_dataset
from learning_agents.ope import ope_report_rows
from learning_agents.policies import HeuristicRouterPolicy, Policy, QTablePolicy, RandomPolicy
from learning_agents.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the off-policy-evaluation runner's command-line flags.

    What + why: OPE estimates a target policy's value from a behaviour log without running the
    target, so the flags govern the log and the ground-truth budget. Exposes ``--output-dir``,
    ``--episodes`` (logged episodes; ``None`` defers to the default), and ``--quick`` (a smaller log
    and ground-truth budget for CI).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace with ``output_dir``, ``episodes``, and ``quick``.

    RL concept:
        Off-policy evaluation -- vetting a candidate policy from logged data only.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run OPE for three targets on one behaviour log and write the estimator-accuracy table.

    What + why: builds a single epsilon-soft heuristic-router log, then estimates the value of three
    targets from it with all four estimators (IS, WIS, FQE direct method, doubly-robust): the
    heuristic router itself (well covered), the DP optimum (a different, better policy the log still
    covers -- the real "vet before deploy" case), and a uniform-random policy (far off-policy and
    poorly covered). Each estimate is graded against the simulator's true value. Writes
    ``artifacts/ope/estimator_comparison.csv``, whose ``abs_error`` column makes the central lesson
    visible: OPE is trustworthy in-support and unreliable when behaviour/target overlap is poor.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Off-policy evaluation accuracy as a function of behaviour/target overlap.
    """
    args = parse_args(argv)
    # --quick shrinks both the log and the ground-truth rollout budget for CI.
    episodes = args.episodes if args.episodes is not None else (300 if args.quick else 800)
    truth_episodes = 12 if args.quick else 40
    # epsilon=0.3 keeps the log close to the deterministic targets so importance weights survive.
    dataset = collect_logged_dataset(episodes=episodes, epsilon=0.3, seed=7)

    targets: list[tuple[str, Policy]] = [
        ("heuristic_router", HeuristicRouterPolicy()),
        ("dp_optimal", QTablePolicy(q_table=optimal_action_values(), name="dp_optimal")),
        ("random", RandomPolicy(seed=1)),
    ]
    rows = ope_report_rows(dataset, targets, episodes_per_scenario=truth_episodes)
    write_csv_artifact(args.output_dir / "ope" / "estimator_comparison.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
