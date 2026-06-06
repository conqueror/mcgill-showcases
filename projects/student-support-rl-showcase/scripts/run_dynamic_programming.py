#!/usr/bin/env python3
"""Compute exact optimal Q* and the learned-vs-optimal gap, then write the DP artifacts.

This is the planning rung: it solves the known finite-horizon MDP exactly by backward
induction, then trains model-free Q-learning and measures how far the learned table sits
from the ground-truth optimum. See docs/value-based-learning.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from student_support_rl.dynamic_programming import (
    gap_rows,
    optimal_action_value_rows,
    optimal_action_values,
    q_learning_gap,
)
from student_support_rl.q_learning import train_q_learning
from student_support_rl.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the dynamic-programming runner's command-line flags.

    Exposes ``--output-dir``, ``--episodes`` (Q-learning training length for the comparison;
    ``None`` defers to the default chosen in ``main``), and ``--quick`` (a shorter run for CI).

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
    """Solve the MDP exactly, train Q-learning, and write the learned-vs-optimal gap.

    Computes the ground-truth optimum Q*(s,a)=E[R_{t+1}+gamma*max_a' Q*(s',a')] by backward
    induction, trains model-free Q-learning, prints the max/mean absolute gap, then writes
    ``artifacts/dp/optimal_action_values.csv`` (the exact Q*) and
    ``artifacts/dp/q_learning_gap.csv`` (per-state-action |learned - Q*|).

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Dynamic programming / planning vs. learning. See docs/value-based-learning.md.
    """
    args = parse_args(argv)
    # --quick caps training at 600 episodes for CI; otherwise honor --episodes or the 5000 default.
    episodes = args.episodes if args.episodes is not None else (600 if args.quick else 5000)
    q_star = optimal_action_values()  # exact optimum via backward induction (planning)
    learned = train_q_learning(episodes=episodes).q_table  # model-free estimate (learning)
    gap = q_learning_gap(learned, q_star)  # how far learning sits from the planned optimum
    print(
        f"Q-learning vs optimal Q*: max_abs_gap={gap['max_abs_gap']} "
        f"mean_abs_gap={gap['mean_abs_gap']} over {gap['num_states']} shared states"
    )
    write_csv_artifact(
        args.output_dir / "dp" / "optimal_action_values.csv",
        optimal_action_value_rows(q_star),
    )
    write_csv_artifact(
        args.output_dir / "dp" / "q_learning_gap.csv",
        gap_rows(learned, q_star),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
