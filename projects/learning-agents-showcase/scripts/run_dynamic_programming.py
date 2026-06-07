#!/usr/bin/env python3
"""Compute the exact optimal Q* by backward induction and the learned-vs-optimal gap."""

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

from learning_agents.dynamic_programming import (
    gap_rows,
    optimal_action_value_rows,
    optimal_action_values,
)
from learning_agents.q_learning import train_q_learning
from learning_agents.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the dynamic-programming runner's command-line flags.

    Exposes ``--output-dir`` (where the DP artifacts land), ``--episodes`` (the model-free
    Q-learning training length used for the comparison; ``None`` defers to the default chosen
    in ``main``), and ``--quick`` (a shorter run for CI smoke checks).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.

    RL concept:
        Planning needs no episodes, but learning does; ``--episodes`` controls only the
        Q-learning side of the optimal-vs-learned comparison.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Solve the routing MDP exactly, train Q-learning, and write the learned-vs-optimal gap.

    Computes the ground-truth optimum Q*(s,a)=E[R_{t+1}+gamma*max_a' Q*(s',a')] by backward
    induction over the known finite-horizon MDP (planning), then trains model-free Q-learning
    (learning) and measures how far the learned table sits from that optimum. Writes
    ``artifacts/dp/optimal_action_values.csv`` (the exact Q*) and
    ``artifacts/dp/q_learning_gap.csv`` (per-state-action |learned - Q*|).

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Dynamic programming / planning vs. learning. Backward induction yields the exact Q*
        when the model is known, giving a reference the model-free learner can be scored against.
    """
    args = parse_args(argv)
    # --quick caps Q-learning at 120 episodes for CI; otherwise honor --episodes or the 400 default.
    episodes = args.episodes if args.episodes is not None else (120 if args.quick else 400)
    optimal_values = optimal_action_values()  # exact optimum via backward induction (planning)
    write_csv_artifact(
        args.output_dir / "dp" / "optimal_action_values.csv",
        optimal_action_value_rows(optimal_values),
    )
    q_result = train_q_learning(episodes=episodes)  # model-free estimate (learning)
    write_csv_artifact(
        args.output_dir / "dp" / "q_learning_gap.csv",
        gap_rows(q_result.q_table, optimal_values),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
