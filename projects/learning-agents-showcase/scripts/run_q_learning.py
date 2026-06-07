#!/usr/bin/env python3
"""Train the tabular Q-learning orchestration policy and write its artifacts."""

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

from learning_agents.q_learning import q_table_rows, train_q_learning
from learning_agents.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the Q-learning runner's command-line flags.

    What + why: this runner trains the value-based rung of the showcase ladder, so it needs to let
    a caller choose how long to learn for. It exposes ``--output-dir`` (where artifacts land),
    ``--episodes`` (training episodes; ``None`` defers to the default chosen in ``main``), and
    ``--quick`` (a shorter run for CI smoke checks).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.

    RL concept: episode count is the training budget for off-policy TD control (Q-learning).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train tabular Q-learning and write its training curve and learned Q-table.

    What + why: Q-learning is off-policy temporal-difference control -- the agent explores with an
    epsilon-greedy behaviour policy but bootstraps every backup from ``max_a' Q(s', a')``, the
    value of the greedy next action, so it chases the Bellman optimality action values Q*(s,a)
    regardless of how it explored. This script trains that table over the agent-decision MDP and
    writes ``artifacts/q_learning/training_curve.csv`` (return per episode) and
    ``artifacts/q_learning/q_table.csv`` (the learned action values Q(s,a)) so the learned
    orchestration policy is reproducible and inspectable.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept: off-policy TD control (Q-learning); the greedy backup separates it from on-policy
    SARSA and points the estimate at Q*(s,a).
    """
    args = parse_args(argv)
    # --quick caps training at 120 episodes for CI; otherwise honor --episodes or the 400 default.
    episodes = args.episodes if args.episodes is not None else (120 if args.quick else 400)
    result = train_q_learning(episodes=episodes)
    write_csv_artifact(
        args.output_dir / "q_learning" / "training_curve.csv",
        result.training_curve,
    )
    write_csv_artifact(
        args.output_dir / "q_learning" / "q_table.csv",
        q_table_rows(result.q_table),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
