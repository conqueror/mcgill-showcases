#!/usr/bin/env python3
"""Train the tabular Q-learning policy and write its artifacts.

This is the value-based control rung: Q-learning is off-policy temporal-difference control that
bootstraps toward the greedy action, so it learns Q*(s,a)=E[R_{t+1}+gamma*max_a' Q*(s',a')]
regardless of how it explores. The script trains the table and writes the training curve plus the
final learned Q-table. See docs/value-based-learning.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from student_support_rl.q_learning import q_table_rows, train_q_learning
from student_support_rl.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the Q-learning runner's command-line flags.

    Exposes ``--output-dir``, ``--episodes`` (training episodes; ``None`` defers to the default
    chosen in ``main``), and ``--quick`` (a shorter run for CI).

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
    """Train tabular Q-learning and write its training curve and learned Q-table.

    Writes ``artifacts/q_learning/training_curve.csv`` (return per training block) and
    ``artifacts/q_learning/q_table.csv`` (the learned action values Q(s,a)).

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Off-policy TD control (Q-learning). See docs/value-based-learning.md.
    """
    args = parse_args(argv)
    # --quick caps training at 600 episodes for CI; otherwise honor --episodes or the 5000 default.
    episodes = args.episodes if args.episodes is not None else (600 if args.quick else 5000)
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
