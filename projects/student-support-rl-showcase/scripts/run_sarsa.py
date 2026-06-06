#!/usr/bin/env python3
"""Train tabular SARSA (on-policy TD control) and write its training-curve and Q-table.

SARSA is the on-policy sibling of Q-learning: it bootstraps from the action it actually
takes next rather than the greedy max. Compare its artifacts against the q_learning ones.
See docs/value-based-learning.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from student_support_rl.q_learning import q_table_rows
from student_support_rl.reporting import write_csv_artifact
from student_support_rl.sarsa import train_sarsa


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the SARSA runner's command-line flags.

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
    """Train tabular SARSA and write its training curve and learned Q-table.

    Writes ``artifacts/sarsa/training_curve.csv`` and ``artifacts/sarsa/q_table.csv`` so the
    on-policy SARSA estimates can be diffed against the off-policy Q-learning table; the TD error
    here is delta = target - Q(s,A) with the target bootstrapped from the *next action actually
    taken*, not the greedy max.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        On-policy TD control (SARSA). See docs/value-based-learning.md.
    """
    args = parse_args(argv)
    # --quick caps training at 600 episodes for CI; otherwise honor --episodes or the 5000 default.
    episodes = args.episodes if args.episodes is not None else (600 if args.quick else 5000)
    result = train_sarsa(episodes=episodes)
    write_csv_artifact(
        args.output_dir / "sarsa" / "training_curve.csv",
        result.training_curve,
    )
    write_csv_artifact(
        args.output_dir / "sarsa" / "q_table.csv",
        q_table_rows(result.q_table),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
