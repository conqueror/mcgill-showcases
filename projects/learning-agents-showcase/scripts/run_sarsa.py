#!/usr/bin/env python3
"""Train tabular SARSA (on-policy TD control) and write its training-curve and Q-table."""

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

from learning_agents.q_learning import q_table_rows
from learning_agents.reporting import write_csv_artifact
from learning_agents.sarsa import train_sarsa


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the SARSA runner's command-line flags.

    What + why: SARSA is the on-policy sibling of Q-learning, and this runner trains it so its
    estimates can be diffed against the off-policy Q-learning artifacts. The flags stay minimal and
    mirror the Q-learning runner: ``--output-dir`` (where the two CSVs land), ``--episodes`` (the
    number of training episodes; ``None`` defers to the default chosen in ``main`` so ``--quick``
    can select it), and ``--quick`` (a shorter run for CI smoke checks).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace with ``output_dir``, ``episodes``, and ``quick``.

    RL concept:
        On-policy TD control (SARSA) -- the value-based rung beside off-policy Q-learning.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train tabular SARSA and write its training curve and learned Q-table.

    What + why: writes ``artifacts/sarsa/training_curve.csv`` and ``artifacts/sarsa/q_table.csv`` so
    the on-policy SARSA estimates can be diffed against the off-policy Q-learning table. The only
    structural difference from Q-learning is the backup target: the TD error here is
    ``delta = target - Q(s, A)`` with the target bootstrapped from the *next action actually taken*
    under the epsilon-greedy behaviour, not the greedy max -- so SARSA learns the value of the
    policy it follows (exploration included).

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        On-policy TD control (SARSA): target = R + gamma * Q(s', A') for the chosen next action A',
        contrasted with off-policy Q-learning's target = R + gamma * max_a' Q(s', a').
    """
    args = parse_args(argv)
    # --quick caps training at 120 episodes for CI; otherwise honor --episodes or the 400 default.
    episodes = args.episodes if args.episodes is not None else (120 if args.quick else 400)
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
