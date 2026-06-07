#!/usr/bin/env python3
"""Compare independent vs joint-action multi-agent learning and write the Lane C artifacts."""

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

from learning_agents.marl import (
    CLIMBING_GAME,
    marl_comparison_rows,
    train_independent_q_learning,
    train_joint_action_learner,
)
from learning_agents.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the MARL runner's command-line flags.

    What + why: the comparison averages a success rate over many seeds, so the flags size the sweep.
    Exposes ``--output-dir`` and ``--quick`` (fewer seeds and shorter runs for CI; the qualitative
    coordination gap holds either way).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace with ``output_dir`` and ``quick``.

    RL concept:
        Locus of learning C -- multi-agent coordination (independent vs centralised learning).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the independent-vs-joint MARL comparison and write the Lane C artifacts.

    What + why: trains independent Q-learners and a joint-action (centralised) learner on the
    cooperative Climbing game and writes ``artifacts/marl/coordination_comparison.csv`` (per-method
    coordination success rate, final joint action, and team reward vs the optimum) and
    ``artifacts/marl/training_curves.csv`` (each method's greedy-team-reward convergence). Together
    they show the coordination gap: independent learning miscoordinates onto a safe, suboptimal
    joint action while centralised joint-action learning reaches the optimum.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Decentralised (independent) vs centralised (joint-action) multi-agent learning.
    """
    args = parse_args(argv)
    seeds = 8 if args.quick else 30
    episodes = 1500 if args.quick else 4000

    write_csv_artifact(
        args.output_dir / "marl" / "coordination_comparison.csv",
        marl_comparison_rows(CLIMBING_GAME, seeds=seeds, episodes=episodes),
    )

    # Convergence curves from one representative run of each learner (greedy team reward over time).
    curve_rows: list[dict[str, int | float | str]] = []
    for method, result in (
        ("independent", train_independent_q_learning(CLIMBING_GAME, episodes=episodes, seed=0)),
        ("joint", train_joint_action_learner(CLIMBING_GAME, episodes=episodes, seed=0)),
    ):
        for row in result.training_curve:
            curve_rows.append(
                {
                    "method": method,
                    "episode": row["episode"],
                    "greedy_team_reward": row["greedy_team_reward"],
                }
            )
    write_csv_artifact(args.output_dir / "marl" / "training_curves.csv", curve_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
