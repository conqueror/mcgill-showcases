#!/usr/bin/env python3
"""Train tabular REINFORCE (Monte-Carlo policy gradient) and write its training curve."""

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

from learning_agents.policy_gradient import train_reinforce
from learning_agents.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the policy-gradient runner's command-line flags.

    What + why: this runner trains the policy-gradient rung of the showcase ladder, so it needs to
    let a caller choose how long to learn for. It exposes ``--output-dir`` (where artifacts land),
    ``--episodes`` (training episodes; ``None`` defers to the default chosen in ``main``), and
    ``--quick`` (a shorter run for CI smoke checks).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.

    RL concept: episode count is the training budget for Monte-Carlo policy gradient (REINFORCE).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train tabular REINFORCE and write its training curve.

    What + why: REINFORCE is Monte-Carlo policy gradient -- it optimizes the orchestration policy
    pi_theta *directly* by gradient ascent on expected return, instead of first learning action
    values and acting greedily as Q-learning does. Keeping it tabular (one logit per state-action,
    no neural net) exposes the exact score-function update that PPO scales up. This script rolls out
    on-policy episodes over the agent-decision MDP and writes
    ``artifacts/policy_gradient/training_curve.csv`` (return per episode, with the mean-return
    baseline) so the learned policy is reproducible and the learning curve is inspectable.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept: Monte-Carlo policy gradient (REINFORCE); the policy is optimized directly via
    grad J = E[grad log pi(A|s) (G_t - b)], and the mean-return baseline b is the simplest variance
    reduction and the seed of the actor-critic "critic".
    """
    args = parse_args(argv)
    # --quick caps training at 400 episodes for CI; otherwise honor --episodes or the 1500 default.
    episodes = args.episodes if args.episodes is not None else (400 if args.quick else 1500)
    result = train_reinforce(episodes=episodes)
    write_csv_artifact(
        args.output_dir / "policy_gradient" / "training_curve.csv",
        result.training_curve,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
