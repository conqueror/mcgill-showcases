#!/usr/bin/env python3
"""Train tabular REINFORCE (Monte-Carlo policy gradient) and write its training curve.

REINFORCE optimizes the policy directly instead of learning values first. This tabular,
no-neural-net version exposes the exact update PPO scales up. See
docs/policy-gradient-and-actor-critic.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from student_support_rl.policy_gradient import train_reinforce
from student_support_rl.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the policy-gradient runner's command-line flags.

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
    """Train tabular REINFORCE and write its training curve.

    Writes ``artifacts/policy_gradient/training_curve.csv``, the return per training block under
    the Monte-Carlo policy-gradient update grad J = E[grad log pi(A|s) (G_t - b)] applied to a
    softmax policy pi(a|s)=exp(theta_{s,a})/sum exp(theta).

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Monte-Carlo policy gradient (REINFORCE). See docs/policy-gradient-and-actor-critic.md.
    """
    args = parse_args(argv)
    # --quick caps training at 400 episodes for CI; otherwise honor --episodes or the 2000 default.
    episodes = args.episodes if args.episodes is not None else (400 if args.quick else 2000)
    result = train_reinforce(episodes=episodes)
    write_csv_artifact(
        args.output_dir / "policy_gradient" / "training_curve.csv",
        result.training_curve,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
