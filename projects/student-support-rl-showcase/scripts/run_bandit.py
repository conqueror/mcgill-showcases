#!/usr/bin/env python3
"""Run the exploration-versus-exploitation bandit warm-up.

This is the first rung of the ladder: a contextual bandit has no state transitions, so the only
tension is exploration versus exploitation. The script trains the bandit and writes its per-step
reward and cumulative-regret traces so learners can watch regret flatten as the learner stops
paying to explore. See docs/exploration-and-bandits.md.
"""

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

from student_support_rl.bandit import run_bandit_experiment
from student_support_rl.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bandit runner's command-line flags.

    Exposes ``--output-dir`` (where artifacts land), ``--steps`` (bandit pulls; ``None`` means
    use the default chosen in ``main``), and ``--quick`` (a smaller, faster horizon for CI).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train the contextual bandit and write its reward and regret traces.

    Writes ``artifacts/bandit/reward_trace.csv`` (per-step reward) and
    ``artifacts/bandit/regret_trace.csv`` (cumulative regret_T = sum_t [mu*(x_t) - mu_{a_t}(x_t)]),
    the two curves that demonstrate the exploration/exploitation trade-off.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Contextual bandit and regret. See docs/exploration-and-bandits.md.
    """
    args = parse_args(argv)
    # --quick shrinks the horizon for CI; otherwise honor --steps or the 600-pull default.
    steps = args.steps if args.steps is not None else (120 if args.quick else 600)
    result = run_bandit_experiment(steps=steps)
    write_csv_artifact(args.output_dir / "bandit" / "reward_trace.csv", result.reward_trace)
    write_csv_artifact(args.output_dir / "bandit" / "regret_trace.csv", result.regret_trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
