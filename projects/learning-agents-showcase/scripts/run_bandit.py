#!/usr/bin/env python3
"""Run the contextual-bandit agent-routing warm-up and log reward and regret."""

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

from learning_agents.bandit import run_bandit_experiment
from learning_agents.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the bandit runner's command-line flags.

    What + why: this is the first rung of the RL ladder, so the runner only needs to control the
    exploration horizon and where artifacts land. It exposes ``--output-dir`` (where the CSV traces
    are written), ``--steps`` (number of routing decisions to simulate; ``None`` defers to the
    default chosen in ``main``), and ``--quick`` (a smaller, faster horizon for CI smoke runs).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.

    RL concept:
        Exploration horizon for a contextual bandit. See docs/exploration-and-bandits.md.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train the contextual bandit router and write its reward and regret traces.

    What + why: a contextual bandit has no state transitions, so the only tension is exploration
    versus exploitation when routing each request to an agent. This runner simulates ``steps``
    one-shot routing decisions and writes ``artifacts/bandit/reward_trace.csv`` (per-step expected
    reward) and ``artifacts/bandit/regret_trace.csv`` (cumulative regret_T = sum_t [mu*(x_t) -
    mu_{a_t}(x_t)]), the two curves that let learners watch regret flatten as the learner stops
    paying to explore.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Contextual bandit and cumulative regret. See docs/exploration-and-bandits.md.
    """
    args = parse_args(argv)
    # --quick shrinks the horizon for CI; otherwise honor --steps or the 600-step default.
    steps = args.steps if args.steps is not None else (120 if args.quick else 600)
    result = run_bandit_experiment(steps=steps)
    write_csv_artifact(args.output_dir / "bandit" / "reward_trace.csv", result.reward_trace)
    write_csv_artifact(args.output_dir / "bandit" / "regret_trace.csv", result.regret_trace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
