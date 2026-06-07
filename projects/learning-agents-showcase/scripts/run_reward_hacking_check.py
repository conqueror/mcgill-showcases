#!/usr/bin/env python3
"""Contrast the aligned judge rubric with a hackable proxy and write reward-design artifacts."""

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

from learning_agents.policies import AlwaysEscalatePolicy, HeuristicRouterPolicy
from learning_agents.reporting import write_text_artifact
from learning_agents.reward_study import (
    compare_reward_models,
    reward_hacking_report,
    reward_model_specs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the reward-hacking runner's command-line flags.

    What + why: this runner contrasts two reward models over a fixed scenario sweep rather than
    training anything, so its CLI only needs an output location. ``--quick`` is accepted purely so
    every runner in the showcase shares one flag surface; because the sweep is already deterministic
    and tiny, the flag has no effect and is discarded in ``main``.

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace with ``output_dir`` and ``quick`` attributes.

    RL concept:
        Reward design and reward hacking -- the misspecified proxy reverses the policy ranking.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Compare reward models on fixed scenarios and write the reward-design artifacts.

    What + why: every algorithm on the RL ladder optimizes whatever reward it is handed, so a
    misspecified reward gets *exploited* rather than merely tolerated. Here the hackable proxy
    overpays escalation, which flips the ranking so that the blunt ``AlwaysEscalatePolicy``
    ("always_escalate") looks better than the targeted ``HeuristicRouterPolicy``
    ("heuristic_router") even though the aligned judge rubric prefers the router. The script scores
    both policies under each reward model across a fixed scenario sweep, then writes the hacking
    report that exposes the gap plus the good and bad reward specifications.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Reward design and reward hacking -- swapping only the objective, while holding policies and
        scenarios fixed, isolates the reward as the cause of the reversed ranking.
    """
    args = parse_args(argv)
    del args.quick  # deterministic scenario sweep; the --quick flag does not apply here
    comparison_rows = compare_reward_models(
        policies=[AlwaysEscalatePolicy(), HeuristicRouterPolicy()],
        scenario_ids=(0, 1, 2, 3, 4),
    )
    specs = reward_model_specs()
    write_text_artifact(
        args.output_dir / "reward" / "reward_hacking_report.md",
        reward_hacking_report(comparison_rows),
    )
    write_text_artifact(
        args.output_dir / "reward" / "reward_spec_good.md",
        specs["good"],
    )
    write_text_artifact(
        args.output_dir / "reward" / "reward_spec_bad.md",
        specs["bad"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
