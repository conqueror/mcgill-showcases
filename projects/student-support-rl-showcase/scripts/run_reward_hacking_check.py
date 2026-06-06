#!/usr/bin/env python3
"""Compare bad and good reward models and write the reward-design artifacts.

This is the reward-design rung. Because every algorithm on the ladder optimizes whatever reward
it is given, a misspecified reward gets *exploited*: here an advisor-heavy policy scores well
under a naive reward yet serves students poorly. The script contrasts a good and a bad reward
model across fixed scenarios and writes the hacking report plus both reward specs. See
docs/reward-design-and-hacking.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from student_support_rl.policies import AdvisorHeavyPolicy, HeuristicPolicy
from student_support_rl.reporting import write_text_artifact
from student_support_rl.reward_design import (
    compare_reward_models,
    reward_hacking_report,
    reward_model_specs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the reward-hacking runner's command-line flags.

    Exposes ``--output-dir`` and ``--quick``. The comparison is over fixed scenarios with no
    training loop, so ``--quick`` is accepted for a uniform CLI but has no effect here (it is
    discarded in ``main``).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Compare reward models on fixed scenarios and write the reward-design artifacts.

    Scores an advisor-heavy policy and the heuristic policy under both reward models, then writes
    ``artifacts/reward/reward_hacking_report.md`` (the gap that exposes the hack) plus
    ``artifacts/reward/reward_spec_good.md`` and ``artifacts/reward/reward_spec_bad.md``.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        Reward design and reward hacking. See docs/reward-design-and-hacking.md.
    """
    args = parse_args(argv)
    del args.quick  # deterministic scenario sweep; the --quick flag does not apply here
    comparison_rows = compare_reward_models(
        policies=[AdvisorHeavyPolicy(), HeuristicPolicy()],
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
