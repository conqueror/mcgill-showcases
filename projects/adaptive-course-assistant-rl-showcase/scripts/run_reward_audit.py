#!/usr/bin/env python3
"""Run the reward-hacking comparison lane."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_course_assistant_rl.policies import InterventionHeavyPolicy, RuleBasedPolicy
from adaptive_course_assistant_rl.reporting import write_text_artifact
from adaptive_course_assistant_rl.reward_design import (
    compare_reward_models,
    reward_hacking_report,
    reward_model_specs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    comparison_rows = compare_reward_models(
        policies=[InterventionHeavyPolicy(), RuleBasedPolicy()],
        scenario_ids=(0, 1, 2, 3, 4),
    )
    specs = reward_model_specs()
    write_text_artifact(
        args.output_dir / "reward" / "reward_hacking_report.md",
        reward_hacking_report(comparison_rows),
    )
    write_text_artifact(args.output_dir / "reward" / "reward_spec_good.md", specs["good"])
    write_text_artifact(args.output_dir / "reward" / "reward_spec_bad.md", specs["bad"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
