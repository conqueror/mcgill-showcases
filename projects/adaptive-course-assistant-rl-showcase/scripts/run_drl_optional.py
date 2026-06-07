#!/usr/bin/env python3
"""Run the optional DQN and PPO comparison lane."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_course_assistant_rl.config import FULL_DRL_TIMESTEPS, QUICK_DRL_TIMESTEPS
from adaptive_course_assistant_rl.drl import OptionalDRLError, run_drl_comparison
from adaptive_course_assistant_rl.reporting import (
    optional_drl_validation_errors,
    write_csv_artifact,
    write_text_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help="Exit successfully when the optional DRL extras are unavailable.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_drl_comparison(
            timesteps=QUICK_DRL_TIMESTEPS if args.quick else FULL_DRL_TIMESTEPS,
            quick=args.quick,
        )
    except OptionalDRLError as exc:
        print(f"Optional DRL extras unavailable: {exc}")
        return 0 if args.allow_skip else 1
    write_csv_artifact(
        args.output_dir / "drl_optional" / "rl_family_comparison.csv",
        result.comparison_rows,
    )
    write_csv_artifact(
        args.output_dir / "drl_optional" / "scenario_rollups.csv",
        result.scenario_rows,
    )
    write_csv_artifact(
        args.output_dir / "drl_optional" / "dqn_training_summary.csv",
        result.dqn_training_rows,
    )
    write_csv_artifact(
        args.output_dir / "drl_optional" / "ppo_training_summary.csv",
        result.ppo_training_rows,
    )
    write_text_artifact(
        args.output_dir / "drl_optional" / "policy_gradient_bridge_notes.md",
        result.policy_gradient_notes,
    )
    validation_errors = optional_drl_validation_errors(artifacts_dir=args.output_dir)
    if validation_errors:
        print("Optional DRL artifact validation errors:")
        for error in validation_errors:
            print(f"- {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
