#!/usr/bin/env python3
"""Run the optional Gymnasium plus Stable-Baselines3 DRL bridge.

This is the deep-RL rung. It wraps the tabular environment in a Gymnasium interface and trains
DQN (value-based, the neural-network successor to Q-learning) and PPO (the production-grade
actor-critic policy-gradient method), then compares the RL families on common scenarios. The
heavy dependencies are optional: if they are missing the script writes an explanatory bridge
report and still exits 0. See docs/deep-rl.md.
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

from student_support_rl.drl import OptionalDRLError, run_drl_comparison
from student_support_rl.reporting import write_csv_artifact, write_text_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the optional-DRL runner's command-line flags.

    Exposes ``--output-dir``, ``--timesteps`` (environment steps per DRL learner; ``None`` defers
    to the default chosen in ``main``), and ``--quick`` (a shorter run for CI).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train the optional DQN/PPO learners and write the deep-RL comparison artifacts.

    On success, writes ``artifacts/drl_optional/`` files: ``rl_family_comparison.csv``,
    ``scenario_rollups.csv``, ``training_summary.csv``, ``policy_gradient_notes.md``, and
    ``bridge_report.md``. If the optional Gymnasium/SB3 extras are not installed,
    ``run_drl_comparison`` raises ``OptionalDRLError``; the script then writes only an
    explanatory ``bridge_report.md`` and still returns ``0`` so the pipeline never hard-fails.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success and also when optional deps are missing.

    RL concept:
        Deep RL (DQN and PPO) and graceful optional dependencies. See docs/deep-rl.md.
    """
    args = parse_args(argv)
    # --quick shortens the per-learner training budget for CI.
    timesteps = args.timesteps if args.timesteps is not None else (1200 if args.quick else 3600)
    drl_output_dir = args.output_dir / "drl_optional"

    try:
        result = run_drl_comparison(
            timesteps=timesteps,
            output_dir=drl_output_dir,
            quick=args.quick,
        )
    except (
        OptionalDRLError
    ) as exc:  # pragma: no cover - exercised only when optional deps are missing
        write_text_artifact(
            drl_output_dir / "bridge_report.md",
            (
                "# Optional DRL Bridge\n\n"
                "The optional Gymnasium/SB3 path could not run, so the DQN and PPO comparison "
                "artifacts were not generated.\n\n"
                "Install the optional extras with `make sync-drl` and rerun "
                "`make run-drl-optional` to generate the full comparison set.\n\n"
                f"Import error: {exc}\n"
            ),
        )
        return 0

    write_csv_artifact(drl_output_dir / "rl_family_comparison.csv", result.comparison_rows)
    write_csv_artifact(drl_output_dir / "scenario_rollups.csv", result.scenario_rows)
    write_csv_artifact(drl_output_dir / "training_summary.csv", result.training_rows)
    write_text_artifact(
        drl_output_dir / "policy_gradient_notes.md",
        result.policy_gradient_notes,
    )
    write_text_artifact(drl_output_dir / "bridge_report.md", result.bridge_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
