#!/usr/bin/env python3
"""Sweep a cost-aware cascade's effort budget and write its cost-vs-quality frontier."""

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

from learning_agents.cost_cascade import cost_cascade_curve
from learning_agents.reporting import write_csv_artifact


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the cost-cascade runner's command-line flags.

    What + why: the cascade sweep is a deterministic evaluation over fixed scenarios, so the flags
    only size the evaluation. Exposes ``--output-dir`` and ``--quick`` (fewer rollouts per scenario
    for CI; the effort levels swept are fixed so the frontier shape is stable).

    Args:
        argv: Optional argument vector; falls back to ``sys.argv`` when ``None``.

    Returns:
        The populated argparse namespace with ``output_dir`` and ``quick``.

    RL concept:
        Cost-aware control -- sweeping a cost knob to trace the cost/quality frontier.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Sweep the cascade's effort budget and write the cost-vs-quality curve artifact.

    What + why: evaluates an :class:`~learning_agents.cost_cascade.EffortCascadePolicy` at each
    effort level and records realised money cost, latency (steps), combined ``total_cost``, reward,
    escalation rate, and solved rate. Writes ``artifacts/cost_cascade/cost_quality_curve.csv`` --
    the operating curve a practitioner reads to choose a cost/quality operating point on the
    non-dominated frontier.

    Args:
        argv: Optional argument vector forwarded to ``parse_args``.

    Returns:
        Process exit code; ``0`` on success.

    RL concept:
        The cost/quality (money + latency vs reward) Pareto frontier of a cost-aware cascade.
    """
    args = parse_args(argv)
    # --quick shortens the per-scenario evaluation budget; the swept effort levels stay fixed.
    episodes_per_scenario = 4 if args.quick else 12
    rows = cost_cascade_curve(
        effort_levels=(0, 1, 2, 3, 4),
        episodes_per_scenario=episodes_per_scenario,
    )
    write_csv_artifact(args.output_dir / "cost_cascade" / "cost_quality_curve.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
