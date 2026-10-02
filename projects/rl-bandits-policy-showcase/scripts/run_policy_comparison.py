#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from rl_bandits_showcase.evaluation import run_policy_suite
from rl_bandits_showcase.policy_report import build_recommendation_markdown, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build policy comparison summary")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    horizon = 120 if args.quick else 600
    _, summary = run_policy_suite(
        arm_probs=[0.15, 0.35, 0.50, 0.62],
        horizon=horizon,
        seed=args.seed,
    )

    summary_path = root / "artifacts/sim/policy_comparison.csv"
    report_path = root / "artifacts/sim/policy_recommendation.md"

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    report_path.write_text(build_recommendation_markdown(summary), encoding="utf-8")

    write_manifest(root / "artifacts/manifest.json")


if __name__ == "__main__":
    main()
