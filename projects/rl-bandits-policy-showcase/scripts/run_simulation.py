#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_bandits_showcase.evaluation import run_policy_suite

REQUIRED_ARTIFACTS = [
    "artifacts/sim/reward_trace.csv",
    "artifacts/sim/regret_trace.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bandit policy simulations")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    horizon = 120 if args.quick else 600
    trace, _ = run_policy_suite(arm_probs=[0.15, 0.35, 0.50, 0.62], horizon=horizon, seed=args.seed)

    reward_trace = trace.loc[:, ["round", "strategy", "reward", "cumulative_reward"]]
    regret_trace = trace.loc[:, ["round", "strategy", "instant_regret", "cumulative_regret"]]

    reward_path = root / "artifacts/sim/reward_trace.csv"
    regret_path = root / "artifacts/sim/regret_trace.csv"

    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_trace.to_csv(reward_path, index=False)
    regret_trace.to_csv(regret_path, index=False)

    manifest_path = root / "artifacts/manifest.json"
    manifest_path.write_text(
        json.dumps({"version": 1, "required_files": REQUIRED_ARTIFACTS}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
