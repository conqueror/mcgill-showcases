#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_release_showcase.rollout import evaluate_canary

REQUIRED_FILES = [
    "artifacts/registry/model_versions.json",
    "artifacts/rollout/canary_eval.csv",
    "artifacts/rollout/decision_log.json",
    "artifacts/rollout/rollback_plan.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model rollout simulation")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    rng = np.random.default_rng(args.seed)

    n = 200 if args.quick else 900
    champion_scores = pd.Series(rng.normal(loc=0.71, scale=0.05, size=n), name="champion_auc")
    challenger_scores = pd.Series(rng.normal(loc=0.725, scale=0.05, size=n), name="challenger_auc")

    decision = evaluate_canary(
        champion_scores,
        challenger_scores,
        min_gain=0.005,
        max_regression=0.01,
    )

    eval_df = pd.DataFrame(
        {
            "champion_auc": champion_scores,
            "challenger_auc": challenger_scores,
            "delta": challenger_scores - champion_scores,
        }
    )

    canary_path = root / "artifacts/rollout/canary_eval.csv"
    decision_path = root / "artifacts/rollout/decision_log.json"
    registry_path = root / "artifacts/registry/model_versions.json"
    rollback_path = root / "artifacts/rollout/rollback_plan.md"
    manifest_path = root / "artifacts/manifest.json"

    canary_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    eval_df.to_csv(canary_path, index=False)

    decision_payload = {
        "decision": decision.decision,
        "reason": decision.reason,
        "champion_mean_auc": float(champion_scores.mean()),
        "challenger_mean_auc": float(challenger_scores.mean()),
        "mean_delta": float((challenger_scores - champion_scores).mean()),
    }
    decision_path.write_text(json.dumps(decision_payload, indent=2), encoding="utf-8")

    registry_payload = {
        "champion": "v1.4.0",
        "challenger": "v1.5.0",
        "decision": decision.decision,
    }
    registry_path.write_text(json.dumps(registry_payload, indent=2), encoding="utf-8")

    rollback_plan = "\n".join(
        [
            "# Rollback Plan",
            "",
            "1. Route 100% traffic back to champion version.",
            "2. Freeze challenger rollout and capture failure evidence.",
            "3. Open incident review and patch challenger issues.",
        ]
    )
    rollback_path.write_text(rollback_plan + "\n", encoding="utf-8")

    manifest_path.write_text(
        json.dumps({"version": 1, "required_files": REQUIRED_FILES}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
