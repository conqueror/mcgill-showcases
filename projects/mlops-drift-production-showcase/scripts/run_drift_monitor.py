#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlops_drift_showcase.data import generate_incoming_data
from mlops_drift_showcase.drift import compute_drift_report
from mlops_drift_showcase.policy import recommend_action

REQUIRED_FILES = [
    "artifacts/drift/drift_report.csv",
    "artifacts/drift/drift_summary.md",
    "artifacts/policy/retrain_recommendation.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature drift monitor")
    parser.add_argument("--quick", action="store_true", help="Use quicker settings for smoke runs")
    parser.add_argument("--seed", type=int, default=99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    reference_path = root / "artifacts/reference/train_features.csv"
    if not reference_path.exists():
        raise SystemExit("Reference features are missing. Run scripts/run_pipeline.py first.")

    import pandas as pd

    reference = pd.read_csv(reference_path)
    shift = 0.4 if args.quick else 0.65
    incoming = generate_incoming_data(reference, shift_strength=shift, random_state=args.seed)

    drift_report = compute_drift_report(reference, incoming)
    recommendation = recommend_action(drift_report)

    report_path = root / "artifacts/drift/drift_report.csv"
    summary_path = root / "artifacts/drift/drift_summary.md"
    recommendation_path = root / "artifacts/policy/retrain_recommendation.json"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    recommendation_path.parent.mkdir(parents=True, exist_ok=True)

    drift_report.to_csv(report_path, index=False)

    drifted = drift_report[drift_report["drift_flag"] == 1]["feature"].tolist()
    summary = [
        "# Drift Summary",
        "",
        f"Drifted features: {len(drifted)}",
        f"Recommended action: {recommendation['action']}",
        "",
        "## Top Drifted Features",
    ]
    if drifted:
        summary.extend(f"- {name}" for name in drifted[:10])
    else:
        summary.append("- none")
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")

    recommendation_path.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

    manifest_path = root / "artifacts/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"version": 1, "required_files": []}

    required = set(manifest.get("required_files", []))
    required.update(REQUIRED_FILES)
    manifest["required_files"] = sorted(required)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
