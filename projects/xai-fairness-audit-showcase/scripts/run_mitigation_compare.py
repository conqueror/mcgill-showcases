#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from xai_fairness_showcase.data import make_audit_dataset
from xai_fairness_showcase.mitigation import run_mitigation_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fairness mitigation comparison")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    n_samples = 900 if args.quick else 1800
    split = make_audit_dataset(n_samples=n_samples, random_state=args.seed)
    table = run_mitigation_benchmark(
        split.x_train,
        split.y_train,
        split.g_train,
        split.x_test,
        split.y_test,
        split.g_test,
    )

    output_path = root / "artifacts/mitigation/mitigation_tradeoff_table.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)

    manifest_path = root / "artifacts/manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"version": 1, "required_files": []}

    required = set(manifest.get("required_files", []))
    required.add("artifacts/mitigation/mitigation_tradeoff_table.csv")
    manifest["required_files"] = sorted(required)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
