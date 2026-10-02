#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "artifacts/manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Missing artifacts/manifest.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = manifest.get("required_files")
    if not isinstance(required, list) or not required or not all(
        isinstance(path, str) and path for path in required
    ):
        raise SystemExit("required_files must be a nonempty list of paths")

    missing = [path for path in required if not (root / path).is_file()
               or (root / path).stat().st_size == 0]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    metrics = pd.read_csv(root / "artifacts/fairness/group_metrics.csv")
    denominators = {"selection_rate": "count", "tpr": "positive_count",
                    "fnr": "positive_count", "fpr": "negative_count",
                    "precision": "predicted_positive_count"}
    if metrics.empty or not {"group", *denominators, *denominators.values()}.issubset(
        metrics.columns
    ) or metrics["group"].isna().any():
        raise SystemExit("Subgroup metrics must contain groups, rates, and denominator counts.")
    for metric, denominator in denominators.items():
        counts = pd.to_numeric(metrics[denominator]).to_numpy(dtype=float)
        rates = pd.to_numeric(metrics[metric]).to_numpy(dtype=float)
        if not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any():
            raise SystemExit("Subgroup denominator counts must be nonnegative integers.")
        supported = counts > 0
        if metric == "selection_rate" and not supported.all():
            raise SystemExit("Every reported group must contain observations.")
        if not np.isfinite(rates[supported]).all() or (
            (rates[supported] < 0) | (rates[supported] > 1)
        ).any() or not np.isnan(rates[~supported]).all():
            raise SystemExit("Rates must be in [0, 1], or undefined when the denominator is zero.")

    print("All required artifacts exist.")


if __name__ == "__main__":
    main()
