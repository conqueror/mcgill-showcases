#!/usr/bin/env python
from __future__ import annotations

import json
from math import isclose
from pathlib import Path

import numpy as np
import pandas as pd

from model_release_showcase.rollout import evaluate_canary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "artifacts/manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = payload.get("required_files")
    if (
        not isinstance(required, list)
        or not required
        or not all(isinstance(p, str) for p in required)
    ):
        raise SystemExit("Manifest required_files must be a nonempty list of paths")
    missing = [path for path in required if not (root / path).exists()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    try:
        for name in required:
            path = root / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Empty or invalid artifact: {name}")
        evaluation = pd.read_csv(root / "artifacts/rollout/canary_eval.csv")
        decision = json.loads((root / "artifacts/rollout/decision_log.json").read_text())
        registry = json.loads((root / "artifacts/registry/model_versions.json").read_text())
        expected = evaluate_canary(
            evaluation["champion_score"],
            evaluation["challenger_score"],
            min_gain=decision["min_gain"],
            max_regression=decision["max_regression"],
        )
        if decision["decision"] != expected.decision or decision["reason"] != expected.reason:
            raise ValueError("Decision does not match recorded scores and thresholds")
        for key, value in (
            ("champion_mean_score", evaluation["champion_score"].mean()),
            ("challenger_mean_score", evaluation["challenger_score"].mean()),
            ("mean_delta", (evaluation["challenger_score"] - evaluation["champion_score"]).mean()),
        ):
            if not isclose(decision[key], float(value), abs_tol=1e-12):
                raise ValueError(f"Recorded {key} does not match the scores")
        if not np.allclose(
            evaluation["delta"], evaluation["challenger_score"] - evaluation["champion_score"]
        ):
            raise ValueError("Recorded deltas do not match the scores")
        if (
            decision["metric"] != "synthetic_score"
            or decision["metric_direction"] != "higher_is_better"
        ):
            raise ValueError("Invalid metric definition")
        active = registry["challenger"] if expected.decision == "promote" else registry["champion"]
        if (
            registry["active_version"] != active
            or registry["rollback_target"] != registry["champion"]
            or registry["decision"] != expected.decision
        ):
            raise ValueError("Registry active version or rollback target is inconsistent")
    except Exception as exc:
        raise SystemExit(f"Invalid artifact contents: {exc}") from exc


if __name__ == "__main__":
    main()
