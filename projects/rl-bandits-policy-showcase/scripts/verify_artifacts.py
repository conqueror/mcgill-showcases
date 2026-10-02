#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rl_bandits_showcase.policy_report import REQUIRED_ARTIFACTS, build_recommendation_markdown


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "artifacts/manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("Missing artifacts/manifest.json")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Invalid artifact manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise SystemExit("Artifact manifest must be a version 1 object")
    required = manifest.get("required_files")
    if (
        not isinstance(required, list)
        or not all(isinstance(path, str) for path in required)
        or sorted(required) != sorted(REQUIRED_ARTIFACTS)
    ):
        raise SystemExit("Manifest required_files must match the canonical artifact bundle")

    missing = [path for path in REQUIRED_ARTIFACTS if not (root / path).is_file()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    schemas = {
        "reward_trace.csv": ["round", "strategy", "reward", "cumulative_reward"],
        "regret_trace.csv": ["round", "strategy", "instant_regret", "cumulative_regret"],
        "policy_comparison.csv": ["strategy", "cumulative_reward", "cumulative_regret"],
    }
    strategies = {"epsilon_greedy", "ucb1", "thompson_sampling"}
    frames: dict[str, pd.DataFrame] = {}
    for name, columns in schemas.items():
        try:
            frame = pd.read_csv(root / "artifacts/sim" / name)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Invalid {name}: {exc}") from exc
        if frame.empty or not set(columns).issubset(frame.columns):
            raise SystemExit(f"{name} must have data rows and columns {columns}")
        if set(frame["strategy"]) != strategies:
            raise SystemExit(f"{name} must contain exactly the three supported strategies")
        numeric_columns = [column for column in columns if column != "strategy"]
        try:
            numbers = frame[numeric_columns].apply(
                pd.to_numeric, errors="raise"
            ).to_numpy(dtype=float)
        except ValueError as exc:
            raise SystemExit(f"{name} contains non-numeric values") from exc
        if not np.isfinite(numbers).all() or (numbers < 0).any():
            raise SystemExit(f"{name} must contain finite, nonnegative numbers")
        frames[name] = frame

    reward_trace = frames["reward_trace.csv"]
    regret_trace = frames["regret_trace.csv"]
    summary = frames["policy_comparison.csv"]
    if not reward_trace["reward"].isin([0.0, 1.0]).all():
        raise SystemExit("Reward trace must contain binary rewards")
    if len(summary) != len(strategies):
        raise SystemExit("Policy comparison must have one row per strategy")
    horizon = len(reward_trace) // len(strategies)
    for strategy in strategies:
        rewards = reward_trace[reward_trace["strategy"] == strategy]
        regrets = regret_trace[regret_trace["strategy"] == strategy]
        rounds = list(range(1, horizon + 1))
        if rewards["round"].tolist() != rounds or regrets["round"].tolist() != rounds:
            raise SystemExit("Traces must contain consecutive rounds with the same horizon")
        if not np.allclose(
            rewards["cumulative_reward"], rewards["reward"].cumsum(), rtol=1e-10, atol=1e-8
        ) or not np.allclose(
            regrets["cumulative_regret"], regrets["instant_regret"].cumsum(), rtol=1e-10, atol=1e-8
        ):
            raise SystemExit("Trace cumulative totals do not match their per-round values")
        row = summary[summary["strategy"] == strategy].iloc[0]
        if not np.allclose(
            [row["cumulative_reward"], row["cumulative_regret"]],
            [rewards["cumulative_reward"].iloc[-1], regrets["cumulative_regret"].iloc[-1]],
            rtol=1e-10,
            atol=1e-8,
        ):
            raise SystemExit("Policy comparison totals do not match the traces")

    report_path = root / "artifacts/sim/policy_recommendation.md"
    if (
        report_path.read_text(encoding="utf-8").strip()
        != build_recommendation_markdown(summary).strip()
    ):
        raise SystemExit("Policy report does not match the comparison")
    print("All required artifacts are valid.")


if __name__ == "__main__":
    main()
