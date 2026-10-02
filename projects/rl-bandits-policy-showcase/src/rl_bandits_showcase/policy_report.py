from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_ARTIFACTS = (
    "artifacts/sim/reward_trace.csv",
    "artifacts/sim/regret_trace.csv",
    "artifacts/sim/policy_comparison.csv",
    "artifacts/sim/policy_recommendation.md",
)


def write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps({"version": 1, "required_files": list(REQUIRED_ARTIFACTS)}, indent=2),
        encoding="utf-8",
    )


def build_recommendation_markdown(summary: pd.DataFrame) -> str:
    best = summary.iloc[0]
    lines = [
        "# Policy Recommendation",
        "",
        f"Highest observed reward in this run: **{best['strategy']}**",
        f"Final cumulative reward: **{best['cumulative_reward']:.2f}**",
        f"Final cumulative regret: **{best['cumulative_regret']:.2f}**",
        "",
        "Each policy was evaluated in one seeded simulation with a different environment seed. "
        "This ranking does not establish general superiority or quantify uncertainty.",
        "",
        "## Policy Table",
        "",
        "| strategy | cumulative_reward | cumulative_regret |",
        "|---|---:|---:|",
    ]

    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['strategy']} | {row['cumulative_reward']:.2f} | "
            f"{row['cumulative_regret']:.2f} |"
        )

    lines.append("")
    return "\n".join(lines)
