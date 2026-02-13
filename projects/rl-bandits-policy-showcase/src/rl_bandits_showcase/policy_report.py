from __future__ import annotations

import pandas as pd


def build_recommendation_markdown(summary: pd.DataFrame) -> str:
    best = summary.iloc[0]
    lines = [
        "# Policy Recommendation",
        "",
        f"Recommended policy: **{best['strategy']}**",
        f"Final cumulative reward: **{best['cumulative_reward']:.2f}**",
        f"Final cumulative regret: **{best['cumulative_regret']:.2f}**",
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
