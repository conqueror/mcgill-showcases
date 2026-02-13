from __future__ import annotations

import pandas as pd


def build_disparity_markdown(disparities: pd.DataFrame) -> str:
    lines = [
        "# Fairness Disparity Summary",
        "",
        "| Metric | Max | Min | Gap |",
        "|---|---:|---:|---:|",
    ]
    for row in disparities.to_dict(orient="records"):
        lines.append(
            f"| {row['metric']} | {row['max']:.4f} | {row['min']:.4f} | {row['gap']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)
