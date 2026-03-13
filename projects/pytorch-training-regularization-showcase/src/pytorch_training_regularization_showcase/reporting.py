"""Artifact-writing helpers for the PyTorch showcase."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def expected_artifact_paths(output_dir: Path) -> dict[str, Path]:
    """Return the stable artifact path contract for the showcase."""

    return {
        "baseline_metrics": output_dir / "baseline_metrics.json",
        "training_history": output_dir / "training_history.csv",
        "optimizer_comparison": output_dir / "optimizer_comparison.csv",
        "learning_rate_schedule_comparison": (
            output_dir / "learning_rate_schedule_comparison.csv"
        ),
        "regularization_ablation": output_dir / "regularization_ablation.csv",
        "gradient_health_report": output_dir / "gradient_health_report.md",
        "error_analysis": output_dir / "error_analysis.csv",
        "summary": output_dir / "summary.md",
    }


def write_csv_artifact(table: pd.DataFrame, destination: Path) -> None:
    """Write a DataFrame artifact to CSV."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)


def write_json_artifact(content: dict[str, Any], destination: Path) -> None:
    """Write a JSON artifact to disk."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(content, indent=2), encoding="utf-8")


def write_markdown_artifact(content: str, destination: Path) -> None:
    """Write a Markdown artifact to disk."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


def build_summary_markdown(
    project_title: str,
    highlights: list[str],
    next_steps: list[str],
    extra_sections: dict[str, list[str]] | None = None,
) -> str:
    """Build the top-level summary artifact for the project."""

    lines = [f"# {project_title}", "", "## Highlights"]
    lines.extend(f"- {item}" for item in highlights)
    lines.extend(["", "## Next Steps"])
    lines.extend(f"- {item}" for item in next_steps)

    for section_title, items in (extra_sections or {}).items():
        lines.extend(["", f"## {section_title}"])
        lines.extend(f"- {item}" for item in items)

    return "\n".join(lines) + "\n"


def build_gradient_health_report_markdown(
    gradient_table: pd.DataFrame,
    dataset_name: str,
) -> str:
    """Summarize per-parameter gradient norms in Markdown."""

    strongest = gradient_table.sort_values("gradient_norm", ascending=False).iloc[0]
    weakest = gradient_table.sort_values("gradient_norm", ascending=True).iloc[0]

    lines = [
        "# Gradient Health Report",
        "",
        f"- Dataset: {dataset_name}",
        (
            f"- Strongest gradient: {strongest['parameter']} "
            f"({strongest['gradient_norm']:.6f})"
        ),
        f"- Weakest gradient: {weakest['parameter']} ({weakest['gradient_norm']:.6f})",
        "",
        "## Interpretation",
        (
            "Large gradients suggest parameters receiving strong learning signal, "
            "while tiny gradients can hint at saturation or weak feature flow."
        ),
        "",
        "## Parameter Snapshot",
    ]
    lines.extend(
        f"- {row.parameter}: grad={row.gradient_norm:.6f}, weight={row.weight_norm:.6f}"
        for row in gradient_table.itertuples(index=False)
    )
    return "\n".join(lines) + "\n"


def to_highlight(metric_name: str, value: Any) -> str:
    """Format a short highlight line for the summary artifact."""

    return f"{metric_name}: {value}"
