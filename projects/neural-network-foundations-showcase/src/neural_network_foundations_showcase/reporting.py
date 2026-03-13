"""Artifact writing helpers for the neural network foundations showcase."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def expected_artifact_paths(output_dir: Path) -> dict[str, Path]:
    """Return the stable artifact path contract for the showcase."""

    return {
        "activation_comparison": output_dir / "activation_comparison.csv",
        "loss_function_comparison": output_dir / "loss_function_comparison.csv",
        "backprop_gradient_trace": output_dir / "backprop_gradient_trace.csv",
        "initialization_comparison": output_dir / "initialization_comparison.csv",
        "underfit_overfit_examples": output_dir / "underfit_overfit_examples.csv",
        "training_curves": output_dir / "training_curves.csv",
        "decision_boundary_summary": output_dir / "decision_boundary_summary.csv",
        "decision_boundaries": output_dir / "decision_boundaries.png",
        "summary": output_dir / "summary.md",
    }


def write_csv_artifact(table: pd.DataFrame, destination: Path) -> None:
    """Write a DataFrame artifact to CSV."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)


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


def to_highlight(metric_name: str, value: Any) -> str:
    """Format a short highlight line for the summary artifact."""

    return f"{metric_name}: {value}"
