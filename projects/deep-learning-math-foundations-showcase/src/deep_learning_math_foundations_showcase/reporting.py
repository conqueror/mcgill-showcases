"""Artifact writing helpers for the deep learning math foundations showcase."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd


def expected_artifact_paths(output_dir: Path) -> dict[str, Path]:
    """Return the stable artifact path contract for the showcase."""

    return {
        "vector_operations": output_dir / "vector_operations.csv",
        "matrix_transformations": output_dir / "matrix_transformations.csv",
        "derivative_examples": output_dir / "derivative_examples.csv",
        "gradient_descent_trace": output_dir / "gradient_descent_trace.csv",
        "probability_simulations": output_dir / "probability_simulations.csv",
        "information_theory_summary": output_dir / "information_theory_summary.md",
        "summary": output_dir / "summary.md",
        "manifest": output_dir / "manifest.json",
    }


def write_artifact_manifest(
    artifact_paths: dict[str, Path],
    command: list[str],
    random_states: list[int],
    parameters: dict[str, Any] | None = None,
) -> None:
    """Bind generated outputs to the command, source, seeds, and environment."""

    project_dir = Path(__file__).resolve().parents[2]
    outputs = [path for key, path in artifact_paths.items() if key != "manifest"]
    manifest = {
        "required_files": [path.name for path in outputs],
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs
        },
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_dir,
            text=True,
        ).strip(),
        "source_sha256": {
            str(path.relative_to(project_dir)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for folder in ("src", "scripts")
            for path in sorted((project_dir / folder).rglob("*.py"))
        },
        "command": command,
        "random_states": random_states,
        "parameters": parameters or {},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in ["numpy", "pandas", "matplotlib", "sympy"]
            },
        },
    }
    artifact_paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
    """Format a short highlight line for a metric."""

    return f"{metric_name}: {value}"
