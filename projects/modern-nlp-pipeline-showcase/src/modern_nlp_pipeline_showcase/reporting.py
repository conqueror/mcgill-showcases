"""Artifact writing and verification helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def required_artifact_paths() -> list[str]:
    """Return the artifact contract for this showcase."""
    return [
        "artifacts/manifest.json",
        "artifacts/data/corpus_overview.csv",
        "artifacts/data/topic_distribution.csv",
        "artifacts/classification/metrics_summary.csv",
        "artifacts/retrieval/retrieval_metrics.csv",
        "artifacts/retrieval/retrieval_examples.json",
        "artifacts/generation/qa_outputs.csv",
        "artifacts/generation/query_summaries.json",
        "artifacts/summary.md",
    ]


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def build_manifest() -> dict[str, Any]:
    return {"required_files": required_artifact_paths()}


def verify_required_artifacts(project_root: Path, required_paths: list[str]) -> list[str]:
    """Return relative paths that are missing from the artifact contract."""
    missing: list[str] = []
    for relative_path in required_paths:
        if not (project_root / relative_path).exists():
            missing.append(relative_path)
    return missing
