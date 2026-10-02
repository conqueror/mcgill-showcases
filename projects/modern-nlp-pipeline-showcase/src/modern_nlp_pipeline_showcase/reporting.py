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
    """Return paths with missing, empty, or malformed artifact content."""
    columns = {
        "corpus_overview.csv": {"paper_id", "title", "topic", "abstract_words", "summary_words"},
        "topic_distribution.csv": {"topic", "paper_count"},
        "metrics_summary.csv": {"model", "accuracy", "macro_f1", "train_rows", "test_rows"},
        "retrieval_metrics.csv": {"strategy", "recall_at_k", "mrr_at_k", "top_k"},
        "qa_outputs.csv": {
            "query_id",
            "backend",
            "source_strategy",
            "predicted_answer",
            "expected_answer",
        },
        "retrieval_examples.json": {
            "query_id",
            "query",
            "strategy",
            "relevant_paper_id",
            "hit_rank",
            "retrieved_paper_ids",
            "top_chunk_id",
            "top_chunk_text",
            "top_passages",
        },
        "query_summaries.json": {"query_id", "backend", "source_strategy", "summary_text"},
    }
    missing: list[str] = []
    for relative_path in required_paths:
        path = project_root / relative_path
        try:
            content = path.read_text(encoding="utf-8")
            valid = bool(content.strip())
            required = columns.get(path.name, set())
            if path.suffix == ".json":
                payload = json.loads(content)
                if path.name == "manifest.json":
                    valid = payload == build_manifest()
                else:
                    valid = (
                        isinstance(payload, list)
                        and bool(payload)
                        and all(
                            isinstance(row, dict)
                            and required <= row.keys()
                            and isinstance(row.get("query_id"), str)
                            and bool(row["query_id"])
                            and all(
                                isinstance(row[key], str)
                                for key in required
                                - {
                                    "hit_rank",
                                    "retrieved_paper_ids",
                                    "top_chunk_id",
                                    "top_chunk_text",
                                    "top_passages",
                                }
                            )
                            for row in payload
                        )
                    )
            elif path.suffix == ".csv":
                frame = pd.read_csv(path)
                valid = not frame.empty and required <= set(frame.columns)
                valid = valid and bool(
                    frame.loc[:, list(required - {"predicted_answer"})].notna().all().all()
                )
                for metric in {"accuracy", "macro_f1", "recall_at_k", "mrr_at_k"} & required:
                    valid = valid and bool(
                        pd.to_numeric(frame[metric], errors="raise").between(0, 1).all()
                    )
                for count in {"top_k", "train_rows", "test_rows", "paper_count"} & required:
                    values = pd.to_numeric(frame[count], errors="raise")
                    valid = valid and bool(((values > 0) & (values % 1 == 0)).all())
        except (OSError, ValueError, UnicodeError, KeyError):
            valid = False
        if not valid:
            missing.append(relative_path)
    return missing
