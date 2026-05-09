"""Artifact manifest helpers for the agentic course assistant showcase."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

BASE_REQUIRED_FILES: tuple[str, ...] = (
    "artifacts/course_assistant_response.md",
    "artifacts/agent_trace.json",
    "artifacts/resource_matches.csv",
    "artifacts/concepts/agentic_concepts.csv",
    "artifacts/concepts/openai_vs_adk_concepts.json",
    "artifacts/concepts/refined_questions.md",
    "artifacts/concepts/student_learning_path.md",
    "artifacts/evals/agent_judge_rubric.json",
    "artifacts/evals/concept_coverage.json",
)

REQUIRED_HARNESS_FILES: tuple[str, ...] = (
    "artifacts/harness/run_ledger.jsonl",
    "artifacts/harness/trace_schema.json",
    "artifacts/harness/eval_cases.jsonl",
    "artifacts/harness/judge_verdicts.json",
    "artifacts/harness/failure_injection_report.md",
)


def all_required_files() -> tuple[str, ...]:
    """Return the full artifact contract for the showcase."""

    return BASE_REQUIRED_FILES + REQUIRED_HARNESS_FILES


def merge_required_files(manifest_path: Path, required_files: Iterable[str]) -> list[str]:
    """Merge required files into the manifest and return the sorted contract."""

    existing_required: list[str] = []
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_required = payload.get("required_files", [])
        if isinstance(raw_required, list) and all(isinstance(path, str) for path in raw_required):
            existing_required = raw_required

    merged = sorted(set(existing_required) | set(required_files))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"version": 1, "required_files": merged}, indent=2) + "\n",
        encoding="utf-8",
    )
    return merged
