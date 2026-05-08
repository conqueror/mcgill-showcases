"""Artifact contract validation for the course assistant showcase."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from agentic_course_assistant.concept_atlas import (
    EXPECTED_CONCEPT_COLUMNS,
    REQUESTED_CONCEPT_IDS,
)

EXPECTED_TRACE_KEYS = {
    "agent_name",
    "guardrails",
    "intent",
    "question",
    "resource_ids",
    "trace",
}
EXPECTED_RESOURCE_COLUMNS = ["resource_id", "title", "topic", "level", "kind", "skills"]
EXPECTED_INTENTS = {"concept", "exercise", "debug", "project"}
MIN_CONCEPT_COUNT = 25


def verify(root: Path) -> list[str]:
    """Return artifact contract errors for the given project root."""

    errors: list[str] = []
    manifest_path = root / "artifacts/manifest.json"
    if not manifest_path.exists():
        return ["Missing artifacts/manifest.json"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid artifacts/manifest.json: {exc}"]

    required_files = manifest.get("required_files")
    if not isinstance(required_files, list) or not all(
        isinstance(path, str) for path in required_files
    ):
        return ["artifacts/manifest.json must define a string list named required_files"]

    missing = [path for path in required_files if not (root / path).exists()]
    if missing:
        errors.append(f"Missing required artifacts: {missing}")
        return errors

    errors.extend(_validate_response_markdown(root / "artifacts/course_assistant_response.md"))
    errors.extend(_validate_trace(root / "artifacts/agent_trace.json"))
    errors.extend(_validate_resource_matches(root / "artifacts/resource_matches.csv"))
    errors.extend(_validate_concept_csv(root / "artifacts/concepts/agentic_concepts.csv"))
    errors.extend(
        _validate_comparison_json(root / "artifacts/concepts/openai_vs_adk_concepts.json")
    )
    errors.extend(_validate_refined_questions(root / "artifacts/concepts/refined_questions.md"))
    errors.extend(_validate_learning_path(root / "artifacts/concepts/student_learning_path.md"))
    errors.extend(_validate_judge_rubric(root / "artifacts/evals/agent_judge_rubric.json"))
    errors.extend(_validate_coverage(root / "artifacts/evals/concept_coverage.json"))
    return errors


def _validate_response_markdown(response_path: Path) -> list[str]:
    text = response_path.read_text(encoding="utf-8")
    required_sections = (
        "# Agentic Course Assistant Response",
        "## Question",
        "## Route",
        "## Answer",
        "## Resource Matches",
        "## Trace",
    )
    missing_sections = [section for section in required_sections if section not in text]
    if missing_sections:
        return [f"course_assistant_response.md missing sections: {missing_sections}"]
    return []


def _validate_trace(trace_path: Path) -> list[str]:
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid artifacts/agent_trace.json: {exc}"]

    errors: list[str] = []
    missing_keys = sorted(EXPECTED_TRACE_KEYS - set(trace))
    if missing_keys:
        errors.append(f"agent_trace.json missing keys: {missing_keys}")
    if trace.get("intent") not in EXPECTED_INTENTS:
        errors.append("agent_trace.json has an unknown intent")
    if not isinstance(trace.get("trace"), list) or not trace.get("trace"):
        errors.append("agent_trace.json trace must be a non-empty list")
    if not isinstance(trace.get("resource_ids"), list) or not trace.get("resource_ids"):
        errors.append("agent_trace.json resource_ids must be a non-empty list")
    return errors


def _validate_resource_matches(matches_path: Path) -> list[str]:
    with matches_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_RESOURCE_COLUMNS:
            return [f"resource_matches.csv columns must be {EXPECTED_RESOURCE_COLUMNS}"]
        rows = list(reader)

    if not rows:
        return ["resource_matches.csv must contain at least one resource row"]
    if any(not row.get("resource_id") or not row.get("title") for row in rows):
        return ["resource_matches.csv rows must include resource_id and title"]
    return []


def _validate_concept_csv(concepts_path: Path) -> list[str]:
    with concepts_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_CONCEPT_COLUMNS:
            return [f"agentic_concepts.csv columns must be {EXPECTED_CONCEPT_COLUMNS}"]
        rows = list(reader)

    errors: list[str] = []
    if len(rows) < MIN_CONCEPT_COUNT:
        errors.append(f"agentic_concepts.csv must contain at least {MIN_CONCEPT_COUNT} rows")
    concept_ids = {row.get("concept_id", "") for row in rows}
    missing = sorted(REQUESTED_CONCEPT_IDS - concept_ids)
    if missing:
        errors.append(f"agentic_concepts.csv missing requested concepts: {missing}")
    required_text_fields = ["name", "definition", "openai_agents_sdk", "google_adk"]
    if any(not row.get(field) for row in rows for field in required_text_fields):
        errors.append("agentic_concepts.csv rows must include framework comparison text")
    return errors


def _validate_comparison_json(comparison_path: Path) -> list[str]:
    try:
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid openai_vs_adk_concepts.json: {exc}"]

    errors: list[str] = []
    frameworks = comparison.get("frameworks")
    concepts = comparison.get("concepts")
    if not isinstance(frameworks, dict):
        errors.append("openai_vs_adk_concepts.json must define frameworks")
    else:
        for framework in ("openai_agents_sdk", "google_adk"):
            if framework not in frameworks:
                errors.append(f"openai_vs_adk_concepts.json missing {framework}")
    if not isinstance(concepts, list) or len(concepts) < MIN_CONCEPT_COUNT:
        errors.append("openai_vs_adk_concepts.json must include the concept atlas")
    return errors


def _validate_refined_questions(questions_path: Path) -> list[str]:
    text = questions_path.read_text(encoding="utf-8")
    required_terms = (
        "tools",
        "guardrails",
        "tracing",
        "evals",
        "A2A",
        "sessions",
        "memory",
        "skills",
        "harnesses",
    )
    missing_terms = [term for term in required_terms if term.lower() not in text.lower()]
    if missing_terms:
        return [f"refined_questions.md missing terms: {missing_terms}"]
    return []


def _validate_learning_path(learning_path: Path) -> list[str]:
    text = learning_path.read_text(encoding="utf-8")
    required_phrases = (
        "# Student Learning Path",
        "Stage 1: Offline Workflow",
        "Stage 2: SDK Shape",
        "Stage 5: Evaluation And Harness",
        "A2A",
        "memory",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in text]
    if missing_phrases:
        return [f"student_learning_path.md missing phrases: {missing_phrases}"]
    return []


def _validate_judge_rubric(rubric_path: Path) -> list[str]:
    try:
        rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid agent_judge_rubric.json: {exc}"]

    criteria = rubric.get("criteria")
    if not isinstance(criteria, list) or len(criteria) < 5:
        return ["agent_judge_rubric.json must define at least five criteria"]
    if any("question" not in criterion or "weight" not in criterion for criterion in criteria):
        return ["agent_judge_rubric.json criteria must include question and weight"]
    return []


def _validate_coverage(coverage_path: Path) -> list[str]:
    try:
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid concept_coverage.json: {exc}"]

    if coverage.get("verdict") != "pass":
        return ["concept_coverage.json verdict must be pass"]
    covered = coverage.get("requested_concepts_covered")
    if not isinstance(covered, dict):
        return ["concept_coverage.json must define requested_concepts_covered"]
    missing = [
        concept_id
        for concept_id in REQUESTED_CONCEPT_IDS
        if covered.get(concept_id) is not True
    ]
    if missing:
        return [f"concept_coverage.json missing requested concepts: {sorted(missing)}"]
    return []
