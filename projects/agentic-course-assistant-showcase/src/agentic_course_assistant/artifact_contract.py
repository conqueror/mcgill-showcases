"""Artifact contract validation for the course assistant showcase."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from agentic_course_assistant.artifact_manifest import BASE_REQUIRED_FILES, REQUIRED_HARNESS_FILES
from agentic_course_assistant.concept_atlas import (
    EXPECTED_CONCEPT_COLUMNS,
    REQUESTED_CONCEPT_IDS,
)

EXPECTED_TRACE_KEYS = {
    "agent_name",
    "guardrails",
    "harness_events",
    "intent",
    "question",
    "resource_ids",
    "trace",
}
EXPECTED_HARNESS_EVENTS = {
    "guardrail_check",
    "handoff",
    "tool_call",
}
EXPECTED_HARNESS_LIFECYCLE_ARTIFACTS = {
    "artifacts/harness/judge_verdicts.json",
    "artifacts/harness/run_ledger.jsonl",
}
EXPECTED_EVAL_CASES: dict[str, dict[str, object]] = {
    "route_project_agent": {
        "expected_workflow": "router_triage",
        "expected_intent": "project",
    },
    "block_secret_request": {
        "expected_workflow": "custom_policy_agent",
        "expected_allowed": False,
        "expected_guardrail": "Do not paste secrets.",
    },
    "sequential_plan_resources": {
        "expected_workflow": "sequential_course_plan",
        "expected_min_resources": 3,
    },
    "parallel_review_count": {
        "expected_workflow": "parallel_resource_review",
        "expected_review_count": 3,
    },
    "bounded_loop": {
        "expected_workflow": "loop_refinement",
        "expected_min_rounds": 2,
    },
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

    canonical_required = set(BASE_REQUIRED_FILES) | set(REQUIRED_HARNESS_FILES)
    missing_from_manifest = sorted(canonical_required - set(required_files))
    if missing_from_manifest:
        errors.append(f"artifacts/manifest.json missing canonical files: {missing_from_manifest}")

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
    errors.extend(_validate_harness_artifacts(root))
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
    harness_events = trace.get("harness_events")
    if not isinstance(harness_events, list) or not harness_events:
        errors.append("agent_trace.json harness_events must be a non-empty list")
    else:
        event_types: set[str] = {
            str(event.get("event_type"))
            for event in harness_events
            if isinstance(event, dict) and isinstance(event.get("event_type"), str)
        }
        missing_events = sorted(EXPECTED_HARNESS_EVENTS - event_types)
        if missing_events:
            errors.append(f"agent_trace.json missing harness events: {missing_events}")
        unexpected_events = sorted(event_types - EXPECTED_HARNESS_EVENTS)
        if unexpected_events:
            errors.append(f"agent_trace.json has unexpected harness events: {unexpected_events}")
        errors.extend(_validate_harness_events_match_trace(trace, harness_events))
    return errors


def _validate_harness_events_match_trace(
    trace: dict[str, object],
    harness_events: list[object],
) -> list[str]:
    trace_steps = trace.get("trace")
    if not isinstance(trace_steps, list):
        return []
    trace_text = " ".join(str(step) for step in trace_steps)
    event_names = {
        event.get("event_type"): event.get("name")
        for event in harness_events
        if isinstance(event, dict)
    }
    required_trace_pairs = {
        "tool_call": "course_catalog_tool.search_resources",
        "guardrail_check": "guardrail.scope_check",
    }
    errors: list[str] = []
    for event_type, expected_trace_step in required_trace_pairs.items():
        if event_names.get(event_type) != expected_trace_step:
            errors.append(f"agent_trace.json {event_type} event is not tied to the trace")
        if expected_trace_step not in trace_text:
            errors.append(f"agent_trace.json trace missing step for {event_type}")
    handoff_name = event_names.get("handoff")
    if not isinstance(handoff_name, str) or handoff_name not in trace_text:
        errors.append("agent_trace.json handoff event is not tied to the trace")
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


def _validate_harness_artifacts(root: Path) -> list[str]:
    harness_dir = root / "artifacts/harness"
    errors: list[str] = []
    trace_schema_path = harness_dir / "trace_schema.json"
    trace_schema = _load_harness_trace_schema(trace_schema_path)
    if isinstance(trace_schema, list):
        errors.extend(trace_schema)
    else:
        errors.extend(_validate_harness_trace_schema(trace_schema))
        errors.extend(
            _validate_trace_matches_harness_schema(
                root / "artifacts/agent_trace.json",
                trace_schema,
            )
        )
    errors.extend(_validate_harness_eval_cases(harness_dir / "eval_cases.jsonl"))
    errors.extend(_validate_harness_judge_verdicts(harness_dir / "judge_verdicts.json"))
    errors.extend(_validate_harness_run_ledger(harness_dir / "run_ledger.jsonl"))
    errors.extend(_validate_harness_failure_report(harness_dir / "failure_injection_report.md"))
    return errors


def _load_harness_trace_schema(trace_schema_path: Path) -> dict[str, object] | list[str]:
    if not trace_schema_path.exists():
        return ["Missing artifacts/harness/trace_schema.json"]
    try:
        loaded = json.loads(trace_schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid trace_schema.json: {exc}"]
    if not isinstance(loaded, dict):
        return ["trace_schema.json must be an object"]
    return loaded


def _validate_harness_trace_schema(schema: dict[str, object]) -> list[str]:
    required_keys = schema.get("required_top_level_keys")
    workflow_patterns = schema.get("workflow_patterns")
    required_events = schema.get("required_harness_events")
    lifecycle_artifacts = schema.get("harness_lifecycle_artifacts")
    errors: list[str] = []
    if not isinstance(required_keys, list) or set(EXPECTED_TRACE_KEYS) - set(required_keys):
        errors.append("trace_schema.json must include the assistant trace keys")
    if not isinstance(workflow_patterns, list) or len(workflow_patterns) < 5:
        errors.append("trace_schema.json must include the five offline workflow patterns")
    if not isinstance(required_events, list) or set(required_events) != EXPECTED_HARNESS_EVENTS:
        errors.append("trace_schema.json must include the required harness events")
    if (
        not isinstance(lifecycle_artifacts, list)
        or set(lifecycle_artifacts) != EXPECTED_HARNESS_LIFECYCLE_ARTIFACTS
    ):
        errors.append("trace_schema.json must include the harness lifecycle artifacts")
    return errors


def _validate_trace_matches_harness_schema(
    trace_path: Path,
    trace_schema: dict[str, object],
) -> list[str]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    required_keys = trace_schema.get("required_top_level_keys")
    required_events = trace_schema.get("required_harness_events")
    errors: list[str] = []
    if isinstance(required_keys, list):
        missing_keys = sorted(set(required_keys) - set(trace))
        if missing_keys:
            errors.append(f"agent_trace.json missing schema-required keys: {missing_keys}")
    if isinstance(required_events, list):
        event_types: set[str] = {
            str(event.get("event_type"))
            for event in trace.get("harness_events", [])
            if isinstance(event, dict) and isinstance(event.get("event_type"), str)
        }
        required_event_set = {str(event) for event in required_events}
        missing_events = sorted(required_event_set - event_types)
        if missing_events:
            errors.append(f"agent_trace.json missing schema-required events: {missing_events}")
        unexpected_events = sorted(event_types - required_event_set)
        if unexpected_events:
            errors.append(
                f"agent_trace.json has events outside the trace schema: {unexpected_events}"
            )
    return errors


def _validate_harness_eval_cases(eval_cases_path: Path) -> list[str]:
    if not eval_cases_path.exists():
        return ["Missing artifacts/harness/eval_cases.jsonl"]
    lines = [line for line in eval_cases_path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) < 5:
        return ["eval_cases.jsonl must include at least five eval cases"]

    errors: list[str] = []
    cases_by_id: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid eval_cases.jsonl line {line_number}: {exc}")
            continue
        if not payload.get("case_id") or not payload.get("prompt"):
            errors.append(f"eval_cases.jsonl line {line_number} must include case_id and prompt")
        if not payload.get("expected_workflow"):
            errors.append(f"eval_cases.jsonl line {line_number} must include expected_workflow")
        case_id = payload.get("case_id")
        if isinstance(case_id, str):
            if case_id in cases_by_id:
                errors.append(f"eval_cases.jsonl duplicate case_id: {case_id}")
            cases_by_id[case_id] = payload

    expected_case_ids = set(EXPECTED_EVAL_CASES)
    observed_case_ids = set(cases_by_id)
    missing_case_ids = sorted(expected_case_ids - observed_case_ids)
    unexpected_case_ids = sorted(observed_case_ids - expected_case_ids)
    if missing_case_ids:
        errors.append(f"eval_cases.jsonl missing canonical eval cases: {missing_case_ids}")
    if unexpected_case_ids:
        errors.append(f"eval_cases.jsonl has non-canonical eval cases: {unexpected_case_ids}")
    for case_id, expected_fields in EXPECTED_EVAL_CASES.items():
        payload = cases_by_id.get(case_id)
        if payload is None:
            continue
        for field, expected_value in expected_fields.items():
            if payload.get(field) != expected_value:
                errors.append(
                    f"eval_cases.jsonl case {case_id} has non-canonical {field}: "
                    f"{payload.get(field)!r}"
                )
    return errors


def _validate_harness_judge_verdicts(judge_verdicts_path: Path) -> list[str]:
    if not judge_verdicts_path.exists():
        return ["Missing artifacts/harness/judge_verdicts.json"]
    try:
        payload = json.loads(judge_verdicts_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"Invalid judge_verdicts.json: {exc}"]

    verdicts = payload.get("verdicts")
    summary = payload.get("summary")
    if not isinstance(verdicts, list) or len(verdicts) < 10:
        return ["judge_verdicts.json must include workflow and eval-case verdicts"]
    errors: list[str] = []
    if not isinstance(summary, dict) or summary.get("failed") != 0:
        errors.append("judge_verdicts.json summary must report zero failures")
    case_ids = {
        verdict.get("case_id")
        for verdict in verdicts
        if isinstance(verdict.get("case_id"), str)
    }
    expected_case_ids = set(EXPECTED_EVAL_CASES)
    missing_case_ids = sorted(expected_case_ids - case_ids)
    if missing_case_ids:
        errors.append(f"judge_verdicts.json missing eval case verdicts: {missing_case_ids}")
    for verdict in verdicts:
        if verdict.get("verdict") != "pass":
            errors.append("judge_verdicts.json contains a non-pass verdict")
        if not isinstance(verdict.get("checks"), dict) or not verdict["checks"]:
            errors.append("judge_verdicts.json verdicts must include checks")
    return errors


def _validate_harness_run_ledger(run_ledger_path: Path) -> list[str]:
    if not run_ledger_path.exists():
        return ["Missing artifacts/harness/run_ledger.jsonl"]
    lines = [line for line in run_ledger_path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        return ["run_ledger.jsonl must contain at least one run entry"]

    errors: list[str] = []
    latest_status: str | None = None
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid run_ledger.jsonl line {line_number}: {exc}")
            continue
        if payload.get("event") != "harness_lab_run":
            errors.append(f"run_ledger.jsonl line {line_number} has unexpected event")
        status = payload.get("status")
        if status not in {"pass", "fail"}:
            errors.append(f"run_ledger.jsonl line {line_number} must report pass or fail status")
        else:
            latest_status = status
    if latest_status != "pass":
        errors.append("run_ledger.jsonl latest run must report pass status")
    return errors


def _validate_harness_failure_report(report_path: Path) -> list[str]:
    if not report_path.exists():
        return ["Missing artifacts/harness/failure_injection_report.md"]
    text = report_path.read_text(encoding="utf-8")
    required_phrases = (
        "# Harness Failure Injection Report",
        "Tool failure",
        "Guardrail trip",
        "Routing ambiguity",
        "Loop runaway",
        "Trace corruption",
    )
    missing = [phrase for phrase in required_phrases if phrase not in text]
    if missing:
        return [f"failure_injection_report.md missing phrases: {missing}"]
    return []
