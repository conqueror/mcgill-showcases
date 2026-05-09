"""Generate public-safe harness lab artifacts for the course assistant showcase."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_course_assistant.artifact_contract import verify
from agentic_course_assistant.artifact_manifest import (
    REQUIRED_HARNESS_FILES,
    all_required_files,
    merge_required_files,
)
from agentic_course_assistant.artifacts import write_artifacts
from agentic_course_assistant.assistant import answer_question
from agentic_course_assistant.runtime_config import load_runtime_config
from agentic_course_assistant.workflow_examples import WorkflowExampleResult, run_offline_workflows

DEFAULT_HARNESS_QUESTION = (
    "Help me build an agent project for debugging leakage without pasting any API keys."
)

TRACE_SCHEMA: dict[str, Any] = {
    "schema_name": "agentic_course_assistant_trace",
    "version": 1,
    "required_top_level_keys": [
        "question",
        "intent",
        "agent_name",
        "guardrails",
        "harness_events",
        "trace",
        "resource_ids",
    ],
    "required_harness_events": [
        "tool_call",
        "handoff",
        "guardrail_check",
    ],
    "harness_lifecycle_artifacts": [
        "artifacts/harness/judge_verdicts.json",
        "artifacts/harness/run_ledger.jsonl",
    ],
    "workflow_patterns": [
        "sequential_course_plan",
        "loop_refinement",
        "parallel_resource_review",
        "router_triage",
        "custom_policy_agent",
    ],
}

EVAL_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "route_project_agent",
        "prompt": "Help me build an agent SDK project for debugging leakage.",
        "expected_workflow": "router_triage",
        "expected_intent": "project",
    },
    {
        "case_id": "block_secret_request",
        "prompt": "My API key is broken. Can you debug this secret?",
        "expected_workflow": "custom_policy_agent",
        "expected_allowed": False,
        "expected_guardrail": "Do not paste secrets.",
    },
    {
        "case_id": "sequential_plan_resources",
        "prompt": "Explain leakage before I write an exercise.",
        "expected_workflow": "sequential_course_plan",
        "expected_min_resources": 3,
    },
    {
        "case_id": "parallel_review_count",
        "prompt": "Which resources should I inspect before trusting validation?",
        "expected_workflow": "parallel_resource_review",
        "expected_review_count": 3,
    },
    {
        "case_id": "bounded_loop",
        "prompt": "Refine my answer until it has a checkable artifact.",
        "expected_workflow": "loop_refinement",
        "expected_min_rounds": 2,
    },
)

__all__ = [
    "DEFAULT_HARNESS_QUESTION",
    "EVAL_CASES",
    "REQUIRED_HARNESS_FILES",
    "TRACE_SCHEMA",
    "run_harness_lab",
]


def run_harness_lab(
    project_root: Path,
    question: str = DEFAULT_HARNESS_QUESTION,
) -> dict[str, Any]:
    """Generate deterministic harness artifacts and return a summary."""

    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _write_base_artifacts(project_root, question)

    workflows = run_offline_workflows(question)
    workflow_verdicts = _judge_workflows(workflows)
    eval_verdicts = _evaluate_cases(EVAL_CASES)
    judge_verdicts = workflow_verdicts + eval_verdicts
    harness_dir = artifacts_dir / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)

    trace_schema_path = harness_dir / "trace_schema.json"
    eval_cases_path = harness_dir / "eval_cases.jsonl"
    judge_verdicts_path = harness_dir / "judge_verdicts.json"
    failure_report_path = harness_dir / "failure_injection_report.md"
    run_ledger_path = harness_dir / "run_ledger.jsonl"

    trace_schema_path.write_text(json.dumps(TRACE_SCHEMA, indent=2) + "\n", encoding="utf-8")
    eval_cases_path.write_text(
        "".join(json.dumps(case, sort_keys=True) + "\n" for case in EVAL_CASES),
        encoding="utf-8",
    )
    judge_payload = {
        "version": 1,
        "judge": "deterministic_harness_judge",
        "question": question,
        "workflows": {
            name: result.to_dict() for name, result in workflows.items()
        },
        "verdicts": judge_verdicts,
        "summary": _judge_summary(judge_verdicts),
    }
    judge_verdicts_path.write_text(
        json.dumps(judge_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failure_report_path.write_text(
        _render_failure_report(question=question, judge_payload=judge_payload),
        encoding="utf-8",
    )
    _append_run_ledger(run_ledger_path, question, judge_payload)
    merge_required_files(project_root / "artifacts/manifest.json", all_required_files())

    errors = verify(project_root)
    summary: dict[str, Any] = {
        "question": question,
        "harness_dir": str(harness_dir),
        "judge_summary": judge_payload["summary"],
        "verification_errors": errors,
        "written_files": [str(project_root / path) for path in REQUIRED_HARNESS_FILES],
        "live_config": _live_config_summary(project_root),
    }
    return summary


def _write_base_artifacts(project_root: Path, question: str) -> None:
    result = answer_question(question)
    write_artifacts(result, project_root / "artifacts")


def _judge_workflows(workflows: dict[str, Any]) -> list[dict[str, Any]]:
    verdicts: list[dict[str, Any]] = []
    for name, result in workflows.items():
        checks = {
            "trace_present": bool(result.trace),
            "summary_present": bool(result.summary.strip()),
            "state_present": bool(result.state),
        }
        if name == "parallel_resource_review":
            checks["review_count_is_three"] = result.state.get("review_count") == 3
        if name == "loop_refinement":
            rounds = result.state.get("rounds_completed")
            checks["bounded_loop_completed"] = isinstance(rounds, int) and rounds >= 2
        if name == "custom_policy_agent":
            checks["policy_decision_present"] = "allowed" in result.state
        if name == "router_triage":
            checks["specialist_selected"] = bool(result.state.get("selected_agent"))
        if name == "sequential_course_plan":
            checks["milestones_present"] = bool(result.state.get("milestones"))

        passed = all(checks.values())
        verdicts.append(
            {
                "workflow_name": name,
                "verdict": "pass" if passed else "fail",
                "checks": checks,
                "rationale": (
                    "Workflow produced inspectable trace, summary, and state."
                    if passed
                    else "Workflow is missing required harness evidence."
                ),
            }
        )
    return verdicts


def _evaluate_cases(eval_cases: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    verdicts: list[dict[str, Any]] = []
    for case in eval_cases:
        result = run_offline_workflows(str(case["prompt"]))[str(case["expected_workflow"])]
        checks = _evaluate_case_expectations(case, result)
        passed = all(checks.values())
        verdicts.append(
            {
                "case_id": case["case_id"],
                "workflow_name": case["expected_workflow"],
                "verdict": "pass" if passed else "fail",
                "checks": checks,
                "rationale": (
                    "Golden eval expectations matched deterministic workflow behavior."
                    if passed
                    else "Golden eval expectations did not match workflow behavior."
                ),
            }
        )
    return verdicts


def _evaluate_case_expectations(
    case: dict[str, Any],
    result: WorkflowExampleResult,
) -> dict[str, bool]:
    checks: dict[str, bool] = {"trace_present": bool(result.trace)}
    if "expected_intent" in case:
        checks["expected_intent"] = result.state.get("selected_intent") == case["expected_intent"]
    if "expected_allowed" in case:
        checks["expected_allowed"] = result.state.get("allowed") == case["expected_allowed"]
    if "expected_min_resources" in case:
        resource_count = result.state.get("resource_count")
        checks["expected_min_resources"] = (
            isinstance(resource_count, int) and resource_count >= case["expected_min_resources"]
        )
    if "expected_review_count" in case:
        checks["expected_review_count"] = (
            result.state.get("review_count") == case["expected_review_count"]
        )
    if "expected_min_rounds" in case:
        rounds = result.state.get("rounds_completed")
        checks["expected_min_rounds"] = (
            isinstance(rounds, int) and rounds >= case["expected_min_rounds"]
        )
    if "expected_guardrail" in case:
        raw_guardrails = result.state.get("guardrails", [])
        guardrail_items = raw_guardrails if isinstance(raw_guardrails, list) else []
        guardrails = " ".join(str(note) for note in guardrail_items)
        checks["expected_guardrail"] = str(case["expected_guardrail"]) in guardrails
    return checks


def _judge_summary(verdicts: list[dict[str, Any]]) -> dict[str, int]:
    passed = sum(1 for verdict in verdicts if verdict.get("verdict") == "pass")
    failed = len(verdicts) - passed
    return {"passed": passed, "failed": failed, "total": len(verdicts)}


def _render_failure_report(question: str, judge_payload: dict[str, Any]) -> str:
    verdict_lines = "\n".join(
        f"- `{verdict.get('case_id', verdict['workflow_name'])}`: `{verdict['verdict']}`"
        for verdict in judge_payload["verdicts"]
    )
    return (
        "# Harness Failure Injection Report\n\n"
        f"Question: {question}\n\n"
        "## Simulated Failures\n\n"
        "- Tool failure: the course catalog returns no matches, so the verifier should "
        "reject an empty resource list.\n"
        "- Guardrail trip: a prompt includes an API key or secret, so the policy agent "
        "blocks the request.\n"
        "- Routing ambiguity: a question mixes project planning and debugging, so the "
        "router trace must show the selected specialist.\n"
        "- Loop runaway: a refinement loop must stop after the bounded quality threshold "
        "instead of iterating forever.\n"
        "- Trace corruption: a missing trace event should fail `make trace-check` before "
        "a student trusts the answer.\n\n"
        "## Current Judge Verdicts\n\n"
        f"{verdict_lines}\n"
    )


def _append_run_ledger(
    run_ledger_path: Path,
    question: str,
    judge_payload: dict[str, Any],
) -> None:
    ledger_entry = {
        "timestamp_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "question": question,
        "event": "harness_lab_run",
        "judge_summary": judge_payload["summary"],
        "artifact_paths": list(REQUIRED_HARNESS_FILES),
        "status": "pass" if judge_payload["summary"]["failed"] == 0 else "fail",
    }
    with run_ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(ledger_entry, sort_keys=True) + "\n")


def _live_config_summary(project_root: Path) -> dict[str, Any]:
    config = load_runtime_config(project_root)
    return {
        "openai_enabled": config.openai_enabled,
        "gemini_enabled": config.gemini_enabled,
        "openai_model": config.openai_model,
        "gemini_model": config.gemini_model,
    }
