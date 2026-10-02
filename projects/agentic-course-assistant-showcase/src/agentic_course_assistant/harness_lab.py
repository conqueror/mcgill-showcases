"""Generate public-safe harness lab artifacts for the course assistant showcase."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from agentic_course_assistant.artifact_contract import verify
from agentic_course_assistant.artifact_manifest import (
    BASE_REQUIRED_FILES,
    REQUIRED_HARNESS_FILES,
    _run_identity,
    all_required_files,
    merge_required_files,
)
from agentic_course_assistant.artifacts import write_artifacts
from agentic_course_assistant.assistant import answer_question
from agentic_course_assistant.runtime_config import load_runtime_config
from agentic_course_assistant.workflow_examples import (
    MAX_REFINEMENT_ROUNDS,
    WorkflowExampleResult,
    run_offline_workflows,
)

DEFAULT_HARNESS_QUESTION = (
    "Help me build an agent project for debugging leakage without pasting any API keys."
)

TRACE_SCHEMA: dict[str, Any] = {
    "schema_name": "agentic_course_assistant_trace",
    "version": 1,
    "required_top_level_keys": [
        "question",
        "run",
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
        "expected_max_rounds": MAX_REFINEMENT_ROUNDS,
        "expected_stop_reason": "quality_threshold",
    },
)

__all__ = [
    "DEFAULT_HARNESS_QUESTION",
    "EVAL_CASES",
    "REQUIRED_HARNESS_FILES",
    "TRACE_SCHEMA",
    "run_harness_lab",
    "write_showcase_artifacts",
]


def run_harness_lab(
    project_root: Path,
    question: str = DEFAULT_HARNESS_QUESTION,
) -> dict[str, Any]:
    """Generate deterministic harness artifacts and return a summary."""

    summary = write_showcase_artifacts(question, project_root / "artifacts")
    summary["verification_errors"] = verify(project_root)
    summary["live_config"] = _live_config_summary(project_root)
    return summary


def write_showcase_artifacts(question: str, artifacts_dir: Path) -> dict[str, Any]:
    """Refresh the assistant and harness artifacts together for one question."""

    base_paths = write_artifacts(answer_question(question), artifacts_dir)

    workflows = run_offline_workflows(question)
    run = _run_identity(question)
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

    trace_schema_path.write_text(
        json.dumps({**TRACE_SCHEMA, "run": run}, indent=2) + "\n", encoding="utf-8"
    )
    eval_cases_path.write_text(
        "".join(json.dumps({**case, "run": run}, sort_keys=True) + "\n" for case in EVAL_CASES),
        encoding="utf-8",
    )
    judge_payload = {
        "version": 1,
        "judge": "deterministic_harness_judge",
        "question": question,
        "run": run,
        "workflows": {
            name: result.to_dict() for name, result in workflows.items()
        },
        "verdicts": judge_verdicts,
        "summary": _judge_summary(judge_verdicts),
        "failure_injections": _run_failure_injections(question, workflows),
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
    manifest_path = artifacts_dir / "manifest.json"
    merge_required_files(manifest_path, all_required_files())

    return {
        "question": question,
        "harness_dir": str(harness_dir),
        "judge_summary": judge_payload["summary"],
        "written_files": [
            str(path) for path in (
                *base_paths,
                trace_schema_path,
                eval_cases_path,
                judge_verdicts_path,
                failure_report_path,
                run_ledger_path,
                manifest_path,
            )
        ],
    }


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
            checks["bounded_loop_completed"] = (
                type(rounds) is int and 1 <= rounds <= MAX_REFINEMENT_ROUNDS
            )
            checks["stop_reason_present"] = result.state.get("stop_reason") in {
                "quality_threshold", "round_limit"
            }
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
        result = run_offline_workflows(str(case["prompt"])).get(str(case["expected_workflow"]))
        checks = (
            _evaluate_case_expectations(case, result)
            if result is not None else {"workflow_allowed": False}
        )
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
    if "expected_max_rounds" in case:
        rounds = result.state.get("rounds_completed")
        checks["expected_max_rounds"] = (
            type(rounds) is int and 1 <= rounds <= case["expected_max_rounds"]
        )
    if "expected_stop_reason" in case:
        checks["expected_stop_reason"] = (
            result.state.get("stop_reason") == case["expected_stop_reason"]
        )
    if "expected_guardrail" in case:
        raw_guardrails = result.state.get("guardrails", [])
        guardrail_items = raw_guardrails if isinstance(raw_guardrails, list) else []
        guardrails = " ".join(str(note) for note in guardrail_items)
        checks["expected_guardrail"] = str(case["expected_guardrail"]) in guardrails
    return checks


def _judge_summary(verdicts: list[dict[str, Any]]) -> dict[str, int]:
    passed = sum(
        1 for verdict in verdicts
        if verdict.get("checks") and all(value is True for value in verdict["checks"].values())
    )
    failed = len(verdicts) - passed
    return {"passed": passed, "failed": failed, "total": len(verdicts)}


def _run_failure_injections(
    question: str, workflows: dict[str, WorkflowExampleResult]
) -> dict[str, dict[str, object]]:
    """Exercise broken outputs at the verifier, policy, and judge boundaries."""

    outcomes: dict[str, dict[str, object]] = {}
    with TemporaryDirectory(prefix="course-assistant-injections-") as directory:
        root = Path(directory)
        artifacts_dir = root / "artifacts"
        result = answer_question(question)
        write_artifacts(replace(result, resources=()), artifacts_dir)
        merge_required_files(artifacts_dir / "manifest.json", BASE_REQUIRED_FILES)
        errors = verify(root, require_harness=False)
        outcomes["Tool failure"] = {
            "passed": any("resource" in error for error in errors),
            "observed": errors,
        }
        write_artifacts(result, artifacts_dir)
        trace_path = artifacts_dir / "agent_trace.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        trace["trace"].remove("course_catalog_tool.search_resources")
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        errors = verify(root, require_harness=False)
        outcomes["Trace corruption"] = {
            "passed": any("trace missing step for tool_call" in error for error in errors),
            "observed": errors,
        }

    secret_case = next(case for case in EVAL_CASES if case["case_id"] == "block_secret_request")
    try:
        answer_question(str(secret_case["prompt"]))
    except ValueError as exc:
        outcomes["Guardrail trip"] = {"passed": True, "observed": str(exc)}
    else:
        outcomes["Guardrail trip"] = {"passed": False, "observed": "Input was accepted."}

    ambiguous = run_offline_workflows("project debug")["router_triage"]
    outcomes["Routing ambiguity"] = {
        "passed": ambiguous.state["selected_intent"] == "debug",
        "observed": f"Tied project/debug keywords selected {ambiguous.state['selected_intent']}.",
    }
    loop = workflows["loop_refinement"]
    runaway = replace(loop, state={**loop.state, "rounds_completed": MAX_REFINEMENT_ROUNDS + 1})
    verdict = _judge_workflows({"loop_refinement": runaway})[0]
    outcomes["Loop runaway"] = {
        "passed": verdict["verdict"] == "fail",
        "observed": verdict["checks"],
    }
    return outcomes


def _render_failure_report(question: str, judge_payload: dict[str, Any]) -> str:
    verdict_lines = "\n".join(
        f"- `{verdict.get('case_id', verdict['workflow_name'])}`: `{verdict['verdict']}`"
        for verdict in judge_payload["verdicts"]
    )
    injection_lines = "\n".join(
        f"- {name}: `{'pass' if outcome['passed'] else 'fail'}`; "
        f"observed: {json.dumps(outcome['observed'], sort_keys=True)}"
        for name, outcome in judge_payload["failure_injections"].items()
    )
    return (
        "# Harness Failure Injection Report\n\n"
        f"Question: {question}\n\n"
        f"Run ID: {judge_payload['run']['run_id']}\n\n"
        "## Executed Boundary Checks\n\n"
        "The tool and trace checks inject corrupt outputs into the artifact verifier. "
        "The loop check injects an excessive round count into the workflow judge. "
        "The policy check sends a flagged prompt through the assistant, and the routing "
        "check exercises a keyword tie.\n\n"
        f"{injection_lines}\n\n"
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
        "run": judge_payload["run"],
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
