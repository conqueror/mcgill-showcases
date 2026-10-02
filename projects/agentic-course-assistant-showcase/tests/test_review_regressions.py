from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from agentic_course_assistant.artifact_contract import verify
from agentic_course_assistant.artifacts import write_artifacts
from agentic_course_assistant.assistant import answer_question
from agentic_course_assistant.course_catalog import search_resources
from agentic_course_assistant.harness_lab import (
    EVAL_CASES,
    _evaluate_case_expectations,
    _judge_workflows,
    run_harness_lab,
)
from agentic_course_assistant.openai_live_artifacts import run_live_openai_bundle
from agentic_course_assistant.runtime_config import apply_live_environment, load_runtime_config
from agentic_course_assistant.workflow_examples import loop_refinement, run_offline_workflows

SECRET_QUESTION = "My API key is synthetic-lane02-secret. Help me debug it."


def test_secret_is_rejected_before_lookup_or_persistence(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    result = answer_question("Explain leakage")

    def forbidden_lookup(*args: object, **kwargs: object) -> None:
        raise AssertionError("blocked input reached the catalog")

    monkeypatch.setattr("agentic_course_assistant.assistant.search_resources", forbidden_lookup)
    with pytest.raises(ValueError, match="sensitive"):
        answer_question(SECRET_QUESTION)
    with pytest.raises(ValueError, match="sensitive"):
        write_artifacts(replace(result, question=SECRET_QUESTION), tmp_path)
    assert not list(tmp_path.iterdir())


def test_policy_blocks_sibling_workflows(monkeypatch: MonkeyPatch) -> None:
    def forbidden_plan(question: str) -> None:
        raise AssertionError("blocked input reached a sibling workflow")

    monkeypatch.setattr(
        "agentic_course_assistant.workflow_examples.sequential_course_plan", forbidden_plan
    )
    results = run_offline_workflows(SECRET_QUESTION)
    assert set(results) == {"custom_policy_agent"}
    assert results["custom_policy_agent"].state["allowed"] is False


def test_secret_is_rejected_before_hosted_call(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    calls: list[str] = []

    async def fake_runner(question: str, **kwargs: object) -> str:
        calls.append(question)
        return "Synthetic hosted answer."

    monkeypatch.setattr(
        "agentic_course_assistant.openai_live_artifacts.run_openai_specialist_course_assistant",
        fake_runner,
    )
    with pytest.raises(ValueError, match="sensitive"):
        asyncio.run(run_live_openai_bundle(SECRET_QUESTION, tmp_path))
    assert calls == []
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "runner_name",
    ["run_openai_agents_course_assistant", "run_openai_specialist_course_assistant"],
)
def test_direct_sdk_entry_points_reject_secrets(
    runner_name: str, monkeypatch: MonkeyPatch
) -> None:
    from agentic_course_assistant import openai_agents_example

    def forbidden_bundle() -> None:
        raise AssertionError("blocked input reached SDK construction")

    monkeypatch.setattr(openai_agents_example, "_build_agents_bundle", forbidden_bundle)
    runner = getattr(openai_agents_example, runner_name)
    with pytest.raises(ValueError, match="sensitive"):
        asyncio.run(runner(SECRET_QUESTION))


def test_empty_environment_mapping_is_isolated(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-process-key")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-process-gemini-key")
    monkeypatch.setenv("OPENAI_MODEL", "synthetic-process-model")
    config = load_runtime_config(tmp_path, environ={})
    assert config.openai_api_key is None
    assert config.gemini_api_key is None
    assert config.openai_model != "synthetic-process-model"
    target: dict[str, str] = {}
    apply_live_environment(tmp_path, environ=target)
    assert "OPENAI_API_KEY" not in target
    assert "GEMINI_API_KEY" not in target


@pytest.mark.parametrize("corrupt_trace", ["{", "[]", "null", "42", '{"harness_events": 4}'])
def test_corrupt_trace_returns_errors(tmp_path: Path, corrupt_trace: str) -> None:
    run_harness_lab(tmp_path)
    (tmp_path / "artifacts/agent_trace.json").write_text(corrupt_trace, encoding="utf-8")
    errors = verify(tmp_path)
    assert any("agent_trace.json" in error for error in errors)


@pytest.mark.parametrize("check_value", [False, "true", 1, None])
def test_passing_label_cannot_hide_bad_checks(tmp_path: Path, check_value: object) -> None:
    run_harness_lab(tmp_path)
    path = tmp_path / "artifacts/harness/judge_verdicts.json"
    payload = json.loads(path.read_text())
    payload["verdicts"][0]["checks"]["trace_present"] = check_value
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert any("checks" in error for error in verify(tmp_path))


def test_judge_summary_is_derived_from_checks(tmp_path: Path) -> None:
    run_harness_lab(tmp_path)
    path = tmp_path / "artifacts/harness/judge_verdicts.json"
    payload = json.loads(path.read_text())
    payload["summary"] = {"passed": 999, "failed": 0, "total": 999}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert any("summary" in error for error in verify(tmp_path))


def test_fresh_base_cannot_use_stale_harness(tmp_path: Path) -> None:
    run_harness_lab(tmp_path, question="Explain leakage")
    write_artifacts(answer_question("Help me debug validation"), tmp_path / "artifacts")
    assert any("run identity" in error for error in verify(tmp_path))


@pytest.mark.parametrize("artifact", ["judge_verdicts.json", "run_ledger.jsonl"])
def test_harness_requires_current_source_and_scenario(tmp_path: Path, artifact: str) -> None:
    run_harness_lab(tmp_path)
    path = tmp_path / "artifacts/harness" / artifact
    lines = path.read_text().splitlines()
    payload = json.loads(lines[-1] if artifact.endswith("jsonl") else path.read_text())
    payload["run"] = {"run_id": "stale", "source_sha256": "stale", "scenario_version": 0}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert any("run identity" in error for error in verify(tmp_path))


def test_bounded_loop_judges_reject_runaway_count() -> None:
    result = loop_refinement("Refine the plan")
    runaway = replace(result, state={**result.state, "rounds_completed": 100})
    assert _judge_workflows({"loop_refinement": runaway})[0]["verdict"] == "fail"
    case = next(case for case in EVAL_CASES if case["case_id"] == "bounded_loop")
    assert not all(_evaluate_case_expectations(case, runaway).values())


def test_refinement_stops_when_quality_never_passes(monkeypatch: MonkeyPatch) -> None:
    import agentic_course_assistant.workflow_examples as workflows

    calls: list[str] = []

    def never_ready(draft: str) -> bool:
        calls.append(draft)
        return False

    monkeypatch.setattr(workflows, "_draft_meets_quality", never_ready, raising=False)
    result = workflows.loop_refinement("Refine the plan")
    assert len(calls) == 2
    assert result.state["rounds_completed"] == 2
    assert result.state["stop_reason"] == "round_limit"
    drafts = result.state["draft_rounds"]
    assert isinstance(drafts, list)
    assert len(drafts) == 2


def test_failure_report_records_executed_injections(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    import agentic_course_assistant.harness_lab as harness

    # A broken verifier must make the injected empty catalog and corrupt trace fail.
    monkeypatch.setattr(harness, "verify", lambda *args, **kwargs: [])
    harness.run_harness_lab(tmp_path)
    payload = json.loads((tmp_path / "artifacts/harness/judge_verdicts.json").read_text())
    injections = payload["failure_injections"]
    assert injections["Tool failure"]["passed"] is False
    assert injections["Trace corruption"]["passed"] is False
    report = (tmp_path / "artifacts/harness/failure_injection_report.md").read_text()
    assert "Tool failure: `fail`" in report
    assert "Trace corruption: `fail`" in report


def test_catalog_abstains_without_overlap() -> None:
    assert search_resources("zzzz-unmatched") == []
    assert search_resources("") == []
    result = answer_question("zzzz-unmatched")
    assert result.resources == ()
    assert "No matching course resources" in result.answer


def test_live_persisted_paths_do_not_depend_on_bundle_location(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    async def fake_runner(question: str, **kwargs: object) -> str:
        return "Synthetic hosted answer."

    monkeypatch.setattr(
        "agentic_course_assistant.openai_live_artifacts.run_openai_specialist_course_assistant",
        fake_runner,
    )
    for name in ("first", "second"):
        asyncio.run(run_live_openai_bundle("Explain leakage", tmp_path / name))
    first = (tmp_path / "first/artifacts/openai_run_summary.json").read_bytes()
    second = (tmp_path / "second/artifacts/openai_run_summary.json").read_bytes()
    assert first == second


def test_quickstart_refreshes_harness_before_verification() -> None:
    project_root = Path(__file__).resolve().parents[1]
    for document in (project_root / "README.md", project_root / "docs/lab-guide.md"):
        text = document.read_text()
        assert "make smoke\nmake eval\nmake verify" in text


@pytest.mark.parametrize(
    "statement",
    [
        "does not test implemented capabilities",
        "reviewer records are sequential simulations",
        "neither example registers an SDK guardrail callback",
    ],
)
def test_readme_states_the_teaching_boundaries(statement: str) -> None:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    assert statement in readme.read_text()
