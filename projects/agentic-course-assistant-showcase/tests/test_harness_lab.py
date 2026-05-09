from __future__ import annotations

import json
import tomllib
from pathlib import Path

from pytest import MonkeyPatch

from agentic_course_assistant.artifact_contract import verify
from agentic_course_assistant.artifacts import write_artifacts
from agentic_course_assistant.assistant import answer_question
from agentic_course_assistant.harness_lab import REQUIRED_HARNESS_FILES, run_harness_lab
from agentic_course_assistant.runtime_config import apply_live_environment, load_runtime_config
from agentic_course_assistant.workflow_examples import custom_policy_agent, run_offline_workflows


def test_offline_workflows_cover_required_patterns() -> None:
    results = run_offline_workflows("Help me build an agent project for debugging leakage.")

    assert set(results) == {
        "sequential_course_plan",
        "loop_refinement",
        "parallel_resource_review",
        "router_triage",
        "custom_policy_agent",
    }
    assert results["router_triage"].state["selected_intent"] == "project"
    assert results["parallel_resource_review"].state["review_count"] == 3
    rounds_completed = results["loop_refinement"].state["rounds_completed"]
    assert isinstance(rounds_completed, int)
    assert rounds_completed >= 2
    assert all(result.trace for result in results.values())


def test_custom_policy_agent_blocks_secret_requests() -> None:
    result = custom_policy_agent("My API key is broken. Can you fix it with this secret?")

    assert result.state["allowed"] is False
    assert "blocked" in result.summary.lower()


def test_load_runtime_config_reads_env_file_and_defaults(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-openai-key",
                "GEMINI_API_KEY=test-gemini-key",
                "OPENAI_MODEL=gpt-5.4-mini",
                "GEMINI_MODEL=gemini-3.1-flash-lite-preview",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_runtime_config(tmp_path)

    assert config.openai_api_key == "test-openai-key"
    assert config.gemini_api_key == "test-gemini-key"
    assert config.openai_model == "gpt-5.4-mini"
    assert config.gemini_model == "gemini-3.1-flash-lite-preview"


def test_apply_live_environment_exports_sdk_aliases(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=test-openai-key",
                "GEMINI_API_KEY=test-gemini-key",
                "OPENAI_MODEL=gpt-5.4-mini",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    environ: dict[str, str] = {}

    config = apply_live_environment(tmp_path, environ=environ)

    assert config.openai_enabled is True
    assert config.gemini_enabled is True
    assert environ["OPENAI_API_KEY"] == "test-openai-key"
    assert environ["GEMINI_API_KEY"] == "test-gemini-key"
    assert environ["GOOGLE_API_KEY"] == "test-gemini-key"
    assert environ["OPENAI_DEFAULT_MODEL"] == "gpt-5.4-mini"


def test_apply_live_environment_accepts_direct_google_api_key(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "GOOGLE_API_KEY=test-google-key",
                "GEMINI_MODEL=gemini-3.1-flash-lite-preview",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    environ: dict[str, str] = {}

    config = apply_live_environment(tmp_path, environ=environ)

    assert config.gemini_enabled is True
    assert config.gemini_api_key == "test-google-key"
    assert environ["GEMINI_API_KEY"] == "test-google-key"
    assert environ["GOOGLE_API_KEY"] == "test-google-key"


def test_run_harness_lab_writes_manifest_and_harness_artifacts(tmp_path: Path) -> None:
    summary = run_harness_lab(tmp_path)

    manifest = json.loads((tmp_path / "artifacts/manifest.json").read_text(encoding="utf-8"))

    for relative_path in REQUIRED_HARNESS_FILES:
        assert relative_path in manifest["required_files"]
        assert (tmp_path / relative_path).exists()

    assert summary["judge_summary"]["passed"] == 10
    errors = verify(tmp_path)
    assert errors == []


def test_verify_artifacts_rejects_corrupted_harness_outputs(tmp_path: Path) -> None:
    run_harness_lab(tmp_path)
    (tmp_path / "artifacts/harness/trace_schema.json").write_text("{}", encoding="utf-8")
    (tmp_path / "artifacts/harness/run_ledger.jsonl").write_text("", encoding="utf-8")

    errors = verify(tmp_path)

    assert any("trace_schema.json" in error for error in errors)
    assert any("run_ledger.jsonl" in error for error in errors)


def test_run_harness_lab_enforces_golden_eval_expectations(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    import agentic_course_assistant.harness_lab as harness_lab

    impossible_cases = tuple(
        {
            **case,
            "expected_workflow": "router_triage",
            "expected_intent": "not-a-real-intent",
        }
        for case in harness_lab.EVAL_CASES
    )
    monkeypatch.setattr(harness_lab, "EVAL_CASES", impossible_cases)

    summary = harness_lab.run_harness_lab(tmp_path)

    assert summary["judge_summary"]["failed"] >= 5
    assert summary["verification_errors"]


def test_run_harness_lab_refreshes_base_trace_for_current_question(tmp_path: Path) -> None:
    first_question = "Explain leakage as a concept."
    second_question = "Help me build an agent project for debugging leakage."

    run_harness_lab(tmp_path, question=first_question)
    run_harness_lab(tmp_path, question=second_question)

    trace_payload = json.loads((tmp_path / "artifacts/agent_trace.json").read_text())
    assert trace_payload["question"] == second_question


def test_verify_artifacts_rejects_trace_missing_schema_events(tmp_path: Path) -> None:
    run_harness_lab(tmp_path)
    trace_path = tmp_path / "artifacts/agent_trace.json"
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_payload["harness_events"] = []
    trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")

    errors = verify(tmp_path)

    assert any("schema-required events" in error or "harness_events" in error for error in errors)


def test_verify_artifacts_rejects_harness_event_trace_drift(tmp_path: Path) -> None:
    run_harness_lab(tmp_path)
    trace_path = tmp_path / "artifacts/agent_trace.json"
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_payload["trace"] = [
        step for step in trace_payload["trace"] if step != "course_catalog_tool.search_resources"
    ]
    trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")

    errors = verify(tmp_path)

    assert any("trace missing step for tool_call" in error for error in errors)


def test_verify_artifacts_rejects_lifecycle_events_in_agent_trace(tmp_path: Path) -> None:
    run_harness_lab(tmp_path)
    trace_path = tmp_path / "artifacts/agent_trace.json"
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_payload["harness_events"].append(
        {"event_type": "judge_verdict", "name": "deterministic_artifact_contract"}
    )
    trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")

    errors = verify(tmp_path)

    assert any("unexpected harness events" in error for error in errors)


def test_verify_artifacts_rejects_missing_lifecycle_artifact_schema(tmp_path: Path) -> None:
    run_harness_lab(tmp_path)
    trace_schema_path = tmp_path / "artifacts/harness/trace_schema.json"
    trace_schema = json.loads(trace_schema_path.read_text(encoding="utf-8"))
    trace_schema.pop("harness_lifecycle_artifacts")
    trace_schema_path.write_text(json.dumps(trace_schema), encoding="utf-8")

    errors = verify(tmp_path)

    assert any("harness lifecycle artifacts" in error for error in errors)


def test_verify_artifacts_rejects_manifest_harness_opt_out(tmp_path: Path) -> None:
    run_harness_lab(tmp_path)
    manifest_path = tmp_path / "artifacts/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["required_files"] = [
        path for path in manifest["required_files"] if not path.startswith("artifacts/harness/")
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    errors = verify(tmp_path)

    assert any("missing canonical files" in error for error in errors)


def test_verify_artifacts_rejects_non_canonical_eval_cases(tmp_path: Path) -> None:
    run_harness_lab(tmp_path)
    eval_cases_path = tmp_path / "artifacts/harness/eval_cases.jsonl"
    drift_cases = [
        {
            "case_id": f"drift_{index}",
            "prompt": "This should not replace the canonical harness eval set.",
            "expected_workflow": "router_triage",
        }
        for index in range(5)
    ]
    eval_cases_path.write_text(
        "".join(json.dumps(case, sort_keys=True) + "\n" for case in drift_cases),
        encoding="utf-8",
    )

    errors = verify(tmp_path)

    assert any("missing canonical eval cases" in error for error in errors)
    assert any("non-canonical eval cases" in error for error in errors)


def test_harness_events_are_derived_from_trace(tmp_path: Path) -> None:
    result = answer_question("Help me debug leakage.")
    altered_trace_result = result.__class__(
        question=result.question,
        intent=result.intent,
        agent_name=result.agent_name,
        answer=result.answer,
        resources=result.resources,
        guardrails=result.guardrails,
        trace=tuple(
            step for step in result.trace if step != "course_catalog_tool.search_resources"
        ),
    )

    try:
        write_artifacts(altered_trace_result, tmp_path / "artifacts")
    except ValueError as exc:
        assert "course_catalog_tool.search_resources" in str(exc)
    else:
        raise AssertionError("write_artifacts should reject missing tool-call trace evidence")


def test_project_files_document_harness_targets_and_env_defaults() -> None:
    project_root = Path(__file__).resolve().parents[1]
    makefile_text = (project_root / "Makefile").read_text(encoding="utf-8")
    env_example_text = (project_root / ".env.example").read_text(encoding="utf-8")
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert "eval:" in makefile_text
    assert "trace-check:" in makefile_text
    assert "harness-report:" in makefile_text
    assert "OPENAI_API_KEY=" in env_example_text
    assert "GEMINI_API_KEY=" in env_example_text
    assert "OPENAI_MODEL=gpt-5.4-mini" in env_example_text
    assert "GEMINI_MODEL=gemini-3.1-flash-lite-preview" in env_example_text
    assert "live" in pyproject["project"]["optional-dependencies"]
