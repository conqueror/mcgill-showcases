from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

from pytest import MonkeyPatch

from agentic_course_assistant.artifact_contract import verify
from agentic_course_assistant.openai_live_artifacts import LIVE_TRACE_SOURCE, run_live_openai_bundle


def test_openai_agents_module_imports_without_optional_sdk() -> None:
    module = importlib.import_module("agentic_course_assistant.openai_agents_example")

    assert hasattr(module, "run_openai_agents_course_assistant")
    assert hasattr(module, "run_openai_specialist_course_assistant")


def test_live_openai_bundle_reuses_offline_contract(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    async def fake_specialist_runner(
        question: str,
        *,
        intent: str | None = None,
        resource_context: str | None = None,
    ) -> str:
        assert question
        assert intent == "debug"
        assert resource_context
        return "Hosted debug mentor answer grounded in the local course catalog."

    monkeypatch.setattr(
        "agentic_course_assistant.openai_live_artifacts.run_openai_specialist_course_assistant",
        fake_specialist_runner,
    )

    summary = asyncio.run(
        run_live_openai_bundle(
            "How should I debug a suspicious validation score?",
            tmp_path,
        )
    )

    assert summary["verification_errors"] == []
    trace_payload = json.loads(
        (tmp_path / "artifacts/agent_trace.json").read_text(encoding="utf-8")
    )
    assert trace_payload["trace_source"] == LIVE_TRACE_SOURCE
    assert trace_payload["runtime"] == "openai_agents_sdk"
    assert "not a raw openai trace" in trace_payload["sdk_trace_note"].lower()
    assert any("teaching_adapter." in step for step in trace_payload["trace"])
    assert trace_payload["resource_ids"]
    response_text = (tmp_path / "artifacts/course_assistant_response.md").read_text(
        encoding="utf-8"
    )
    assert "The local teaching adapter performs the intent selection" in response_text
    assert verify(tmp_path, require_harness=False) == []

    manifest = json.loads((tmp_path / "artifacts/manifest.json").read_text(encoding="utf-8"))
    assert "artifacts/course_assistant_response.md" in manifest["required_files"]
    assert "artifacts/evals/agent_judge_rubric.json" in manifest["required_files"]
