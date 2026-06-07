"""Build the opt-in hosted artifact bundle for the OpenAI Agents SDK path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_course_assistant.artifact_contract import verify
from agentic_course_assistant.artifact_manifest import BASE_REQUIRED_FILES, merge_required_files
from agentic_course_assistant.artifacts import write_artifacts
from agentic_course_assistant.assistant import (
    AGENT_BY_INTENT,
    AssistantResult,
    classify_question,
    guardrail_notes,
)
from agentic_course_assistant.course_catalog import CourseResource, search_resources
from agentic_course_assistant.openai_agents_example import run_openai_specialist_course_assistant

LIVE_TRACE_SOURCE = "hosted_response_with_local_teaching_adapter"


async def run_live_openai_bundle(question: str, bundle_root: Path) -> dict[str, Any]:
    """Write the hosted bundle in the same format students inspect offline."""

    bundle_root.mkdir(parents=True, exist_ok=True)
    artifacts_dir = bundle_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    intent = classify_question(question)
    agent_name = AGENT_BY_INTENT[intent]
    resources = tuple(search_resources(question))
    answer = await run_openai_specialist_course_assistant(
        question,
        intent=intent,
        resource_context=_render_resource_context(resources),
    )
    result = AssistantResult(
        question=question,
        intent=intent,
        agent_name=agent_name,
        answer=answer,
        resources=resources,
        guardrails=guardrail_notes(question),
        trace=(
            "teaching_adapter.received_question",
            f"teaching_adapter.selected_intent:{intent}",
            "course_catalog_tool.search_resources",
            f"teaching_adapter.selected_specialist:{agent_name.lower().replace(' ', '_')}",
            f"{agent_name.lower().replace(' ', '_')}.hosted_response",
            "guardrail.scope_check",
        ),
    )
    written = write_artifacts(
        result,
        artifacts_dir,
        response_note=(
            "This bundle keeps the offline teaching contract while sourcing only the final answer "
            "from a hosted OpenAI Agents SDK specialist. The local teaching adapter performs the "
            "intent selection and course-catalog grounding steps before the hosted call. This "
            "trace is a comparable student artifact, not a raw OpenAI platform trace dump."
        ),
        trace_extras={
            "runtime": "openai_agents_sdk",
            "trace_source": LIVE_TRACE_SOURCE,
            "sdk_trace_note": (
                "Comparable student-facing trace for a hosted OpenAI Agents SDK run. "
                "Intent selection and course-catalog grounding are performed by the local "
                "teaching adapter before the hosted specialist call. This is not a raw OpenAI "
                "trace export."
            ),
        },
    )
    run_summary_path = artifacts_dir / "openai_run_summary.json"
    run_summary = {
        "question": question,
        "intent": intent,
        "agent_name": agent_name,
        "resource_ids": [resource.resource_id for resource in resources],
        "runtime": "openai_agents_sdk",
        "trace_source": LIVE_TRACE_SOURCE,
        "bundle_root": str(bundle_root),
        "written_files": [str(path) for path in written],
    }
    run_summary_path.write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    merge_required_files(artifacts_dir / "manifest.json", BASE_REQUIRED_FILES)
    verification_errors = verify(bundle_root, require_harness=False)
    return {
        **run_summary,
        "manifest_path": str(artifacts_dir / "manifest.json"),
        "verification_errors": verification_errors,
    }


def _render_resource_context(resources: tuple[CourseResource, ...]) -> str:
    resource_lines = "\n".join(
        f"- {resource.title}: {resource.summary}" for resource in resources
    )
    return (
        "Ground the answer in the course resources below when they are relevant.\n\n"
        f"{resource_lines}"
    )
