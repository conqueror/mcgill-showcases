"""Artifact writers for the agentic course assistant showcase."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from agentic_course_assistant.assistant import AssistantResult
from agentic_course_assistant.concept_atlas import write_concept_artifacts


def write_artifacts(result: AssistantResult, output_dir: Path) -> list[Path]:
    """Write stable, student-readable artifacts for one assistant run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "course_assistant_response.md"
    trace_path = output_dir / "agent_trace.json"
    matches_path = output_dir / "resource_matches.csv"
    concept_paths = write_concept_artifacts(output_dir)

    response_path.write_text(_render_markdown(result), encoding="utf-8")
    trace_path.write_text(
        json.dumps(
            {
                "question": result.question,
                "intent": result.intent,
                "agent_name": result.agent_name,
                "guardrails": list(result.guardrails),
                "harness_events": _harness_events(result),
                "trace": list(result.trace),
                "resource_ids": [resource.resource_id for resource in result.resources],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with matches_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["resource_id", "title", "topic", "level", "kind", "skills"],
        )
        writer.writeheader()
        for resource in result.resources:
            writer.writerow(
                {
                    "resource_id": resource.resource_id,
                    "title": resource.title,
                    "topic": resource.topic,
                    "level": resource.level,
                    "kind": resource.kind,
                    "skills": "; ".join(resource.skills),
                }
            )
    return [response_path, trace_path, matches_path, *concept_paths]


def _harness_events(result: AssistantResult) -> list[dict[str, str]]:
    trace_text = " ".join(result.trace)
    return [
        {
            "event_type": "tool_call",
            "name": _require_trace_step(trace_text, "course_catalog_tool.search_resources"),
        },
        {
            "event_type": "handoff",
            "name": _require_trace_step(trace_text, result.agent_name.lower().replace(" ", "_")),
        },
        {
            "event_type": "guardrail_check",
            "name": _require_trace_step(trace_text, "guardrail.scope_check"),
        },
    ]


def _require_trace_step(trace_text: str, expected_step: str) -> str:
    if expected_step not in trace_text:
        raise ValueError(f"trace is missing required step: {expected_step}")
    return expected_step


def _render_markdown(result: AssistantResult) -> str:
    resource_lines = "\n".join(
        f"- `{resource.resource_id}`: {resource.title} ({resource.kind})"
        for resource in result.resources
    )
    trace_lines = "\n".join(f"- `{step}`" for step in result.trace)
    return (
        "# Agentic Course Assistant Response\n\n"
        f"## Question\n{result.question}\n\n"
        f"## Route\n- Intent: `{result.intent}`\n- Specialist: `{result.agent_name}`\n\n"
        f"## Answer\n{result.answer}\n\n"
        f"## Resource Matches\n{resource_lines}\n\n"
        f"## Trace\n{trace_lines}\n"
    )
