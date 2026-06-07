"""Optional OpenAI Agents SDK version of the course assistant.

Install the `openai` optional extra and set `OPENAI_API_KEY` before importing
this module. The default showcase path stays offline, but this file gives
students a small hosted example to compare with the deterministic implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from agentic_course_assistant.assistant import Intent, classify_question
from agentic_course_assistant.course_catalog import search_resources
from agentic_course_assistant.runtime_config import apply_live_environment

_RUNTIME_CONFIG = apply_live_environment()


def lookup_course_resources(question: str) -> str:
    """Return course resources that match a student question."""

    resources = search_resources(question)
    return "\n".join(
        f"{resource.resource_id}: {resource.title} - {resource.summary}" for resource in resources
    )


@dataclass(frozen=True)
class OpenAIAgentsBundle:
    """Small container for the lazily built SDK agents and runner."""

    runner: Any
    triage_agent: Any
    agents_by_intent: dict[Intent, Any]


def _load_openai_agents_sdk() -> tuple[Any, Any, Any]:
    try:
        module = import_module("agents")
    except ImportError as exc:  # pragma: no cover - optional SDK dependency
        raise RuntimeError(
            "Install the optional OpenAI Agents SDK with "
            "`uv sync --extra dev --extra openai`."
        ) from exc
    return module.Agent, module.Runner, module.function_tool


def _build_agents_bundle() -> OpenAIAgentsBundle:
    Agent, Runner, function_tool = _load_openai_agents_sdk()
    resource_tool = function_tool(lookup_course_resources)
    concept_coach = Agent(
        name="Concept coach",
        model=_RUNTIME_CONFIG.openai_model,
        handoff_description="Explains ML and agentic AI concepts clearly.",
        instructions=(
            "Explain the concept, name the artifact to inspect, and keep the answer concise."
        ),
        tools=[resource_tool],
    )
    practice_designer = Agent(
        name="Practice designer",
        model=_RUNTIME_CONFIG.openai_model,
        handoff_description="Creates small practice tasks for students.",
        instructions="Turn the question into a runnable exercise with one expected output.",
        tools=[resource_tool],
    )
    debug_mentor = Agent(
        name="Debug mentor",
        model=_RUNTIME_CONFIG.openai_model,
        handoff_description="Helps debug suspicious metrics, leakage, and split issues.",
        instructions="Ask for evidence, inspect likely failure modes, and never request secrets.",
        tools=[resource_tool],
    )
    project_planner = Agent(
        name="Project planner",
        model=_RUNTIME_CONFIG.openai_model,
        handoff_description="Scopes student portfolio projects and showcase extensions.",
        instructions="Keep the build small, testable, and artifact-driven.",
        tools=[resource_tool],
    )
    agents_by_intent: dict[Intent, Any] = {
        "concept": concept_coach,
        "exercise": practice_designer,
        "debug": debug_mentor,
        "project": project_planner,
    }
    triage_agent = Agent(
        name="Course assistant triage",
        model=_RUNTIME_CONFIG.openai_model,
        instructions=(
            "Route each student question to the best specialist. Use the course lookup tool when "
            "grounding the answer in project resources."
        ),
        handoffs=list(agents_by_intent.values()),
        tools=[resource_tool],
    )
    return OpenAIAgentsBundle(
        runner=Runner,
        triage_agent=triage_agent,
        agents_by_intent=agents_by_intent,
    )


async def run_openai_agents_course_assistant(question: str) -> str:
    """Run the hosted triage agent and return its final answer."""

    bundle = _build_agents_bundle()
    _require_openai_runtime()
    result = await bundle.runner.run(bundle.triage_agent, question)
    return _coerce_final_output(result)


async def run_openai_specialist_course_assistant(
    question: str,
    *,
    intent: Intent | None = None,
    resource_context: str | None = None,
) -> str:
    """Run one hosted specialist for the live teaching bundle."""

    bundle = _build_agents_bundle()
    _require_openai_runtime()
    selected_intent = intent or classify_question(question)
    selected_agent = bundle.agents_by_intent[selected_intent]
    prompt = _build_specialist_prompt(question, resource_context=resource_context)
    result = await bundle.runner.run(selected_agent, prompt)
    return _coerce_final_output(result)


def _require_openai_runtime() -> None:
    if not _RUNTIME_CONFIG.openai_enabled:
        raise RuntimeError(
            "Set OPENAI_API_KEY in your environment or project .env before running "
            "the optional OpenAI Agents SDK example."
        )


def _build_specialist_prompt(question: str, *, resource_context: str | None) -> str:
    if not resource_context:
        return question
    return f"{question}\n\n{resource_context}"


def _coerce_final_output(result: Any) -> str:
    return str(result.final_output)
