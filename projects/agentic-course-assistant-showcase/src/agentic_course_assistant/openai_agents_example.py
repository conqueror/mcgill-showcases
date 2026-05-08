"""Optional OpenAI Agents SDK version of the course assistant.

Install the `openai` optional extra and set `OPENAI_API_KEY` before importing
this module. The default showcase path stays offline, but this file gives
students a real SDK shape to compare against the deterministic implementation.
"""

from __future__ import annotations

try:
    from agents import Agent, Runner, function_tool
except ImportError as exc:  # pragma: no cover - optional SDK dependency
    raise RuntimeError(
        "Install the optional OpenAI Agents SDK with "
        "`uv sync --extra dev --extra openai`."
    ) from exc

from agentic_course_assistant.course_catalog import search_resources


@function_tool  # type: ignore[untyped-decorator, unused-ignore]
def lookup_course_resources(question: str) -> str:
    """Return course resources that match a student question."""

    resources = search_resources(question)
    return "\n".join(
        f"{resource.resource_id}: {resource.title} - {resource.summary}" for resource in resources
    )


concept_coach = Agent(
    name="Concept coach",
    handoff_description="Explains ML and agentic AI concepts clearly.",
    instructions="Explain the concept, name the artifact to inspect, and keep the answer concise.",
    tools=[lookup_course_resources],
)

practice_designer = Agent(
    name="Practice designer",
    handoff_description="Creates small practice tasks for students.",
    instructions="Turn the question into a runnable exercise with one expected output.",
    tools=[lookup_course_resources],
)

debug_mentor = Agent(
    name="Debug mentor",
    handoff_description="Helps debug suspicious metrics, leakage, and split issues.",
    instructions="Ask for evidence, inspect likely failure modes, and never request secrets.",
    tools=[lookup_course_resources],
)

project_planner = Agent(
    name="Project planner",
    handoff_description="Scopes student portfolio projects and showcase extensions.",
    instructions="Keep the build small, testable, and artifact-driven.",
    tools=[lookup_course_resources],
)

triage_agent = Agent(
    name="Course assistant triage",
    instructions=(
        "Route each student question to the best specialist. Use the course lookup tool when "
        "grounding the answer in project resources."
    ),
    handoffs=[concept_coach, practice_designer, debug_mentor, project_planner],
    tools=[lookup_course_resources],
)


async def run_openai_agents_course_assistant(question: str) -> str:
    """Run the SDK agent and return its final answer."""

    result = await Runner.run(triage_agent, question)
    return str(result.final_output)
