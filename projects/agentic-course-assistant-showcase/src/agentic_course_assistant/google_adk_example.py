"""Optional Google ADK version of the course assistant.

Install the `adk` optional extra and set the required Gemini or Vertex
credentials before running with `adk run`. ADK discovers the `root_agent`
object below.
"""

from __future__ import annotations

try:
    from google.adk.agents.llm_agent import Agent
except ImportError as exc:  # pragma: no cover - optional SDK dependency
    raise RuntimeError(
        "Install the optional Google ADK package with "
        "`uv sync --extra dev --extra adk`."
    ) from exc

from agentic_course_assistant.course_catalog import search_resources
from agentic_course_assistant.runtime_config import apply_live_environment

_RUNTIME_CONFIG = apply_live_environment()


def lookup_course_resources(question: str) -> dict[str, object]:
    """Return course resources that match a student question."""

    resources = search_resources(question)
    return {
        "status": "success",
        "resources": [
            {
                "resource_id": resource.resource_id,
                "title": resource.title,
                "topic": resource.topic,
                "summary": resource.summary,
            }
            for resource in resources
        ],
    }


root_agent = Agent(
    model=_RUNTIME_CONFIG.gemini_model,
    name="course_assistant",
    description="Routes student ML questions to a grounded course-assistant response.",
    instruction=(
        "Help students learn from public course artifacts. Use lookup_course_resources before "
        "answering. Do not request secrets, API keys, or private data."
    ),
    tools=[lookup_course_resources],
)
