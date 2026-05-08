"""ADK-discoverable wrapper for the optional Google ADK demo.

Run from this project root after installing the `adk` optional extra:

    uv sync --extra dev --extra adk
    uv run adk run adk_course_assistant
"""

from __future__ import annotations

from agentic_course_assistant.google_adk_example import root_agent

__all__ = ["root_agent"]
