"""Educational helpers for the autoresearch showcase."""

from .agent_brief import render_agent_brief
from .decision_policy import default_scenarios, recommend_status
from .platforms import get_profile, list_profiles
from .reporting import build_showcase

__all__ = [
    "build_showcase",
    "default_scenarios",
    "get_profile",
    "list_profiles",
    "recommend_status",
    "render_agent_brief",
]
