"""Deterministic agent workflow used by the default showcase path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentic_course_assistant.course_catalog import CourseResource, search_resources, tokenize

Intent = Literal["concept", "exercise", "debug", "project"]


@dataclass(frozen=True)
class AssistantResult:
    """Serializable result for artifact writing and tests."""

    question: str
    intent: Intent
    agent_name: str
    answer: str
    resources: tuple[CourseResource, ...]
    guardrails: tuple[str, ...]
    trace: tuple[str, ...]


ROUTE_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    "concept": ("why", "what", "explain", "understand", "intuition", "confused"),
    "exercise": ("practice", "exercise", "quiz", "homework", "example", "try"),
    "debug": ("debug", "broken", "error", "too good", "leak", "leakage", "wrong"),
    "project": ("build", "project", "showcase", "portfolio", "agent", "sdk", "adk"),
}

ROUTE_PRIORITY: dict[Intent, int] = {
    "debug": 4,
    "project": 3,
    "exercise": 2,
    "concept": 1,
}

AGENT_BY_INTENT: dict[Intent, str] = {
    "concept": "Concept coach",
    "exercise": "Practice designer",
    "debug": "Debug mentor",
    "project": "Project planner",
}

SENSITIVE_TERMS = {"password", "token", "api key", "secret", "credential"}


def classify_question(question: str) -> Intent:
    """Route a student question to a specialist intent."""

    normalized = question.lower()
    terms = tokenize(question)
    route_scores: dict[Intent, int] = {}
    for intent, keywords in ROUTE_KEYWORDS.items():
        route_scores[intent] = sum(
            1 for keyword in keywords if _keyword_matches(keyword, normalized, terms)
        )
    best_intent, best_score = max(
        route_scores.items(), key=lambda item: (item[1], ROUTE_PRIORITY[item[0]])
    )
    return best_intent if best_score > 0 else "concept"


def guardrail_notes(question: str) -> tuple[str, ...]:
    """Return safety notes that keep the assistant scoped to public learning work."""

    normalized = question.lower()
    terms = tokenize(question)
    notes: list[str] = ["Scope locked to course learning support and public artifacts."]
    if any(_keyword_matches(term, normalized, terms) for term in SENSITIVE_TERMS):
        notes.append(
            "Do not paste secrets. Replace credentials with placeholders before debugging."
        )
    return tuple(notes)


def _keyword_matches(keyword: str, normalized_text: str, terms: set[str]) -> bool:
    """Match keywords by tokens so words like projection do not trigger project."""

    if " " not in keyword:
        return keyword in terms

    keyword_phrase = " ".join(_ordered_tokens(keyword))
    normalized_phrase = " ".join(_ordered_tokens(normalized_text))
    return f" {keyword_phrase} " in f" {normalized_phrase} "


def _ordered_tokens(text: str) -> tuple[str, ...]:
    """Return lowercase word tokens in their original order."""

    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return tuple(part for part in cleaned.split() if part)


def answer_question(question: str, limit: int = 3) -> AssistantResult:
    """Answer a course question with routing, tool lookup, and trace metadata."""

    if not question.strip():
        raise ValueError("question must not be empty")

    intent = classify_question(question)
    agent_name = AGENT_BY_INTENT[intent]
    resources = tuple(search_resources(question, limit=limit))
    notes = guardrail_notes(question)
    trace = (
        "triage_agent.received_question",
        f"triage_agent.selected_intent:{intent}",
        "course_catalog_tool.search_resources",
        f"{agent_name.lower().replace(' ', '_')}.draft_answer",
        "guardrail.scope_check",
    )
    answer = _compose_answer(question, intent, agent_name, resources, notes)
    return AssistantResult(
        question=question,
        intent=intent,
        agent_name=agent_name,
        answer=answer,
        resources=resources,
        guardrails=notes,
        trace=trace,
    )


def _compose_answer(
    question: str,
    intent: Intent,
    agent_name: str,
    resources: tuple[CourseResource, ...],
    notes: tuple[str, ...],
) -> str:
    lead = {
        "concept": "Start by separating the concept from the implementation details.",
        "exercise": "Turn this into a small practice loop with one observable output.",
        "debug": "Treat the surprising result as a traceability problem before changing the model.",
        "project": (
            "Keep the first build small: one router, one tool, one specialist, one verifier."
        ),
    }[intent]
    resource_lines = "\n".join(
        f"- {resource.title}: {resource.summary}" for resource in resources
    )
    next_step = _next_step(intent, question)
    guardrail_lines = "\n".join(f"- {note}" for note in notes)
    return (
        f"{agent_name}: {lead}\n\n"
        f"Suggested resources:\n{resource_lines}\n\n"
        f"Next step:\n- {next_step}\n\n"
        f"Guardrails:\n{guardrail_lines}"
    )


def _next_step(intent: Intent, question: str) -> str:
    terms = tokenize(question)
    if intent == "debug":
        if "leakage" in terms or "leak" in terms:
            return "List every feature and mark whether it is known before prediction time."
        return "Re-run the baseline after checking split overlap and preprocessing fit scope."
    if intent == "exercise":
        return "Write one tiny example, predict the answer by hand, then run the script."
    if intent == "project":
        return (
            "Implement the offline path first, then add one SDK adapter behind an optional command."
        )
    return "Explain the idea in one sentence, then connect it to a generated artifact."
