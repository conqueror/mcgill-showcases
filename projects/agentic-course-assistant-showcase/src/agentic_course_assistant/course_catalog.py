"""Small course catalog used as a deterministic tool target."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CourseResource:
    """A compact teaching resource that can be retrieved by the assistant."""

    resource_id: str
    title: str
    topic: str
    level: str
    kind: str
    skills: tuple[str, ...]
    summary: str


COURSE_RESOURCES: tuple[CourseResource, ...] = (
    CourseResource(
        resource_id="eda-leakage-001",
        title="Spot leakage before model training",
        topic="data leakage",
        level="beginner-intermediate",
        kind="diagnostic",
        skills=("eda", "splits", "feature engineering"),
        summary=(
            "Check whether a feature uses target, future, or post-outcome information before it "
            "enters the training matrix."
        ),
    ),
    CourseResource(
        resource_id="splits-002",
        title="Choose a split strategy that matches deployment",
        topic="validation design",
        level="beginner-intermediate",
        kind="concept",
        skills=("train validation test", "time split", "group split"),
        summary=(
            "Use random, stratified, group, or time-based splits according to how the model "
            "will be used after training."
        ),
    ),
    CourseResource(
        resource_id="metrics-003",
        title="Interpret validation metrics with a baseline",
        topic="evaluation",
        level="beginner",
        kind="exercise",
        skills=("metrics", "baseline", "confusion matrix"),
        summary=(
            "Compare model results against a simple baseline before trusting improvements from a "
            "more complex model."
        ),
    ),
    CourseResource(
        resource_id="debug-004",
        title="Debug suspiciously high validation scores",
        topic="model debugging",
        level="intermediate",
        kind="debug checklist",
        skills=("debugging", "leakage", "cross validation"),
        summary=(
            "Audit target leakage, duplicated rows, split overlap, preprocessing fit scope, and "
            "metric calculation when validation looks too good."
        ),
    ),
    CourseResource(
        resource_id="agents-005",
        title="Design a small agent workflow before adding SDKs",
        topic="agentic AI",
        level="intermediate",
        kind="project pattern",
        skills=("agent routing", "tool use", "guardrails"),
        summary=(
            "Start with a deterministic router, one data lookup tool, specialist behaviors, and "
            "observable trace records before calling hosted models."
        ),
    ),
)


def tokenize(text: str) -> set[str]:
    """Return lowercase word tokens without punctuation."""

    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return {part for part in cleaned.split() if part}


def search_resources(query: str, limit: int = 3) -> list[CourseResource]:
    """Rank resources by simple lexical overlap for offline reproducibility."""

    query_terms = tokenize(query)
    scored: list[tuple[int, int, CourseResource]] = []
    for index, resource in enumerate(COURSE_RESOURCES):
        searchable = " ".join(
            [
                resource.title,
                resource.topic,
                resource.kind,
                resource.summary,
                " ".join(resource.skills),
            ]
        )
        overlap = len(query_terms & tokenize(searchable))
        scored.append((overlap, index, resource))

    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    matches = [resource for score, _, resource in ranked if score > 0]
    if not matches:
        matches = [resource for _, _, resource in ranked]
    return matches[:limit]
