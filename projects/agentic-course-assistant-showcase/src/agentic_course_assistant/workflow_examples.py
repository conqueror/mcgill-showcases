"""Deterministic workflow examples that mirror common ADK orchestration patterns."""

from __future__ import annotations

from dataclasses import dataclass

from agentic_course_assistant.assistant import AGENT_BY_INTENT, classify_question, guardrail_notes
from agentic_course_assistant.course_catalog import (
    COURSE_RESOURCES,
    CourseResource,
    search_resources,
)


@dataclass(frozen=True)
class WorkflowExampleResult:
    """Serializable workflow output for harness evaluation and teaching artifacts."""

    workflow_name: str
    summary: str
    trace: tuple[str, ...]
    state: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_name": self.workflow_name,
            "summary": self.summary,
            "trace": list(self.trace),
            "state": self.state,
        }


def sequential_course_plan(question: str) -> WorkflowExampleResult:
    """Build a fixed-order study plan from retrieved resources."""

    resources = _resource_trio(question)
    milestones = [
        "Name the learning goal in one sentence.",
        f"Study `{resources[0].resource_id}` before touching code.",
        f"Use `{resources[1].resource_id}` to design a sanity-check exercise.",
        "Write down one artifact that proves the workflow worked.",
    ]
    trace = (
        "sequential_course_plan.capture_goal",
        "sequential_course_plan.collect_resources",
        "sequential_course_plan.order_milestones",
        "sequential_course_plan.publish_plan",
    )
    return WorkflowExampleResult(
        workflow_name="sequential_course_plan",
        summary="A step-by-step plan that keeps the agent workflow visible to students.",
        trace=trace,
        state={
            "resource_ids": [resource.resource_id for resource in resources],
            "milestones": milestones,
            "resource_count": len(resources),
        },
    )


def loop_refinement(question: str) -> WorkflowExampleResult:
    """Show an iterative draft-review-revise loop with deterministic rounds."""

    draft_rounds = [
        "Draft 1: answer the question in one paragraph and point to one artifact.",
        "Draft 2: shorten the answer and add one concrete verification step.",
        "Final: keep the recommendation small, testable, and traceable.",
    ]
    trace = (
        "loop_refinement.draft_round_1",
        "loop_refinement.review_round_1",
        "loop_refinement.revise_round_2",
        "loop_refinement.stop_after_quality_threshold",
    )
    return WorkflowExampleResult(
        workflow_name="loop_refinement",
        summary="A bounded refinement loop that improves the plan without becoming open-ended.",
        trace=trace,
        state={
            "question": question,
            "draft_rounds": draft_rounds,
            "rounds_completed": 2,
        },
    )


def parallel_resource_review(question: str) -> WorkflowExampleResult:
    """Review several resources in parallel and then merge the findings."""

    resources = _resource_trio(question)
    reviewer_names = ("signal_reviewer", "risk_reviewer", "teaching_reviewer")
    reviews = [
        {
            "reviewer": reviewer_name,
            "resource_id": resource.resource_id,
            "takeaway": f"Use {resource.kind} material to teach {resource.topic}.",
        }
        for reviewer_name, resource in zip(reviewer_names, resources, strict=True)
    ]
    trace = (
        "parallel_resource_review.spawn_reviews",
        "parallel_resource_review.collect_reviews",
        "parallel_resource_review.rank_consensus",
    )
    return WorkflowExampleResult(
        workflow_name="parallel_resource_review",
        summary="Three lightweight reviewers compare resources before the workflow picks a lead.",
        trace=trace,
        state={
            "reviews": reviews,
            "review_count": len(reviews),
            "consensus_resource_id": resources[0].resource_id,
        },
    )


def router_triage(question: str) -> WorkflowExampleResult:
    """Mirror a router that hands a question to a specialist."""

    intent = classify_question(question)
    agent_name = AGENT_BY_INTENT[intent]
    trace = (
        "router_triage.read_question",
        f"router_triage.select_intent:{intent}",
        f"router_triage.handoff:{agent_name.lower().replace(' ', '_')}",
    )
    return WorkflowExampleResult(
        workflow_name="router_triage",
        summary="A triage router picks the best specialist instead of doing every job itself.",
        trace=trace,
        state={
            "selected_intent": intent,
            "selected_agent": agent_name,
        },
    )


def custom_policy_agent(question: str) -> WorkflowExampleResult:
    """Apply a deterministic policy gate before the workflow proceeds."""

    notes = guardrail_notes(question)
    blocked = any("secrets" in note.lower() for note in notes)
    status = "blocked" if blocked else "approved"
    trace = (
        "custom_policy_agent.read_question",
        f"custom_policy_agent.policy_check:{status}",
        "custom_policy_agent.return_policy_decision",
    )
    summary = (
        "Blocked the request because it contains a secret-handling risk."
        if blocked
        else "Approved the request after the policy check stayed inside public-safe scope."
    )
    return WorkflowExampleResult(
        workflow_name="custom_policy_agent",
        summary=summary,
        trace=trace,
        state={
            "allowed": not blocked,
            "guardrails": list(notes),
        },
    )


def run_offline_workflows(question: str) -> dict[str, WorkflowExampleResult]:
    """Return the full deterministic workflow set used by the harness lab."""

    return {
        "sequential_course_plan": sequential_course_plan(question),
        "loop_refinement": loop_refinement(question),
        "parallel_resource_review": parallel_resource_review(question),
        "router_triage": router_triage(question),
        "custom_policy_agent": custom_policy_agent(question),
    }


def _resource_trio(question: str) -> list[CourseResource]:
    resources = search_resources(question, limit=3)
    if len(resources) < 3:
        seen_ids = {resource.resource_id for resource in resources}
        resources.extend(
            resource for resource in COURSE_RESOURCES if resource.resource_id not in seen_ids
        )
    return resources[:3]
