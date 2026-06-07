"""Map learned policy outputs back into a deterministic assistant contract."""

from __future__ import annotations

from collections.abc import Sequence

from adaptive_course_assistant_rl.environment import ACTION_LABELS, BANDIT_ACTIONS


def policy_router_payload() -> dict[str, object]:
    """Return the deterministic routing payload exported by the showcase."""
    return {
        "router_version": 1,
        "export_kind": "assistant_side_action_contract",
        "decision_boundary": "pedagogical_intervention_only",
        "exports_learned_weights": False,
        "exports_champion_policy_parameters": False,
        "allowed_actions": list(ACTION_LABELS.values()),
        "bandit_subset": [ACTION_LABELS[action] for action in BANDIT_ACTIONS],
        "notes": [
            "The deterministic assistant already understands the request.",
            "The learned layer chooses the next intervention, not the answer text itself.",
            "This export is a teaching contract for assistant-side routing, not a model-weight dump.",
            "Escalation remains available when the state stays risky late in the episode.",
        ],
    }


def action_mapping_markdown() -> str:
    """Explain how each learned action maps back into assistant behavior."""
    rows = [
        ("ask_clarifying_question", "The assistant asks for missing context before continuing."),
        ("retrieve_course_note", "The assistant grounds the next turn in course material."),
        ("give_hint", "The assistant nudges the student without solving the task for them."),
        ("give_worked_example", "The assistant shows a fuller path when the student is stuck."),
        ("assign_targeted_practice", "The assistant turns the next step into a small exercise."),
        ("check_understanding", "The assistant tests whether the issue is actually resolved."),
        ("slow_down_and_rephrase", "The assistant reduces overload and restates the idea more plainly."),
        ("escalate_to_human", "The assistant stops and hands off when the state stays risky."),
    ]
    body = "\n".join(f"- `{action}`: {meaning}" for action, meaning in rows)
    return "# Action Mapping\n\n" + body + "\n"


def intervention_decision_rows(summary_rows: Sequence[dict[str, int | float | str]]) -> list[dict[str, object]]:
    """Convert policy summaries into a small student-facing decision table."""
    interpretations = {
        "rule_based": "Hand-written baseline used as a readable classroom default.",
        "random": "Untrained random baseline used for sanity checking.",
    }
    return [
        {
            "policy": row["policy"],
            "avg_reward": row["avg_reward"],
            "solved_rate": row["solved_rate"],
            "interpretation": interpretations.get(
                str(row["policy"]),
                "Learned policy compared against the rule and random baselines.",
            ),
        }
        for row in summary_rows
    ]
