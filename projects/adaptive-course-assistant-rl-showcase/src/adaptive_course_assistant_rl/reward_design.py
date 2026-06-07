"""Reward audit helpers for the adaptive course assistant RL showcase."""

from __future__ import annotations

from collections.abc import Sequence

from adaptive_course_assistant_rl.environment import ACTION_COSTS, AssistantState, default_reward
from adaptive_course_assistant_rl.evaluation import evaluate_policies
from adaptive_course_assistant_rl.policies import Policy


def bad_reward(
    previous_state: AssistantState,
    action: int,
    next_state: AssistantState,
    done: bool,
) -> float:
    """A bad proxy reward that overpays for high-effort interventions."""
    del done
    confidence_gain = max(0, next_state.confidence_level - previous_state.confidence_level)
    speed_bonus = 1.2 if action in (3, 4) else 0.0
    return round((1.3 * confidence_gain) + speed_bonus - (0.1 * ACTION_COSTS[action]), 4)


GOOD_REWARD = default_reward
BAD_REWARD = bad_reward


def compare_reward_models(
    *,
    policies: Sequence[Policy],
    scenario_ids: Sequence[int],
    horizon: int = 5,
) -> list[dict[str, int | float | str]]:
    """Evaluate the same policies under good and bad reward designs."""
    rows: list[dict[str, int | float | str]] = []
    for reward_name, reward_fn in (("bad", BAD_REWARD), ("good", GOOD_REWARD)):
        summary_rows, _ = evaluate_policies(
            policies=policies,
            scenario_ids=scenario_ids,
            horizon=horizon,
            reward_fn=reward_fn,
        )
        for row in summary_rows:
            tagged = dict(row)
            tagged["reward_model"] = reward_name
            rows.append(tagged)
    return rows


def reward_model_specs() -> dict[str, str]:
    """Return plain-language descriptions of the aligned and misaligned rewards."""
    return {
        "good": (
            "# Good Reward Design\n\n"
            "- Reward grounded resolution.\n"
            "- Reward safer, lower-risk tutoring states.\n"
            "- Charge for extra turns and intervention cost.\n"
            "- Penalize ungrounded answers and missed escalation.\n"
        ),
        "bad": (
            "# Bad Reward Design\n\n"
            "- Reward short-term confidence only.\n"
            "- Add a bonus for heavier interventions.\n"
            "- Ignore whether the help was grounded or safe.\n"
        ),
    }


def reward_hacking_report(comparison_rows: Sequence[dict[str, int | float | str]]) -> str:
    """Explain the rank reversal between the good and bad reward designs."""
    by_key = {(str(row["reward_model"]), str(row["policy"])): row for row in comparison_rows}
    bad_heavy = by_key[("bad", "intervention_heavy")]
    bad_rule = by_key[("bad", "rule_based")]
    good_heavy = by_key[("good", "intervention_heavy")]
    good_rule = by_key[("good", "rule_based")]
    return (
        "# Reward Hacking Report\n\n"
        "The bad reward makes the heavy-handed policy look better than it really is.\n\n"
        f"- Bad reward, intervention-heavy policy: {bad_heavy['avg_reward']}\n"
        f"- Bad reward, rule-based policy: {bad_rule['avg_reward']}\n"
        f"- Good reward, intervention-heavy policy: {good_heavy['avg_reward']}\n"
        f"- Good reward, rule-based policy: {good_rule['avg_reward']}\n\n"
        "That ranking flip is the point. A policy can look good on a sloppy proxy and still do a poor job on the real objective.\n"
    )
