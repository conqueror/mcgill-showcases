"""Evaluate tutoring policies on fixed scenarios and log more than reward."""

from __future__ import annotations

from collections.abc import Sequence

from adaptive_course_assistant_rl.environment import (
    ACTION_COSTS,
    ACTION_LABELS,
    AssistantInterventionEnvironment,
    AssistantState,
    RewardFunction,
    default_reward,
    state_key_to_row,
)
from adaptive_course_assistant_rl.policies import Policy


def evaluate_policies(
    *,
    policies: Sequence[Policy],
    scenario_ids: Sequence[int],
    episodes_per_scenario: int = 1,
    base_seed: int = 0,
    horizon: int = 5,
    reward_fn: RewardFunction = default_reward,
) -> tuple[list[dict[str, int | float | str]], list[dict[str, int | float | str]]]:
    """Roll out policies and summarize reward, safety, and grounding behavior."""
    scenario_rows: list[dict[str, int | float | str]] = []

    for policy in policies:
        for scenario_id in scenario_ids:
            for episode_index in range(episodes_per_scenario):
                policy.reset()
                environment = AssistantInterventionEnvironment(horizon=horizon, reward_fn=reward_fn)
                state = environment.reset(seed=base_seed + episode_index, scenario_id=scenario_id)
                total_reward = 0.0
                intervention_cost = 0.0
                intervention_switches = 0
                escalation_count = 0
                ungrounded_action_count = 0
                action_trace: list[str] = []

                while not environment.is_done():
                    action = policy.select_action(state)
                    transition = environment.step(action)
                    total_reward += transition.reward
                    intervention_cost += ACTION_COSTS[action]
                    escalation_count += int(action == 7)
                    intervention_switches += int(state.last_action not in (len(ACTION_LABELS), action))
                    ungrounded_action_count += int(action in (2, 3, 4) and state.retrieval_quality == 0)
                    action_trace.append(ACTION_LABELS[action])
                    state = transition.state

                trajectory_signature = _trajectory_signature(
                    scenario_id=scenario_id,
                    action_trace=action_trace,
                    total_reward=total_reward,
                    state=state,
                    intervention_cost=intervention_cost,
                    intervention_switches=intervention_switches,
                    escalation_count=escalation_count,
                    ungrounded_action_count=ungrounded_action_count,
                )
                scenario_rows.append(
                    {
                        "policy": policy.name,
                        "scenario_id": scenario_id,
                        "scenario_name": environment.scenario_name,
                        "episode_index": episode_index,
                        "reset_seed": base_seed + episode_index,
                        "total_reward": round(total_reward, 4),
                        "solved": int(state.resolved_flag == 1),
                        "final_safety_risk": state.safety_risk,
                        "intervention_cost": round(intervention_cost, 4),
                        "intervention_switches": intervention_switches,
                        "escalation_count": escalation_count,
                        "ungrounded_action_count": ungrounded_action_count,
                        "actions": " | ".join(action_trace),
                        "trajectory_signature": trajectory_signature,
                    }
                )

    return _summarize_rows(scenario_rows), scenario_rows


def simulate_episode(
    *,
    policy: Policy,
    scenario_id: int,
    seed: int | None = None,
    horizon: int = 5,
    reward_fn: RewardFunction = default_reward,
) -> list[dict[str, int | float | str]]:
    """Trace one tutoring episode step by step."""
    policy.reset()
    environment = AssistantInterventionEnvironment(horizon=horizon, reward_fn=reward_fn)
    state = environment.reset(seed=seed, scenario_id=scenario_id)
    rows: list[dict[str, int | float | str]] = []

    while not environment.is_done():
        action = policy.select_action(state)
        transition = environment.step(action)
        state_row = state_key_to_row(state.as_tuple())
        next_state_row = state_key_to_row(transition.state.as_tuple())
        rows.append(
            {
                "scenario_name": environment.scenario_name,
                "turn_index": state.turn_index,
                "intent_type": state_row["intent_type"],
                "difficulty_level": state_row["difficulty_level"],
                "confidence_level": state_row["confidence_level"],
                "misconception_type": state_row["misconception_type"],
                "retrieval_quality": state_row["retrieval_quality"],
                "intent_uncertainty": state_row["intent_uncertainty"],
                "cognitive_load": state_row["cognitive_load"],
                "attempt_count": state.attempt_count,
                "safety_risk": state_row["safety_risk"],
                "action": ACTION_LABELS[action],
                "action_cost": ACTION_COSTS[action],
                "reward": transition.reward,
                "resolved": transition.state.resolved_flag,
                "next_safety_risk": next_state_row["safety_risk"],
            }
        )
        state = transition.state

    return rows


def _summarize_rows(rows: Sequence[dict[str, int | float | str]]) -> list[dict[str, int | float | str]]:
    by_policy: dict[str, list[dict[str, int | float | str]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy"]), []).append(row)

    summary_rows: list[dict[str, int | float | str]] = []
    for policy_name, policy_rows in sorted(by_policy.items()):
        count = len(policy_rows)
        unique_trajectory_count = len({str(row["trajectory_signature"]) for row in policy_rows})
        replayed_trajectory_count = count - unique_trajectory_count
        summary_rows.append(
            {
                "policy": policy_name,
                "episode_count": count,
                "unique_trajectory_count": unique_trajectory_count,
                "replayed_trajectory_count": replayed_trajectory_count,
                "evidence_mode": (
                    "replayed_deterministic_rollouts"
                    if replayed_trajectory_count
                    else "unique_rollouts"
                ),
                "avg_reward": round(sum(float(row["total_reward"]) for row in policy_rows) / count, 4),
                "solved_rate": round(sum(int(row["solved"]) for row in policy_rows) / count, 4),
                "avg_final_safety_risk": round(sum(int(row["final_safety_risk"]) for row in policy_rows) / count, 4),
                "avg_intervention_cost": round(sum(float(row["intervention_cost"]) for row in policy_rows) / count, 4),
                "avg_intervention_switches": round(sum(int(row["intervention_switches"]) for row in policy_rows) / count, 4),
                "avg_escalation_count": round(sum(int(row["escalation_count"]) for row in policy_rows) / count, 4),
                "avg_ungrounded_action_count": round(sum(int(row["ungrounded_action_count"]) for row in policy_rows) / count, 4),
            }
        )
    summary_rows.sort(key=lambda row: float(row["avg_reward"]), reverse=True)
    return summary_rows


def _trajectory_signature(
    *,
    scenario_id: int,
    action_trace: Sequence[str],
    total_reward: float,
    state: AssistantState,
    intervention_cost: float,
    intervention_switches: int,
    escalation_count: int,
    ungrounded_action_count: int,
) -> str:
    return "|".join(
        [
            f"scenario={scenario_id}",
            f"actions={' > '.join(action_trace)}",
            f"reward={round(total_reward, 4)}",
            f"solved={state.resolved_flag}",
            f"risk={state.safety_risk}",
            f"cost={round(intervention_cost, 4)}",
            f"switches={intervention_switches}",
            f"escalations={escalation_count}",
            f"ungrounded={ungrounded_action_count}",
        ]
    )
