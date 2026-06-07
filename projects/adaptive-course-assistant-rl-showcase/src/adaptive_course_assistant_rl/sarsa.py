"""Tabular on-policy SARSA for the tutoring intervention MDP."""

from __future__ import annotations

import random
from dataclasses import dataclass

from adaptive_course_assistant_rl.environment import (
    ACTION_LABELS,
    AssistantInterventionEnvironment,
    default_reward,
)
from adaptive_course_assistant_rl.policies import QLearningPolicy, greedy_action
from adaptive_course_assistant_rl.q_learning import StateKey


@dataclass
class SarsaResult:
    """Learned SARSA table plus per-episode reward trace."""

    q_table: dict[StateKey, list[float]]
    training_curve: list[dict[str, int | float]]

    def greedy_policy(self) -> QLearningPolicy:
        return QLearningPolicy(q_table=self.q_table, name="sarsa")


def train_sarsa(
    *,
    episodes: int = 450,
    seed: int = 11,
    scenario_ids: tuple[int, ...] = (0, 1, 2, 3, 4),
    alpha: float = 0.28,
    gamma: float = 0.9,
    epsilon: float = 0.35,
    epsilon_decay: float = 0.97,
    epsilon_min: float = 0.05,
    horizon: int = 5,
) -> SarsaResult:
    """Train SARSA with the same environment family as Q-learning."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    rng = random.Random(seed)
    q_table: dict[StateKey, list[float]] = {}
    training_curve: list[dict[str, int | float]] = []
    current_epsilon = epsilon

    for episode in range(1, episodes + 1):
        scenario_id = scenario_ids[(episode - 1) % len(scenario_ids)]
        environment = AssistantInterventionEnvironment(horizon=horizon, reward_fn=default_reward)
        state = environment.reset(seed=seed + episode, scenario_id=scenario_id)
        state_key = state.as_tuple()
        q_table.setdefault(state_key, [0.0] * len(ACTION_LABELS))
        action = _epsilon_greedy_action(q_table[state_key], current_epsilon, rng)
        total_reward = 0.0
        steps = 0

        while not environment.is_done():
            transition = environment.step(action)
            next_key = transition.state.as_tuple()
            q_table.setdefault(next_key, [0.0] * len(ACTION_LABELS))
            old_value = q_table[state_key][action]
            if transition.done:
                next_action = action
                future_value = 0.0
            else:
                next_action = _epsilon_greedy_action(q_table[next_key], current_epsilon, rng)
                future_value = q_table[next_key][next_action]
            target = transition.reward + gamma * future_value
            q_table[state_key][action] = old_value + alpha * (target - old_value)
            total_reward += transition.reward
            steps += 1
            state_key = next_key
            action = next_action

        training_curve.append(
            {
                "episode": episode,
                "scenario_id": scenario_id,
                "total_reward": round(total_reward, 4),
                "epsilon": round(current_epsilon, 4),
                "steps": steps,
            }
        )
        current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay)

    return SarsaResult(q_table=q_table, training_curve=training_curve)


def _epsilon_greedy_action(action_values: list[float], epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(action_values))
    return greedy_action(action_values)
