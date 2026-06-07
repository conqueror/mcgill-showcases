"""Real contextual bandit for the first intervention choice."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from adaptive_course_assistant_rl.environment import (
    ACTION_LABELS,
    BANDIT_ACTIONS,
    ScenarioDefinition,
    scenario_catalog,
)

CONTEXTUAL_REWARD_WEIGHTS: tuple[tuple[float, ...], ...] = (
    (0.8, -0.1, -0.2, 0.5, 0.2, 0.1, 0.4, -0.2),
    (0.2, 0.7, 0.5, 0.0, 0.2, -0.1, 0.0, 0.0),
    (-0.1, 0.2, 0.6, 0.4, -0.2, 0.3, -0.1, 0.2),
    (0.1, 0.3, 0.8, -0.4, 0.1, -0.1, -0.2, 0.1),
    (0.4, -0.2, -0.1, 0.2, 0.5, 0.5, 0.6, 0.3),
)


@dataclass(frozen=True)
class BanditRunResult:
    """Reward trace, regret trace, and action mix from one bandit run."""

    metrics_rows: list[dict[str, int | float | str]]
    regret_rows: list[dict[str, int | float | str]]
    action_rows: list[dict[str, int | float | str]]


def run_bandit_experiment(*, steps: int = 400, epsilon: float = 0.2, seed: int = 7) -> BanditRunResult:
    """Run epsilon-greedy ridge regression over first-turn tutoring interventions."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    rng = random.Random(seed)
    feature_count = len(CONTEXTUAL_REWARD_WEIGHTS[0])
    design_matrices = [np.eye(feature_count, dtype=float) for _ in BANDIT_ACTIONS]
    reward_vectors = [np.zeros(feature_count, dtype=float) for _ in BANDIT_ACTIONS]
    counts = [0 for _ in BANDIT_ACTIONS]
    cumulative_reward = 0.0
    cumulative_regret = 0.0
    metrics_rows: list[dict[str, int | float | str]] = []
    regret_rows: list[dict[str, int | float | str]] = []
    scenarios = scenario_catalog()

    for step in range(1, steps + 1):
        scenario = scenarios[(step - 1) % len(scenarios)]
        context = context_vector(scenario)
        expected_rewards = [_expected_reward(context, weights) for weights in CONTEXTUAL_REWARD_WEIGHTS]
        optimal_local = max(range(len(expected_rewards)), key=lambda idx: expected_rewards[idx])

        if min(counts) == 0 or rng.random() < epsilon:
            local_action = rng.randrange(len(BANDIT_ACTIONS))
        else:
            local_action = max(
                range(len(BANDIT_ACTIONS)),
                key=lambda idx: _estimated_reward(
                    context=context,
                    design_matrix=design_matrices[idx],
                    reward_vector=reward_vectors[idx],
                ),
            )

        action = BANDIT_ACTIONS[local_action]
        reward_probability = expected_rewards[local_action]
        reward = 1.0 if rng.random() < reward_probability else 0.0
        counts[local_action] += 1
        vector = np.array(context, dtype=float)
        design_matrices[local_action] += np.outer(vector, vector)
        reward_vectors[local_action] += reward * vector
        cumulative_reward += reward
        instantaneous_regret = expected_rewards[optimal_local] - reward_probability
        cumulative_regret += instantaneous_regret

        metrics_rows.append(
            {
                "step": step,
                "scenario_name": scenario.name,
                "context_signature": scenario_signature(scenario),
                "action": ACTION_LABELS[action],
                "reward": reward,
                "expected_reward": round(reward_probability, 4),
                "cumulative_reward": round(cumulative_reward, 4),
                "estimated_value": round(
                    _estimated_reward(
                        context=context,
                        design_matrix=design_matrices[local_action],
                        reward_vector=reward_vectors[local_action],
                    ),
                    4,
                ),
                "optimal_action": ACTION_LABELS[BANDIT_ACTIONS[optimal_local]],
            }
        )
        regret_rows.append(
            {
                "step": step,
                "scenario_name": scenario.name,
                "action": ACTION_LABELS[action],
                "optimal_action": ACTION_LABELS[BANDIT_ACTIONS[optimal_local]],
                "instantaneous_regret": round(instantaneous_regret, 4),
                "cumulative_regret": round(cumulative_regret, 4),
            }
        )

    action_rows: list[dict[str, int | float | str]] = [
        {
            "action": ACTION_LABELS[action],
            "pull_count": counts[index],
            "pull_share": round(counts[index] / float(steps), 4),
        }
        for index, action in enumerate(BANDIT_ACTIONS)
    ]
    return BanditRunResult(metrics_rows=metrics_rows, regret_rows=regret_rows, action_rows=action_rows)


def context_vector(scenario: ScenarioDefinition) -> tuple[float, ...]:
    """Build a small feature vector from the first-turn tutoring context."""
    return (
        1.0,
        scenario.intent_type / 3.0,
        scenario.difficulty_level / 2.0,
        scenario.confidence_level / 2.0,
        scenario.misconception_type / 3.0,
        scenario.retrieval_quality / 2.0,
        scenario.intent_uncertainty / 2.0,
        scenario.cognitive_load / 2.0,
    )


def scenario_signature(scenario: ScenarioDefinition) -> str:
    """Return a compact human-readable signature for one scenario."""
    return (
        f"intent={scenario.intent_type}|difficulty={scenario.difficulty_level}|"
        f"confidence={scenario.confidence_level}|misconception={scenario.misconception_type}"
    )


def _expected_reward(context: tuple[float, ...], weights: tuple[float, ...]) -> float:
    score = sum(left * right for left, right in zip(context, weights, strict=True))
    return 1.0 / (1.0 + math.exp(-score))


def _estimated_reward(
    *,
    context: tuple[float, ...],
    design_matrix: np.ndarray,
    reward_vector: np.ndarray,
) -> float:
    theta = np.linalg.solve(design_matrix, reward_vector)
    return float(np.dot(theta, np.array(context, dtype=float)))
