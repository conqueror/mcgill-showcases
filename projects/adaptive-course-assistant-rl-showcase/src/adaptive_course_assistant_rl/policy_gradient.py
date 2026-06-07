"""Small, readable REINFORCE implementation for the tutoring environment."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from adaptive_course_assistant_rl.environment import (
    ACTION_LABELS,
    AssistantInterventionEnvironment,
    AssistantState,
    default_reward,
)
from adaptive_course_assistant_rl.policies import greedy_action

StateKey = tuple[int, ...]


def softmax(logits: list[float]) -> list[float]:
    """Convert logits into a probability distribution."""
    if not logits:
        raise ValueError("logits must be non-empty")
    shift = max(logits)
    exps = [math.exp(value - shift) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


@dataclass
class ReinforcePolicy:
    """Greedy readout from learned REINFORCE logits."""

    theta: dict[StateKey, list[float]]
    name: str = "reinforce"

    def reset(self) -> None:
        return None

    def select_action(self, state: AssistantState) -> int:
        logits = self.theta.get(state.as_tuple())
        if logits is None:
            return 0
        return greedy_action(logits)


@dataclass
class ReinforceResult:
    """Learned logits plus training curve."""

    theta: dict[StateKey, list[float]]
    training_curve: list[dict[str, int | float]]

    def greedy_policy(self) -> ReinforcePolicy:
        return ReinforcePolicy(theta=self.theta)


def train_reinforce(
    *,
    episodes: int = 700,
    seed: int = 17,
    scenario_ids: tuple[int, ...] = (0, 1, 2, 3, 4),
    alpha: float = 0.08,
    gamma: float = 0.9,
    horizon: int = 5,
) -> ReinforceResult:
    """Train a tabular softmax policy with REINFORCE."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    rng = random.Random(seed)
    action_count = len(ACTION_LABELS)
    theta: dict[StateKey, list[float]] = {}
    training_curve: list[dict[str, int | float]] = []

    for episode in range(1, episodes + 1):
        scenario_id = scenario_ids[(episode - 1) % len(scenario_ids)]
        environment = AssistantInterventionEnvironment(horizon=horizon, reward_fn=default_reward)
        state = environment.reset(seed=seed + episode, scenario_id=scenario_id)
        trajectory: list[tuple[StateKey, int, float]] = []
        total_reward = 0.0

        while not environment.is_done():
            state_key = state.as_tuple()
            theta.setdefault(state_key, [0.0] * action_count)
            action = _sample_action(theta[state_key], rng)
            transition = environment.step(action)
            trajectory.append((state_key, action, transition.reward))
            total_reward += transition.reward
            state = transition.state

        returns = [0.0] * len(trajectory)
        running_return = 0.0
        for index in range(len(trajectory) - 1, -1, -1):
            running_return = trajectory[index][2] + gamma * running_return
            returns[index] = running_return

        baseline = sum(returns) / len(returns)
        for index, (state_key, action, _reward) in enumerate(trajectory):
            probabilities = softmax(theta[state_key])
            advantage = returns[index] - baseline
            for candidate in range(action_count):
                indicator = 1.0 if candidate == action else 0.0
                theta[state_key][candidate] += alpha * advantage * (indicator - probabilities[candidate])

        training_curve.append(
            {
                "episode": episode,
                "scenario_id": scenario_id,
                "total_reward": round(total_reward, 4),
                "baseline": round(baseline, 4),
                "steps": len(trajectory),
            }
        )

    return ReinforceResult(theta=theta, training_curve=training_curve)


def _sample_action(logits: list[float], rng: random.Random) -> int:
    probabilities = softmax(logits)
    threshold = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold <= cumulative:
            return index
    return len(probabilities) - 1
