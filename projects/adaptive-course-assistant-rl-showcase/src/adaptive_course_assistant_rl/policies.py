"""Policies compared in the adaptive course assistant RL showcase."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, SupportsInt

from adaptive_course_assistant_rl.environment import ACTION_LABELS, AssistantState


class Policy(Protocol):
    """Minimal interface shared by rule-based and learned policies."""

    name: str

    def reset(self) -> None: ...

    def select_action(self, state: AssistantState) -> int: ...


class PredictModel(Protocol):
    """Tiny structural type for SB3-like models."""

    def predict(
        self,
        observation: Sequence[float],
        deterministic: bool = True,
    ) -> tuple[SupportsInt, object]: ...


def greedy_action(action_values: list[float]) -> int:
    """Return the first index attaining the maximum action value."""
    best_value = max(action_values)
    for index, value in enumerate(action_values):
        if value == best_value:
            return index
    raise RuntimeError("greedy_action requires a non-empty list")


@dataclass
class RandomPolicy:
    """Uniform-random baseline."""

    seed: int = 7
    name: str = "random"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        # Keep one seeded RNG stream across repeated rollouts so the evaluation
        # baseline stays reproducible without replaying the exact same episode.
        return None

    def select_action(self, state: AssistantState) -> int:
        del state
        return self._rng.randrange(len(ACTION_LABELS))


@dataclass
class RuleBasedPolicy:
    """Reasonable hand-written tutoring policy for comparison."""

    name: str = "rule_based"

    def reset(self) -> None:
        return None

    def select_action(self, state: AssistantState) -> int:
        if state.safety_risk == 2 and state.turn_index >= 3:
            return 7
        if state.intent_uncertainty >= 2:
            return 0
        if state.retrieval_quality == 0:
            return 1
        if state.cognitive_load >= 2:
            return 6
        if state.misconception_type >= 2:
            return 3
        if state.confidence_level == 0:
            return 2
        if state.turn_index >= 2:
            return 5
        return 4


@dataclass
class InterventionHeavyPolicy:
    """Deliberately shallow policy that overuses worked examples."""

    name: str = "intervention_heavy"

    def reset(self) -> None:
        return None

    def select_action(self, state: AssistantState) -> int:
        del state
        return 3


@dataclass
class QLearningPolicy:
    """Greedy wrapper around a learned Q-table."""

    q_table: dict[tuple[int, ...], list[float]]
    name: str = "q_learning"

    def reset(self) -> None:
        return None

    def select_action(self, state: AssistantState) -> int:
        return greedy_action(self.q_table.get(state.as_tuple(), [0.0] * len(ACTION_LABELS)))


@dataclass
class ModelPolicy:
    """Bridge a deep model into the same evaluation harness as tabular policies."""

    model: PredictModel
    observation_fn: Callable[[AssistantState], Sequence[float]]
    name: str
    deterministic: bool = True

    def reset(self) -> None:
        return None

    def select_action(self, state: AssistantState) -> int:
        action, _ = self.model.predict(
            self.observation_fn(state),
            deterministic=self.deterministic,
        )
        return int(action)
