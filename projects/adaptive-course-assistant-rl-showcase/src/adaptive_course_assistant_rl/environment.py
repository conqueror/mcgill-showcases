"""Deterministic tutoring environment for the adaptive course assistant showcase.

This module is the shared world every policy learns against. The setup is simple on
purpose: a deterministic assistant has already classified the student's request and
retrieved some initial context, and the learned policy chooses the next pedagogical
intervention. That is the seam this showcase teaches.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from adaptive_course_assistant_rl.config import DEFAULT_HORIZON

INTENT_LABELS = {
    0: "concept_help",
    1: "debug_help",
    2: "study_plan",
    3: "exam_review",
}
DIFFICULTY_LABELS = {0: "intro", 1: "intermediate", 2: "advanced"}
CONFIDENCE_LABELS = {0: "low", 1: "medium", 2: "high"}
MISCONCEPTION_LABELS = {
    0: "none",
    1: "notation",
    2: "conceptual",
    3: "procedural",
}
QUALITY_LABELS = {0: "poor", 1: "partial", 2: "strong"}
LEVEL_LABELS = {0: "low", 1: "medium", 2: "high"}
RISK_LABELS = {0: "low", 1: "medium", 2: "high"}

ACTION_LABELS = {
    0: "ask_clarifying_question",
    1: "retrieve_course_note",
    2: "give_hint",
    3: "give_worked_example",
    4: "assign_targeted_practice",
    5: "check_understanding",
    6: "slow_down_and_rephrase",
    7: "escalate_to_human",
}
ACTION_COSTS = {
    0: 0.5,
    1: 0.7,
    2: 1.0,
    3: 1.3,
    4: 1.1,
    5: 0.6,
    6: 0.8,
    7: 1.8,
}
NONE_ACTION = len(ACTION_LABELS)

BANDIT_ACTIONS = (0, 1, 2, 3, 6)

STATE_FIELD_NAMES = (
    "intent_type",
    "difficulty_level",
    "confidence_level",
    "misconception_type",
    "retrieval_quality",
    "intent_uncertainty",
    "cognitive_load",
    "turn_index",
    "attempt_count",
    "last_action",
    "safety_risk",
    "resolved_flag",
)


@dataclass(frozen=True)
class ScenarioDefinition:
    """Starting tutoring context for one named scenario."""

    scenario_id: int
    name: str
    intent_type: int
    difficulty_level: int
    confidence_level: int
    misconception_type: int
    retrieval_quality: int
    intent_uncertainty: int
    cognitive_load: int
    attempt_count: int


SCENARIOS: tuple[ScenarioDefinition, ...] = (
    ScenarioDefinition(0, "lost_in_notation", 0, 0, 0, 1, 0, 2, 1, 0),
    ScenarioDefinition(1, "debugging_spiral", 1, 1, 0, 3, 1, 1, 2, 1),
    ScenarioDefinition(2, "overloaded_exam_reviewer", 3, 2, 0, 2, 0, 1, 2, 1),
    ScenarioDefinition(3, "study_plan_drift", 2, 1, 1, 0, 1, 1, 1, 0),
    ScenarioDefinition(4, "advanced_but_uncertain", 0, 2, 1, 2, 0, 2, 1, 1),
)


@dataclass(frozen=True)
class AssistantState:
    """Compact tutoring state used by the bandit, tabular RL, and DRL bridge."""

    intent_type: int
    difficulty_level: int
    confidence_level: int
    misconception_type: int
    retrieval_quality: int
    intent_uncertainty: int
    cognitive_load: int
    turn_index: int
    attempt_count: int
    last_action: int
    safety_risk: int
    resolved_flag: int

    def as_tuple(self) -> tuple[int, ...]:
        """Return a stable discrete key for tabular methods."""
        return (
            self.intent_type,
            self.difficulty_level,
            self.confidence_level,
            self.misconception_type,
            self.retrieval_quality,
            self.intent_uncertainty,
            self.cognitive_load,
            self.turn_index,
            self.attempt_count,
            self.last_action,
            self.safety_risk,
            self.resolved_flag,
        )

    def as_normalized_vector(self, *, horizon: int) -> tuple[float, ...]:
        """Return a simple [0, 1]-scaled view for function approximation."""
        return (
            round(self.intent_type / 3.0, 6),
            round(self.difficulty_level / 2.0, 6),
            round(self.confidence_level / 2.0, 6),
            round(self.misconception_type / 3.0, 6),
            round(self.retrieval_quality / 2.0, 6),
            round(self.intent_uncertainty / 2.0, 6),
            round(self.cognitive_load / 2.0, 6),
            round(self.turn_index / max(1, horizon - 1), 6),
            round(self.attempt_count / 3.0, 6),
            round(self.last_action / max(1, NONE_ACTION), 6),
            round(self.safety_risk / 2.0, 6),
            float(self.resolved_flag),
        )


@dataclass(frozen=True)
class TransitionResult:
    """Bundle the outcome of one environment step."""

    state: AssistantState
    reward: float
    done: bool
    info: dict[str, int | float | str]


def clamp(value: int, lower: int, upper: int) -> int:
    """Clamp an integer into a closed interval."""
    return max(lower, min(upper, value))


def lowered_misconception(level: int, strength: int = 1) -> int:
    """Reduce misconception severity toward ``none``."""
    return max(0, level - strength)


def safety_risk_from_state(
    *,
    difficulty_level: int,
    confidence_level: int,
    intent_uncertainty: int,
    cognitive_load: int,
    attempt_count: int,
    misconception_type: int,
    resolved_flag: int,
) -> int:
    """Compute a simple safety/escalation risk score."""
    if resolved_flag:
        return 0
    score = difficulty_level + intent_uncertainty + cognitive_load + max(0, misconception_type - 1)
    if confidence_level == 0:
        score += 1
    if attempt_count >= 2:
        score += 1
    if score >= 5:
        return 2
    if score >= 3:
        return 1
    return 0


def state_key_to_row(state_key: tuple[int, ...]) -> dict[str, int | str]:
    """Decode a tabular state key into named columns for CSV artifacts."""
    (
        intent_type,
        difficulty_level,
        confidence_level,
        misconception_type,
        retrieval_quality,
        intent_uncertainty,
        cognitive_load,
        turn_index,
        attempt_count,
        last_action,
        safety_risk,
        resolved_flag,
    ) = state_key
    return {
        "intent_type": INTENT_LABELS[intent_type],
        "difficulty_level": DIFFICULTY_LABELS[difficulty_level],
        "confidence_level": CONFIDENCE_LABELS[confidence_level],
        "misconception_type": MISCONCEPTION_LABELS[misconception_type],
        "retrieval_quality": QUALITY_LABELS[retrieval_quality],
        "intent_uncertainty": LEVEL_LABELS[intent_uncertainty],
        "cognitive_load": LEVEL_LABELS[cognitive_load],
        "turn_index": turn_index,
        "attempt_count": attempt_count,
        "last_action": "none" if last_action == NONE_ACTION else ACTION_LABELS[last_action],
        "safety_risk": RISK_LABELS[safety_risk],
        "resolved_flag": resolved_flag,
    }


def default_reward(
    previous_state: AssistantState,
    action: int,
    next_state: AssistantState,
    done: bool,
) -> float:
    """Reward grounded resolution while charging for drag, churn, and unsafe shortcuts."""
    confidence_gain = next_state.confidence_level - previous_state.confidence_level
    retrieval_gain = next_state.retrieval_quality - previous_state.retrieval_quality
    safety_gain = previous_state.safety_risk - next_state.safety_risk
    resolved_bonus = 8.0 if next_state.resolved_flag and next_state.retrieval_quality >= 1 else 5.0 if next_state.resolved_flag else 0.0
    escalation_bonus = 2.0 if action == 7 and previous_state.safety_risk == 2 else 0.0
    turn_penalty = 1.0
    switch_penalty = 2.0 if previous_state.last_action not in (NONE_ACTION, action) and not next_state.resolved_flag else 0.0
    ungrounded_penalty = 4.0 if action in (2, 3, 4) and previous_state.retrieval_quality == 0 else 0.0
    horizon_penalty = 5.0 if done and not next_state.resolved_flag and action != 7 else 0.0
    missed_escalation_penalty = 7.0 if previous_state.safety_risk == 2 and action != 7 and done and not next_state.resolved_flag else 0.0
    reward = (
        resolved_bonus
        + 3.0 * confidence_gain
        + 1.0 * retrieval_gain
        + 2.0 * safety_gain
        + escalation_bonus
        - turn_penalty
        - ACTION_COSTS[action]
        - switch_penalty
        - ungrounded_penalty
        - horizon_penalty
        - missed_escalation_penalty
    )
    return round(reward, 4)


RewardFunction = Callable[[AssistantState, int, AssistantState, bool], float]


@dataclass
class AssistantInterventionEnvironment:
    """Finite-horizon tutoring environment used across the project."""

    horizon: int = DEFAULT_HORIZON
    reward_fn: RewardFunction = default_reward

    def __post_init__(self) -> None:
        self._state: AssistantState | None = None
        self._done = False
        self._scenario_name = ""

    def reset(self, seed: int | None = None, scenario_id: int = 0) -> AssistantState:
        """Start a new tutoring episode from one named scenario."""
        del seed  # deterministic by scenario in the core environment
        scenario = SCENARIOS[scenario_id]
        self._scenario_name = scenario.name
        self._state = AssistantState(
            intent_type=scenario.intent_type,
            difficulty_level=scenario.difficulty_level,
            confidence_level=scenario.confidence_level,
            misconception_type=scenario.misconception_type,
            retrieval_quality=scenario.retrieval_quality,
            intent_uncertainty=scenario.intent_uncertainty,
            cognitive_load=scenario.cognitive_load,
            turn_index=0,
            attempt_count=scenario.attempt_count,
            last_action=NONE_ACTION,
            safety_risk=safety_risk_from_state(
                difficulty_level=scenario.difficulty_level,
                confidence_level=scenario.confidence_level,
                intent_uncertainty=scenario.intent_uncertainty,
                cognitive_load=scenario.cognitive_load,
                attempt_count=scenario.attempt_count,
                misconception_type=scenario.misconception_type,
                resolved_flag=0,
            ),
            resolved_flag=0,
        )
        self._done = False
        return self._state

    def observe(self) -> AssistantState:
        """Return the current state."""
        if self._state is None:
            raise RuntimeError("environment must be reset before observe")
        return self._state

    def is_done(self) -> bool:
        """Report whether the episode has ended."""
        return self._done

    @property
    def scenario_name(self) -> str:
        """Return the current scenario label."""
        return self._scenario_name

    def step(self, action: int) -> TransitionResult:
        """Advance the tutoring state by one intervention choice."""
        if action not in ACTION_LABELS:
            raise ValueError(f"unknown action: {action}")
        if self._done:
            raise RuntimeError("episode is done; call reset before stepping again")
        previous_state = self.observe()
        next_state, escalated = self._transition(previous_state, action)
        done = bool(escalated or next_state.resolved_flag or next_state.turn_index >= self.horizon)
        reward = self.reward_fn(previous_state, action, next_state, done)
        self._state = next_state
        self._done = done
        return TransitionResult(
            state=next_state,
            reward=reward,
            done=done,
            info={
                "action": action,
                "action_label": ACTION_LABELS[action],
                "action_cost": ACTION_COSTS[action],
                "scenario_name": self._scenario_name,
                "resolved": next_state.resolved_flag,
                "safety_risk": next_state.safety_risk,
            },
        )

    def _transition(self, state: AssistantState, action: int) -> tuple[AssistantState, bool]:
        """Apply one pedagogical intervention to the current tutoring state."""
        confidence = state.confidence_level
        misconception = state.misconception_type
        retrieval = state.retrieval_quality
        uncertainty = state.intent_uncertainty
        load = state.cognitive_load
        attempts = state.attempt_count
        resolved = state.resolved_flag
        escalated = False

        if action == 0:
            uncertainty = clamp(uncertainty - 1, 0, 2)
            load = clamp(load - 1, 0, 2)
        elif action == 1:
            retrieval = clamp(retrieval + 1, 0, 2)
            confidence = clamp(confidence + (1 if uncertainty == 0 else 0), 0, 2)
        elif action == 2:
            attempts = clamp(attempts + 1, 0, 3)
            if retrieval >= 1:
                confidence = clamp(confidence + 1, 0, 2)
                misconception = lowered_misconception(misconception)
                if uncertainty == 0 and misconception == 0 and state.difficulty_level <= 1:
                    resolved = 1
            else:
                load = clamp(load + 1, 0, 2)
        elif action == 3:
            attempts = clamp(attempts + 1, 0, 3)
            confidence = clamp(confidence + 1, 0, 2)
            misconception = lowered_misconception(misconception, strength=2)
            if retrieval >= 1:
                load = clamp(load - 1, 0, 2)
            else:
                load = clamp(load + 1, 0, 2)
            if uncertainty == 0 and retrieval >= 1 and misconception <= 1:
                resolved = 1
        elif action == 4:
            attempts = clamp(attempts + 1, 0, 3)
            if uncertainty == 0 and misconception == 0 and confidence >= 1:
                resolved = 1
            else:
                confidence = clamp(confidence - 1, 0, 2)
        elif action == 5:
            if confidence == 2 and uncertainty == 0 and misconception == 0 and retrieval >= 1:
                resolved = 1
            else:
                confidence = clamp(confidence - 1, 0, 2)
        elif action == 6:
            load = clamp(load - 1, 0, 2)
            uncertainty = clamp(uncertainty - 1, 0, 2)
            confidence = clamp(confidence + 1, 0, 2)
        else:
            escalated = True

        next_state = AssistantState(
            intent_type=state.intent_type,
            difficulty_level=state.difficulty_level,
            confidence_level=confidence,
            misconception_type=misconception,
            retrieval_quality=retrieval,
            intent_uncertainty=uncertainty,
            cognitive_load=load,
            turn_index=state.turn_index + 1,
            attempt_count=attempts,
            last_action=action,
            safety_risk=safety_risk_from_state(
                difficulty_level=state.difficulty_level,
                confidence_level=confidence,
                intent_uncertainty=uncertainty,
                cognitive_load=load,
                attempt_count=attempts,
                misconception_type=misconception,
                resolved_flag=resolved,
            ),
            resolved_flag=resolved,
        )
        return next_state, escalated


def scenario_catalog() -> tuple[ScenarioDefinition, ...]:
    """Return the immutable scenario list used across the project."""
    return SCENARIOS
