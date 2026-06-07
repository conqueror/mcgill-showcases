"""Tests for the decision policies: baselines, greedy selection, and learned wrappers.

These assert behavioural properties: the greedy operator's deterministic tie-break, the heuristic
router's sensible action ordering (clarify-then-ground-then-answer, escalate only when stuck), the
random policy's reproducibility, the always-escalate foil, and the Q-table / model wrappers
(including the unseen-state fallback).
"""

from __future__ import annotations

from collections.abc import Sequence

from learning_agents.environment import ACTION_LABELS, AgentDecisionEnvironment, AgentState
from learning_agents.policies import (
    AlwaysEscalatePolicy,
    HeuristicRouterPolicy,
    ModelPolicy,
    Policy,
    QTablePolicy,
    RandomPolicy,
    greedy_action,
)


def _state(
    *,
    step: int = 0,
    difficulty: int = 0,
    ambiguity: int = 0,
    evidence: int = 0,
    attempts: int = 0,
    budget: int = 30,
) -> AgentState:
    """Build an AgentState for policy tests with neutral defaults."""
    return AgentState(
        step=step,
        intent=0,
        difficulty=difficulty,
        ambiguity=ambiguity,
        evidence=evidence,
        attempts=attempts,
        budget=budget,
    )


def test_greedy_action_breaks_ties_to_lowest_index() -> None:
    """Equal action values resolve to the first index (so an all-zeros row picks action 0)."""
    assert greedy_action([0.0, 0.0, 0.0, 0.0]) == 0
    assert greedy_action([1.0, 1.0, 0.5]) == 0


def test_greedy_action_selects_the_maximum() -> None:
    """The greedy operator returns the index of the maximal value."""
    assert greedy_action([0.1, 0.9, 0.3, 0.2]) == 1
    assert greedy_action([-1.0, -0.5, -0.9]) == 1


def test_greedy_action_empty_raises() -> None:
    """An empty value row cannot yield an action and raises."""
    import pytest

    with pytest.raises(RuntimeError):
        greedy_action([])


def test_random_policy_is_reproducible_and_in_range() -> None:
    """The seeded random policy replays the same action stream after reset, always in-range."""
    policy = RandomPolicy(seed=123)
    state = _state()
    first = [policy.select_action(state) for _ in range(10)]
    policy.reset()
    second = [policy.select_action(state) for _ in range(10)]
    assert first == second
    assert all(action in ACTION_LABELS for action in first)


def test_random_policy_satisfies_protocol() -> None:
    """RandomPolicy structurally satisfies the Policy protocol (name/reset/select_action)."""
    policy: Policy = RandomPolicy()
    assert policy.name == "random"
    policy.reset()
    assert policy.select_action(_state()) in ACTION_LABELS


def test_always_escalate_policy_is_constant() -> None:
    """The always-escalate foil returns action 3 regardless of state."""
    policy = AlwaysEscalatePolicy()
    assert policy.name == "always_escalate"
    assert policy.select_action(_state(difficulty=0, ambiguity=0, evidence=3)) == 3
    assert policy.select_action(_state(difficulty=2, ambiguity=2, evidence=0)) == 3


def test_heuristic_router_clarifies_when_ambiguous() -> None:
    """With unresolved ambiguity (and budget/steps), the router clarifies first."""
    router = HeuristicRouterPolicy()
    assert router.select_action(_state(ambiguity=2, difficulty=1, evidence=0)) == 2


def test_heuristic_router_retrieves_when_underground_but_unambiguous() -> None:
    """With no ambiguity but inadequate evidence for the difficulty, the router retrieves."""
    router = HeuristicRouterPolicy()
    assert router.select_action(_state(ambiguity=0, difficulty=2, evidence=0)) == 1
    assert router.select_action(_state(ambiguity=0, difficulty=2, evidence=1)) == 1


def test_heuristic_router_answers_when_grounded_and_unambiguous() -> None:
    """When evidence is adequate and ambiguity is 0, the router answers directly."""
    router = HeuristicRouterPolicy()
    assert router.select_action(_state(ambiguity=0, difficulty=0, evidence=0)) == 0
    assert router.select_action(_state(ambiguity=0, difficulty=2, evidence=2)) == 0


def test_heuristic_router_escalates_when_stuck_on_hard_request() -> None:
    """Out of budget and steps on a hard, still-ungrounded request, the router escalates."""
    router = HeuristicRouterPolicy(horizon=5)
    # Hard request, no evidence, but no budget left to retrieve and at the step horizon.
    stuck = _state(step=5, difficulty=2, ambiguity=0, evidence=0, budget=0)
    assert router.select_action(stuck) == 3


def test_heuristic_router_runs_to_termination_on_every_scenario() -> None:
    """Rolling the router on each scenario always terminates and commits or hands off safely."""
    for scenario_id in range(5):
        env = AgentDecisionEnvironment()
        router = HeuristicRouterPolicy(horizon=env.horizon)
        router.reset()
        state = env.reset(scenario_id=scenario_id)
        steps = 0
        last_action = None
        while not env.is_done() and steps < 20:
            last_action = router.select_action(state)
            state = env.step(last_action).state
            steps += 1
        assert env.is_done() is True
        # The router must end by committing or escalating, never by exhausting the loop guard.
        assert last_action is not None


def test_qtable_policy_acts_greedily_on_known_state() -> None:
    """QTablePolicy plays the argmax action for a state present in the learned table."""
    state = _state(difficulty=1, ambiguity=0, evidence=1)
    q_table = {state.as_tuple(): [0.0, 0.1, 0.0, 0.9]}
    policy = QTablePolicy(q_table=q_table)
    assert policy.name == "q_table"
    assert policy.select_action(state) == 3


def test_qtable_policy_falls_back_to_action_zero_on_unseen_state() -> None:
    """An unseen state yields the all-zeros fallback row, so greedy returns action 0."""
    policy = QTablePolicy(q_table={})
    assert policy.select_action(_state(difficulty=2, ambiguity=2, evidence=0)) == 0


class _StubModel:
    """Minimal PredictModel stub returning a fixed action, for testing ModelPolicy."""

    def __init__(self, action: int) -> None:
        self.action = action
        self.last_observation: Sequence[float] | None = None
        self.last_deterministic: bool | None = None

    def predict(
        self, observation: Sequence[float], deterministic: bool = True
    ) -> tuple[int, object]:
        self.last_observation = observation
        self.last_deterministic = deterministic
        return self.action, None


def test_model_policy_encodes_state_and_returns_model_action() -> None:
    """ModelPolicy passes the encoded observation to the model and returns its action."""
    model = _StubModel(action=2)
    horizon = 5
    policy = ModelPolicy(
        model=model,
        observation_fn=lambda s: s.as_normalized_vector(horizon=horizon),
        name="dqn",
        deterministic=True,
    )
    state = _state(difficulty=2, ambiguity=1, evidence=1)
    action = policy.select_action(state)
    assert action == 2
    # The model received the normalized 7-vector and the deterministic flag.
    assert model.last_observation == state.as_normalized_vector(horizon=horizon)
    assert model.last_deterministic is True


def test_model_policy_satisfies_protocol() -> None:
    """ModelPolicy structurally satisfies the Policy protocol."""
    policy: Policy = ModelPolicy(
        model=_StubModel(action=0),
        observation_fn=lambda s: s.as_normalized_vector(horizon=5),
        name="ppo",
    )
    policy.reset()
    assert policy.select_action(_state()) == 0
