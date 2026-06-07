"""Tests for the agent-decision MDP: dynamics, determinism, and termination logic.

These tests assert real MDP properties rather than smoke-checking imports: that the dynamics are
deterministic given (state, action), that each action moves exactly the variable it should, that the
three termination paths (commit, horizon, budget) fire correctly, and that ``reset`` jitter is
confined to the start state.
"""

from __future__ import annotations

import pytest

from learning_agents.environment import (
    ACTION_COSTS,
    ACTION_LABELS,
    MAX_AMBIGUITY,
    MAX_EVIDENCE,
    STARTING_BUDGET,
    AgentDecisionEnvironment,
    AgentState,
    RequestScenario,
    evidence_is_adequate,
    scenario_catalog,
)


def test_action_tables_are_consistent() -> None:
    """The label and cost tables cover the same four actions, and costs rise toward escalate."""
    assert set(ACTION_LABELS) == set(ACTION_COSTS) == {0, 1, 2, 3}
    assert ACTION_LABELS == {0: "answer_direct", 1: "retrieve", 2: "clarify", 3: "escalate"}
    # answer_direct is free to attempt; escalate is strictly the most expensive action.
    assert ACTION_COSTS[0] == 0.0
    assert ACTION_COSTS[3] == max(ACTION_COSTS.values())
    assert ACTION_COSTS[3] > ACTION_COSTS[1] > ACTION_COSTS[2] > ACTION_COSTS[0]


def test_scenario_catalog_has_five_varied_requests() -> None:
    """There are five scenarios spanning a range of difficulty and ambiguity."""
    catalog = scenario_catalog()
    assert len(catalog) == 5
    assert all(isinstance(s, RequestScenario) for s in catalog)
    assert [s.scenario_id for s in catalog] == [0, 1, 2, 3, 4]
    names = {s.name for s in catalog}
    assert {"easy_factual", "ambiguous_query", "needs_escalation"} <= names
    # The catalog must actually vary difficulty and ambiguity (not all identical requests).
    assert len({s.difficulty for s in catalog}) > 1
    assert len({s.ambiguity for s in catalog}) > 1


def test_reset_initializes_start_state() -> None:
    """Reset produces a clean S_0: step/evidence/attempts at 0 and full starting budget."""
    env = AgentDecisionEnvironment()
    state = env.reset(scenario_id=3)
    scenario = scenario_catalog()[3]
    assert state.step == 0
    assert state.evidence == 0
    assert state.attempts == 0
    assert state.budget == STARTING_BUDGET
    assert state.intent == scenario.intent
    assert state.difficulty == scenario.difficulty
    assert state.ambiguity == scenario.ambiguity
    assert env.scenario_name == scenario.name
    assert env.is_done() is False


def test_reset_rejects_out_of_range_scenario() -> None:
    """An invalid scenario id raises rather than silently clamping."""
    env = AgentDecisionEnvironment()
    with pytest.raises(ValueError):
        env.reset(scenario_id=99)
    with pytest.raises(ValueError):
        env.reset(scenario_id=-1)


def test_observe_before_reset_raises() -> None:
    """Observing before the first reset is an error (no state exists yet)."""
    env = AgentDecisionEnvironment()
    with pytest.raises(RuntimeError):
        env.observe()


def test_dynamics_are_deterministic_given_state_and_action() -> None:
    """Two environments stepped through the same actions yield identical state trajectories."""
    actions = [2, 1, 1, 0]
    trajectories = []
    for _ in range(2):
        env = AgentDecisionEnvironment()
        env.reset(scenario_id=2)
        states: list[tuple[int, ...]] = []
        for action in actions:
            if env.is_done():
                break
            result = env.step(action)
            states.append(result.state.as_tuple())
        trajectories.append(states)
    assert trajectories[0] == trajectories[1]


def test_reset_seed_only_jitters_start_not_transitions() -> None:
    """The reset seed perturbs only S_0; transitions remain deterministic afterwards."""
    env = AgentDecisionEnvironment()
    # Same seed -> identical start state (seed is consumed entirely in reset).
    first = env.reset(seed=7, scenario_id=4)
    second = env.reset(seed=7, scenario_id=4)
    assert first == second
    # Jitter stays within the valid caps for difficulty and ambiguity.
    assert 0 <= first.difficulty
    assert 0 <= first.ambiguity <= MAX_AMBIGUITY
    # After a seeded reset, stepping is still deterministic (seed never reaches step()).
    env.reset(seed=7, scenario_id=4)
    t1 = env.step(1).state.as_tuple()
    env.reset(seed=7, scenario_id=4)
    t2 = env.step(1).state.as_tuple()
    assert t1 == t2


def test_retrieve_adds_evidence_and_spends_budget() -> None:
    """retrieve increments evidence (capped) and attempts, debits budget, and is non-terminal."""
    env = AgentDecisionEnvironment()
    start = env.reset(scenario_id=3)
    result = env.step(1)
    assert result.done is False
    assert result.state.evidence == start.evidence + 1
    assert result.state.attempts == start.attempts + 1
    assert result.state.ambiguity == start.ambiguity  # retrieve does not touch ambiguity
    assert result.state.budget == start.budget - int(round(ACTION_COSTS[1] * 10))
    assert result.state.step == start.step + 1


def test_evidence_is_capped() -> None:
    """Repeated retrieves saturate evidence at MAX_EVIDENCE rather than growing without bound."""
    env = AgentDecisionEnvironment()
    env.reset(scenario_id=3)
    state = env.observe()
    for _ in range(MAX_EVIDENCE + 3):
        if env.is_done():
            break
        state = env.step(1).state
    assert state.evidence == MAX_EVIDENCE


def test_clarify_reduces_ambiguity_and_floors_at_zero() -> None:
    """clarify decrements ambiguity (floored at 0) and attempts, and is non-terminal."""
    env = AgentDecisionEnvironment()
    start = env.reset(scenario_id=2)  # ambiguity 2
    result = env.step(2)
    assert result.done is False
    assert result.state.ambiguity == start.ambiguity - 1
    assert result.state.attempts == start.attempts + 1
    assert result.state.evidence == start.evidence  # clarify does not touch evidence
    # Drive ambiguity to its floor; it must not go negative.
    state = result.state
    for _ in range(5):
        if env.is_done():
            break
        state = env.step(2).state
    assert state.ambiguity == 0


def test_answer_direct_is_terminal() -> None:
    """answer_direct ends the episode immediately and reports the matching termination reason."""
    env = AgentDecisionEnvironment()
    env.reset(scenario_id=0)
    result = env.step(0)
    assert result.done is True
    assert env.is_done() is True
    assert result.info["termination"] == "answer_direct"


def test_escalate_is_terminal_and_costly() -> None:
    """escalate ends the episode and debits the largest action cost."""
    env = AgentDecisionEnvironment()
    start = env.reset(scenario_id=4)
    result = env.step(3)
    assert result.done is True
    assert result.info["termination"] == "escalate"
    assert result.state.budget == start.budget - int(round(ACTION_COSTS[3] * 10))


def test_horizon_terminates_a_non_committing_agent() -> None:
    """An agent that only retrieves/clarifies is force-terminated once step exceeds the horizon."""
    env = AgentDecisionEnvironment(horizon=5)
    env.reset(scenario_id=3)
    last = None
    # Alternate cheap non-terminal actions so budget is not the cause of termination.
    for action in (2, 2, 1, 2, 2, 1, 2):
        if env.is_done():
            break
        last = env.step(action)
    assert last is not None
    assert last.done is True
    assert last.state.step <= env.horizon + 1
    assert last.info["termination"] in {"horizon", "budget_exhausted"}


def test_budget_violation_ends_episode_without_applying_action() -> None:
    """When an action's cost would overdraw the budget, it is not applied and the episode ends."""
    # Horizon high enough that the clock will not interfere; small budget forces the violation.
    env = AgentDecisionEnvironment(horizon=50)
    env.reset(scenario_id=3)
    last = env.observe()
    steps = 0
    # Repeatedly escalate-cost is terminal, so retrieve (cost 5 tenths) until budget can't pay.
    while not env.is_done() and steps < 100:
        before = env.observe()
        result = env.step(1)
        steps += 1
        if result.info["termination"] == "budget_exhausted":
            # The blocked action did not change evidence/attempts/budget (only the clock moved).
            assert result.done is True
            assert result.state.evidence == before.evidence
            assert result.state.attempts == before.attempts
            assert result.state.budget == before.budget
            assert result.state.budget - int(round(ACTION_COSTS[1] * 10)) < 0
            break
        last = result.state
    assert env.is_done() is True
    assert last is not None


def test_step_after_done_raises() -> None:
    """Stepping a terminated episode raises rather than silently continuing."""
    env = AgentDecisionEnvironment()
    env.reset(scenario_id=0)
    env.step(0)  # terminal
    with pytest.raises(RuntimeError):
        env.step(1)


def test_step_rejects_unknown_action() -> None:
    """An action outside the defined set raises a ValueError."""
    env = AgentDecisionEnvironment()
    env.reset(scenario_id=0)
    with pytest.raises(ValueError):
        env.step(7)


def test_state_as_tuple_round_trips_field_order() -> None:
    """as_tuple exposes the seven fields in declaration order for use as a table key."""
    state = AgentState(
        step=1, intent=2, difficulty=2, ambiguity=1, evidence=3, attempts=4, budget=12
    )
    assert state.as_tuple() == (1, 2, 2, 1, 3, 4, 12)


def test_state_normalized_vector_is_seven_unit_floats() -> None:
    """The normalized observation has seven coordinates, all within [0, 1]."""
    state = AgentState(
        step=5, intent=4, difficulty=2, ambiguity=2, evidence=3, attempts=9, budget=STARTING_BUDGET
    )
    vector = state.as_normalized_vector(horizon=5)
    assert len(vector) == 7
    assert all(0.0 <= component <= 1.0 for component in vector)
    # Fields at their maxima map to ~1.0; a fresh budget maps to ~1.0.
    assert vector[2] == pytest.approx(1.0)  # difficulty / MAX_DIFFICULTY
    assert vector[4] == pytest.approx(1.0)  # evidence / MAX_EVIDENCE
    assert vector[6] == pytest.approx(1.0)  # full starting budget


def test_normalized_vector_step_is_clamped_past_horizon() -> None:
    """A horizon-terminated state has step == horizon + 1, yet the step coord stays clamped to 1.0.

    Regression guard: a horizon-5 episode ends on a state with step 6, so step / horizon = 1.2
    without clamping; the documented [0, 1] contract must hold for this reachable terminal state.
    """
    env = AgentDecisionEnvironment(horizon=5)
    env.reset(scenario_id=3)
    last = None
    for action in (2, 2, 1, 2, 2, 1, 2):
        if env.is_done():
            break
        last = env.step(action)
    assert last is not None
    assert last.info["termination"] == "horizon"
    assert last.state.step == env.horizon + 1
    vector = last.state.as_normalized_vector(horizon=env.horizon)
    assert vector[0] == pytest.approx(1.0)
    assert all(0.0 <= component <= 1.0 for component in vector)


def test_evidence_adequacy_is_monotone_in_difficulty() -> None:
    """Harder requests require strictly more evidence to count as adequately grounded."""
    assert evidence_is_adequate(evidence=0, difficulty=0) is True
    assert evidence_is_adequate(evidence=0, difficulty=1) is False
    assert evidence_is_adequate(evidence=1, difficulty=1) is True
    assert evidence_is_adequate(evidence=1, difficulty=2) is False
    assert evidence_is_adequate(evidence=2, difficulty=2) is True
