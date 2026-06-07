"""Tests for the vendored deep-RL lane (NumPy DQN and actor-critic PPO).

These pin the mechanics (MLP forward/backward, feature encoding), the learning behaviour (both
deep methods clear the random floor and mostly solve the scenarios), determinism by seed, and the
optional-DRL artifact schema (the q_learning/dqn/ppo family comparison and per-scenario rollups).
Budgets are deliberately small so the suite stays fast; the qualitative claims hold regardless.
"""

from __future__ import annotations

import numpy as np

from learning_agents.deep_rl import (
    MLP,
    DQNModel,
    FamilyEntry,
    PPOModel,
    build_model_policy,
    family_comparison_rows,
    state_features,
    train_dqn,
    train_ppo,
)
from learning_agents.environment import ACTION_LABELS, AgentState
from learning_agents.policies import QTablePolicy
from learning_agents.q_learning import train_q_learning

_SAMPLE_STATE = AgentState(
    step=0, intent=1, difficulty=1, ambiguity=0, evidence=0, attempts=0, budget=30
)


def test_state_features_are_unit_scaled() -> None:
    """The feature encoding returns a 7-vector in [0, 1] (network-ready inputs)."""
    features = state_features(_SAMPLE_STATE, horizon=5)
    assert features.shape == (7,)
    assert float(features.min()) >= 0.0
    assert float(features.max()) <= 1.0


def test_mlp_backward_reduces_mse() -> None:
    """One MLP can fit a fixed target by gradient descent (forward/backward are consistent)."""
    rng = np.random.default_rng(0)
    network = MLP.initialize(rng, in_dim=7, hidden=16, out_dim=4)
    inputs = rng.random((8, 7))
    target = rng.random((8, 4))

    def mse() -> float:
        prediction, _ = network.forward(inputs)
        return float(np.mean((prediction - target) ** 2))

    start = mse()
    for _ in range(200):
        prediction, cache = network.forward(inputs)
        d_output = (2.0 / inputs.shape[0]) * (prediction - target)
        network.backward(cache, d_output, learning_rate=0.1)
    assert mse() < start * 0.5  # loss at least halves


def test_train_dqn_learns_and_is_deterministic() -> None:
    """DQN clears the random floor, mostly solves the scenarios, and is reproducible by seed."""
    model, curve = train_dqn(episodes=120, epsilon=0.2, seed=0)
    assert curve  # a training curve was recorded
    policy = build_model_policy(model, name="dqn", horizon=5)
    comparison, _ = family_comparison_rows([FamilyEntry(policy, "value_based_deep")])
    row = comparison[0]
    assert float(row["avg_reward"]) > 0.3  # well above the random floor (~ -1.18)
    assert float(row["solved_rate"]) >= 0.8

    again, _ = train_dqn(episodes=120, epsilon=0.2, seed=0)
    assert np.allclose(model.action_values(_SAMPLE_STATE), again.action_values(_SAMPLE_STATE))


def test_train_ppo_learns_above_floor() -> None:
    """PPO learns a policy clearly better than random (even if it lands in a safe local optimum)."""
    model, curve = train_ppo(iterations=15, episodes_per_iteration=12, seed=0)
    assert curve
    policy = build_model_policy(model, name="ppo", horizon=5)
    comparison, _ = family_comparison_rows([FamilyEntry(policy, "actor_critic_policy_gradient")])
    row = comparison[0]
    assert float(row["avg_reward"]) > 0.0  # above the random floor
    assert float(row["solved_rate"]) >= 0.6


def test_dqn_model_predict_returns_valid_action() -> None:
    """A trained DQN's ``predict`` returns a valid action index for the SB3-style interface."""
    model, _ = train_dqn(episodes=60, seed=0)
    action, _ = model.predict(state_features(_SAMPLE_STATE, horizon=5).tolist())
    assert action in ACTION_LABELS


def test_ppo_predict_deterministic_is_the_mode() -> None:
    """PPO deterministic ``predict`` returns argmax of pi; sampling stays within the action set."""
    model, _ = train_ppo(iterations=10, episodes_per_iteration=10, seed=0)
    observation = state_features(_SAMPLE_STATE, horizon=5).tolist()
    probabilities = model.action_probabilities(_SAMPLE_STATE)
    greedy, _ = model.predict(observation, deterministic=True)
    assert greedy == int(np.argmax(probabilities))
    sampled, _ = model.predict(observation, deterministic=False)
    assert sampled in ACTION_LABELS


def test_family_comparison_schema_and_three_families() -> None:
    """The comparison contrasts q_learning, dqn, and ppo with the contract's columns + rollups."""
    q_table = train_q_learning(episodes=200, seed=0).q_table
    q_policy = QTablePolicy(q_table=q_table, name="q_learning")
    dqn_model, _ = train_dqn(episodes=80, seed=0)
    ppo_model, _ = train_ppo(iterations=8, episodes_per_iteration=10, seed=0)
    entries = [
        FamilyEntry(q_policy, "tabular_value_based"),
        FamilyEntry(build_model_policy(dqn_model, name="dqn", horizon=5), "value_based_deep"),
        FamilyEntry(
            build_model_policy(ppo_model, name="ppo", horizon=5), "actor_critic_policy_gradient"
        ),
    ]
    comparison, rollups = family_comparison_rows(entries, episodes_per_scenario=2)

    assert {row["policy"] for row in comparison} == {"q_learning", "dqn", "ppo"}
    for row in comparison:
        assert set(row) == {"policy", "family", "avg_reward", "avg_escalation_rate", "solved_rate"}
    assert rollups
    for row in rollups:
        assert set(row) == {"policy", "scenario_id", "scenario_name", "total_reward"}


def test_models_satisfy_policy_protocol() -> None:
    """The wrapped DQN/PPO models behave as ``Policy`` objects (name/reset/select_action)."""
    dqn_model, _ = train_dqn(episodes=40, seed=0)
    policy = build_model_policy(dqn_model, name="dqn", horizon=5)
    assert policy.name == "dqn"
    policy.reset()
    assert policy.select_action(_SAMPLE_STATE) in ACTION_LABELS


def test_predict_models_are_typed() -> None:
    """Smoke-check the dataclasses expose the documented fields for ``ModelPolicy`` wrapping."""
    dqn_model, _ = train_dqn(episodes=20, seed=0)
    ppo_model, _ = train_ppo(iterations=4, episodes_per_iteration=6, seed=0)
    assert isinstance(dqn_model, DQNModel)
    assert isinstance(ppo_model, PPOModel)
    assert dqn_model.horizon == 5
    assert ppo_model.horizon == 5
