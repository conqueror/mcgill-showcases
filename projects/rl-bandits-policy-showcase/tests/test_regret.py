from __future__ import annotations

import pytest

from rl_bandits_showcase.bandits import EpsilonGreedy
from rl_bandits_showcase.environment import BernoulliBanditEnvironment
from rl_bandits_showcase.simulation import run_policy_simulation


def test_cumulative_regret_non_negative() -> None:
    env = BernoulliBanditEnvironment(arm_probs=[0.2, 0.4, 0.5], seed=8)
    policy = EpsilonGreedy(n_arms=3, epsilon=0.1, seed=8)
    trace = run_policy_simulation(policy, env, horizon=100)
    assert (trace["cumulative_regret"] >= 0).all()
    assert (trace["instant_regret"] >= 0).all()
    assert (trace["cumulative_regret"].diff().dropna() >= 0).all()


def test_pseudo_regret_uses_the_chosen_mean_on_successes_and_failures() -> None:
    env = BernoulliBanditEnvironment(arm_probs=[0.25, 0.75], seed=1)
    policy = EpsilonGreedy(n_arms=2, epsilon=0.0, seed=1)
    trace = run_policy_simulation(policy, env, horizon=4)

    assert trace["arm"].tolist() == [0, 0, 0, 0]
    assert set(trace["reward"]) == {0.0, 1.0}
    # The best mean is 3/4 and the chosen mean is 1/4: each gap is 1/2.
    assert trace["instant_regret"].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert trace["cumulative_regret"].tolist() == pytest.approx([0.5, 1.0, 1.5, 2.0])


def test_optimal_arm_has_zero_pseudo_regret_even_when_its_reward_is_zero() -> None:
    env = BernoulliBanditEnvironment(arm_probs=[0.75, 0.25], seed=1)
    policy = EpsilonGreedy(n_arms=2, epsilon=0.0, seed=1)
    trace = run_policy_simulation(policy, env, horizon=4)

    assert set(trace["reward"]) == {0.0, 1.0}
    assert trace["instant_regret"].tolist() == [0.0, 0.0, 0.0, 0.0]
