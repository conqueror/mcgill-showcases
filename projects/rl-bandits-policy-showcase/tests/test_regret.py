from __future__ import annotations

from rl_bandits_showcase.bandits import EpsilonGreedy
from rl_bandits_showcase.environment import BernoulliBanditEnvironment
from rl_bandits_showcase.simulation import run_policy_simulation


def test_cumulative_regret_non_negative() -> None:
    env = BernoulliBanditEnvironment(arm_probs=[0.2, 0.4, 0.5], seed=8)
    policy = EpsilonGreedy(n_arms=3, epsilon=0.1, seed=8)
    trace = run_policy_simulation(policy, env, horizon=100)
    assert (trace["cumulative_regret"] >= 0).all()
