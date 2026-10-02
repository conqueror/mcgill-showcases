from __future__ import annotations

import pytest

from rl_bandits_showcase.bandits import UCB1, BanditPolicy, EpsilonGreedy, ThompsonSampling
from rl_bandits_showcase.environment import BernoulliBanditEnvironment


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), float("inf")])
def test_environment_rejects_invalid_probabilities_at_construction(probability: float) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        BernoulliBanditEnvironment(arm_probs=[probability])


@pytest.mark.parametrize("epsilon", [-0.1, 1.1, float("nan"), float("inf")])
def test_epsilon_greedy_rejects_invalid_exploration_probabilities(epsilon: float) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        EpsilonGreedy(n_arms=2, epsilon=epsilon)


def test_policies_require_at_least_one_arm() -> None:
    with pytest.raises(ValueError, match="n_arms"):
        EpsilonGreedy(n_arms=0, epsilon=0.1)
    with pytest.raises(ValueError, match="n_arms"):
        UCB1(n_arms=0)
    with pytest.raises(ValueError, match="n_arms"):
        ThompsonSampling(n_arms=0)


@pytest.mark.parametrize("arm", [-1, 2])
def test_environment_rejects_out_of_range_arm_indices(arm: int) -> None:
    env = BernoulliBanditEnvironment(arm_probs=[0.0, 1.0])
    with pytest.raises(ValueError, match="arm"):
        env.pull(arm)


@pytest.mark.parametrize(
    "policy", [EpsilonGreedy(n_arms=2, epsilon=0.1), UCB1(n_arms=2), ThompsonSampling(n_arms=2)]
)
@pytest.mark.parametrize("arm", [-1, 2])
def test_policy_updates_reject_out_of_range_arm_indices(policy: BanditPolicy, arm: int) -> None:
    with pytest.raises(ValueError, match="arm"):
        policy.update(arm, 1.0)


@pytest.mark.parametrize(
    "policy", [EpsilonGreedy(n_arms=2, epsilon=0.1), UCB1(n_arms=2), ThompsonSampling(n_arms=2)]
)
@pytest.mark.parametrize("reward", [-0.1, 0.5, 1.1, float("nan"), float("inf")])
def test_policy_updates_require_binary_rewards(policy: BanditPolicy, reward: float) -> None:
    with pytest.raises(ValueError, match="reward"):
        policy.update(0, reward)


@pytest.mark.parametrize("policy", [EpsilonGreedy(n_arms=2, epsilon=0.1), UCB1(n_arms=2)])
def test_sample_means_match_two_successes_in_three_pulls(policy: EpsilonGreedy | UCB1) -> None:
    for reward in (1.0, 0.0, 1.0):
        policy.update(0, reward)
    # A Bernoulli sample mean is successes / pulls = 2/3.
    assert policy.counts.tolist() == [3.0, 0.0]
    assert policy.values.tolist() == pytest.approx([2.0 / 3.0, 0.0])


def test_thompson_posterior_counts_successes_and_failures() -> None:
    policy = ThompsonSampling(n_arms=2)
    for reward in (1.0, 0.0, 1.0):
        policy.update(0, reward)
    # Beta(1,1) plus two successes and one failure is Beta(3,2).
    assert policy.alpha.tolist() == [3.0, 1.0]
    assert policy.beta.tolist() == [2.0, 1.0]
