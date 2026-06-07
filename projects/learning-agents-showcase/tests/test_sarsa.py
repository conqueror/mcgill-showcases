"""Tests for tabular SARSA -- on-policy temporal-difference control.

These assert the load-bearing properties of on-policy TD control on the agent-decision MDP:

* SARSA *learns* -- its greedy policy beats the uniform-random baseline on the judge-rubric reward
  across the whole scenario distribution (otherwise the algorithm is not doing its job).
* SARSA is *on-policy* -- its backup bootstraps from the value of the action it actually takes next
  (Q[s'][A']), which differs from the off-policy Q-learning target max_a' Q[s'][a'] whenever the
  chosen next action is not the greedy one. This is the single structural difference between the two
  TD methods, asserted directly on a controlled transition.
* The run is *deterministic* given the seed, the per-episode training curve follows the documented
  schema, epsilon decays to its floor, and the result wraps into a working greedy policy.
"""

from __future__ import annotations

import random
import statistics

import pytest

from learning_agents.environment import (
    ACTION_LABELS,
    AgentDecisionEnvironment,
    AgentState,
    scenario_catalog,
)
from learning_agents.policies import Policy, QTablePolicy, RandomPolicy, greedy_action
from learning_agents.sarsa import (
    SarsaResult,
    _sarsa_backup_target,
    train_sarsa,
)


def _mean_return(
    policy: Policy, *, seed_base: int, episodes_per_scenario: int, horizon: int
) -> float:
    """Roll a policy across every scenario and return its mean undiscounted episode return.

    What + why: a single comparable scalar for "how good is this policy" -- the same rollout code
    scores a learned and a baseline policy identically, which is exactly how the evaluation harness
    decides whether learning helped. RL concept: Monte-Carlo policy evaluation by averaging episode
    returns over the start-state distribution.

    Args:
        policy: Any object satisfying the Policy protocol.
        seed_base: Base reset seed; offset per episode so each draws a reproducible start state.
        episodes_per_scenario: Episodes rolled per scenario.
        horizon: Episode length H for the environment.

    Returns:
        The mean total reward across all scenario/episode rollouts.
    """
    totals: list[float] = []
    for scenario_id in range(len(scenario_catalog())):
        for episode in range(episodes_per_scenario):
            environment = AgentDecisionEnvironment(horizon=horizon)
            state = environment.reset(seed=seed_base + episode, scenario_id=scenario_id)
            policy.reset()
            total = 0.0
            while not environment.is_done():
                action = policy.select_action(state)
                transition = environment.step(action)
                total += transition.reward
                state = transition.state
            totals.append(total)
    return statistics.mean(totals)


def test_train_sarsa_rejects_nonpositive_episodes() -> None:
    """Training with zero or negative episodes is a usage error and raises."""
    with pytest.raises(ValueError):
        train_sarsa(episodes=0)
    with pytest.raises(ValueError):
        train_sarsa(episodes=-5)


def test_train_sarsa_returns_well_formed_result() -> None:
    """The result holds a Q-table over the 7-int state key and a per-episode training curve."""
    result = train_sarsa(episodes=40, seed=7)
    assert isinstance(result, SarsaResult)
    assert result.q_table  # at least one state visited
    for key, values in result.q_table.items():
        # The tabular key is the agent-decision state's 7-tuple; each row has one value per action.
        assert isinstance(key, tuple)
        assert len(key) == 7
        assert len(values) == len(ACTION_LABELS)
    assert len(result.training_curve) == 40


def test_training_curve_has_documented_schema_and_decays_epsilon() -> None:
    """Every curve row exposes the documented keys; epsilon is non-increasing toward its floor."""
    epsilon_min = 0.05
    result = train_sarsa(
        episodes=120, seed=7, epsilon=0.4, epsilon_decay=0.97, epsilon_min=epsilon_min
    )
    expected_keys = {"episode", "scenario_id", "total_reward", "epsilon", "steps"}
    epsilons: list[float] = []
    for index, row in enumerate(result.training_curve, start=1):
        assert set(row.keys()) == expected_keys
        assert row["episode"] == index
        assert row["scenario_id"] in range(len(scenario_catalog()))
        assert row["steps"] >= 1  # at least one committing action per episode
        assert row["epsilon"] >= epsilon_min
        epsilons.append(float(row["epsilon"]))
    # Epsilon-greedy exploration is annealed: the schedule never increases and ends at the floor.
    consecutive = zip(epsilons, epsilons[1:], strict=False)  # offset pairing: lengths differ by 1
    assert all(later <= earlier + 1e-9 for earlier, later in consecutive)
    assert epsilons[-1] == pytest.approx(epsilon_min)


def test_train_sarsa_is_deterministic_under_fixed_seed() -> None:
    """A fixed seed reproduces the Q-table and the training curve exactly (seeded RNG)."""
    first = train_sarsa(episodes=80, seed=11)
    second = train_sarsa(episodes=80, seed=11)
    assert first.q_table == second.q_table
    assert first.training_curve == second.training_curve


def test_sarsa_greedy_policy_beats_random_baseline() -> None:
    """The learned greedy policy outscores uniform-random on the judge reward across scenarios.

    RL concept: the whole point of value-based control -- learning Q(s, a) must yield a policy that
    is strictly better than acting at random, otherwise the learner adds nothing.
    """
    horizon = 5
    result = train_sarsa(episodes=400, seed=7, horizon=horizon)
    learned = result.greedy_policy()
    assert isinstance(learned, QTablePolicy)
    learned_mean = _mean_return(learned, seed_base=1000, episodes_per_scenario=30, horizon=horizon)
    random_mean = _mean_return(
        RandomPolicy(seed=42), seed_base=1000, episodes_per_scenario=30, horizon=horizon
    )
    assert learned_mean > random_mean


def test_sarsa_greedy_policy_handles_unseen_states() -> None:
    """The greedy policy is total: an unseen state falls back to all-zeros -> action 0."""
    result = train_sarsa(episodes=20, seed=7)
    policy = result.greedy_policy()
    # A budget that can never arise during training (caps at STARTING_BUDGET) is an unseen state.
    unseen = AgentState(
        step=0, intent=0, difficulty=2, ambiguity=2, evidence=0, attempts=0, budget=999
    )
    assert unseen.as_tuple() not in result.q_table
    assert policy.select_action(unseen) == 0


def test_sarsa_backup_is_on_policy_not_max() -> None:
    """SARSA's TD target uses the CHOSEN next action A', not max_a' Q(s', a') (off-policy).

    What + why: this is the one structural difference between SARSA and Q-learning, so we assert it
    directly on the *production* backup (:func:`learning_agents.sarsa._sarsa_backup_target`) -- the
    exact function the training loop calls -- rather than on a formula re-derived in the test (which
    would pass even if the implementation secretly used ``max`` and was really Q-learning). On a
    controlled non-terminal transition we pick a next action A' that is NOT the greedy/argmax action
    and confirm the real target equals R + gamma*Q(s', A') and *differs* from the off-policy
    R + gamma*max_a' Q(s', a').

    RL concept: on-policy (SARSA) vs off-policy (Q-learning) TD control -- same sampled transition,
    different bootstrap target.
    """
    gamma = 0.9
    environment = AgentDecisionEnvironment(horizon=5)
    environment.reset(scenario_id=1)  # howto_medium: a non-terminal start

    # Take a non-terminal action (retrieve) so a genuine s' with future value exists.
    transition = environment.step(1)
    assert transition.done is False  # retrieve is non-terminal -> we bootstrap from s'

    # Seed Q(s', .) so the greedy next action (argmax, index 0) is NOT the action our forced
    # epsilon-greedy picks. With epsilon=1.0 the behaviour policy explores uniformly; a fixed RNG
    # makes that exploratory pick deterministic and (here) different from the argmax.
    next_values = [5.0, 1.0, 0.5, 0.25]
    assert greedy_action(next_values) == 0  # off-policy max would bootstrap from this index
    chosen_next_action = _force_epsilon_greedy(next_values, random.Random(0))
    assert chosen_next_action != 0  # the on-policy A' differs from the greedy/max action

    # The PRODUCTION backup target -- this is what the training loop actually computes.
    actual_target = _sarsa_backup_target(
        reward=transition.reward,
        gamma=gamma,
        next_action_values=next_values,
        next_action=chosen_next_action,
        done=transition.done,
    )
    on_policy_target = transition.reward + gamma * next_values[chosen_next_action]
    off_policy_target = transition.reward + gamma * max(next_values)  # what Q-learning would use

    # The real implementation bootstraps from the CHOSEN action A' (on-policy)...
    assert actual_target == pytest.approx(on_policy_target)
    # ...and is genuinely DIFFERENT from the off-policy max target on this transition. If the
    # implementation regressed to ``max`` (i.e. became Q-learning), this assertion fails.
    assert actual_target != pytest.approx(off_policy_target)


def test_sarsa_backup_target_zeros_future_on_terminal() -> None:
    """On a terminal transition the production backup is exactly R (no bootstrap), per the math.

    What + why: SARSA's target is R + gamma*Q(s', A') only while the episode continues; once
    ``done`` is True there is no next action to follow, so the future term must vanish and the agent
    must not bootstrap past the horizon. We assert this on the real backup function for both a
    terminal and a non-terminal case, since mishandling the terminal flag silently corrupts every
    end-of-episode update.

    RL concept: episodic bootstrapping -- the terminal state has zero continuation value.
    """
    next_values = [5.0, 1.0, 0.5, 0.25]
    # Terminal: future is dropped, target collapses to the immediate reward regardless of gamma/A'.
    terminal_target = _sarsa_backup_target(
        reward=2.0, gamma=0.9, next_action_values=next_values, next_action=0, done=True
    )
    assert terminal_target == pytest.approx(2.0)
    # Non-terminal sanity: the same inputs DO bootstrap, proving the terminal case is special-cased.
    nonterminal_target = _sarsa_backup_target(
        reward=2.0, gamma=0.9, next_action_values=next_values, next_action=0, done=False
    )
    assert nonterminal_target == pytest.approx(2.0 + 0.9 * 5.0)


def _force_epsilon_greedy(action_values: list[float], rng: random.Random) -> int:
    """Mirror SARSA's epsilon-greedy under forced exploration (epsilon = 1.0) for on-policy testing.

    What + why: with epsilon = 1.0 the behaviour policy always explores, so the chosen next action
    is a uniform-random draw -- the same code path SARSA uses to pick A'. A fixed RNG makes the draw
    deterministic so the test can compare on-policy vs off-policy targets exactly.

    Args:
        action_values: The Q(s', .) row (used only for its length under forced exploration).
        rng: Seeded RNG producing the deterministic exploratory action.

    Returns:
        A uniform-random action index in ``range(len(action_values))``.

    RL concept: the exploratory branch of the epsilon-greedy behaviour policy SARSA evaluates.
    """
    return rng.randrange(len(action_values))
