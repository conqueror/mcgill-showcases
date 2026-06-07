"""Pin tabular Q-learning's training contract, its edge over random, and its off-policy target.

These tests fix the value-based control rung
(contextual bandit -> MDP -> Q-learning -> DQN -> ...). They assert real properties, not just that
code runs:

1. Training emits the documented per-episode curve and a populated Q-table (the TD loop ran).
2. The greedy Q-policy beats a uniform-random policy on average return across the scenario bank --
   the headline value-of-learning claim.
3. Q-learning is genuinely *off-policy*: its greedy backup makes the learned value of a terminal
   commit approach the Bellman-optimal value (here, the immediate reward, since terminal actions
   carry no bootstrap), independent of the exploratory behaviour that generated the data.
4. ``q_table_rows`` flattens the table into the exact long-format CSV schema the artifact pipeline
   relies on.

RL concept:
    Off-policy temporal-difference control (Q-learning) and its Bellman-optimality target.

Math:
    TD error delta = target - Q(s,A) with target = R_{t+1} + gamma*max_a' Q(s',a'); fixed point is
    the Bellman optimality value Q*(s,a) = E[R_{t+1} + gamma*max_a' Q*(s',a')].
"""

from __future__ import annotations

from collections.abc import Sequence

from learning_agents.environment import (
    ACTION_LABELS,
    AgentDecisionEnvironment,
    AgentState,
    default_reward,
)
from learning_agents.policies import Policy, RandomPolicy
from learning_agents.q_learning import QLearningResult, q_table_rows, train_q_learning

SCENARIO_BANK: tuple[int, ...] = (0, 1, 2, 3, 4)


def _rollout_return(
    policy: Policy,
    *,
    scenario_ids: Sequence[int],
    horizon: int = 5,
    seed: int | None = None,
) -> float:
    """Sum the undiscounted return of a policy rolled out once on each scenario.

    Re-simulates the policy inside the known agent-decision MDP and accumulates R_{t+1} over every
    step, which estimates the policy's finite-horizon value under the scenario distribution. This is
    a self-contained evaluation helper (the showcase has no separate evaluation module yet at this
    rung), kept deterministic via the optional start-state seed.

    Args:
        policy: The policy to evaluate (must satisfy the ``Policy`` protocol).
        scenario_ids: Scenarios to roll the policy out on, one episode each.
        horizon: Episode length H for the environment.
        seed: Optional start-state jitter seed; ``None`` uses each scenario's exact start.

    Returns:
        The total undiscounted return summed across the scenarios.

    RL concept: simulator-based policy evaluation -- estimating G_t = sum_k R_{t+k+1} by rollout.
    """
    total = 0.0
    for scenario_id in scenario_ids:
        policy.reset()
        environment = AgentDecisionEnvironment(horizon=horizon)
        state = environment.reset(seed=seed, scenario_id=scenario_id)
        while not environment.is_done():
            transition = environment.step(policy.select_action(state))
            total += transition.reward
            state = transition.state
    return total


def test_q_learning_training_returns_curve_and_q_table() -> None:
    """Training emits one curve row per episode and a non-empty Q-table.

    Pins the output contract of ``train_q_learning``: exactly ``episodes`` training-curve rows, a
    populated tabular Q(s,a), and the per-episode bookkeeping columns
    (episode, scenario_id, total_reward, epsilon, steps) needed to plot learning and the epsilon
    schedule. Structural -- it pins that the agent-environment loop ran and recorded values,
    independent of how good the learned policy is.

    RL concept:
        Tabular action-value storage Q(s,a) populated by TD updates.
    """
    result = train_q_learning(episodes=24, seed=11, scenario_ids=SCENARIO_BANK)

    assert isinstance(result, QLearningResult)
    # One curve row per episode; Q-table populated => the TD loop actually ran.
    assert len(result.training_curve) == 24
    assert result.q_table
    assert {"episode", "scenario_id", "total_reward", "epsilon", "steps"}.issubset(
        result.training_curve[0]
    )
    # Every learned row holds exactly one value per action in ACTION_LABELS.
    assert all(len(row) == len(ACTION_LABELS) for row in result.q_table.values())


def test_train_q_learning_rejects_nonpositive_episodes() -> None:
    """Zero or negative episode counts are rejected before any learning runs."""
    import pytest

    with pytest.raises(ValueError):
        train_q_learning(episodes=0)


def test_trained_q_policy_outperforms_random_baseline() -> None:
    """The learned greedy policy beats uniform-random on average return across the scenario bank.

    Pins the core value-of-learning claim: after enough decayed-epsilon episodes the greedy Q-policy
    earns strictly higher return than ``RandomPolicy`` on the exact scenarios and, averaged over a
    spread of jittered starts, generalizes the same way. Comparing against a baseline is how we tell
    learning actually helped rather than memorizing one lucky start.

    RL concept:
        Policy improvement from value learning, evaluated against a baseline on matched scenarios.
    """
    result = train_q_learning(
        episodes=600,
        seed=7,
        scenario_ids=SCENARIO_BANK,
        epsilon=0.35,
        epsilon_decay=0.96,
    )
    learned = result.greedy_policy()
    assert learned.name == "q_table"
    random_policy = RandomPolicy(seed=7)

    # On the exact (un-jittered) scenarios the learned policy strictly beats random.
    learned_exact = _rollout_return(learned, scenario_ids=SCENARIO_BANK)
    random_exact = _rollout_return(random_policy, scenario_ids=SCENARIO_BANK)
    assert learned_exact > random_exact

    # And it generalizes: averaged over jittered starts it still wins by a clear margin.
    learned_mean = sum(
        _rollout_return(learned, scenario_ids=SCENARIO_BANK, seed=s) for s in range(8)
    ) / 8.0
    random_mean = sum(
        _rollout_return(random_policy, scenario_ids=SCENARIO_BANK, seed=s) for s in range(8)
    ) / 8.0
    assert learned_mean > random_mean


def test_q_learning_is_off_policy_and_approaches_bellman_optimal_value() -> None:
    """The greedy backup makes a terminal commit's learned value approach its Bellman-optimal value.

    The off-policy signature of Q-learning is that its TD target uses ``max_a' Q(s',a')`` rather
    than the action the exploratory behaviour took. For a *terminal* commit the bootstrap term is
    zero, so the Bellman-optimal action value is exactly the immediate reward R(s, a, s', True).
    After enough visits the learned Q for ``answer_direct`` at the easy-factual start (grounded:
    difficulty 0, ambiguity 0) must converge to that immediate reward -- regardless of how much the
    epsilon-greedy behaviour wandered. This is the Bellman-optimality fixed point the off-policy
    target chases.

    RL concept:
        Off-policy control: the greedy (max) backup targets Q*(s,a) independent of the behaviour
        policy; for a terminal action Q*(s,a) = R_{t+1}.
    """
    result = train_q_learning(
        episodes=600,
        seed=7,
        scenario_ids=SCENARIO_BANK,
        epsilon=0.35,
        epsilon_decay=0.96,
    )

    environment = AgentDecisionEnvironment()
    start = environment.reset(scenario_id=0)  # easy_factual: difficulty 0, ambiguity 0, evidence 0
    learned_row = result.q_table[start.as_tuple()]

    # answer_direct (0) is terminal here; its only situational change is the clock advancing.
    answered = AgentState(
        step=start.step + 1,
        intent=start.intent,
        difficulty=start.difficulty,
        ambiguity=start.ambiguity,
        evidence=start.evidence,
        attempts=start.attempts,
        budget=start.budget,
    )
    bellman_optimal = default_reward(start, 0, answered, True)

    # Learned greedy value of the terminal commit approaches its Bellman-optimal (immediate) value.
    assert abs(learned_row[0] - bellman_optimal) < 1e-3
    # And answering is the greedy choice on an already-grounded, unambiguous request.
    assert max(range(len(learned_row)), key=lambda a: learned_row[a]) == 0


def test_q_table_rows_emits_long_format_schema() -> None:
    """``q_table_rows`` flattens the table to one deterministic row per (state, action) pair.

    Pins the artifact contract: every state key expands into one row per action carrying the seven
    unpacked state fields, the action index, and its rounded Q-value. Rows are emitted in sorted
    state order so the CSV is deterministic and diff-friendly.

    RL concept:
        Reading off the learned action-value function Q(s,a) for export/inspection.
    """
    state_a = AgentState(
        step=0, intent=0, difficulty=0, ambiguity=0, evidence=0, attempts=0, budget=30
    )
    state_b = AgentState(
        step=1, intent=2, difficulty=1, ambiguity=2, evidence=0, attempts=0, budget=27
    )
    q_table = {
        state_b.as_tuple(): [0.0, 0.5, 0.25, 1.0],
        state_a.as_tuple(): [2.0, -0.1, 0.0, -0.5],
    }

    rows = q_table_rows(q_table)

    expected_columns = {
        "step",
        "intent",
        "difficulty",
        "ambiguity",
        "evidence",
        "attempts",
        "budget",
        "action",
        "q_value",
    }
    assert len(rows) == 2 * len(ACTION_LABELS)
    assert all(set(row) == expected_columns for row in rows)
    # Deterministic order: the smaller state tuple (state_a) comes first, actions ascending.
    assert [row["action"] for row in rows[: len(ACTION_LABELS)]] == [0, 1, 2, 3]
    assert rows[0]["step"] == 0 and rows[0]["budget"] == 30
    assert rows[0]["action"] == 0 and rows[0]["q_value"] == 2.0
    # The unpacked fields round-trip the original state key exactly.
    assert (
        rows[0]["step"],
        rows[0]["intent"],
        rows[0]["difficulty"],
        rows[0]["ambiguity"],
        rows[0]["evidence"],
        rows[0]["attempts"],
        rows[0]["budget"],
    ) == state_a.as_tuple()
