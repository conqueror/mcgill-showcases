"""Pin offline (batch) RL: logging a behaviour policy and learning from the fixed log by FQI.

These tests anchor the offline-RL rung -- learning from a *frozen* log with no new environment
interaction. They pin that: the logged dataset is reproducible and carries a valid behaviour-policy
propensity per row; the epsilon-soft behaviour distribution is well-formed; Fitted-Q Iteration
converges to a fixed point on the log; the offline-learned policy is *good* (it improves on the
heuristic behaviour policy by extracting the optimum the log covers); and -- the central honesty
of offline RL -- coverage is partial, so states the log never decides from keep an all-zeros value
(no evidence, no learning).

RL concept:
    Offline / batch RL and Fitted-Q Iteration, plus the data-coverage constraint.
"""

from __future__ import annotations

import statistics

from learning_agents.dynamic_programming import optimal_action_values
from learning_agents.environment import ACTION_LABELS, AgentDecisionEnvironment, scenario_catalog
from learning_agents.evaluation import evaluate_policies
from learning_agents.offline_rl import (
    behavior_action_probabilities,
    collect_logged_dataset,
    fitted_q_iteration,
)
from learning_agents.policies import HeuristicRouterPolicy, QTablePolicy

ALL_SCENARIOS = tuple(range(len(scenario_catalog())))
NUM_ACTIONS = len(ACTION_LABELS)


def test_behavior_action_probabilities_are_epsilon_soft() -> None:
    """The behaviour distribution sums to 1, stays positive, and peaks on the base action.

    Pins the epsilon-soft contract used to log propensities: every action keeps mass ``epsilon/|A|``
    (so the log has full action support for later importance sampling), the base policy's greedy
    action additionally carries ``1 - epsilon``, and the probabilities form a valid distribution.
    """
    state = AgentDecisionEnvironment().reset(scenario_id=2)  # ambiguous_query -> base picks clarify
    base = HeuristicRouterPolicy()
    probs = behavior_action_probabilities(base, state, epsilon=0.3)

    assert abs(sum(probs) - 1.0) < 1e-9
    assert all(p > 0.0 for p in probs)  # epsilon > 0 keeps full support
    greedy = base.select_action(state)
    assert probs[greedy] == max(probs)  # the base action carries the exploit mass
    # epsilon = 0 collapses to a deterministic (degenerate) distribution on the base action.
    greedy_probs = behavior_action_probabilities(base, state, epsilon=0.0)
    assert greedy_probs[greedy] == 1.0


def test_logged_dataset_is_reproducible_and_well_formed() -> None:
    """Two logs with the same seed are identical, and every row carries a valid propensity.

    Pins reproducibility (offline experiments must be replayable from the same seed) and the row
    contract: each logged action is legal and its recorded behaviour probability lies in (0, 1].
    """
    first = collect_logged_dataset(episodes=50, epsilon=0.3, seed=11)
    second = collect_logged_dataset(episodes=50, epsilon=0.3, seed=11)

    assert len(first) == len(second) and len(first) > 0
    assert first.transitions == second.transitions  # frozen dataclasses -> value equality
    assert first.behavior_policy_name == "heuristic_router"
    for row in first.transitions:
        assert row.action in ACTION_LABELS
        assert 0.0 < row.behavior_action_prob <= 1.0


def test_fitted_q_iteration_converges_to_a_fixed_point() -> None:
    """FQI drives the Bellman residual to ~0 on the fixed log (a batch fixed point).

    Pins that sweeping the static dataset converges: the final sweep's residual is below tolerance,
    and the curve is non-increasing (each batch sweep contracts toward the log's fixed point).
    """
    dataset = collect_logged_dataset(episodes=300, epsilon=0.6, seed=7)
    result = fitted_q_iteration(dataset, gamma=0.9, sweeps=100, tolerance=1e-6)

    residuals = [float(row["bellman_residual"]) for row in result.training_curve]
    assert residuals[-1] < 1e-6  # reached the fixed point
    # batch FQI on a finite-horizon deterministic model contracts: residuals never increase
    assert all(
        later <= earlier + 1e-9
        for earlier, later in zip(residuals, residuals[1:], strict=False)
    )


def test_offline_policy_improves_on_the_behaviour_policy() -> None:
    """The FQI policy learned from the log beats the heuristic behaviour policy offline.

    This is the offline-RL payoff: from a log of a decent-but-exploring behaviour policy, batch RL
    extracts the optimum the log covers and so *improves* on the policy that generated the data. We
    evaluate the greedy FQI policy and the heuristic router on the fixed scenarios and assert the
    learned policy scores at least as well -- offline policy improvement, not mere imitation.
    """
    dataset = collect_logged_dataset(episodes=600, epsilon=0.6, seed=7)
    result = fitted_q_iteration(dataset, gamma=0.9)
    offline_policy = QTablePolicy(q_table=result.q_table, name="offline_fqi")

    summary, _ = evaluate_policies(
        policies=[offline_policy, HeuristicRouterPolicy()],
        scenario_ids=ALL_SCENARIOS,
        episodes_per_scenario=12,
    )
    reward = {str(row["policy"]): float(row["avg_reward"]) for row in summary}
    assert reward["offline_fqi"] >= reward["heuristic_router"]


def test_fqi_approaches_optimum_on_covered_states() -> None:
    """On the states the log covers, FQI's values track the exact DP optimum Q*.

    Pins that offline learning is *correct where it has data*: restricted to the states shared with
    the backward-induction Q*, the mean absolute value gap is small. It is not zero -- thin coverage
    of some successors leaves a residual gap -- which is exactly why the metric is reported per the
    covered support rather than claimed globally.
    """
    dataset = collect_logged_dataset(episodes=600, epsilon=0.6, seed=7)
    result = fitted_q_iteration(dataset, gamma=0.9)
    q_star = optimal_action_values()

    shared = [key for key in result.q_table if key in q_star]
    gaps = [
        abs(result.q_table[key][a] - q_star[key][a])
        for key in shared
        for a in range(NUM_ACTIONS)
    ]
    assert shared  # the log covers some optimal-table states
    assert statistics.mean(gaps) < 0.6  # close on covered support (not exact: partial coverage)


def test_coverage_is_partial_and_uncovered_states_stay_zero() -> None:
    """The log covers only part of the state space, and never-decided states keep zero values.

    The defining honesty of offline RL: a state that appears only as a successor (e.g. a terminal
    state, never a logged decision point) is never updated, so its value row stays all-zeros -- the
    learner has no evidence there. We also confirm the logged decision states are a strict subset of
    the full optimal-table state space, so the offline policy is trustworthy only on that support.
    """
    dataset = collect_logged_dataset(episodes=600, epsilon=0.6, seed=7)
    result = fitted_q_iteration(dataset, gamma=0.9)
    q_star = optimal_action_values()

    decision_states = {row.state.as_tuple() for row in dataset.transitions}
    # Partial coverage: the log decides from strictly fewer states than the full reachable space.
    assert decision_states < set(q_star)
    # A successor-only state (never a logged decision) keeps its all-zeros initialisation.
    successor_only = {
        row.next_state.as_tuple()
        for row in dataset.transitions
        if row.next_state.as_tuple() not in decision_states
    }
    assert successor_only  # terminal states qualify
    sample = next(iter(successor_only))
    assert result.q_table[sample] == [0.0] * NUM_ACTIONS
