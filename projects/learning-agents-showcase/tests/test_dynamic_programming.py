"""Tests that backward-induction Q* is exact and that model-free Q-learning converges to it.

These tests anchor the *planning* rung that sits just below model-free control on the ladder
(contextual bandit -> MDP -> Q-learning -> SARSA -> REINFORCE -> actor-critic -> PPO). The module
under test solves the finite-horizon deterministic agent-decision MDP exactly by backward induction,
producing the ground-truth optimum Q*(s, a) that tabular Q-learning only *estimates* from samples.

Q* is guarded two complementary ways: a same-table Bellman self-consistency sweep, and a fully
independent forward recursion (``_independent_v_star``) that cannot share a bug with the backward
sweep. The remaining tests pin determinism, the gamma=0 collapse to the immediate reward, the
step-equals-depth invariant that makes the backward sweep sound, the CSV row schemas, and the
central learning claim -- that an off-policy Q-learner's table moves toward Q* with more training.

To keep this a focused port of ``dynamic_programming.py`` with no dependency on a separate learning
module, the convergence test trains a small *self-contained* tabular Q-learner
(``_train_q_learning``) defined in this file. It is ordinary off-policy TD control against the real
environment, so "model-free control approaches the model-based optimum" is exercised, not stubbed.

RL concept:
    Dynamic programming / value iteration on a known MDP versus model-free Q-learning.

Math:
    Bellman optimality (deterministic transition s -> s'):
        Q*(s, a) = E[R_{t+1} + gamma * max_a' Q*(s', a')] = R_{t+1} + (0 if done else gamma*V*(s'))
    with V*(s) = max_a Q*(s, a), return G_t = sum_k gamma^k R_{t+k+1}, and TD error
    delta = target - Q(s, A) equal to 0 for every (s, a) at the exact fixed point.
"""

from __future__ import annotations

import random
from functools import cache

from learning_agents.dynamic_programming import (
    StateKey,
    gap_rows,
    model_step,
    optimal_action_value_rows,
    optimal_action_values,
    q_learning_gap,
    reachable_acting_states,
)
from learning_agents.environment import (
    ACTION_LABELS,
    SCENARIOS,
    AgentDecisionEnvironment,
    AgentState,
)

HORIZON = 5
GAMMA = 0.9
NUM_ACTIONS = len(ACTION_LABELS)
# Exact reachable acting-state count for the 5 scenarios at H=5, BFS-seeded from the full +/-1
# difficulty/ambiguity jitter box that reset() can draw (26 distinct start states); a literal guard
# against silent state-space drift (a changed transition/horizon/jitter would move this number).
EXPECTED_NUM_STATES = 371


def _q_star() -> dict[StateKey, list[float]]:
    """Build the exact backward-induction Q* used as the shared ground truth across tests."""
    return optimal_action_values(horizon=HORIZON, gamma=GAMMA)


def _state(key: StateKey) -> AgentState:
    """Rebuild an :class:`AgentState` from its 7-int key (the inverse of ``as_tuple``)."""
    return AgentState(*key)


@cache
def _independent_v_star(key: StateKey) -> float:
    """Ground-truth V*(s) by full forward recursion to the horizon.

    Deliberately re-derived from scratch (only the env *model* via ``model_step`` is reused), so it
    cannot share a bug with ``optimal_action_values``'s backward-induction sweep. Used to catch a
    wrong successor mapping or ordering that a same-table self-consistency check misses.
    """
    state = _state(key)
    best: float | None = None
    for action in ACTION_LABELS:
        transition = model_step(state, action, horizon=HORIZON)
        successor = transition.state.as_tuple()
        future = 0.0 if transition.done else GAMMA * _independent_v_star(successor)
        candidate = transition.reward + future
        best = candidate if best is None else max(best, candidate)
    assert best is not None  # ACTION_LABELS is non-empty
    return best


def _train_q_learning(
    *,
    episodes: int,
    seed: int,
    alpha: float = 0.5,
    epsilon: float = 0.9,
) -> dict[StateKey, list[float]]:
    """Train a minimal tabular off-policy Q-learner against the real environment.

    What + why: kept inside the test so this file ports ``dynamic_programming.py`` alone, with no
    dependency on a separate learning module. It is textbook epsilon-greedy Q-learning -- the
    off-policy TD update ``Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))`` -- so the
    convergence claim is exercised against the same MDP the planner solves, not a stub. Every RNG is
    seeded, so the run is deterministic.

    Exploration is high (epsilon defaults to 0.9) on purpose: ``answer_direct`` and ``escalate`` are
    terminal commits, so a greedy-leaning learner ends episodes early and never reaches the deep
    acting states behind those commits. Heavy exploration lets coverage approach the full reachable
    set, which is what makes "approaches Q* across the state space" a meaningful claim.

    Args:
        episodes: Number of training episodes (rollouts across all scenarios in round-robin).
        seed: Seed for the action-selection RNG, making the run reproducible.
        alpha: TD learning rate in (0, 1].
        epsilon: Exploration probability for the epsilon-greedy behaviour policy (high by default so
            terminal commits do not starve deep states of visits).

    Returns:
        The learned Q-table mapping a state's 7-int key to its list of per-action values.
    """
    rng = random.Random(seed)
    q_table: dict[StateKey, list[float]] = {}
    env = AgentDecisionEnvironment(horizon=HORIZON)
    for episode in range(episodes):
        # Round-robin over scenarios so every start state gets visited as training proceeds.
        scenario_id = episode % len(SCENARIOS)
        state = env.reset(scenario_id=scenario_id)
        while not env.is_done():
            key = state.as_tuple()
            row = q_table.setdefault(key, [0.0] * NUM_ACTIONS)
            if rng.random() < epsilon:
                action = rng.randrange(NUM_ACTIONS)  # explore
            else:
                best = max(row)  # exploit: greedy w.r.t. current estimates (first argmax)
                action = next(a for a, value in enumerate(row) if value == best)
            result = env.step(action)
            next_key = result.state.as_tuple()
            future = (
                0.0
                if result.done
                else GAMMA * max(q_table.setdefault(next_key, [0.0] * NUM_ACTIONS))
            )
            target = result.reward + future  # off-policy TD target: bootstrap on max_a' Q(s',a')
            row[action] += alpha * (target - row[action])
            state = result.state
    return q_table


def test_reachable_states_respect_step_invariant() -> None:
    """Every acting state reached at BFS depth t has step == t, so backward induction is sound.

    The enumerated acting states span steps ``0..H`` exactly (a fresh episode starts at step 0), and
    step-H states are the base case (every action from them terminates the episode -- a commit, a
    horizon cutoff, or a budget give-up). This step-equals-depth invariant is what lets the solver
    sweep states by descending step in a single pass with no fixed-point iteration.

    RL concept:
        Reachable state space of a finite-horizon MDP.
    """
    states = reachable_acting_states(horizon=HORIZON)
    assert states
    # acting states span steps 0..H (reset is step 0, the last decision is at step H)
    assert {key[0] for key in states} == set(range(0, HORIZON + 1))
    # step-H acting states are the base case: every action from them terminates the episode
    for key in states:
        if key[0] == HORIZON:
            assert all(model_step(_state(key), a, horizon=HORIZON).done for a in ACTION_LABELS)


def test_bellman_optimality_self_consistency() -> None:
    """Q* satisfies its own Bellman optimality fixed point: every residual is ~0.

    For each reachable (s, a) the stored value must equal its one-step Bellman target
    ``r + (0 if done else gamma * max_a' Q*(s', a'))``. A correct backward-induction sweep
    reproduces this exactly (max residual < 1e-9). This is necessary but not sufficient -- it
    re-derives Q* from the SAME table, so a wrong successor mapping could pass; the
    independent-recursion test below closes that gap.

    RL concept:
        Bellman optimality equation.

    Math:
        Q*(s, a) = R_{t+1} + (0 if done else gamma * max_a' Q*(s', a')).
    """
    q_star = _q_star()
    max_residual = 0.0
    for state_key, action_values in q_star.items():
        state = _state(state_key)
        for action in ACTION_LABELS:
            transition = model_step(state, action, horizon=HORIZON)
            future = 0.0 if transition.done else GAMMA * max(q_star[transition.state.as_tuple()])
            target = transition.reward + future
            max_residual = max(max_residual, abs(action_values[action] - target))
    # one ordered backward-induction sweep reproduces the Bellman target exactly
    assert max_residual < 1e-9


def test_q_star_matches_independent_forward_recursion() -> None:
    """Q* matches a fully independent forward recursion, catching a wrong successor mapping.

    Cross-validation against ``_independent_v_star`` -- a from-scratch recursive V* that reuses only
    the env *model*, never the backward-induction code path -- so the two cannot share a bug. Every
    entry must agree (max diff < 1e-9), and the table must have exactly ``EXPECTED_NUM_STATES``
    reachable acting states, guarding against silent state-space drift.

    RL concept:
        Independent re-derivation of the DP optimum.
    """
    q_star = _q_star()
    assert len(q_star) == EXPECTED_NUM_STATES
    max_diff = 0.0
    for state_key, action_values in q_star.items():
        state = _state(state_key)
        for action in ACTION_LABELS:
            transition = model_step(state, action, horizon=HORIZON)
            future = (
                0.0
                if transition.done
                else GAMMA * _independent_v_star(transition.state.as_tuple())
            )
            independent_q = transition.reward + future
            max_diff = max(max_diff, abs(action_values[action] - independent_q))
    assert max_diff < 1e-9


def test_optimal_action_values_are_deterministic() -> None:
    """Solving the MDP twice yields identical Q*: planning is a deterministic computation.

    Unlike sampling-based learning, dynamic programming on a known model has no RNG, so two solves
    must return exactly equal tables. This makes Q* a stable reference to diff learned tables.

    RL concept:
        Determinism of model-based planning.
    """
    first = optimal_action_values(horizon=HORIZON, gamma=GAMMA)
    second = optimal_action_values(horizon=HORIZON, gamma=GAMMA)
    assert first == second


def test_optimal_policy_routes_each_scenario_sensibly() -> None:
    """The greedy policy induced by Q* picks the orchestration move each scenario was designed for.

    A behavioural sanity check on the joint (reward + dynamics + solver): from each scenario's start
    state, ``argmax_a Q*(s, a)`` should be the move that scenario rewards -- answer an easy factual
    request, retrieve for a medium how-to, clarify an ambiguous query, and escalate the genuinely
    hard/needs-human cases. If the reward or the backup were wrong, these would not line up.

    RL concept:
        The optimal greedy policy pi(s) = argmax_a Q*(s, a) derived from the planned optimum.
    """
    q_star = _q_star()
    expected_action = {
        "easy_factual": 0,  # adequately grounded + unambiguous -> answer directly
        "howto_medium": 1,  # needs one unit of grounding -> retrieve
        "ambiguous_query": 2,  # under-specified -> clarify first
        "hard_debug": 3,  # hard enough that a human hand-off wins
        "needs_escalation": 3,  # hard AND ambiguous -> escalate
    }
    for scenario in SCENARIOS:
        start = AgentDecisionEnvironment(horizon=HORIZON).reset(scenario_id=scenario.scenario_id)
        values = q_star[start.as_tuple()]
        greedy = max(range(NUM_ACTIONS), key=lambda a: values[a])
        assert greedy == expected_action[scenario.name], scenario.name


def test_optimal_action_value_rows_schema_and_count() -> None:
    """The flattened Q* CSV has one row per (state, action) with the documented columns.

    Pins the artifact contract: row count equals ``|states| * |actions|`` and the column set is the
    seven state fields plus ``action`` and ``optimal_q_value``. Stable schemas let the optimal-value
    dump be diffed against a learned-Q dump downstream.

    RL concept:
        Tabular Q* artifact for inspection.
    """
    q_star = _q_star()
    rows = optimal_action_value_rows(q_star)
    assert len(rows) == len(q_star) * NUM_ACTIONS
    assert set(rows[0]) == {
        "step",
        "intent",
        "difficulty",
        "ambiguity",
        "evidence",
        "attempts",
        "budget",
        "action",
        "optimal_q_value",
    }
    # rows are sorted by state then action, so the first four actions belong to one state
    assert [row["action"] for row in rows[:NUM_ACTIONS]] == list(range(NUM_ACTIONS))


def test_gamma_zero_collapses_to_immediate_reward() -> None:
    """With gamma = 0 the bootstrap term vanishes, so Q*(s, a) equals the one-step reward.

    Checks the discounting boundary case for EVERY reachable acting state (not a sample): when
    gamma = 0 the agent is fully myopic and ``Q*(s, a) = R_{t+1}``. Verifying this globally confirms
    the future-value term is gated correctly everywhere, not just on average.

    RL concept:
        Role of the discount factor gamma.

    Math:
        gamma = 0  =>  Q*(s, a) = R_{t+1}.
    """
    q_star = optimal_action_values(horizon=HORIZON, gamma=0.0)
    assert q_star  # non-empty
    for state_key in q_star:
        state = _state(state_key)
        for action in ACTION_LABELS:
            transition = model_step(state, action, horizon=HORIZON)
            assert abs(q_star[state_key][action] - transition.reward) < 1e-9


def test_q_learning_converges_toward_optimum() -> None:
    """Model-free Q-learning approaches the model-based optimum Q* as training proceeds.

    The headline learning claim, asserted two ways. Absolute: a well-trained table reaches the great
    majority of states with a small mean ``|Q_learned - Q*|`` (Q* spans roughly [-1.5, +2.0], so the
    0.5 bar still fails if the backup regresses to no learning). Relative: on the states shared by a
    well-trained and a barely-trained (3-episode) table, more training strictly cuts the mean gap by
    a non-trivial margin. The shared-state restriction keeps the comparison apples-to-apples since
    the two tables visit different sets.

    RL concept:
        Convergence of off-policy TD control to Q*.

    Math:
        gap = |Q_learned(s, a) - Q*(s, a)|; training drives the mean gap toward 0.
    """
    q_star = _q_star()

    trained = _train_q_learning(episodes=5000, seed=7)
    barely_trained = _train_q_learning(episodes=3, seed=7)

    trained_gap = q_learning_gap(trained, q_star)
    # absolute convergence: a well-trained table sits close to the exact optimum. Q* spans roughly
    # [-1.5, 2.0]; measured mean_abs_gap is ~0.33, so the 0.5 bar keeps headroom yet still fails if
    # the backup regresses to a no-learning baseline.
    # The in-test learner trains on the noise-free scenario starts, so it visits the ~105 states
    # reachable from those centres -- a subset of Q*'s 371 jitter-inclusive states. Heavy
    # exploration reaches almost all of that noise-free-reachable subset, so >90 shared states hold.
    assert int(trained_gap["num_states"]) > 90
    assert float(trained_gap["mean_abs_gap"]) < 0.5

    # apples-to-apples improvement on the states BOTH tables cover (Q* ∩ trained ∩ barely):
    common = set(trained) & set(barely_trained)
    common_trained = {key: trained[key] for key in common}
    common_barely = {key: barely_trained[key] for key in common}
    trained_common = q_learning_gap(common_trained, q_star)
    barely_common = q_learning_gap(common_barely, q_star)
    assert int(trained_common["num_states"]) > 0
    # training strictly reduces the mean approximation error toward Q* on shared states, by a
    # non-trivial margin (a flaky == tie or a regressed backup would not clear this bar).
    improvement = float(barely_common["mean_abs_gap"]) - float(trained_common["mean_abs_gap"])
    assert improvement > 0.25


def test_q_learning_gap_is_zero_against_itself() -> None:
    """A table compared to itself has zero gap, fixing the metric's lower bound.

    Sanity-anchors :func:`q_learning_gap`: feeding Q* in for the learned table must yield
    ``max_abs_gap == mean_abs_gap == 0`` over all ``EXPECTED_NUM_STATES`` shared states, so any
    non-zero gap elsewhere is genuine approximation error, not a metric artefact.

    RL concept:
        The approximation-error metric is exact at the optimum (gap to Q* is 0 for Q* itself).
    """
    q_star = _q_star()
    gap = q_learning_gap(q_star, q_star)
    assert gap["num_states"] == EXPECTED_NUM_STATES
    assert float(gap["max_abs_gap"]) == 0.0
    assert float(gap["mean_abs_gap"]) == 0.0


def test_q_learning_gap_handles_disjoint_tables() -> None:
    """With no shared states the gap summary reports zeros and a zero state count.

    Guards the documented empty-intersection contract: a learned table that shares no state with Q*
    yields ``{"max_abs_gap": 0.0, "mean_abs_gap": 0.0, "num_states": 0}`` rather than raising.

    RL concept:
        Restricting the comparison to visited states keeps it apples-to-apples.
    """
    q_star = _q_star()
    disjoint: dict[StateKey, list[float]] = {(99, 99, 99, 99, 99, 99, 99): [0.0] * NUM_ACTIONS}
    gap = q_learning_gap(disjoint, q_star)
    assert gap == {"max_abs_gap": 0.0, "mean_abs_gap": 0.0, "num_states": 0}
    assert gap_rows(disjoint, q_star) == []


def test_gap_rows_schema_matches_shared_states() -> None:
    """Per-(state, action) gap rows cover exactly the shared states and agree with the summary.

    Pins the convergence-artifact contract: there is one row per (state, action) over states present
    in BOTH the learned and optimal tables, the column set matches the documented schema (state
    fields + learned/optimal values + ``abs_gap``), and the row-level maximum ``abs_gap`` equals the
    summary's ``max_abs_gap``. Lets learners sort by ``abs_gap`` to find the decisions Q-learning
    still gets wrong.

    RL concept:
        State-resolved value-based approximation error.
    """
    q_star = _q_star()
    trained = _train_q_learning(episodes=40, seed=7)
    rows = gap_rows(trained, q_star)
    summary = q_learning_gap(trained, q_star)

    shared = {key for key in q_star if key in trained}
    assert rows  # training visits at least some states
    assert len(rows) == len(shared) * NUM_ACTIONS
    assert set(rows[0]) == {
        "step",
        "intent",
        "difficulty",
        "ambiguity",
        "evidence",
        "attempts",
        "budget",
        "action",
        "learned_q_value",
        "optimal_q_value",
        "abs_gap",
    }
    # the row-level max abs_gap agrees with the summary's max_abs_gap
    assert abs(max(float(row["abs_gap"]) for row in rows) - float(summary["max_abs_gap"])) < 1e-6
