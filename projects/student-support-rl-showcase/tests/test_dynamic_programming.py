"""Tests that backward-induction Q* is exact and that model-free Q-learning converges to it.

These tests anchor the *planning* rung that sits just below model-free control on the ladder
(contextual bandit -> MDP -> Q-learning -> DQN -> policy gradient -> actor-critic -> PPO). The
module under test solves the finite-horizon deterministic MDP exactly by backward induction,
producing the ground-truth optimum Q*(s,a) that tabular Q-learning only *estimates* from
samples. Two complementary correctness checks guard Q*: a same-table Bellman self-consistency
sweep, and a fully independent forward recursion (``_independent_v_star``) that cannot share a
bug with the backward sweep. The remaining tests pin determinism, the gamma=0 collapse to the
immediate reward, the CSV row schemas, and the central learning claim -- that Q-learning's
table moves measurably toward Q* with more training.

RL concept:
    Dynamic programming / value iteration on a known MDP versus model-free Q-learning; see
    docs/value-based-learning.md, docs/mdp-and-environment.md and docs/math-notes.md.

Math:
    Bellman optimality (deterministic transition s -> s'):
        Q*(s,a) = E[R_{t+1} + gamma * max_a' Q*(s',a')] = R_{t+1} + (0 if done else gamma * V*(s'))
    with V*(s) = max_a Q*(s,a), return G_t = sum_k gamma^k R_{t+k+1}, and TD error
    delta = target - Q(s,A) equal to 0 for every (s,a) at the exact fixed point.
"""

from __future__ import annotations

from functools import cache

from student_support_rl.dynamic_programming import (
    gap_rows,
    model_step,
    optimal_action_value_rows,
    optimal_action_values,
    q_learning_gap,
    reachable_acting_states,
)
from student_support_rl.environment import ACTION_LABELS, StudentState
from student_support_rl.q_learning import StateKey, train_q_learning

HORIZON = 6
GAMMA = 0.9


def _q_star() -> dict[StateKey, list[float]]:
    """Build the exact backward-induction Q* used as the shared ground truth across tests."""
    return optimal_action_values(horizon=HORIZON, gamma=GAMMA)


@cache
def _independent_v_star(key: StateKey) -> float:
    """Ground-truth V*(s) by full forward recursion to the horizon.

    Deliberately re-derived from scratch (only the env *model* via ``model_step`` is reused),
    so it cannot share a bug with ``optimal_action_values``'s backward-induction sweep. Used to
    catch a wrong successor mapping or ordering that a same-table self-consistency check misses.
    """
    state = __state(key)
    best: float | None = None
    for action in ACTION_LABELS:
        transition = model_step(state, action, horizon=HORIZON)
        successor = transition.state.as_tuple()
        future = 0.0 if transition.done else GAMMA * _independent_v_star(successor)
        candidate = transition.reward + future
        best = candidate if best is None else max(best, candidate)
    assert best is not None  # ACTION_LABELS is non-empty
    return best


def test_reachable_states_respect_week_invariant() -> None:
    """Every acting state at decision step t has week == t, so backward induction is sound.

    The enumerated acting states span weeks ``1..H`` exactly, and week-H states are the
    base case (every action from them terminates the episode). This week-equals-step
    invariant is what lets the solver sweep states by descending week in a single pass
    with no fixed-point iteration.

    RL concept:
        Reachable state space of a finite-horizon MDP (docs/mdp-and-environment.md).
    """
    states = reachable_acting_states(horizon=HORIZON)
    assert states
    # every acting state reached at decision step t has week == t, so weeks span 1..H
    assert {key[0] for key in states} == set(range(1, HORIZON + 1))
    # week-H acting states are the base case: every action from them terminates the episode
    for key in states:
        if key[0] == HORIZON:
            assert all(model_step(__state(key), a, horizon=HORIZON).done for a in ACTION_LABELS)


def __state(key: StateKey) -> StudentState:
    return StudentState(*key)


def test_bellman_optimality_self_consistency() -> None:
    """Q* satisfies its own Bellman optimality fixed point: every residual is ~0.

    For each reachable (s, a) the stored value must equal its one-step Bellman target
    ``r + (0 if done else gamma * max_a' Q*(s', a'))``. A correct backward-induction sweep
    reproduces this exactly (max residual < 1e-9). This is necessary but not sufficient --
    it re-derives Q* from the SAME table, so a systematically wrong successor mapping could
    pass; the independent-recursion test below closes that gap.

    RL concept:
        Bellman optimality equation (docs/math-notes.md, docs/value-based-learning.md).

    Math:
        Q*(s, a) = R_{t+1} + (0 if done else gamma * max_a' Q*(s', a')).
    """
    q_star = _q_star()
    # for every reachable (s, a): Q*(s,a) == r + (0 if done else gamma * max_a' Q*(s',a'))
    max_residual = 0.0
    for state_key, action_values in q_star.items():
        state = __state(state_key)
        for action in ACTION_LABELS:
            transition = model_step(state, action, horizon=HORIZON)
            future = 0.0 if transition.done else GAMMA * max(q_star[transition.state.as_tuple()])
            target = transition.reward + future
            max_residual = max(max_residual, abs(action_values[action] - target))
    # one ordered backward-induction sweep reproduces the Bellman target exactly
    assert max_residual < 1e-9


def test_q_star_matches_independent_forward_recursion() -> None:
    """Q* matches a fully independent forward recursion, catching a wrong successor mapping.

    Cross-validation against ``_independent_v_star`` -- a from-scratch recursive V* that reuses
    only the env *model*, never the backward-induction code path -- so the two cannot share a
    bug. Every entry must agree (max diff < 1e-9), and the table must have exactly 612
    reachable acting states (the 5 scenarios at H=6), guarding against silent state-space drift.

    RL concept:
        Independent re-derivation of the DP optimum (docs/value-based-learning.md).
    """
    # Self-consistency above re-derives Q* from the SAME stored table, so it cannot detect a
    # systematically wrong successor mapping. Here we compare every entry against a fully
    # independent recursive V* (built without the backward-induction code path at all).
    q_star = _q_star()
    assert len(q_star) == 612  # exact reachable acting-state count for the 5 scenarios at H=6
    max_diff = 0.0
    for state_key, action_values in q_star.items():
        state = __state(state_key)
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

    Unlike sampling-based learning, dynamic programming on a known model has no RNG, so two
    solves must return exactly equal tables. This makes Q* a stable reference to diff learned
    tables against.

    RL concept:
        Determinism of model-based planning (docs/value-based-learning.md).
    """
    first = optimal_action_values(horizon=HORIZON, gamma=GAMMA)
    second = optimal_action_values(horizon=HORIZON, gamma=GAMMA)
    assert first == second


def test_optimal_action_value_rows_schema_and_count() -> None:
    """The flattened Q* CSV has one row per (state, action) with the documented columns.

    Pins the artifact contract: row count equals ``|states| * |actions|`` and the column set
    is the six state fields plus ``action`` and ``optimal_q_value``. Stable schemas let the
    optimal-value dump be diffed against the learned-Q dump downstream.

    RL concept:
        Tabular Q* artifact for inspection (docs/value-based-learning.md).
    """
    q_star = _q_star()
    rows = optimal_action_value_rows(q_star)
    assert len(rows) == len(q_star) * len(ACTION_LABELS)
    assert set(rows[0]) == {
        "week",
        "engagement",
        "completion",
        "pressure",
        "risk",
        "prior_interventions",
        "action",
        "optimal_q_value",
    }


def test_gamma_zero_collapses_to_immediate_reward() -> None:
    """With gamma = 0 the bootstrap term vanishes, so Q*(s,a) equals the one-step reward.

    Checks the discounting boundary case for EVERY reachable acting state (not a sample): when
    gamma = 0 the agent is fully myopic and ``Q*(s, a) = R_{t+1}``. Verifying this globally
    confirms the future-value term is gated correctly everywhere, not just on average.

    RL concept:
        Role of the discount factor gamma (docs/mdp-and-environment.md, docs/math-notes.md).

    Math:
        gamma = 0  =>  Q*(s, a) = R_{t+1}.
    """
    # with gamma = 0 there is no bootstrapping, so Q*(s,a) is exactly the one-step reward
    # for EVERY reachable acting state (not a sample) -- the bootstrap term must vanish globally.
    q_star = optimal_action_values(horizon=HORIZON, gamma=0.0)
    assert q_star  # non-empty
    for state_key in q_star:
        state = __state(state_key)
        for action in ACTION_LABELS:
            transition = model_step(state, action, horizon=HORIZON)
            assert abs(q_star[state_key][action] - transition.reward) < 1e-9


def test_q_learning_converges_toward_optimum() -> None:
    """Model-free Q-learning approaches the model-based optimum Q* as training proceeds.

    The headline learning claim, asserted two ways. Absolute: a well-trained table
    (2500 episodes) reaches >400 states with mean ``|Q_learned - Q*|`` < 4.0 (Q* spans roughly
    [-17, +9]; the 4.0 bar leaves headroom yet fails if the backup regresses to no learning).
    Relative: on the states shared by a well-trained and a barely-trained (5-episode) table,
    more training strictly cuts the mean gap by a non-trivial margin (> 0.25). The shared-state
    restriction keeps the comparison apples-to-apples since the two tables visit different sets.

    RL concept:
        Convergence of off-policy TD control to Q* (docs/value-based-learning.md).

    Math:
        gap = |Q_learned(s, a) - Q*(s, a)|; training drives the mean gap toward 0.
    """
    q_star = _q_star()

    trained = train_q_learning(
        episodes=2500,
        seed=7,
        epsilon=0.5,
        epsilon_decay=0.999,
        epsilon_min=0.1,
    ).q_table
    barely_trained = train_q_learning(episodes=5, seed=7).q_table

    trained_gap = q_learning_gap(trained, q_star)
    # absolute convergence: a well-trained table sits close to the exact optimum. Measured
    # mean_abs_gap is ~3.0-3.18 across seeds (Q* spans roughly [-17, +9]); 4.0 keeps headroom
    # while still failing if the backup/target ever regresses to a no-learning baseline.
    assert trained_gap["num_states"] > 400  # 2500 episodes reach the great majority of states
    assert float(trained_gap["mean_abs_gap"]) < 4.0

    # apples-to-apples improvement on the states BOTH tables cover (Q* ∩ trained ∩ barely):
    common = set(trained) & set(barely_trained)
    common_trained = {key: trained[key] for key in common}
    common_barely = {key: barely_trained[key] for key in common}
    trained_common = q_learning_gap(common_trained, q_star)
    barely_common = q_learning_gap(common_barely, q_star)
    assert trained_common["num_states"] > 0
    # training strictly reduces the mean approximation error toward Q* on shared states, by a
    # non-trivial margin (a flaky == tie or a regressed backup would not clear this bar).
    improvement = float(barely_common["mean_abs_gap"]) - float(trained_common["mean_abs_gap"])
    assert improvement > 0.25


def test_gap_rows_schema_matches_shared_states() -> None:
    """Per-(state, action) gap rows cover exactly the shared states and agree with the summary.

    Pins the convergence-artifact contract: there is one row per (state, action) over states
    present in BOTH the learned and optimal tables, the column set matches the documented
    schema (state fields + learned/optimal values + ``abs_gap``), and the row-level maximum
    ``abs_gap`` equals the summary's ``max_abs_gap``. Lets learners sort by ``abs_gap`` to find
    the decisions Q-learning still gets wrong.

    RL concept:
        State-resolved value-based approximation error (docs/value-based-learning.md).
    """
    q_star = _q_star()
    trained = train_q_learning(episodes=40, seed=7).q_table
    rows = gap_rows(trained, q_star)
    summary = q_learning_gap(trained, q_star)

    shared = {key for key in q_star if key in trained}
    assert len(rows) == len(shared) * len(ACTION_LABELS)
    assert set(rows[0]) == {
        "week",
        "engagement",
        "completion",
        "pressure",
        "risk",
        "prior_interventions",
        "action",
        "learned_q_value",
        "optimal_q_value",
        "abs_gap",
    }
    # the row-level max abs_gap agrees with the summary's max_abs_gap
    assert abs(max(float(row["abs_gap"]) for row in rows) - float(summary["max_abs_gap"])) < 1e-6
