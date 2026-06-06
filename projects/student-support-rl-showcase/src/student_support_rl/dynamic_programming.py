"""Compute exact ground-truth optimal action values Q* by backward induction.

What + why: model-free Q-learning estimates Q(s, a) by sampling transitions without
ever seeing the environment's rules. This module instead *uses the rules* -- it treats
the real :class:`StudentSupportEnvironment` as a known model and solves the finite-horizon
deterministic MDP exactly. The resulting Q* is the model-based optimum that tabular
Q-learning converges toward, so learners can lay the two tables side by side and watch the
gap shrink as training proceeds.

RL concept: this is the *planning* / dynamic-programming rung that sits just below
model-free control on the ladder
(contextual bandit -> MDP -> Q-learning -> DQN -> policy gradient -> actor-critic -> PPO).
Backward induction is value iteration specialized to a finite horizon: because every
acting state reached at decision step ``t`` has ``week == t`` and the post-terminal done
state is never acted on, solving states by descending ``week`` is well-defined and
collision-free -- each future value is already known when we need it, so a single sweep
suffices (no iteration to a fixed point required).

RL concept: see docs/mdp-and-environment.md (the MDP + model), docs/value-based-learning.md
(why Q-learning approximates this Q*), docs/math-notes.md (Bellman optimality derivation),
and docs/glossary.md (planning vs. learning).

Math:
    Bellman optimality, deterministic transition s -> s' = T(s, a):
        Q*(s, a) = R_{t+1} + (0 if done else gamma * max_a' Q*(s', a'))
    Optimal state value: V*(s) = max_a Q*(s, a).
    Return being maximized: G_t = sum_k gamma^k R_{t+k+1}.
"""

from __future__ import annotations

from collections import deque

from student_support_rl.environment import (
    ACTION_LABELS,
    SCENARIOS,
    StudentState,
    StudentSupportEnvironment,
    TransitionResult,
    default_reward,
)
from student_support_rl.environment import RewardFunction as RewardFunction
from student_support_rl.q_learning import StateKey

__all__ = [
    "model_step",
    "reachable_acting_states",
    "optimal_action_values",
    "optimal_action_value_rows",
    "q_learning_gap",
    "gap_rows",
]


def model_step(
    state: StudentState,
    action: int,
    *,
    horizon: int = 6,
    reward_fn: RewardFunction = default_reward,
) -> TransitionResult:
    """Apply one environment transition to an arbitrary state, treating the env as the model.

    What + why: backward induction needs ``(R_{t+1}, s', done)`` for every state-action
    pair, but the public env only steps from its own internal cursor. We inject the known
    state via the documented private attributes ``_state`` / ``_done`` so the *exact same*
    transition, reward, terminal week-clamp, and done-dependent reward term are reused --
    never reimplemented. A fresh env per call keeps this pure and order-independent.

    RL concept: this is the MDP's model T(s, a) -> (s', r) made queryable for planning,
    the ingredient model-free Q-learning is deliberately denied (docs/mdp-and-environment.md).

    Args:
        state: The state to act from (used as the model's current state).
        action: Action index in ``ACTION_LABELS`` (0..3).
        horizon: Episode length H; controls the terminal week-clamp and done flag.
        reward_fn: Reward function R(prev, action, next, done); defaults to the env default.

    Returns:
        The :class:`TransitionResult` (state, reward, done, info) for this single step.

    Raises:
        ValueError: If ``action`` is not a known action (propagated from ``env.step``).
    """
    env = StudentSupportEnvironment(horizon=horizon, reward_fn=reward_fn)
    env._state = state  # documented private-state injection: treat the known env as the model
    env._done = False
    return env.step(action)


def reachable_acting_states(
    *,
    horizon: int = 6,
    reward_fn: RewardFunction = default_reward,
) -> set[StateKey]:
    """Enumerate every reachable non-terminal acting state via BFS from the 5 scenarios.

    What + why: backward induction only needs the states we actually choose an action from.
    We BFS out from the canonical scenario start states; on each transition we recurse into
    the successor only when ``done`` is False, so the post-terminal done state is collected
    by no one and never acted on. This yields the full set of decision points exactly once.

    RL concept: the reachable state space of the MDP (docs/mdp-and-environment.md); only
    these states carry a well-defined Q*(s, .) because only here is an action ever taken.

    Args:
        horizon: Episode length H used to build the model.
        reward_fn: Reward function passed through to :func:`model_step`.

    Returns:
        A set of 6-int state keys for every reachable acting (non-terminal) state. Every
        key satisfies ``1 <= week <= horizon``; states with ``week == horizon`` are the
        backward-induction base case (their step always yields ``done``).
    """
    start_states = [
        StudentSupportEnvironment(horizon=horizon).reset(seed=None, scenario_id=i).as_tuple()
        for i in range(len(SCENARIOS))
    ]
    seen: set[StateKey] = set()
    frontier: deque[StateKey] = deque(start_states)
    while frontier:
        key = frontier.popleft()
        if key in seen:
            continue
        seen.add(key)
        state = StudentState(*key)
        for action in ACTION_LABELS:
            transition = model_step(state, action, horizon=horizon, reward_fn=reward_fn)
            # enqueue successors only while non-terminal: the done state is never acted on
            if not transition.done:
                frontier.append(transition.state.as_tuple())
    return seen


def optimal_action_values(
    *,
    horizon: int = 6,
    gamma: float = 0.9,
    reward_fn: RewardFunction = default_reward,
) -> dict[StateKey, list[float]]:
    """Solve exact Q* for the finite-horizon deterministic MDP by backward induction.

    What + why: this is the ground-truth optimum a learner compares model-free Q-learning
    against. Because acting states reached at step ``t`` always have ``week == t`` and the
    done state is never acted on, we process states by *descending* ``week``: week-H acting
    states first (every action there yields ``done``, so the future term is 0), then H-1,
    ..., 1. By the time we evaluate a state, ``Q*(s', .)`` for its (later-week) successors
    is already computed, so one ordered sweep gives the exact fixed point with no iteration.

    RL concept: value iteration on a finite-horizon MDP -- the planning rung below
    model-free Q-learning on the ladder (docs/value-based-learning.md, docs/math-notes.md).

    Args:
        horizon: Episode length H of the MDP.
        gamma: Discount factor gamma in [0, 1] applied to the successor's optimal value.
        reward_fn: Reward function R(prev, action, next, done); defaults to the env default.

    Returns:
        A dict mapping each reachable acting state key to its list of optimal action values
        ``[Q*(s, 0), ..., Q*(s, 3)]``.

    Math:
        Q*(s, a) = R_{t+1} + (0 if done else gamma * max_a' Q*(s', a')),  with TD error
        delta = target - Q at convergence equal to 0 for every (s, a).
    """
    acting_states = reachable_acting_states(horizon=horizon, reward_fn=reward_fn)
    states_by_week: dict[int, list[StateKey]] = {}
    for key in acting_states:
        states_by_week.setdefault(key[0], []).append(key)

    q_star: dict[StateKey, list[float]] = {}
    # backward induction: descending week -> every successor's Q* is known before use
    for week in sorted(states_by_week, reverse=True):
        for key in states_by_week[week]:
            state = StudentState(*key)
            action_values: list[float] = []
            for action in ACTION_LABELS:
                transition = model_step(state, action, horizon=horizon, reward_fn=reward_fn)
                successor_value = (
                    0.0  # terminal: no future return beyond the done step
                    if transition.done
                    else gamma * max(q_star[transition.state.as_tuple()])  # bootstrap on Q*(s')
                )
                action_values.append(transition.reward + successor_value)
            q_star[key] = action_values
    return q_star


def optimal_action_value_rows(q_star: dict[StateKey, list[float]]) -> list[dict[str, int | float]]:
    """Flatten Q* into one row per (state, action) for the optimal-value CSV artifact.

    What + why: the artifact lets learners read the exact optimum per decision and diff it
    against the learned Q-table, making "what should the agent have done" inspectable.

    RL concept: a tabular Q* dump (docs/value-based-learning.md); compare with the learned
    q_table CSV produced by q_learning.q_table_rows.

    Args:
        q_star: Optimal action-value table from :func:`optimal_action_values`.

    Returns:
        Rows sorted by state then action, each with columns:
        ``week, engagement, completion, pressure, risk, prior_interventions, action,
        optimal_q_value``. Intended relpath: ``artifacts/dp/optimal_action_values.csv``.
    """
    rows: list[dict[str, int | float]] = []
    for state_key in sorted(q_star):
        week, engagement, completion, pressure, risk, prior_interventions = state_key
        for action, value in enumerate(q_star[state_key]):
            rows.append(
                {
                    "week": week,
                    "engagement": engagement,
                    "completion": completion,
                    "pressure": pressure,
                    "risk": risk,
                    "prior_interventions": prior_interventions,
                    "action": action,
                    "optimal_q_value": round(value, 6),
                }
            )
    return rows


def _shared_abs_gaps(
    learned_q: dict[StateKey, list[float]],
    q_star: dict[StateKey, list[float]],
) -> list[tuple[StateKey, int, float]]:
    """Return per-(state, action) absolute gaps over states present in BOTH tables.

    What + why: Q-learning only populates states it has visited, so a fair comparison is
    restricted to the intersection -- otherwise unvisited tail states would dominate.

    Args:
        learned_q: A learned Q-table (e.g. from train_q_learning).
        q_star: The exact optimal action-value table.

    Returns:
        A list of (state_key, action, abs_gap) over the shared states, in sorted state order.
    """
    gaps: list[tuple[StateKey, int, float]] = []
    for state_key in sorted(q_star):
        if state_key not in learned_q:
            continue
        learned_values = learned_q[state_key]
        optimal_values = q_star[state_key]
        for action, optimal_value in enumerate(optimal_values):
            # |Q_learned - Q*| is the per-entry approximation error model-free control leaves
            gaps.append((state_key, action, abs(learned_values[action] - optimal_value)))
    return gaps


def q_learning_gap(
    learned_q: dict[StateKey, list[float]],
    q_star: dict[StateKey, list[float]],
) -> dict[str, int | float]:
    """Summarize how far a learned Q-table sits from the exact optimum Q*.

    What + why: this is the convergence yardstick -- as Q-learning trains, these gaps shrink
    toward 0, making "model-free approaches the model-based optimum" a measurable claim.
    Computed only over states present in both tables for an apples-to-apples comparison.

    RL concept: the approximation error of model-free value-based control versus the
    dynamic-programming optimum (docs/value-based-learning.md, docs/math-notes.md).

    Args:
        learned_q: A learned Q-table mapping state key -> action values.
        q_star: The exact optimal action-value table from :func:`optimal_action_values`.

    Returns:
        A dict with ``max_abs_gap``, ``mean_abs_gap``, and ``num_states`` (number of shared
        states). With no shared states, both gaps are ``0.0`` and ``num_states`` is ``0``.

    Math:
        per-entry gap = |Q_learned(s, a) - Q*(s, a)|; max and mean are taken over all shared
        (s, a) pairs.
    """
    gaps = _shared_abs_gaps(learned_q, q_star)
    shared_states = {state_key for state_key, _, _ in gaps}
    if not gaps:
        return {"max_abs_gap": 0.0, "mean_abs_gap": 0.0, "num_states": 0}
    abs_values = [gap for _, _, gap in gaps]
    return {
        "max_abs_gap": round(max(abs_values), 6),
        "mean_abs_gap": round(sum(abs_values) / len(abs_values), 6),
        "num_states": len(shared_states),
    }


def gap_rows(
    learned_q: dict[StateKey, list[float]],
    q_star: dict[StateKey, list[float]],
) -> list[dict[str, int | float]]:
    """Emit per-(state, action) gap rows comparing a learned Q-table against Q*.

    What + why: drives the convergence artifact -- learners can sort by ``abs_gap`` to find
    exactly which decisions Q-learning still gets wrong relative to the planned optimum.

    RL concept: a state-resolved view of value-based approximation error
    (docs/value-based-learning.md).

    Args:
        learned_q: A learned Q-table mapping state key -> action values.
        q_star: The exact optimal action-value table from :func:`optimal_action_values`.

    Returns:
        Rows over states present in both tables, each with columns:
        ``week, engagement, completion, pressure, risk, prior_interventions, action,
        learned_q_value, optimal_q_value, abs_gap``. Intended relpath:
        ``artifacts/dp/q_learning_gap.csv``.
    """
    rows: list[dict[str, int | float]] = []
    for state_key, action, abs_gap in _shared_abs_gaps(learned_q, q_star):
        week, engagement, completion, pressure, risk, prior_interventions = state_key
        rows.append(
            {
                "week": week,
                "engagement": engagement,
                "completion": completion,
                "pressure": pressure,
                "risk": risk,
                "prior_interventions": prior_interventions,
                "action": action,
                "learned_q_value": round(learned_q[state_key][action], 6),
                "optimal_q_value": round(q_star[state_key][action], 6),
                "abs_gap": round(abs_gap, 6),
            }
        )
    return rows
