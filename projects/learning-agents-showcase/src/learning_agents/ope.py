"""Off-policy evaluation (OPE): estimate a target policy's value from a *fixed log* only.

What + why: :mod:`learning_agents.evaluation` is honest that it is NOT
off-policy evaluation. It re-simulates each policy inside the known
environment. That is fine for a teaching simulator, but the real governance
question is harder: *before* you deploy a new policy, can you estimate how
good it is using only the logs your current (behaviour) policy already
produced, without ever running the new policy? That is OPE, and it is what
lets you vet a candidate offline. This module implements the four canonical
episodic OPE estimators and, because we DO have a simulator, checks each
against the true value so a learner can see which estimators are accurate
and why.

The estimators, in increasing sophistication:

* **Importance sampling (IS)** reweights each logged trajectory's return by the ratio of how likely
  the target vs behaviour policy were to take its actions. Unbiased, but high variance: for a
  deterministic target only trajectories the behaviour happened to follow survive (the rest get
  weight 0).
* **Weighted IS (WIS)** self-normalises those weights. Biased but consistent, with far lower
  variance -- usually the better IS-family choice.
* **Fitted-Q Evaluation (FQE) / direct method** learns Q^pi for the *target* policy from the log
  (like FQI but bootstrapping from the target's action, not the max), then reads the value off the
  start states. Low variance, but biased by model error and coverage.
* **Doubly-robust (DR)** combines the direct method with an IS correction so it is unbiased if
  *either* the weights or the Q-model is right -- the best of both, with variance between IS and the
  direct method.

A note on discounting: these estimators use ``gamma = 1.0`` (undiscounted
finite-horizon return) by default, so the OPE numbers are directly
comparable to the simulator ground truth from
:func:`~learning_agents.evaluation.evaluate_policies`, which reports the
undiscounted episodic return.

RL concept:
    Off-policy evaluation -- estimating E[return | target policy] from a behaviour-policy log via
    importance sampling, a direct (model) estimate, and their doubly-robust combination.

Math:
    Per-step ratio (deterministic target ``pi_e``):  rho_t = 1[a_t = pi_e(s_t)] / pi_beta(a_t | s_t)
    Trajectory weight: w = prod_t rho_t ; return G = sum_t gamma^t r_t.
    IS:  mean_n (w_n G_n).   WIS:  sum_n (w_n G_n) / sum_n w_n.
    Direct: mean over start states of Q^pi(s_0, pi_e(s_0)).
    DR:  mean_n [ V^pi(s_0) + sum_t gamma^t (prod_{i<=t} rho_i)
                 (r_t + gamma * V^pi(s_{t+1}) - Q^pi(s_t, a_t)) ],  with V^pi(s) = Q^pi(s, pi_e(s)).
"""

from __future__ import annotations

from collections.abc import Sequence

from learning_agents.environment import ACTION_LABELS, AgentState, RewardFunction, default_reward
from learning_agents.evaluation import evaluate_policies
from learning_agents.offline_rl import LoggedDataset, LoggedTransition, StateKey
from learning_agents.policies import Policy

__all__ = [
    "direct_method_value",
    "doubly_robust_value",
    "fitted_q_evaluation",
    "importance_sampling_value",
    "ope_estimates",
    "ope_report_rows",
    "target_action_probability",
    "true_policy_value",
]


def target_action_probability(policy: Policy, state: AgentState, action: int) -> float:
    """Return ``pi_e(action | state)`` for a deterministic target policy: 1.0 or 0.0.

    What + why: importance sampling needs the *target* policy's probability of each logged action.
    The showcase's evaluation targets (heuristic router, a learned Q-table policy, the DP optimum)
    are all deterministic, so the distribution is a spike: probability 1 on the action the policy
    would choose, 0 on every other. Centralising this keeps every estimator consistent about what
    ``pi_e(a | s)`` means.

    Args:
        policy: The deterministic target policy ``pi_e``.
        state: The state ``s`` (an :class:`~learning_agents.environment.AgentState`).
        action: The logged action ``a`` to score.

    Returns:
        ``1.0`` if the policy would take ``action`` in ``state``, else ``0.0``.

    RL concept: the target-policy propensity ``pi_e(a | s)`` -- a point mass for a deterministic
    policy, the numerator of every importance ratio.
    """
    return 1.0 if policy.select_action(state) == action else 0.0


def _trajectories(dataset: LoggedDataset) -> list[list[LoggedTransition]]:
    """Split the flat logged transitions back into per-episode trajectories on ``done`` boundaries.

    What + why: the log is stored as a flat sequence of transitions in
    collection order, but the importance-sampling and doubly-robust
    estimators are *per trajectory* because they multiply ratios along an
    episode. Because each episode ends with exactly one ``done=True``
    transition, splitting on those boundaries reconstructs the episodes
    exactly.

    Args:
        dataset: The logged dataset.

    Returns:
        A list of trajectories, each a list of :class:`~learning_agents.offline_rl.LoggedTransition`
        in time order.
    """
    trajectories: list[list[LoggedTransition]] = []
    current: list[LoggedTransition] = []
    for transition in dataset.transitions:
        current.append(transition)
        if transition.done:
            trajectories.append(current)
            current = []
    if current:  # defensive: a trailing unterminated episode (should not occur in practice)
        trajectories.append(current)
    return trajectories


def importance_sampling_value(
    dataset: LoggedDataset,
    target_policy: Policy,
    *,
    gamma: float = 1.0,
    weighted: bool = False,
) -> float:
    """Estimate the target policy's value from the log by (weighted) importance sampling.

    What + why: each logged trajectory's discounted return is reweighted by
    the trajectory importance ratio
    ``w = prod_t pi_e(a_t|s_t)/pi_beta(a_t|s_t)``. For a deterministic target
    this ratio is 0 unless the behaviour policy followed the target at
    *every* step, so only "on-target" trajectories contribute. That is the
    source of IS's high variance. Ordinary IS averages ``w * G``
    (unbiased); weighted IS divides by the sum of weights (biased but
    lower-variance and consistent).

    Args:
        dataset: The behaviour-policy log.
        target_policy: The deterministic target policy ``pi_e`` to evaluate.
        gamma: Discount for the return (default 1.0 = undiscounted finite-horizon return, matching
            the simulator ground truth).
        weighted: If True, self-normalise by the sum of trajectory weights (WIS); else ordinary IS.

    Returns:
        The estimated value. Ordinary IS divides the weighted-return sum by the trajectory count;
        WIS divides by the total weight. Returns ``0.0`` if no trajectory has positive weight (the
        behaviour log never followed the target), which is itself a useful coverage signal.

    RL concept: importance-sampling off-policy evaluation -- unbiased (IS) vs lower-variance,
    self-normalised (WIS).
    """
    weighted_return_sum = 0.0
    weight_sum = 0.0
    trajectories = _trajectories(dataset)
    for trajectory in trajectories:
        weight = 1.0
        discounted_return = 0.0
        for step, transition in enumerate(trajectory):
            ratio = target_action_probability(
                target_policy, transition.state, transition.action
            ) / transition.behavior_action_prob
            weight *= ratio
            discounted_return += (gamma**step) * transition.reward
            if weight == 0.0:
                # Once the target diverges from the logged action, the whole trajectory weight is 0;
                # finish accumulating the (now irrelevant) return for clarity, but it contributes 0.
                discounted_return = discounted_return  # no-op; weight stays 0
        weighted_return_sum += weight * discounted_return
        weight_sum += weight

    if weighted:
        return round(weighted_return_sum / weight_sum, 4) if weight_sum > 0 else 0.0
    return round(weighted_return_sum / len(trajectories), 4) if trajectories else 0.0


def fitted_q_evaluation(
    dataset: LoggedDataset,
    target_policy: Policy,
    *,
    gamma: float = 1.0,
    sweeps: int = 200,
    tolerance: float = 1e-6,
) -> dict[StateKey, list[float]]:
    """Learn Q^pi for a fixed target policy from the log (the direct-method model).

    What + why: this is the model behind the direct and doubly-robust estimators. It is Fitted-Q
    *Evaluation*: like :func:`~learning_agents.offline_rl.fitted_q_iteration` but the bootstrap uses
    the *target policy's* next action instead of the greedy max, so it estimates the value of
    following ``pi_e`` (policy evaluation), not the optimal value (control). It sweeps the fixed log
    to a fixed point, touching no environment.

    Args:
        dataset: The behaviour-policy log to fit on.
        target_policy: The deterministic policy ``pi_e`` whose value is being modelled.
        gamma: Discount factor (default 1.0).
        sweeps: Maximum batch sweeps over the dataset.
        tolerance: Early-stop once a sweep's maximum table change falls below this.

    Returns:
        ``Q^pi`` as a table keyed by the 7-int state key; states seen only as successors keep their
        all-zeros row (the coverage gap, identical to FQI's).

    RL concept: Fitted-Q Evaluation -- the model-based (direct) component of OPE; policy evaluation,
    not control.

    Math:
        Q^pi_{k+1}(s, a) = mean_D[ R + (0 if done else gamma * Q^pi_k(s', pi_e(s'))) ].
    """
    num_actions = len(ACTION_LABELS)
    q_table: dict[StateKey, list[float]] = {}
    for transition in dataset.transitions:
        q_table.setdefault(transition.state.as_tuple(), [0.0] * num_actions)
        q_table.setdefault(transition.next_state.as_tuple(), [0.0] * num_actions)

    grouped: dict[tuple[StateKey, int], list[LoggedTransition]] = {}
    for transition in dataset.transitions:
        grouped.setdefault((transition.state.as_tuple(), transition.action), []).append(transition)

    for _sweep in range(sweeps):
        updates: dict[tuple[StateKey, int], float] = {}
        for (state_key, action), rows in grouped.items():
            target_sum = 0.0
            for row in rows:
                if row.done:
                    future = 0.0
                else:
                    # Bootstrap from the TARGET policy's action at s' (policy evaluation, not max).
                    next_action = target_policy.select_action(row.next_state)
                    future = gamma * q_table[row.next_state.as_tuple()][next_action]
                target_sum += row.reward + future
            updates[(state_key, action)] = target_sum / len(rows)

        residual = 0.0
        for (state_key, action), new_value in updates.items():
            residual = max(residual, abs(new_value - q_table[state_key][action]))
            q_table[state_key][action] = new_value
        if residual < tolerance:
            break

    return q_table


def direct_method_value(
    dataset: LoggedDataset,
    target_policy: Policy,
    q_pi: dict[StateKey, list[float]],
) -> float:
    """Estimate the target value as the mean modelled value over the log's start states.

    What + why: given the fitted ``Q^pi``, the direct method reads the value
    straight off the initial states:
    ``V^pi(s_0) = Q^pi(s_0, pi_e(s_0))`` averaged over the start state of
    every logged trajectory, which is the empirical initial-state
    distribution. It has low variance, but is only as good as the fitted
    model and the log's coverage.

    Args:
        dataset: The log (used for its trajectory start states).
        target_policy: The deterministic target policy ``pi_e``.
        q_pi: The fitted ``Q^pi`` table from :func:`fitted_q_evaluation`.

    Returns:
        The mean ``Q^pi(s_0, pi_e(s_0))`` over trajectory start states; ``0.0`` if the log is empty.

    RL concept: the direct (model) OPE estimator, where value is read from a
    fitted ``Q^pi`` at the start states.
    """
    num_actions = len(ACTION_LABELS)
    trajectories = _trajectories(dataset)
    if not trajectories:
        return 0.0
    total = 0.0
    for trajectory in trajectories:
        start = trajectory[0].state
        action = target_policy.select_action(start)
        total += q_pi.get(start.as_tuple(), [0.0] * num_actions)[action]
    return round(total / len(trajectories), 4)


def doubly_robust_value(
    dataset: LoggedDataset,
    target_policy: Policy,
    q_pi: dict[StateKey, list[float]],
    *,
    gamma: float = 1.0,
) -> float:
    """Estimate the target value with the episodic doubly-robust estimator.

    What + why: DR anchors on the direct method's ``V^pi(s_0)`` and adds an importance-weighted
    correction for the model's per-step Bellman error. It is doubly robust: unbiased if *either* the
    importance weights or the fitted ``Q^pi`` is correct, with variance between IS and the direct
    method. When the importance weight collapses to 0 (the behaviour diverged from the target), the
    correction terms vanish and DR gracefully falls back to the direct-method value -- the property
    that makes it robust on thin coverage.

    Args:
        dataset: The behaviour-policy log.
        target_policy: The deterministic target policy ``pi_e``.
        q_pi: The fitted ``Q^pi`` table from :func:`fitted_q_evaluation`.
        gamma: Discount factor (default 1.0).

    Returns:
        The doubly-robust value estimate, averaged over logged trajectories.

    RL concept: the doubly-robust OPE estimator, a control-variate
    combination of the direct method and importance sampling.

    Math:
        V_DR = mean_n [ V^pi(s_0) + sum_t gamma^t (prod_{i<=t} rho_i)
                        (r_t + gamma * V^pi(s_{t+1}) - Q^pi(s_t, a_t)) ]
        with V^pi(s) = Q^pi(s, pi_e(s)).
    """
    num_actions = len(ACTION_LABELS)

    def value_of(state: AgentState) -> float:
        """V^pi(s) = Q^pi(s, pi_e(s)) for the fitted model (0 for an uncovered state)."""
        action = target_policy.select_action(state)
        return q_pi.get(state.as_tuple(), [0.0] * num_actions)[action]

    trajectories = _trajectories(dataset)
    if not trajectories:
        return 0.0

    total = 0.0
    for trajectory in trajectories:
        estimate = value_of(trajectory[0].state)  # direct-method anchor V^pi(s_0)
        cumulative_weight = 1.0
        for step, transition in enumerate(trajectory):
            ratio = target_action_probability(
                target_policy, transition.state, transition.action
            ) / transition.behavior_action_prob
            cumulative_weight *= ratio
            if cumulative_weight == 0.0:
                break  # all later correction terms are 0; DR falls back to the direct anchor
            q_sa = q_pi.get(transition.state.as_tuple(), [0.0] * num_actions)[transition.action]
            next_value = 0.0 if transition.done else value_of(transition.next_state)
            td_error = transition.reward + gamma * next_value - q_sa
            estimate += (gamma**step) * cumulative_weight * td_error
        total += estimate
    return round(total / len(trajectories), 4)


def true_policy_value(
    target_policy: Policy,
    *,
    scenario_ids: Sequence[int] = (0, 1, 2, 3, 4),
    episodes_per_scenario: int = 40,
    base_seed: int = 0,
    horizon: int = 5,
    reward_fn: RewardFunction = default_reward,
) -> float:
    """Compute the target policy's true value by re-simulating it (the OPE ground truth).

    What + why: because this is a teaching simulator we can *measure* the
    value OPE only estimates by actually rolling the target policy out in
    the environment and averaging its undiscounted return. This is the
    yardstick the IS/WIS/FQE/DR estimates are scored against, so a learner
    can see each estimator's accuracy. In a real deployment this number is
    exactly what you cannot get without running the policy, which is why OPE
    exists.

    Args:
        target_policy: The policy to measure.
        scenario_ids: Scenarios to average over (the start-state distribution).
        episodes_per_scenario: Seeded rollouts per scenario.
        base_seed: Base RNG seed for the env-reset jitter.
        horizon: Episode length H.
        reward_fn: Reward function (defaults to the judge rubric).

    Returns:
        The mean undiscounted episodic return of ``target_policy`` across the scenarios -- the OPE
        ground truth.

    RL concept: the on-policy Monte-Carlo value, used here only as the ground truth to grade OPE.
    """
    summary, _ = evaluate_policies(
        policies=[target_policy],
        scenario_ids=scenario_ids,
        episodes_per_scenario=episodes_per_scenario,
        base_seed=base_seed,
        horizon=horizon,
        reward_fn=reward_fn,
    )
    return round(float(summary[0]["avg_reward"]), 4)


def ope_estimates(
    dataset: LoggedDataset,
    target_policy: Policy,
    *,
    gamma: float = 1.0,
) -> dict[str, float]:
    """Compute all four OPE estimates for one target policy from a single fitted model.

    What + why: fits ``Q^pi`` once and returns the importance-sampling, weighted-IS, direct-method,
    and doubly-robust estimates together, so callers (reports, tests) get the whole comparison from
    one call.

    Args:
        dataset: The behaviour-policy log.
        target_policy: The deterministic target policy to evaluate.
        gamma: Discount factor (default 1.0, undiscounted finite-horizon).

    Returns:
        A mapping with keys ``importance_sampling``, ``weighted_importance_sampling``,
        ``direct_method``, and ``doubly_robust``.

    RL concept: the OPE estimator family side by side -- IS, WIS, direct method, doubly-robust.
    """
    q_pi = fitted_q_evaluation(dataset, target_policy, gamma=gamma)
    return {
        "importance_sampling": importance_sampling_value(
            dataset, target_policy, gamma=gamma, weighted=False
        ),
        "weighted_importance_sampling": importance_sampling_value(
            dataset, target_policy, gamma=gamma, weighted=True
        ),
        "direct_method": direct_method_value(dataset, target_policy, q_pi),
        "doubly_robust": doubly_robust_value(dataset, target_policy, q_pi, gamma=gamma),
    }


def ope_report_rows(
    dataset: LoggedDataset,
    named_targets: Sequence[tuple[str, Policy]],
    *,
    gamma: float = 1.0,
    scenario_ids: Sequence[int] = (0, 1, 2, 3, 4),
    episodes_per_scenario: int = 40,
) -> list[dict[str, int | float | str]]:
    """Build per-target OPE rows: each estimator, the true value, and the absolute error.

    What + why: produces the tabular evidence for the OPE artifact. For each named target policy it
    reports the four estimates from the *log only* alongside the simulator-measured true value and
    each estimator's absolute error, so a reader can see which estimators are accurate on this log.

    Args:
        dataset: The behaviour-policy log every estimate is computed from.
        named_targets: ``(name, policy)`` pairs to evaluate (e.g. the heuristic router, the learned
            Q-table policy, the DP optimum).
        gamma: Discount factor (default 1.0).
        scenario_ids: Scenarios for the ground-truth rollout.
        episodes_per_scenario: Seeded rollouts per scenario for the ground truth.

    Returns:
        One row per (target, estimator) with columns ``target``, ``estimator``, ``estimate``,
        ``true_value``, and ``abs_error``.

    RL concept: an OPE accuracy report -- estimators graded against the known true value.
    """
    rows: list[dict[str, int | float | str]] = []
    for name, policy in named_targets:
        truth = true_policy_value(
            policy,
            scenario_ids=scenario_ids,
            episodes_per_scenario=episodes_per_scenario,
        )
        estimates = ope_estimates(dataset, policy, gamma=gamma)
        for estimator, estimate in estimates.items():
            rows.append(
                {
                    "target": name,
                    "estimator": estimator,
                    "estimate": estimate,
                    "true_value": truth,
                    "abs_error": round(abs(estimate - truth), 4),
                }
            )
    return rows
