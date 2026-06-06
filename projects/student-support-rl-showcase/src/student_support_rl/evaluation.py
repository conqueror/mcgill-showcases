"""Score and trace policies by re-simulating them on fixed, known scenarios.

What + why: once a policy exists (heuristic, tabular Q/SARSA, REINFORCE, or a deep model),
we need to judge it. This module runs each candidate policy across a fixed bank of scenarios
and reports a *vector* of metrics, not just average return -- because reward alone hides
reward hacking and unsafe behaviour. Where this sits on the ladder
(contextual bandit -> MDP -> Q-learning -> DQN -> policy gradient -> actor-critic -> PPO):
this is the *evaluation* stage that follows any rung, the discipline that lets us trust a
learned controller before deployment.

HONESTY -- this is NOT true off-policy evaluation (OPE). True OPE estimates a target
policy's value from a *fixed log of trajectories collected by some other behaviour policy*
(importance sampling, doubly-robust, fitted-Q evaluation, etc.), without ever running the
target. Here we instead have white-box access to ``StudentSupportEnvironment`` and simply
*re-simulate* each policy inside that known model. That is honest, reproducible
simulator-based evaluation, but it inherits the simulator's biases: a policy that looks good
here is only guaranteed good *in this model*, not on real students. Treat the numbers as a
comparison harness, not a deployment certificate.

Why metrics beyond reward: a policy can rack up high ``total_reward`` while behaving badly --
over-intervening on already-supported students, escalating trivial cases to an advisor, or
ignoring a high-risk student. We therefore also surface ``final_risk``,
``intervention_cost``, ``over_intervention_count``, ``escalation_count``,
``unsafe_or_questionable_decisions``, and ``solved_rate`` so reward hacking and unsafe
shortcuts are visible side-by-side with the scalar objective.

RL concept:
    Policy evaluation and governance via simulator rollouts; see
    docs/evaluation-and-governance.md (and docs/reward-design-and-hacking.md for why a single
    reward number is insufficient).

Math:
    Each rollout estimates the (undiscounted, finite-horizon) return G_t = sum_k R_{t+k+1}
    of a fixed policy in a known MDP; averaging over scenarios and seeds approximates that
    policy's value under the scenario distribution.
"""

from __future__ import annotations

from collections.abc import Sequence

from student_support_rl.environment import (
    ACTION_LABELS,
    RewardFunction,
    StudentSupportEnvironment,
    default_reward,
)
from student_support_rl.policies import Policy


def evaluate_policies(
    *,
    policies: Sequence[Policy],
    scenario_ids: Sequence[int],
    episodes_per_scenario: int = 1,
    base_seed: int = 0,
    horizon: int = 6,
    reward_fn: RewardFunction = default_reward,
) -> tuple[list[dict[str, int | float | str]], list[dict[str, int | float | str]]]:
    """Re-simulate every policy across every scenario and return per-episode + summary tables.

    What + why: this is the comparison harness. For each policy we roll out a fresh
    ``StudentSupportEnvironment`` on each requested scenario (optionally several seeded
    episodes per scenario), accumulate a vector of behavioural metrics, then aggregate per
    policy. It is simulator-based evaluation, NOT off-policy evaluation from logged data: we
    re-run each policy inside the known environment rather than estimating its value from
    trajectories some other policy collected (see the module docstring). Seeds are derived
    deterministically as ``base_seed + episode_index`` so runs are reproducible. RL concept:
    policy evaluation -- estimating a fixed policy's return and side-effects before trusting
    it (the stage after any rung of the ladder).

    Args:
        policies: Candidate policies to score; each must satisfy the ``Policy`` protocol
            (``name``, ``reset``, ``select_action``). Evaluated independently.
        scenario_ids: Scenario indices into ``SCENARIOS`` to run each policy on.
        episodes_per_scenario: Number of seeded rollouts per (policy, scenario) pair; >1
            averages over the environment's reset noise.
        base_seed: Base RNG seed; episode ``i`` uses ``base_seed + i`` for the env reset.
        horizon: Episode length (weeks) passed to the environment.
        reward_fn: Reward function injected into the environment; defaults to
            ``default_reward``. Swapping it lets reward-design studies reuse this harness.

    Returns:
        A pair ``(summary_rows, scenario_rows)``. ``scenario_rows`` holds one dict per
        episode (raw per-rollout metrics + the action trace); ``summary_rows`` holds one dict
        per policy, averaged across its episodes and sorted by ``avg_reward`` descending.

    RL concept:
        Policy evaluation and governance; see docs/evaluation-and-governance.md and
        docs/reward-design-and-hacking.md.

    Math:
        Per rollout we sum the finite-horizon return G_t = sum_k R_{t+k+1}; the summary
        averages it (and each side-metric) over episodes to approximate the policy's value.
    """
    scenario_rows: list[dict[str, int | float | str]] = []

    for policy in policies:
        for scenario_id in scenario_ids:
            for episode_index in range(episodes_per_scenario):
                policy.reset()
                environment = StudentSupportEnvironment(horizon=horizon, reward_fn=reward_fn)
                state = environment.reset(
                    seed=base_seed + episode_index,
                    scenario_id=scenario_id,
                )
                total_reward = 0.0
                intervention_cost = 0.0
                over_intervention_count = 0
                escalation_count = 0
                unsafe_or_questionable_decisions = 0
                action_trace: list[str] = []

                while not environment.is_done():
                    action = policy.select_action(state)  # on-policy rollout: pi acts in known MDP
                    transition = environment.step(action)
                    # Monte Carlo return: accumulate R_{t+1} into G_t = sum_k R_{t+k+1}.
                    total_reward += transition.reward
                    # Cost-of-effort metric: sum the action's resource cost (from ACTION_COSTS).
                    intervention_cost += float(transition.info["action_cost"])
                    # Governance metric: count escalations to an advisor meeting (action == 3).
                    escalation_count += int(action == 3)
                    # Reward-hacking probe: intervene again on an already heavily-supported student.
                    over_intervention_count += int(action != 0 and state.prior_interventions >= 2)
                    # Safety probe: do-nothing at max risk, or escalate-to-advisor at minimal risk.
                    unsafe_or_questionable_decisions += int(
                        (action == 0 and state.risk >= 3) or (action == 3 and state.risk <= 1)
                    )
                    action_trace.append(ACTION_LABELS[action])
                    state = transition.state  # advance to s'; loop until terminal

                scenario_rows.append(
                    {
                        "policy": policy.name,
                        "scenario_id": scenario_id,
                        "scenario_name": environment.scenario_name,
                        "episode_index": episode_index,
                        "total_reward": round(total_reward, 4),
                        "final_risk": state.risk,
                        "intervention_cost": round(intervention_cost, 4),
                        "interventions": state.prior_interventions,
                        "over_intervention_count": over_intervention_count,
                        "escalation_count": escalation_count,
                        "unsafe_or_questionable_decisions": unsafe_or_questionable_decisions,
                        # Outcome metric: episode "solved" iff the FINAL risk is low (<= 1).
                        "solved": int(state.risk <= 1),
                        "actions": " | ".join(action_trace),
                    }
                )

    summary_rows = _summarize_scenarios(scenario_rows)
    return summary_rows, scenario_rows


def simulate_episode(
    *,
    policy: Policy,
    scenario_id: int,
    seed: int | None = None,
    horizon: int = 6,
    reward_fn: RewardFunction = default_reward,
) -> list[dict[str, int | float | str]]:
    """Trace one episode step-by-step, returning a row per (s, a, R_{t+1}, s') transition.

    What + why: ``evaluate_policies`` answers "which policy scores better?"; this answers
    "*why* -- what did it actually do, week by week?". It runs a single rollout and logs the
    full state, the chosen action, its cost, the reward R_{t+1}, and the next risk, so a
    reader can audit the decision sequence (e.g. catch a policy that escalates needlessly or
    stalls on a high-risk student). This is qualitative inspection, the companion to the
    aggregate metrics, and -- like the rest of this module -- it re-simulates the known
    environment rather than reading a fixed log (not OPE). RL concept: a trajectory
    tau = (s_1, a_1, R_2, s_2, ...) through the MDP under a fixed policy.

    Args:
        policy: The policy to trace; must satisfy the ``Policy`` protocol.
        scenario_id: Scenario index into ``SCENARIOS`` to start from.
        seed: Optional env-reset seed; ``None`` gives the scenario's noise-free start state.
        horizon: Episode length (weeks) passed to the environment.
        reward_fn: Reward function injected into the environment; defaults to
            ``default_reward``.

    Returns:
        One dict per step with the pre-step state fields, the action label, its cost, the
        step reward, and the resulting ``next_risk`` -- ordered chronologically.

    RL concept:
        Single-trajectory rollout / episode trace; see docs/evaluation-and-governance.md and
        docs/mdp-and-environment.md.
    """
    policy.reset()
    environment = StudentSupportEnvironment(horizon=horizon, reward_fn=reward_fn)
    state = environment.reset(seed=seed, scenario_id=scenario_id)
    rows: list[dict[str, int | float | str]] = []

    while not environment.is_done():
        action = policy.select_action(state)  # fixed policy pi(.|s); no learning here
        transition = environment.step(action)
        rows.append(
            {
                "scenario_name": environment.scenario_name,
                "week": state.week,
                "engagement": state.engagement,
                "completion": state.completion,
                "pressure": state.pressure,
                "risk": state.risk,
                "prior_interventions": state.prior_interventions,
                "action": ACTION_LABELS[action],
                "action_cost": transition.info["action_cost"],
                "reward": transition.reward,
                "next_risk": transition.state.risk,
            }
        )
        state = transition.state

    return rows


def _summarize_scenarios(
    scenario_rows: Sequence[dict[str, int | float | str]],
) -> list[dict[str, int | float | str]]:
    """Aggregate per-episode rows into one mean-per-metric summary row per policy.

    What + why: this is the aggregation step of policy evaluation. It groups the raw
    per-episode rows by policy, averages each metric across that policy's episodes, and
    returns the summaries ranked by ``avg_reward``. Averaging over scenarios and seeds is how
    a few noisy rollouts approximate a policy's value and typical behaviour; reporting the
    side-metrics next to ``avg_reward`` keeps reward hacking and unsafe shortcuts legible
    rather than hidden behind one scalar. RL concept: Monte Carlo policy evaluation -- the
    empirical mean return (and side-effects) estimates the policy's expected value.

    Args:
        scenario_rows: Per-episode rows emitted by ``evaluate_policies`` (each carries a
            ``policy`` name plus the numeric metrics to be averaged).

    Returns:
        One summary dict per policy with ``avg_*`` metrics and ``solved_rate``, sorted by
        ``avg_reward`` descending (best policy first).

    RL concept:
        Monte Carlo estimate of policy value; see docs/evaluation-and-governance.md.

    Math:
        For metric m over a policy's N episodes, the report is the empirical mean
        (1/N) * sum_i m_i -- the unbiased estimate of E[m] under the scenario/seed mix.
    """
    by_policy: dict[str, list[dict[str, int | float | str]]] = {}
    for row in scenario_rows:
        by_policy.setdefault(str(row["policy"]), []).append(row)

    summary_rows: list[dict[str, int | float | str]] = []
    for policy_name, rows in sorted(by_policy.items()):
        count = len(rows)
        avg_reward = sum(float(row["total_reward"]) for row in rows) / count
        avg_final_risk = sum(int(row["final_risk"]) for row in rows) / count
        avg_intervention_cost = sum(float(row["intervention_cost"]) for row in rows) / count
        avg_interventions = sum(int(row["interventions"]) for row in rows) / count
        avg_over_intervention_count = (
            sum(int(row["over_intervention_count"]) for row in rows) / count
        )
        avg_escalation_count = sum(int(row["escalation_count"]) for row in rows) / count
        avg_unsafe_decisions = (
            sum(int(row["unsafe_or_questionable_decisions"]) for row in rows) / count
        )
        solved_rate = sum(int(row["solved"]) for row in rows) / count
        summary_rows.append(
            {
                "policy": policy_name,
                "avg_reward": round(avg_reward, 4),
                "avg_final_risk": round(avg_final_risk, 4),
                "avg_intervention_cost": round(avg_intervention_cost, 4),
                "avg_interventions": round(avg_interventions, 4),
                "avg_over_intervention_count": round(avg_over_intervention_count, 4),
                "avg_escalation_count": round(avg_escalation_count, 4),
                "avg_unsafe_or_questionable_decisions": round(avg_unsafe_decisions, 4),
                "solved_rate": round(solved_rate, 4),
            }
        )
    # Rank policies by estimated value: highest mean return first (the leaderboard order).
    return sorted(summary_rows, key=lambda row: float(row["avg_reward"]), reverse=True)
