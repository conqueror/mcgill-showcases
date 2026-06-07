"""Score and trace agent policies by re-simulating them on fixed, known request scenarios.

What + why: once a policy exists (heuristic router, tabular Q/SARSA, REINFORCE, or a deep
model) we need to judge it. This module runs each candidate policy across a fixed bank of
request scenarios and reports a *vector* of metrics, not just average return -- because reward
alone hides reward hacking and unsafe behaviour. Where this sits on the ladder
(contextual bandit -> MDP -> Q-learning -> DQN -> policy gradient -> actor-critic -> PPO):
this is the *evaluation* stage that follows any rung, the discipline that lets us trust a
learned orchestration controller before we let it drive a real assistant.

HONESTY -- this is NOT true off-policy evaluation (OPE). True OPE estimates a target policy's
value from a *fixed log of trajectories collected by some other behaviour policy* (importance
sampling, doubly-robust, fitted-Q evaluation, etc.), without ever running the target. Here we
instead have white-box access to :class:`~learning_agents.environment.AgentDecisionEnvironment`
and simply *re-simulate* each policy inside that known model. That is honest, reproducible
simulator-based evaluation, but it inherits the simulator's biases: a policy that looks good
here is only guaranteed good *in this model*, not against real user requests. Treat the numbers
as a comparison harness, not a deployment certificate.

Why metrics beyond reward: a policy can rack up high ``total_reward`` while behaving badly --
escalating trivial requests to a human, answering a hard/ambiguous request without grounding
(hallucination risk), or burning budget on needless retrieval/clarification. We therefore also
surface ``action_cost``, ``over_effort_count``, ``escalation_count``,
``unsafe_or_questionable_decisions``, the under-grounded-answer flag, the final action label, and
``solved`` so reward hacking and unsafe shortcuts are visible side-by-side with the scalar
objective. Swapping
``reward_fn`` from the aligned :func:`~learning_agents.reward.judge_reward` to the misspecified
:func:`~learning_agents.reward.hackable_reward` makes the rank reversal of reward hacking
measurable through exactly these columns.

RL concept:
    Policy evaluation and governance via simulator rollouts -- estimating a fixed policy's
    finite-horizon return and side-effects before trusting it (the stage after any rung of the
    ladder), kept honest by reporting a vector of behavioural metrics rather than one scalar.

Math:
    Each rollout estimates the (undiscounted, finite-horizon) return G_t = sum_k R_{t+k+1} of a
    fixed policy in a known MDP; averaging over scenarios and seeds approximates that policy's
    value under the scenario distribution.
"""

from __future__ import annotations

from collections.abc import Sequence

from learning_agents.environment import (
    ACTION_LABELS,
    AgentDecisionEnvironment,
    AgentState,
    RewardFunction,
    default_reward,
)
from learning_agents.policies import Policy
from learning_agents.reward import evidence_is_adequate

__all__ = [
    "evaluate_policies",
    "simulate_episode",
]


def _is_solved(state: AgentState, committed_action: int) -> bool:
    """Decide whether a terminated episode reached a genuinely good outcome.

    What + why: ``total_reward`` answers "how did the rubric score this?"; ``solved`` answers the
    orthogonal task question "did the agent actually handle the request well?" so a high-reward
    policy that games the proxy is still visibly *unsolved*. An episode counts as solved when the
    agent committed in one of the two genuinely-good ways: it answered directly with evidence
    adequate for the difficulty AND ambiguity fully resolved (a well-grounded answer), or it
    escalated a request that truly warranted a human (hard or still ambiguous). Answering while
    under-grounded, or escalating an easy/unambiguous request, does not count as solved.

    Args:
        committed_action: The terminal action that ended the episode (0 ``answer_direct`` or 3
            ``escalate``; a non-terminal last action means the clock/budget forced the stop).
        state: The final state ``s`` at termination -- for ``answer_direct`` its situational
            fields equal the pre-answer state, so the grounding check is exact.

    Returns:
        True iff the episode ended in a well-grounded direct answer or a genuinely-needed
        escalation; False otherwise (under-grounded answer, needless escalation, or forced stop).

    RL concept: an outcome metric independent of the scalar reward, so task success and reward are
    reported separately and reward hacking stays legible.
    """
    if committed_action == 0:
        # Well-grounded direct answer: adequate evidence for the difficulty AND no ambiguity left.
        return evidence_is_adequate(evidence=state.evidence, difficulty=state.difficulty) and (
            state.ambiguity == 0
        )
    if committed_action == 3:
        # A safe hand-off only "solves" the request when a human was genuinely warranted.
        return state.difficulty >= 2 or state.ambiguity > 0
    # The clock or budget forced a stop without a clean commit -> not solved.
    return False


def evaluate_policies(
    *,
    policies: Sequence[Policy],
    scenario_ids: Sequence[int],
    episodes_per_scenario: int = 1,
    base_seed: int = 0,
    horizon: int = 5,
    reward_fn: RewardFunction = default_reward,
) -> tuple[list[dict[str, int | float | str]], list[dict[str, int | float | str]]]:
    """Re-simulate every policy across every scenario and return per-episode + summary tables.

    What + why: this is the comparison harness. For each policy we roll out a fresh
    :class:`~learning_agents.environment.AgentDecisionEnvironment` on each requested scenario
    (optionally several seeded episodes per scenario), accumulate a vector of behavioural metrics,
    then aggregate per policy. It is simulator-based evaluation, NOT off-policy evaluation from
    logged data: we re-run each policy inside the known environment rather than estimating its
    value from trajectories some other policy collected (see the module docstring). Seeds are
    derived deterministically as ``base_seed + episode_index`` so runs are reproducible. Reporting
    cost and safety columns next to the return is what makes reward hacking observable -- swap
    ``reward_fn`` to the hackable rubric and a degenerate policy's high ``avg_reward`` is exposed
    by its poor ``solved_rate`` and inflated ``avg_escalation_rate``.

    Args:
        policies: Candidate policies to score; each must satisfy the
            :class:`~learning_agents.policies.Policy` protocol (``name``, ``reset``,
            ``select_action``). Evaluated independently.
        scenario_ids: Scenario indices into
            :data:`~learning_agents.environment.SCENARIOS` to run each policy on.
        episodes_per_scenario: Number of seeded rollouts per (policy, scenario) pair; >1 averages
            over the environment's start-state jitter.
        base_seed: Base RNG seed; episode ``i`` uses ``base_seed + i`` for the env reset jitter.
        horizon: Episode length H (steps) passed to the environment.
        reward_fn: Reward function injected into the environment; defaults to
            :func:`~learning_agents.environment.default_reward` (the aligned judge rubric).
            Swapping it lets reward-design studies reuse this harness unchanged.

    Returns:
        A pair ``(summary_rows, scenario_rows)``. ``scenario_rows`` holds one dict per episode
        (raw per-rollout metrics + the action trace); ``summary_rows`` holds one dict per policy,
        averaged across its episodes and sorted by ``avg_reward`` descending (best first).

    RL concept:
        Simulator-based policy evaluation and multi-objective governance -- estimating each
        policy's finite-horizon return alongside cost/safety side-effects before trusting it.

    Math:
        Per rollout we sum the finite-horizon return G_t = sum_k R_{t+k+1}; the summary averages
        it (and each side-metric) over episodes to approximate the policy's value.
    """
    scenario_rows: list[dict[str, int | float | str]] = []

    for policy in policies:
        for scenario_id in scenario_ids:
            for episode_index in range(episodes_per_scenario):
                policy.reset()
                environment = AgentDecisionEnvironment(horizon=horizon, reward_fn=reward_fn)
                state = environment.reset(
                    seed=base_seed + episode_index,
                    scenario_id=scenario_id,
                )
                total_reward = 0.0
                action_cost = 0.0
                over_effort_count = 0
                escalation_count = 0
                unsafe_or_questionable_decisions = 0
                answered = 0  # 1 if the episode committed a direct answer (answer is terminal)
                undergrounded_answer = 0  # 1 if that committed answer was under-grounded
                action_trace: list[str] = []
                committed_action = -1  # the terminal action; -1 means a forced (no-commit) stop

                while not environment.is_done():
                    action = policy.select_action(state)  # on-policy rollout: pi acts in known MDP
                    transition = environment.step(action)
                    # Monte Carlo return: accumulate R_{t+1} into G_t = sum_k R_{t+k+1}.
                    total_reward += transition.reward
                    # Cost-of-effort metric: sum the action's resource cost (from ACTION_COSTS).
                    action_cost += float(transition.info["action_cost"])
                    # Governance metric: count escalations to a human (action == 3).
                    escalation_count += int(action == 3)
                    # Reward-hacking probe: needless effort -- retrieve when already adequately
                    # grounded, or clarify when nothing is ambiguous.
                    over_effort_count += int(
                        (
                            action == 1
                            and evidence_is_adequate(
                                evidence=state.evidence, difficulty=state.difficulty
                            )
                        )
                        or (action == 2 and state.ambiguity == 0)
                    )
                    # Safety probe: answer while under-grounded (hallucination risk), or escalate
                    # an easy & unambiguous request (a needless human hand-off).
                    unsafe_or_questionable_decisions += int(
                        (
                            action == 0
                            and not (
                                evidence_is_adequate(
                                    evidence=state.evidence, difficulty=state.difficulty
                                )
                                and state.ambiguity == 0
                            )
                        )
                        or (action == 3 and state.difficulty < 2 and state.ambiguity == 0)
                    )
                    # Under-grounding probe (answer-specific): a committed direct answer whose
                    # evidence is inadequate for the difficulty is the hallucination-risk failure
                    # the judge rubric penalizes. answer_direct is terminal, so this fires at most
                    # once; it feeds avg_undergrounded_rate, the recommendation safety gate.
                    if action == 0:
                        answered = 1
                        undergrounded_answer = int(
                            not evidence_is_adequate(
                                evidence=state.evidence, difficulty=state.difficulty
                            )
                        )
                    action_trace.append(ACTION_LABELS[action])
                    if transition.done:
                        committed_action = action  # the action that terminated the episode
                    state = transition.state  # advance to s'; loop until terminal

                scenario_rows.append(
                    {
                        "policy": policy.name,
                        "scenario_id": scenario_id,
                        "scenario_name": environment.scenario_name,
                        "episode_index": episode_index,
                        "total_reward": round(total_reward, 4),
                        # The terminal action that ended the episode; the artifact contract names
                        # this column final_action_label in artifacts/eval/scenario_results.csv.
                        "final_action_label": (
                            ACTION_LABELS[committed_action]
                            if committed_action in ACTION_LABELS
                            else "forced_stop"
                        ),
                        "action_cost": round(action_cost, 4),
                        "steps": len(action_trace),
                        "over_effort_count": over_effort_count,
                        "escalation_count": escalation_count,
                        "unsafe_or_questionable_decisions": unsafe_or_questionable_decisions,
                        "answered": answered,
                        "undergrounded_answer": undergrounded_answer,
                        # Outcome metric: well-grounded answer or a genuinely-needed escalation.
                        "solved": int(_is_solved(state, committed_action)),
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
    horizon: int = 5,
    reward_fn: RewardFunction = default_reward,
) -> list[dict[str, int | float | str]]:
    """Trace one episode step-by-step, returning a row per (s, a, R_{t+1}, s') transition.

    What + why: :func:`evaluate_policies` answers "which policy scores better?"; this answers
    "*why* -- what did it actually do, step by step?". It runs a single rollout and logs the full
    state, the chosen action, its cost, the reward R_{t+1}, and the resulting evidence/ambiguity,
    so a reader can audit the decision sequence (e.g. catch a policy that escalates needlessly or
    answers before grounding). This is qualitative inspection, the companion to the aggregate
    metrics, and -- like the rest of this module -- it re-simulates the known environment rather
    than reading a fixed log (not OPE).

    Args:
        policy: The policy to trace; must satisfy the
            :class:`~learning_agents.policies.Policy` protocol.
        scenario_id: Scenario index into :data:`~learning_agents.environment.SCENARIOS` to start
            from.
        seed: Optional env-reset jitter seed; ``None`` gives the scenario's noise-free start state.
        horizon: Episode length H (steps) passed to the environment.
        reward_fn: Reward function injected into the environment; defaults to
            :func:`~learning_agents.environment.default_reward`.

    Returns:
        One dict per step with the pre-step state fields, the action label, its cost, the step
        reward, the termination reason, and the resulting ``next_evidence`` / ``next_ambiguity`` --
        ordered chronologically.

    RL concept:
        A single trajectory tau = (s_1, a_1, R_2, s_2, ...) through the MDP under a fixed policy --
        the qualitative trace behind the aggregate evaluation metrics.
    """
    policy.reset()
    environment = AgentDecisionEnvironment(horizon=horizon, reward_fn=reward_fn)
    state = environment.reset(seed=seed, scenario_id=scenario_id)
    rows: list[dict[str, int | float | str]] = []

    while not environment.is_done():
        action = policy.select_action(state)  # fixed policy pi(.|s); no learning here
        transition = environment.step(action)
        rows.append(
            {
                "scenario_name": environment.scenario_name,
                "step": state.step,
                "intent": state.intent,
                "difficulty": state.difficulty,
                "ambiguity": state.ambiguity,
                "evidence": state.evidence,
                "attempts": state.attempts,
                "budget": state.budget,
                # The artifact contract names this column action_label in sample_episodes.csv; it
                # holds the human-readable action label (validated against ACTION_LABELS).
                "action_label": ACTION_LABELS[action],
                "action_cost": transition.info["action_cost"],
                "reward": transition.reward,
                "termination": transition.info["termination"],
                "next_evidence": transition.state.evidence,
                "next_ambiguity": transition.state.ambiguity,
            }
        )
        state = transition.state

    return rows


def _summarize_scenarios(
    scenario_rows: Sequence[dict[str, int | float | str]],
) -> list[dict[str, int | float | str]]:
    """Aggregate per-episode rows into one mean-per-metric summary row per policy.

    What + why: this is the aggregation step of policy evaluation. It groups the raw per-episode
    rows by policy, averages each metric across that policy's episodes, and returns the summaries
    ranked by ``avg_reward``. Averaging over scenarios and seeds is how a few noisy rollouts
    approximate a policy's value and typical behaviour; reporting the side-metrics next to
    ``avg_reward`` keeps reward hacking and unsafe shortcuts legible rather than hidden behind one
    scalar.

    Args:
        scenario_rows: Per-episode rows emitted by :func:`evaluate_policies` (each carries a
            ``policy`` name plus the numeric metrics to be averaged).

    Returns:
        One summary dict per policy with ``avg_*`` metrics and ``solved_rate``, sorted by
        ``avg_reward`` descending (best policy first).

    RL concept:
        Monte Carlo policy evaluation -- the empirical mean return (and side-effects) estimates the
        policy's expected value under the scenario/seed mix.

    Math:
        For metric m over a policy's N episodes, the report is the empirical mean
        (1/N) * sum_i m_i -- the unbiased estimate of E[m] under the scenario/seed distribution.
    """
    by_policy: dict[str, list[dict[str, int | float | str]]] = {}
    for row in scenario_rows:
        by_policy.setdefault(str(row["policy"]), []).append(row)

    summary_rows: list[dict[str, int | float | str]] = []
    for policy_name, rows in sorted(by_policy.items()):
        count = len(rows)
        avg_reward = sum(float(row["total_reward"]) for row in rows) / count
        avg_action_cost = sum(float(row["action_cost"]) for row in rows) / count
        avg_steps = sum(int(row["steps"]) for row in rows) / count
        avg_over_effort_count = sum(int(row["over_effort_count"]) for row in rows) / count
        # Escalation is terminal, so escalation_count is 0/1 per episode and its mean is a *rate*
        # (fraction of episodes that escalated) -- the avg_escalation_rate the contract requires.
        avg_escalation_rate = sum(int(row["escalation_count"]) for row in rows) / count
        avg_unsafe_decisions = (
            sum(int(row["unsafe_or_questionable_decisions"]) for row in rows) / count
        )
        # Under-grounding rate = fraction of *committed direct answers* that were under-grounded
        # (pooled over the policy's episodes), matching reporting.undergrounded_answer_rate. A
        # policy that never answers (e.g. always_escalate) has no answers -> 0.0 by convention.
        total_answered = sum(int(row["answered"]) for row in rows)
        total_undergrounded = sum(int(row["undergrounded_answer"]) for row in rows)
        avg_undergrounded_rate = (
            (total_undergrounded / total_answered) if total_answered else 0.0
        )
        solved_rate = sum(int(row["solved"]) for row in rows) / count
        summary_rows.append(
            {
                "policy": policy_name,
                "avg_reward": round(avg_reward, 4),
                "avg_action_cost": round(avg_action_cost, 4),
                "avg_steps": round(avg_steps, 4),
                "avg_over_effort_count": round(avg_over_effort_count, 4),
                "avg_escalation_rate": round(avg_escalation_rate, 4),
                "avg_undergrounded_rate": round(avg_undergrounded_rate, 4),
                "avg_unsafe_or_questionable_decisions": round(avg_unsafe_decisions, 4),
                "solved_rate": round(solved_rate, 4),
            }
        )
    # Rank policies by estimated value: highest mean return first (the leaderboard order).
    return sorted(summary_rows, key=lambda row: float(row["avg_reward"]), reverse=True)
