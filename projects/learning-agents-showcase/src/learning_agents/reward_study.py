"""Measure reward hacking by scoring fixed policies under the judge rubric vs a hackable proxy.

What + why: :mod:`learning_agents.reward` *defines* two objectives -- the aligned
:func:`~learning_agents.reward.judge_reward` and the deliberately misspecified
:func:`~learning_agents.reward.hackable_reward`. This module turns that pair into a *measurement*:
it re-scores the same fixed policies under both rewards and lays out the rank reversal that is the
diagnostic signature of reward hacking. It is the experiment layer between the reward definitions
and the reward-design artifacts written by ``scripts/run_reward_hacking_check.py``.

The controlled experiment holds the policies, scenarios, and horizon fixed and swaps *only* the
reward function, so any change in the policy ranking must come from the objective, not the dynamics.
Under the judge rubric the :class:`~learning_agents.policies.HeuristicRouterPolicy` (which grounds,
disambiguates, and answers) beats the degenerate
:class:`~learning_agents.policies.AlwaysEscalatePolicy`; under the hackable proxy -- which overpays
escalation -- the always-escalate policy wins despite
solving almost nothing. Seeing the proxy and true objectives disagree on which policy is "best" is
exactly reward hacking, made measurable.

This module imports ONLY from :mod:`learning_agents` so the showcase stays self-contained: the
reward pair from :mod:`~learning_agents.reward`, the simulator-based comparison harness from
:mod:`~learning_agents.evaluation`, and the :class:`~learning_agents.policies.Policy` protocol.

RL concept:
    Reward design and reward hacking -- proxy/true-objective mismatch surfaced as a measurable rank
    reversal across two rewards, evaluated by re-simulation (NOT off-policy evaluation; see
    :mod:`learning_agents.evaluation`).
"""

from __future__ import annotations

from collections.abc import Sequence

from learning_agents.evaluation import evaluate_policies
from learning_agents.policies import Policy
from learning_agents.reward import GOOD_REWARD, HACKABLE_REWARD

__all__ = [
    "compare_reward_models",
    "reward_hacking_report",
    "reward_model_specs",
]


def compare_reward_models(
    *,
    policies: Sequence[Policy],
    scenario_ids: Sequence[int],
    horizon: int = 5,
    episodes_per_scenario: int = 1,
) -> list[dict[str, int | float | str]]:
    """Evaluate the same policies under both the hackable and aligned reward and tag each row.

    What + why: holding the policies, scenarios, and horizon fixed while swapping only ``reward_fn``
    is the controlled experiment that isolates the reward's effect -- any change in the policy
    ranking must come from the objective, not the dynamics. This produces the side-by-side table
    that :func:`reward_hacking_report` reads to show the proxy and true objectives disagree on the
    best policy. The ``"bad"`` rows use :data:`~learning_agents.reward.HACKABLE_REWARD` (the
    misspecified proxy) and the ``"good"`` rows use :data:`~learning_agents.reward.GOOD_REWARD` (the
    judge rubric).

    Args:
        policies: Fixed (non-learning) policies to score under both reward models; each must expose
            a ``name`` (e.g. ``"always_escalate"``, ``"heuristic_router"``).
        scenario_ids: Scenario indices into :data:`~learning_agents.environment.SCENARIOS` to roll
            out per policy.
        horizon: Episode length H used for every rollout.
        episodes_per_scenario: Seeded rollouts per (policy, scenario) pair.

    Returns:
        One summary row per (reward model, policy) pair -- each per-policy summary from
        :func:`~learning_agents.evaluation.evaluate_policies` augmented with a ``reward_model`` key
        whose value is ``"bad"`` (hackable proxy) or ``"good"`` (judge rubric).

    RL concept:
        Simulator-based policy evaluation under competing objectives / reward-model comparison; the
        controlled swap that exposes reward hacking.
    """
    rows: list[dict[str, int | float | str]] = []
    # Controlled swap: identical policies/scenarios/horizon, only the reward function changes.
    for reward_name, reward_fn in (("bad", HACKABLE_REWARD), ("good", GOOD_REWARD)):
        # evaluate_policies returns (summary_rows, scenario_rows); keep the per-policy summary.
        summary_rows, _ = evaluate_policies(
            policies=policies,
            scenario_ids=scenario_ids,
            episodes_per_scenario=episodes_per_scenario,
            horizon=horizon,
            reward_fn=reward_fn,
        )
        for row in summary_rows:
            comparison_row = dict(row)
            comparison_row["reward_model"] = reward_name  # tag which objective produced the row
            rows.append(comparison_row)
    return rows


def reward_model_specs() -> dict[str, str]:
    """Return human-readable Markdown specifications of the aligned and hackable reward designs.

    What + why: reward functions are ultimately *design decisions*, so this returns the
    plain-language intent behind each one -- the aligned judge rubric that rewards a well-grounded
    answer, charges for needless effort, and pays escalation only in proportion to genuine need,
    versus the hackable proxy that overpays escalation and raw retrieval regardless of need. Pairing
    intent with the measured numbers makes the reward-hacking gap legible rather than abstract.

    Returns:
        A mapping with keys ``"good"`` and ``"bad"``, each a Markdown string (heading-led)
        describing that reward model's design goals.

    RL concept:
        Reward specification vs. realized behavior -- the design intent behind the measured gap.
    """
    return {
        "good": (
            "# Good Reward Design (Judge Rubric)\n\n"
            "- Reward a well-grounded direct answer (evidence adequate for the difficulty and "
            "ambiguity resolved).\n"
            "- Penalize an under-grounded answer as a hallucination risk.\n"
            "- Charge a small penalty for needless retrieval or clarification.\n"
            "- Pay escalation only in proportion to genuine need, minus its high human cost.\n"
        ),
        "bad": (
            "# Bad Reward Design (Hackable Proxy)\n\n"
            "- Overpay a flat bonus for escalation regardless of whether a human was needed.\n"
            "- Pay for raw retrieval regardless of need, with no penalty for redundancy.\n"
            "- Under-credit the genuinely good well-grounded answer.\n"
        ),
    }


def reward_hacking_report(comparison_rows: Sequence[dict[str, int | float | str]]) -> str:
    """Summarize the reward-hacking gap by contrasting two policies across two rewards.

    What + why: this is the punchline -- it indexes the comparison table by (reward model, policy)
    and lays out the four ``avg_reward`` numbers that demonstrate the rank reversal. Under the bad
    (proxy) reward the always-escalate policy looks best because the proxy overpays escalation;
    under the good (judge) reward that same policy is penalized for needless escalation and its poor
    grounding, so the heuristic router wins. Reporting ``solved_rate`` beside the reward shows the
    always-escalate policy barely solving anything even when it "wins" the proxy -- the diagnostic
    signature of reward hacking.

    Args:
        comparison_rows: Rows from :func:`compare_reward_models` -- each tagged with
            ``reward_model`` (``"bad"``/``"good"``), ``policy``, ``avg_reward``, and
            ``solved_rate``. Must include the ``"always_escalate"`` and ``"heuristic_router"``
            policies under both reward models.

    Returns:
        A Markdown report (heading-led) stating the hypothesis and listing the always-escalate vs.
        heuristic-router average rewards and solved rates under each reward model.

    Raises:
        KeyError: If a required (reward_model, policy) pair is absent from ``comparison_rows``.

    RL concept:
        Detecting reward hacking via proxy-vs-true objective rank reversal, kept honest by reporting
        ``solved_rate`` next to the scalar reward.
    """
    # Index by (reward_model, policy) so the four cells of the 2x2 comparison are addressable.
    by_key = {(str(row["reward_model"]), str(row["policy"])): row for row in comparison_rows}
    bad_escalate = by_key[("bad", "always_escalate")]
    bad_router = by_key[("bad", "heuristic_router")]
    good_escalate = by_key[("good", "always_escalate")]
    good_router = by_key[("good", "heuristic_router")]

    return (
        "# Reward Hacking Check\n\n"
        "Under the hackable reward, the always-escalate policy looks better because the reward "
        "overpays for escalation regardless of whether a human was needed. Under the aligned judge "
        "rubric the heuristic router wins, and the always-escalate policy's low solved rate "
        "exposes "
        "that its proxy 'win' solved almost nothing.\n\n"
        "## Average reward (proxy vs aligned)\n\n"
        f"- Bad reward always-escalate average reward: {bad_escalate['avg_reward']}\n"
        f"- Bad reward heuristic-router average reward: {bad_router['avg_reward']}\n"
        f"- Good reward always-escalate average reward: {good_escalate['avg_reward']}\n"
        f"- Good reward heuristic-router average reward: {good_router['avg_reward']}\n\n"
        "## Solved rate (the hack, exposed)\n\n"
        f"- Always-escalate solved rate: {good_escalate['solved_rate']}\n"
        f"- Heuristic-router solved rate: {good_router['solved_rate']}\n"
    )
