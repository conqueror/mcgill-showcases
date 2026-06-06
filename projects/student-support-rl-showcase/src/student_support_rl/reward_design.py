"""Contrast a well-aligned reward with a hackable one to expose reward hacking.

Reward design is where an MDP's *objective* lives: the agent optimizes whatever scalar
``R_{t+1}`` we hand it, not what we meant. This module pairs the environment's aligned
``default_reward`` (re-exported as ``GOOD_REWARD``) with a deliberately misspecified
``bad_reward``, then evaluates the same fixed policies under both so a graduate reader can
*measure* reward hacking: a policy that maximizes the proxy reward while degrading the true
objective. The flaw in ``bad_reward`` is intentional and pedagogical -- it pays a bonus for
intervention *intensity* and only counts positive short-term progress, so the advisor-heavy
policy scores best on the proxy yet worst on the aligned reward.

This sits upstream of every method on the ladder (contextual bandit -> MDP -> Q-learning ->
DQN -> policy gradient -> actor-critic -> PPO): all of them inherit the reward, so a bad
reward silently corrupts every learner. The notation follows the project convention: reward
after acting is ``R_{t+1}``, the return is ``G_t = sum_k gamma^k R_{t+k+1}``.

RL concept:
    Reward design, proxy/true-objective mismatch, and reward hacking; see
    docs/reward-design-and-hacking.md and docs/evaluation-and-governance.md.
"""

from __future__ import annotations

from collections.abc import Sequence

from student_support_rl.environment import ACTION_COSTS, StudentState, default_reward
from student_support_rl.evaluation import evaluate_policies
from student_support_rl.policies import Policy


def bad_reward(
    previous_state: StudentState,
    action: int,
    next_state: StudentState,
    done: bool,
) -> float:
    """Compute a deliberately misspecified (hackable) per-step reward.

    This is a teaching counter-example: a proxy reward that *looks* sensible but is
    exploitable. It differs from the aligned ``default_reward`` in three ways, each a classic
    reward-design mistake: (1) it adds an ``intervention_bonus`` proportional to the action
    index, literally paying more for heavier interventions; (2) it clips progress at zero with
    ``max(0, ...)``, so the agent is never charged for backsliding; (3) it ignores intervention
    fatigue (no over-intervention penalty) and unresolved end-of-horizon risk (``done`` is
    discarded). The net effect: an advisor-heavy policy maximizes this proxy ``R_{t+1}`` while
    actually harming the student, the signature of reward hacking.

    Args:
        previous_state: State ``s`` before the action (``S_t``).
        action: Intervention index in ``ACTION_COSTS`` (0 = none ... 3 = advisor meeting);
            larger indices denote heavier, costlier interventions.
        next_state: State ``s'`` after the action (``S_{t+1}``).
        done: Whether the episode terminated this step. Intentionally ignored here -- the bug
            that lets unresolved end-of-horizon risk go unpunished.

    Returns:
        The proxy scalar reward ``R_{t+1}`` (rounded to 4 decimals), upward-biased for large
        ``action`` values and never penalized for declines in engagement or completion.

    RL concept:
        Reward hacking via a misspecified proxy reward; contrast with the aligned
        ``GOOD_REWARD`` and see docs/reward-design-and-hacking.md.

    Math:
        ``R_{t+1} = 1.1 * max(0, progress) + 0.45 * action - 0.1 * cost(action)`` where
        ``progress = (engagement' + completion') - (engagement + completion)``. The ``0.45 *
        action`` term rewards intervention *intensity*, and clipping ``progress`` at 0 removes
        any penalty for regression.
    """
    del done  # discarding `done` is the bug: unresolved end-of-horizon risk goes unpunished
    # Progress is clipped at 0 -> the proxy never charges for engagement/completion declines
    progress = max(
        0,
        (next_state.engagement + next_state.completion)
        - (previous_state.engagement + previous_state.completion),
    )
    # Misalignment: bonus grows with intervention INTENSITY (the action index), not outcomes
    intervention_bonus = 0.45 * action
    # Proxy R_{t+1}: rewards progress + intensity, only lightly discounts the true action cost
    return round((1.1 * progress) + intervention_bonus - (0.1 * ACTION_COSTS[action]), 4)


# GOOD_REWARD is the environment's aligned objective (default_reward): rewards progress and
# risk reduction, charges full action cost, and penalizes over-intervention and unresolved
# end-of-horizon risk -- the true objective the proxy bad_reward fails to track.
GOOD_REWARD = default_reward
BAD_REWARD = bad_reward  # the misspecified proxy reward used to demonstrate reward hacking


def compare_reward_models(
    *,
    policies: Sequence[Policy],
    scenario_ids: Sequence[int],
    horizon: int = 6,
) -> list[dict[str, int | float | str]]:
    """Evaluate the same policies under both the bad and good reward and tag each row.

    Holding the policies, scenarios, and horizon fixed while swapping only ``reward_fn`` is the
    controlled experiment that isolates the reward's effect: any change in the policy ranking
    must come from the reward, not the dynamics. This produces the side-by-side table that
    ``reward_hacking_report`` reads to show the proxy and true objectives disagree.

    Args:
        policies: Fixed (non-learning) policies to score under both reward models; each must
            expose a ``name`` (e.g. ``"heuristic"``, ``"advisor_heavy"``).
        scenario_ids: Environment scenario indices to roll out per policy.
        horizon: Episode length (number of weeks) used for every rollout.

    Returns:
        One summary row per (reward model, policy) pair -- each row is the per-policy summary
        from ``evaluate_policies`` (including ``avg_reward``) augmented with a ``reward_model``
        key whose value is ``"bad"`` or ``"good"``.

    RL concept:
        Simulator-based policy evaluation under competing objectives / reward-model comparison
        (NOT off-policy evaluation -- each policy is re-simulated in the known environment, see
        evaluation.py); see docs/reward-design-and-hacking.md and docs/evaluation-and-governance.md.
    """
    rows: list[dict[str, int | float | str]] = []
    # Controlled swap: identical policies/scenarios/horizon, only the reward function changes
    for reward_name, reward_fn in (("bad", BAD_REWARD), ("good", GOOD_REWARD)):
        # evaluate_policies returns (summary_rows, scenario_rows); keep the per-policy summary
        summary_rows, _ = evaluate_policies(
            policies=policies,
            scenario_ids=scenario_ids,
            horizon=horizon,
            reward_fn=reward_fn,
        )
        for row in summary_rows:
            comparison_row = dict(row)
            comparison_row["reward_model"] = reward_name  # tag which objective produced the row
            rows.append(comparison_row)
    return rows


def reward_model_specs() -> dict[str, str]:
    """Return human-readable Markdown specifications of the good and bad reward designs.

    Reward functions are ultimately *design decisions*, so this returns the plain-language
    intent behind each one -- the aligned design that rewards progress, risk reduction, and
    penalizes fatigue and unresolved risk, versus the hackable design that rewards only
    short-term progress plus an intensity bonus. Pairing intent with the measured numbers makes
    the reward-hacking gap legible rather than abstract.

    Returns:
        A mapping with keys ``"good"`` and ``"bad"``, each holding a Markdown string describing
        that reward model's design goals.

    RL concept:
        Reward specification vs. realized behavior; see docs/reward-design-and-hacking.md.
    """
    return {
        "good": (
            "# Good Reward Design\n\n"
            "- Reward progress in engagement and assignment completion.\n"
            "- Reward risk reduction.\n"
            "- Charge intervention costs.\n"
            "- Penalize repeated interventions and unresolved end-of-horizon risk.\n"
        ),
        "bad": (
            "# Bad Reward Design\n\n"
            "- Reward only short-term progress.\n"
            "- Add bonus points for bigger interventions.\n"
            "- Ignore long-term unresolved risk and intervention fatigue.\n"
        ),
    }


def reward_hacking_report(comparison_rows: Sequence[dict[str, int | float | str]]) -> str:
    """Summarize the reward-hacking gap by contrasting two policies across two rewards.

    This is the punchline of the module: it indexes the comparison table by (reward model,
    policy) and lays out the four ``avg_reward`` numbers that demonstrate the rank reversal.
    Under the bad (proxy) reward the advisor-heavy policy looks best because the proxy overpays
    for intervention intensity; under the good (aligned) reward that same policy is penalized
    for fatigue and unresolved risk, so the ranking flips. Seeing the proxy and true objectives
    disagree on which policy is "best" is exactly the diagnostic signature of reward hacking.

    Args:
        comparison_rows: Rows from ``compare_reward_models`` -- each tagged with ``reward_model``
            (``"bad"``/``"good"``), ``policy``, and ``avg_reward``. Must include the
            ``"advisor_heavy"`` and ``"heuristic"`` policies under both reward models.

    Returns:
        A Markdown report stating the hypothesis and listing the advisor-heavy vs. heuristic
        average rewards under each reward model.

    Raises:
        KeyError: If a required (reward_model, policy) pair is absent from ``comparison_rows``.

    RL concept:
        Detecting reward hacking via proxy-vs-true objective rank reversal; see
        docs/reward-design-and-hacking.md and docs/evaluation-and-governance.md.
    """
    # Index by (reward_model, policy) so the four cells of the 2x2 comparison are addressable
    by_key = {(str(row["reward_model"]), str(row["policy"])): row for row in comparison_rows}
    bad_advisor = by_key[("bad", "advisor_heavy")]
    bad_heuristic = by_key[("bad", "heuristic")]
    good_advisor = by_key[("good", "advisor_heavy")]
    good_heuristic = by_key[("good", "heuristic")]

    return (
        "# Reward Hacking Check\n\n"
        "Under the bad reward, the advisor-heavy policy looks better because the reward "
        "overpays for intervention intensity.\n\n"
        f"- Bad reward advisor-heavy average reward: {bad_advisor['avg_reward']}\n"
        f"- Bad reward heuristic average reward: {bad_heuristic['avg_reward']}\n"
        f"- Good reward advisor-heavy average reward: {good_advisor['avg_reward']}\n"
        f"- Good reward heuristic average reward: {good_heuristic['avg_reward']}\n"
    )
