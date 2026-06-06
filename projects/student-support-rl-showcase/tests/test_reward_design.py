"""Behavioural tests that a misspecified reward invites reward hacking, a good one does not.

These tests pin the headline lesson of reward design: an agent optimizes the reward you
*write down*, not the outcome you *intend*. ``compare_reward_models`` scores the same two
fixed policies under two reward functions -- a ``bad`` reward that overpays for intervention
intensity and a ``good`` reward (the env default) that charges intervention cost and penalizes
unresolved end-of-horizon risk. The over-intervening ``AdvisorHeavyPolicy`` should look
*better* than the ``HeuristicPolicy`` under the bad reward yet *worse* under the good one --
a controlled demonstration of reward hacking that holds the policies fixed and varies only the
objective. The fixed policies stand in for "an optimizer", so no learning is needed to expose
the failure.

RL concept:
    Reward design and reward hacking / specification gaming; see
    docs/reward-design-and-hacking.md and docs/evaluation-and-governance.md.
"""

from __future__ import annotations

from student_support_rl.policies import AdvisorHeavyPolicy, HeuristicPolicy
from student_support_rl.reward_design import compare_reward_models


def test_reward_design_flags_over_intervention() -> None:
    """A bad reward ranks the over-intervening policy above the heuristic; a good one flips it.

    Under ``bad`` reward (intervention-intensity bonus, no fatigue/risk penalty) the
    advisor-heavy policy scores a higher average reward than the heuristic; under ``good``
    reward (the env default, which charges intervention cost and penalizes unresolved risk)
    that ordering reverses. The flip is the signature of reward hacking: the policies are
    unchanged, only the objective changed.

    RL concept:
        Reward hacking / specification gaming (docs/reward-design-and-hacking.md).
    """
    comparison_rows = compare_reward_models(
        policies=[AdvisorHeavyPolicy(), HeuristicPolicy()],
        scenario_ids=(0, 1, 2, 3),
    )

    by_key = {(row["reward_model"], row["policy"]): row for row in comparison_rows}

    # Reward hacking: bad reward overpays for intervention intensity -> advisor_heavy wins
    assert float(by_key[("bad", "advisor_heavy")]["avg_reward"]) > float(
        by_key[("bad", "heuristic")]["avg_reward"]
    )
    # Good reward charges cost + penalizes residual risk -> the ordering flips back
    assert float(by_key[("good", "advisor_heavy")]["avg_reward"]) < float(
        by_key[("good", "heuristic")]["avg_reward"]
    )
