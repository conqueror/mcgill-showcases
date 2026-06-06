"""Pin that simulator-based policy evaluation reports governance metrics, not just reward.

This test fixes the evaluation-and-governance discipline that sits beside the algorithm
ladder: comparing policies on fixed scenarios must expose more than average return. It pins
that both the per-policy summary and the per-scenario breakdown carry cost and safety columns
(intervention cost, escalation count, unsafe-or-questionable decisions), so a policy cannot be
recommended on reward alone. This is the comparison harness run before any rollout.

Note: ``evaluate_policies`` re-simulates each policy inside the known
``StudentSupportEnvironment``; it is NOT off-policy evaluation (OPE) from a fixed log of
trajectories collected by another behaviour policy.

RL concept:
    Simulator-based policy evaluation and multi-objective governance metrics; see
    docs/evaluation-and-governance.md and docs/reward-design-and-hacking.md.
"""

from __future__ import annotations

from student_support_rl.evaluation import evaluate_policies
from student_support_rl.policies import HeuristicPolicy, RandomPolicy
from student_support_rl.q_learning import train_q_learning


def test_policy_evaluation_includes_governance_metrics() -> None:
    """Verify summary and per-scenario rows both expose cost and safety metrics.

    Pins the evaluation contract: across the random, heuristic, and learned policies, the
    summary rows must include average intervention cost, escalation count, and unsafe-or-
    questionable decisions, and the per-scenario rows must carry the un-averaged safety/cost
    fields. Pinning these columns guarantees reward hacking is observable -- a high-reward
    policy that over-escalates or acts unsafely is flagged rather than hidden.

    RL concept:
        Governance/safety metrics in simulator-based policy evaluation; see
        docs/evaluation-and-governance.md.
    """
    q_result = train_q_learning(episodes=40, seed=5)
    summary_rows, scenario_rows = evaluate_policies(
        policies=[RandomPolicy(seed=5), HeuristicPolicy(), q_result.greedy_policy()],
        scenario_ids=(0, 1, 2, 3, 4),
        episodes_per_scenario=2,
    )

    # Summary granularity: per-policy averages must surface cost and safety, not just reward.
    assert {
        "avg_intervention_cost",
        "avg_escalation_count",
        "avg_unsafe_or_questionable_decisions",
    } <= set(summary_rows[0])
    # Per-scenario granularity: the raw safety/cost fields stay inspectable per episode.
    assert {"scenario_name", "unsafe_or_questionable_decisions", "intervention_cost"} <= set(
        scenario_rows[0]
    )
