"""Pin tabular Q-learning's training contract and its edge over a random baseline.

These tests fix the value-based control rung (contextual bandit -> MDP -> Q-learning -> DQN
-> ...). The first pins the shape of what training emits (a per-episode curve and a populated
Q-table). The second pins the headline learning property: after enough episodes the greedy
Q-policy must beat a random policy on both average return and on the count of unsafe-or-
questionable decisions -- learning improves reward *without* trading away safety.

RL concept:
    Off-policy temporal-difference control (Q-learning) and its Bellman-optimality target;
    see docs/value-based-learning.md.

Math:
    TD error delta = target - Q(s,A) with target = R_{t+1} + gamma*max_a' Q(s',a'); fixed
    point is the Bellman optimality value Q*(s,a) = E[R_{t+1} + gamma*max_a' Q*(s',a')].
"""

from __future__ import annotations

from student_support_rl.evaluation import evaluate_policies
from student_support_rl.policies import HeuristicPolicy, RandomPolicy
from student_support_rl.q_learning import train_q_learning


def test_q_learning_training_returns_curve_and_q_table() -> None:
    """Verify training emits one curve row per episode and a non-empty Q-table.

    Pins the output contract of ``train_q_learning``: exactly ``episodes`` training-curve
    rows, a populated tabular Q(s,a), and the per-episode bookkeeping columns (episode,
    scenario_id, total_reward, epsilon) needed to plot learning and the epsilon schedule.
    This is structural -- it pins that the agent-environment loop ran and recorded values,
    independent of how good the learned policy is.

    RL concept:
        Tabular action-value storage Q(s,a) populated by TD updates; see
        docs/value-based-learning.md.
    """
    result = train_q_learning(
        episodes=24,
        seed=11,
        scenario_ids=(0, 1, 2, 3, 4),
    )

    # One curve row per episode; Q-table populated => the TD loop actually ran.
    assert len(result.training_curve) == 24
    assert result.q_table
    assert {"episode", "scenario_id", "total_reward", "epsilon"}.issubset(
        result.training_curve[0]
    )


def test_trained_q_policy_outperforms_random_baseline() -> None:
    """Verify the learned greedy policy beats random on reward AND on unsafe decisions.

    Pins the core value-of-learning claim: after 800 decayed-epsilon episodes the greedy
    Q-policy earns strictly higher average return than ``RandomPolicy`` while making strictly
    fewer unsafe-or-questionable decisions. The joint assertion guards against reward hacking
    -- a higher score must not come from over-intervening or ignoring high-risk students.
    The heuristic policy is included in the comparison set as a reference but is not asserted
    on here.

    RL concept:
        Policy improvement from value learning, evaluated against a baseline under matched
        scenarios; see docs/value-based-learning.md and docs/evaluation-and-governance.md.
    """
    result = train_q_learning(
        episodes=800,
        seed=7,
        scenario_ids=(0, 1, 2, 3, 4),
        epsilon=0.35,
        epsilon_decay=0.96,
    )

    summary_rows, _ = evaluate_policies(
        policies=[
            RandomPolicy(seed=7),
            HeuristicPolicy(),
            result.greedy_policy(),
        ],
        scenario_ids=(0, 1, 2, 3, 4),
    )

    by_policy = {row["policy"]: row for row in summary_rows}

    # Policy improvement: learned greedy policy earns more average return than random.
    assert float(by_policy["q_learning"]["avg_reward"]) > float(by_policy["random"]["avg_reward"])
    # Safety guard: the reward gain must not come from more unsafe/questionable actions.
    assert (
        float(by_policy["q_learning"]["avg_unsafe_or_questionable_decisions"])
        < float(by_policy["random"]["avg_unsafe_or_questionable_decisions"])
    )
