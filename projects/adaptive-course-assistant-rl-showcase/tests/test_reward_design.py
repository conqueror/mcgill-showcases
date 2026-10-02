import pytest

from adaptive_course_assistant_rl.policies import InterventionHeavyPolicy, RuleBasedPolicy
from adaptive_course_assistant_rl.reward_design import compare_reward_models, reward_hacking_report


def test_reward_audit_compares_good_and_bad_rewards() -> None:
    rows = compare_reward_models(
        policies=[InterventionHeavyPolicy(), RuleBasedPolicy()],
        scenario_ids=(0, 1, 2, 3, 4),
    )

    reward_models = {row["reward_model"] for row in rows}
    assert reward_models == {"bad", "good"}
    rewards = {(row["reward_model"], row["policy"]): float(row["avg_reward"]) for row in rows}
    assert rewards[("bad", "intervention_heavy")] > rewards[("bad", "rule_based")]
    assert rewards[("good", "intervention_heavy")] < rewards[("good", "rule_based")]
    assert "ranking flip" in reward_hacking_report(rows)


def test_reward_hacking_report_rejects_a_comparison_without_a_reversal() -> None:
    # In scenario 3, the heavy policy wins under both rewards.
    rows = compare_reward_models(
        policies=[InterventionHeavyPolicy(), RuleBasedPolicy()], scenario_ids=(3,)
    )
    with pytest.raises(ValueError, match="rank reversal"):
        reward_hacking_report(rows)


@pytest.mark.parametrize("bad_heavy,good_heavy", [(1.0, 0.0), (2.0, 1.0)])
def test_reward_hacking_report_rejects_tied_rankings(
    bad_heavy: float, good_heavy: float
) -> None:
    rows: list[dict[str, int | float | str]] = [
        {"reward_model": "bad", "policy": "intervention_heavy", "avg_reward": bad_heavy},
        {"reward_model": "bad", "policy": "rule_based", "avg_reward": 1.0},
        {"reward_model": "good", "policy": "intervention_heavy", "avg_reward": good_heavy},
        {"reward_model": "good", "policy": "rule_based", "avg_reward": 1.0},
    ]
    with pytest.raises(ValueError, match="rank reversal"):
        reward_hacking_report(rows)
