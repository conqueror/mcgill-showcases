from adaptive_course_assistant_rl.policies import InterventionHeavyPolicy, RuleBasedPolicy
from adaptive_course_assistant_rl.reward_design import compare_reward_models


def test_reward_audit_compares_good_and_bad_rewards() -> None:
    rows = compare_reward_models(
        policies=[InterventionHeavyPolicy(), RuleBasedPolicy()],
        scenario_ids=(0, 1),
    )

    reward_models = {row["reward_model"] for row in rows}
    assert reward_models == {"bad", "good"}
