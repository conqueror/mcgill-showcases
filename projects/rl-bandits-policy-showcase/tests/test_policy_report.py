from rl_bandits_showcase.evaluation import run_policy_suite
from rl_bandits_showcase.policy_report import build_recommendation_markdown


def test_policy_report_limits_its_ranking_to_the_observed_run() -> None:
    _, summary = run_policy_suite(arm_probs=[0.25, 0.75], horizon=20, seed=1)
    report = build_recommendation_markdown(summary)

    assert f"Highest observed reward in this run: **{summary.iloc[0]['strategy']}**" in report
    assert "does not establish general superiority" in report
    assert "Recommended policy:" not in report
