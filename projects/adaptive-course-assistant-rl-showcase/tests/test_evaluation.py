from adaptive_course_assistant_rl.evaluation import evaluate_policies, simulate_episode
from adaptive_course_assistant_rl.policies import RandomPolicy, RuleBasedPolicy


def test_evaluation_reports_reward_and_safety_metrics() -> None:
    summary_rows, scenario_rows = evaluate_policies(
        policies=[RandomPolicy(seed=7), RuleBasedPolicy()],
        scenario_ids=(0, 1),
        episodes_per_scenario=1,
    )

    assert summary_rows
    assert scenario_rows
    assert "avg_reward" in summary_rows[0]
    assert "avg_final_safety_risk" in summary_rows[0]
    assert "actions" in scenario_rows[0]


def test_random_policy_varies_across_repeated_rollouts() -> None:
    _summary_rows, scenario_rows = evaluate_policies(
        policies=[RandomPolicy(seed=7)],
        scenario_ids=(0,),
        episodes_per_scenario=3,
    )

    traces = {str(row["actions"]) for row in scenario_rows}
    assert len(traces) > 1


def test_deterministic_replays_are_marked_as_replayed_evidence() -> None:
    summary_rows, scenario_rows = evaluate_policies(
        policies=[RuleBasedPolicy()],
        scenario_ids=(0,),
        episodes_per_scenario=3,
    )

    assert len(scenario_rows) == 3
    assert {int(row["reset_seed"]) for row in scenario_rows} == {0, 1, 2}
    assert len({str(row["trajectory_signature"]) for row in scenario_rows}) == 1

    summary = summary_rows[0]
    assert summary["episode_count"] == 3
    assert summary["unique_trajectory_count"] == 1
    assert summary["replayed_trajectory_count"] == 2
    assert summary["evidence_mode"] == "replayed_deterministic_rollouts"


def test_sample_episode_uses_readable_state_labels() -> None:
    rows = simulate_episode(policy=RuleBasedPolicy(), scenario_id=2)

    assert rows
    first_row = rows[0]
    assert first_row["intent_type"] == "exam_review"
    assert first_row["difficulty_level"] == "advanced"
    assert first_row["confidence_level"] == "low"
    assert first_row["retrieval_quality"] == "poor"
    assert first_row["intent_uncertainty"] == "medium"
    assert first_row["cognitive_load"] == "high"
    assert first_row["safety_risk"] == "high"
    assert first_row["next_safety_risk"] in {"low", "medium", "high"}
