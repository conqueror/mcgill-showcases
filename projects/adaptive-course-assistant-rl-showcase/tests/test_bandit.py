from adaptive_course_assistant_rl.contextual_bandit import run_bandit_experiment


def test_contextual_bandit_writes_real_per_step_logs() -> None:
    result = run_bandit_experiment(steps=12, seed=3)

    assert len(result.metrics_rows) == 12
    assert len(result.regret_rows) == 12
    assert any(float(row["cumulative_regret"]) >= 0.0 for row in result.regret_rows)
    assert {row["action"] for row in result.action_rows}
