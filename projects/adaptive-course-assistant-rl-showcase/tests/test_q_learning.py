from adaptive_course_assistant_rl.q_learning import q_table_rows, train_q_learning


def test_q_learning_returns_a_non_empty_training_curve_and_table() -> None:
    result = train_q_learning(episodes=12, seed=5)

    assert len(result.training_curve) == 12
    rows = q_table_rows(result.q_table)
    assert rows
    assert "intent_type" in rows[0]
    assert "action" in rows[0]
