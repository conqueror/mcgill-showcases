from adaptive_course_assistant_rl.policy_gradient import softmax, train_reinforce


def test_softmax_returns_a_probability_distribution() -> None:
    probabilities = softmax([0.0, 1.0, 2.0])

    assert len(probabilities) == 3
    assert round(sum(probabilities), 6) == 1.0


def test_reinforce_produces_a_training_curve() -> None:
    result = train_reinforce(episodes=12, seed=13)

    assert len(result.training_curve) == 12
    assert "baseline" in result.training_curve[0]
