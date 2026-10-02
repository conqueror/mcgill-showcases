import pytest

from adaptive_course_assistant_rl.environment import AssistantInterventionEnvironment
from adaptive_course_assistant_rl.policy_gradient import softmax, train_reinforce


def test_softmax_returns_a_probability_distribution() -> None:
    probabilities = softmax([0.0, 1.0, 2.0])

    assert len(probabilities) == 3
    assert round(sum(probabilities), 6) == 1.0


def test_reinforce_produces_a_training_curve() -> None:
    result = train_reinforce(episodes=12, seed=13)

    assert len(result.training_curve) == 12
    assert "baseline" in result.training_curve[0]


def test_reinforce_learns_from_a_one_step_episode() -> None:
    result = train_reinforce(episodes=1, scenario_ids=(0,), seed=1, alpha=1.0, horizon=1)
    state = AssistantInterventionEnvironment(horizon=1).reset(scenario_id=0)

    # Seed 1 selects retrieval (action 1) from the uniform eight-action policy.
    # Reward = retrieval gain 1 - turn 1 - cost 0.7 - horizon penalty 5 = -5.7.
    # With no baseline, the selected logit changes by -5.7 * 7/8 = -4.9875;
    # each other logit changes by -5.7 * (-1/8) = 0.7125.
    assert result.theta[state.as_tuple()] == pytest.approx(
        [0.7125, -4.9875, 0.7125, 0.7125, 0.7125, 0.7125, 0.7125, 0.7125]
    )


def test_reinforce_gradient_matches_the_discounted_start_state_return() -> None:
    result = train_reinforce(
        episodes=1, scenario_ids=(0,), seed=1, alpha=1.0, gamma=0.5, horizon=2
    )
    env = AssistantInterventionEnvironment(horizon=2)
    first_state = env.reset(scenario_id=0)
    second_state = env.step(1).state

    # Seed 1 selects actions 1 then 6, each with probability 1/8.
    # r0 = retrieval 1 - turn 1 - cost 0.7 = -0.7.
    # r1 = confidence 3 + safety 2 - turn 1 - cost 0.8 - switch 2 - horizon 5 = -3.8.
    # G0 = -0.7 + 0.5*(-3.8) = -2.6; gamma**1 * G1 = 0.5*(-3.8) = -1.9.
    # Multiply each by the softmax score: 7/8 selected, -1/8 otherwise.
    assert result.theta[first_state.as_tuple()] == pytest.approx(
        [0.325, -2.275, 0.325, 0.325, 0.325, 0.325, 0.325, 0.325]
    )
    assert result.theta[second_state.as_tuple()] == pytest.approx(
        [0.2375, 0.2375, 0.2375, 0.2375, 0.2375, 0.2375, -1.6625, 0.2375]
    )
