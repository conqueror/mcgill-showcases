from adaptive_course_assistant_rl.environment import (
    ACTION_LABELS,
    NONE_ACTION,
    AssistantInterventionEnvironment,
    state_key_to_row,
)


def test_environment_reset_and_step_follow_the_contract() -> None:
    env = AssistantInterventionEnvironment(horizon=4)
    state = env.reset(scenario_id=0)

    assert state.turn_index == 0
    assert state.last_action == NONE_ACTION

    transition = env.step(1)

    assert transition.state.turn_index == 1
    assert transition.state.last_action == 1
    assert transition.info["action_label"] == ACTION_LABELS[1]
    assert env.observe().turn_index == transition.state.turn_index


def test_environment_rejects_unknown_actions() -> None:
    env = AssistantInterventionEnvironment()
    env.reset()
    try:
        env.step(99)
    except ValueError as exc:
        assert "unknown action" in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("expected ValueError for an invalid action")


def test_environment_requires_reset_after_done() -> None:
    env = AssistantInterventionEnvironment(horizon=1)
    env.reset()

    transition = env.step(1)

    assert transition.done is True
    try:
        env.step(1)
    except RuntimeError as exc:
        assert "reset" in str(exc)
    else:  # pragma: no cover - defensive failure branch
        raise AssertionError("expected RuntimeError after terminal transition")


def test_state_key_to_row_uses_domain_specific_labels() -> None:
    row = state_key_to_row((0, 0, 0, 0, 2, 2, 1, 0, 0, NONE_ACTION, 0, 0))

    assert row["retrieval_quality"] == "strong"
    assert row["intent_uncertainty"] == "high"
    assert row["cognitive_load"] == "medium"


def test_targeted_practice_does_not_resolve_while_intent_is_still_uncertain() -> None:
    env = AssistantInterventionEnvironment()
    state = env.reset(scenario_id=3)

    assert state.intent_uncertainty == 1
    assert state.misconception_type == 0
    assert state.confidence_level == 1

    transition = env.step(4)

    assert transition.state.intent_uncertainty == 1
    assert transition.state.resolved_flag == 0
    assert transition.info["resolved"] == 0
    assert transition.done is False
