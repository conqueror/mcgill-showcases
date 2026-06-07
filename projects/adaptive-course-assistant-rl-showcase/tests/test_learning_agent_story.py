from adaptive_course_assistant_rl.learning_agent_story import build_learning_agent_story


def test_learning_agent_story_makes_the_boundary_explicit() -> None:
    story = build_learning_agent_story(quick=True, scenario_id=1)

    assert "# Learning Agent Story" in story.markdown
    assert "OpenAI Agents SDK" in story.markdown
    assert "MARL" in story.markdown
    assert "selected action" in story.markdown
