from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from agentic_course_assistant import answer_question
from agentic_course_assistant.artifact_contract import verify
from agentic_course_assistant.artifacts import write_artifacts
from agentic_course_assistant.assistant import classify_question, guardrail_notes
from agentic_course_assistant.concept_atlas import (
    REQUESTED_CONCEPT_IDS,
    list_concepts,
    required_concept_coverage,
    write_concept_artifacts,
)
from agentic_course_assistant.course_catalog import search_resources


def test_classifies_debug_question() -> None:
    assert classify_question("My validation score is too good. Is this leakage?") == "debug"


def test_classifies_projection_question_as_concept_not_project() -> None:
    assert classify_question("Can you explain projection matrices in PCA?") == "concept"


def test_phrase_keywords_require_adjacent_words() -> None:
    assert classify_question("This model is too expensive but has good recall.") == "concept"


def test_search_resources_prefers_leakage_content() -> None:
    matches = search_resources("feature leakage split")
    assert matches[0].resource_id == "eda-leakage-001"


def test_answer_contains_trace_and_guardrail() -> None:
    result = answer_question("Help me build an agent SDK project without pasting an API key")
    assert result.intent == "project"
    assert result.agent_name == "Project planner"
    assert "triage_agent.selected_intent:project" in result.trace
    assert any("secrets" in note.lower() for note in result.guardrails)


def test_secret_guardrail_requires_secret_phrase() -> None:
    notes = guardrail_notes("The API is clear; which key metric should I optimize?")
    assert len(notes) == 1
    secret_notes = guardrail_notes("I pasted an API key by mistake.")
    assert any("secrets" in note.lower() for note in secret_notes)


def test_empty_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="question must not be empty"):
        answer_question("   ")


def test_write_artifacts_creates_stable_outputs(tmp_path: Path) -> None:
    result = answer_question("Explain data leakage")
    paths = write_artifacts(result, tmp_path)

    assert {path.name for path in paths} == {
        "agent_trace.json",
        "agentic_concepts.csv",
        "agent_judge_rubric.json",
        "concept_coverage.json",
        "course_assistant_response.md",
        "openai_vs_adk_concepts.json",
        "refined_questions.md",
        "resource_matches.csv",
        "student_learning_path.md",
    }
    trace = json.loads((tmp_path / "agent_trace.json").read_text(encoding="utf-8"))
    assert trace["intent"] == "debug"
    assert trace["resource_ids"]


def test_verify_artifacts_validates_trace_and_resource_schema(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "manifest.json").write_text(
        json.dumps(
            {
                "required_files": [
                    "artifacts/course_assistant_response.md",
                    "artifacts/agent_trace.json",
                    "artifacts/resource_matches.csv",
                    "artifacts/concepts/agentic_concepts.csv",
                    "artifacts/concepts/openai_vs_adk_concepts.json",
                    "artifacts/concepts/refined_questions.md",
                    "artifacts/concepts/student_learning_path.md",
                    "artifacts/evals/agent_judge_rubric.json",
                    "artifacts/evals/concept_coverage.json",
                ]
            }
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "course_assistant_response.md").write_text("ok\n", encoding="utf-8")
    (artifacts_dir / "agent_trace.json").write_text(
        json.dumps({"intent": "debug", "trace": []}),
        encoding="utf-8",
    )
    (artifacts_dir / "resource_matches.csv").write_text("bad,columns\n1,2\n", encoding="utf-8")
    write_concept_artifacts(artifacts_dir)

    errors = verify(tmp_path)

    assert any("agent_trace.json missing keys" in error for error in errors)
    assert any("resource_matches.csv columns" in error for error in errors)


def test_adk_wrapper_exposes_agent_file() -> None:
    wrapper = Path(__file__).resolve().parents[1] / "adk_course_assistant/agent.py"
    assert wrapper.exists()
    assert "root_agent" in wrapper.read_text(encoding="utf-8")


def test_pyproject_declares_live_sdk_extras() -> None:
    project_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))

    optional_dependencies = pyproject["project"]["optional-dependencies"]

    assert "openai-agents" in optional_dependencies["openai"]
    assert "google-adk" in optional_dependencies["adk"]
    assert {"openai-agents", "google-adk"} <= set(optional_dependencies["live"])


def test_concept_atlas_covers_requested_topics() -> None:
    concept_ids = {concept.concept_id for concept in list_concepts()}
    assert REQUESTED_CONCEPT_IDS <= concept_ids
    assert len(concept_ids) >= 25
    coverage = required_concept_coverage()
    assert coverage
    assert all(coverage.values())


def test_concept_artifacts_are_written_and_verified(tmp_path: Path) -> None:
    result = answer_question("What is A2A and how is it different from a tool call?")
    write_artifacts(result, tmp_path / "artifacts")
    (tmp_path / "artifacts/manifest.json").write_text(
        json.dumps(
            {
                "required_files": [
                    "artifacts/course_assistant_response.md",
                    "artifacts/agent_trace.json",
                    "artifacts/resource_matches.csv",
                    "artifacts/concepts/agentic_concepts.csv",
                    "artifacts/concepts/openai_vs_adk_concepts.json",
                    "artifacts/concepts/refined_questions.md",
                    "artifacts/concepts/student_learning_path.md",
                    "artifacts/evals/agent_judge_rubric.json",
                    "artifacts/evals/concept_coverage.json",
                ]
            }
        ),
        encoding="utf-8",
    )

    errors = verify(tmp_path)

    assert errors == []
    comparison = json.loads(
        (tmp_path / "artifacts/concepts/openai_vs_adk_concepts.json").read_text(
            encoding="utf-8"
        )
    )
    assert "openai_agents_sdk" in comparison["frameworks"]
    assert "google_adk" in comparison["frameworks"]


def test_verify_artifacts_rejects_corrupted_teaching_markdown(tmp_path: Path) -> None:
    result = answer_question("Explain agent memory and sessions")
    write_artifacts(result, tmp_path / "artifacts")
    (tmp_path / "artifacts/manifest.json").write_text(
        json.dumps(
            {
                "required_files": [
                    "artifacts/course_assistant_response.md",
                    "artifacts/agent_trace.json",
                    "artifacts/resource_matches.csv",
                    "artifacts/concepts/agentic_concepts.csv",
                    "artifacts/concepts/openai_vs_adk_concepts.json",
                    "artifacts/concepts/refined_questions.md",
                    "artifacts/concepts/student_learning_path.md",
                    "artifacts/evals/agent_judge_rubric.json",
                    "artifacts/evals/concept_coverage.json",
                ]
            }
        ),
        encoding="utf-8",
    )

    (tmp_path / "artifacts/course_assistant_response.md").write_text("junk\n", encoding="utf-8")
    (tmp_path / "artifacts/concepts/student_learning_path.md").write_text(
        "junk\n", encoding="utf-8"
    )
    errors = verify(tmp_path)

    assert any("course_assistant_response.md missing sections" in error for error in errors)
    assert any("student_learning_path.md missing phrases" in error for error in errors)
