"""Tests for the required docs set."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = PROJECT_ROOT / "docs"


def test_required_docs_exist_with_expected_headings() -> None:
    """Student-facing docs should be present and structured."""

    required_docs = {
        "learning-flow.md": "## Step-by-Step Flow",
        "concept-learning-map.md": "## Concept Map",
        "code-examples.md": "## Code Examples",
        "domain-use-cases.md": "## Domain Use Cases",
        "checkpoint-answer-key.md": "## Answers",
    }

    for doc_name, required_heading in required_docs.items():
        path = DOCS_ROOT / doc_name
        assert path.exists(), doc_name
        assert required_heading in path.read_text(encoding="utf-8")
