from pathlib import Path


def test_readme_contains_required_sections() -> None:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    content = readme.read_text(encoding="utf-8")

    required_headings = [
        "# Modern NLP Pipeline Showcase",
        "## What You Should Learn",
        "## Prerequisites",
        "## Quickstart",
        "## Key Artifacts",
        "## Common Failure Modes",
        "## Suggested Next Projects",
    ]
    for heading in required_headings:
        assert heading in content


def test_expected_docs_exist() -> None:
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    for relative in (
        "architecture-notes.md",
        "learning-guide.md",
        "model-notes.md",
    ):
        assert (docs_dir / relative).exists()
