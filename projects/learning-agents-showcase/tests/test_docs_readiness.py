"""Pin the project-level docs/readiness contract for the learning-agents showcase."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_readme_points_to_runnable_quickstart_and_local_guide() -> None:
    """The README must advertise the runnable path, not only static quality checks.

    A student should be able to see the quickest honest flow from a clean checkout:
    generate the core artifacts with ``make smoke``, verify them with ``make verify``,
    and use ``make check`` as the code-quality gate. The README should also point to a
    local guide under ``docs/`` because the showcase contract expects that surface.
    """
    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "make smoke" in readme_text
    assert "make verify" in readme_text
    assert "make check" in readme_text
    assert "docs/00-start-here.md" in readme_text
    assert "core runnable path is ready" in readme_text


def test_local_docs_surface_exists() -> None:
    """The project ships the local docs surface promised by the showcase playbook."""
    assert (PROJECT_ROOT / "docs" / "00-start-here.md").is_file()


def test_full_concept_guide_set_is_present() -> None:
    """Regression guard: every concept guide the showcase advertises must exist.

    The guides are cross-linked from ``00-start-here.md`` and the root README and are part of
    the showcase contract, so pin the whole set; an accidental deletion or rename then fails
    loudly instead of silently dropping a guide (and breaking a cross-link).
    """
    expected = [
        "00-start-here.md",
        "locus-of-learning.md",
        "showcase-architecture.md",
        "exploration-and-bandits.md",
        "rl-ladder.md",
        "deep-rl.md",
        "offline-rl-and-ope.md",
        "cost-aware-cascade.md",
        "reward-design-and-hacking.md",
        "evaluation-and-governance.md",
        "lane-a-agent-frameworks.md",
        "lane-b-preference-optimization.md",
        "lane-c-marl.md",
        "glossary.md",
        "math-notes.md",
        "exercises.md",
        "results-dashboard.md",
    ]
    docs_dir = PROJECT_ROOT / "docs"
    missing = [name for name in expected if not (docs_dir / name).is_file()]
    assert not missing, f"missing concept guides: {missing}"
