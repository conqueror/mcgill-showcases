"""Pin the project-level docs/readiness contract for the learning-agents showcase."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("relative_path", "false_claim", "correction"),
    [
        ("README.md", "live SDK gated behind", "SDK construction example"),
        (
            "docs/lane-a-agent-frameworks.md",
            "The framework never inspects these to make a choice",
            "does not receive the learned policy",
        ),
        (
            "docs/lane-a-agent-frameworks.md",
            "This is exactly the shape an Agents-SDK run would log",
            "local simulator trace",
        ),
        (
            "docs/lane-a-agent-frameworks.md",
            "same trace shape a live run would",
            "does not execute the learned policy or prove native trace equivalence",
        ),
        (
            "docs/results-dashboard.md", "with a controlled KL\nleash",
            "GRPO and RLVR report KL but do not penalize it",
        ),
        (
            "docs/results-dashboard.md", "Numbers here are the full-run values",
            "historical examples, not verified results from the corrected code",
        ),
        (
            "docs/exercises.md", "uses the canonical numbers",
            "random-target errors are not evidence of poor overlap",
        ),
        (
            "docs/exercises.md", "roughly two of every three steps",
            "roughly two of every three episodes",
        ),
        (
            "docs/exercises.md", "The gate logic lives in\n`src/learning_agents/evaluation.py`",
            "The gate logic lives in\n`src/learning_agents/reporting.py`",
        ),
        (
            "docs/rl-ladder.md", "here the episode-mean return",
            "mean step return from earlier episodes",
        ),
        (
            "docs/math-notes.md", "here the episode-mean",
            "mean step return from earlier episodes",
        ),
        (
            "src/learning_agents/policy_gradient.py", "safe fallback",
            "does not guarantee safety",
        ),
        (
            "src/learning_agents/sdk_bridge.py", "trace is exactly what an",
            "not a native SDK trace",
        ),
    ],
)
def test_guides_state_the_implemented_behavior(
    relative_path: str, false_claim: str, correction: str
) -> None:
    text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    assert false_claim not in text
    assert correction in text


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
