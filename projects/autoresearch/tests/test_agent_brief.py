from __future__ import annotations

from autoresearch_showcase.agent_brief import render_agent_brief
from autoresearch_showcase.platforms import get_profile


def test_codex_macos_brief_mentions_fixed_and_mutable_surfaces() -> None:
    brief = render_agent_brief(get_profile("macos"), "codex")
    assert "prepare.py" in brief
    assert "train.py" in brief
    assert "program.md" in brief
    assert "Codex" in brief
    assert "miolini/autoresearch-macos" in brief


def test_claude_unix_brief_mentions_repo_and_prompt() -> None:
    brief = render_agent_brief(get_profile("unix"), "claude")
    assert "Claude Code" in brief
    assert "karpathy/autoresearch" in brief
    assert "Create results.tsv if it is missing" in brief
