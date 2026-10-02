from __future__ import annotations

import re
import shlex
from pathlib import Path

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


def test_brief_checks_out_the_snapshot_before_running_tools() -> None:
    profile = get_profile("macos")
    brief = render_agent_brief(profile, "codex")
    commands = [shlex.split(command) for command in re.findall(r"`([^`]+)`", brief)]
    checkout = next((command for command in commands if command[:2] == ["git", "checkout"]),
                    None)
    assert checkout is not None
    assert "--detach" in checkout
    assert checkout[-1] == profile.repo_commit
    assert brief.index("git checkout") < brief.index("uv sync")


def test_brief_labels_platform_information_as_a_dated_snapshot() -> None:
    for platform in ["macos", "unix"]:
        brief = render_agent_brief(get_profile(platform), "codex")
        assert "Source check date: 2026-03-13" in brief
        assert "This generator does not recheck upstream sources" in brief


def test_readme_describes_memory_as_recorded() -> None:
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "memory is recorded for discussion" in readme
    assert "based on quality, memory, and complexity" not in readme
