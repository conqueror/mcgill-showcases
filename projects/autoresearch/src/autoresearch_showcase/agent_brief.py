from __future__ import annotations

from .models import UpstreamProfile


def _agent_label(agent: str) -> str:
    if agent == "codex":
        return "Codex"
    if agent == "claude":
        return "Claude Code"
    raise ValueError(f"Unsupported agent: {agent}")


def _kickoff_prompt(profile: UpstreamProfile) -> str:
    return "\n".join(
        [
            "Read README.md, prepare.py, train.py, and program.md.",
            "Do not modify prepare.py or install new dependencies.",
            "Create results.tsv if it is missing and keep it untracked.",
            "Run one baseline experiment before trying any changes.",
            "Use val_bpb as the primary score and weigh simplicity before keeping changes.",
            (
                f"Treat {profile.mutable_surface} as the editable surface and "
                f"{profile.fixed_surface} as fixed."
            ),
        ]
    )


def render_agent_brief(profile: UpstreamProfile, agent: str) -> str:
    """Render one launch brief for a specific agent and platform."""

    agent_label = _agent_label(agent)
    commands = "\n".join(
        f"{index}. `{command}`"
        for index, command in enumerate(profile.preflight_commands, start=1)
    )
    notes = "\n".join(f"- {note}" for note in profile.notes)
    prompt = _kickoff_prompt(profile)
    return f"""# {agent_label} Launch Brief: {profile.display_name}

## Upstream target

- Repo: `{profile.repo}`
- URL: {profile.repo_url}
- Snapshot commit: `{profile.repo_commit}`

## Why this track

- Hardware expectation: {profile.hardware}
- Attention backend: {profile.attention_backend}
- Compile policy: {profile.compile_policy}

## Preflight

1. Clone the upstream repo.
2. Enter the repo root.
3. Run:

{commands}

## Kickoff prompt

```text
{prompt}
```

## Non-negotiables

- `prepare.py` stays fixed.
- `train.py` is the research surface.
- `program.md` is the operating guide the human can refine over time.
- Keep `results.tsv` local and untracked.
- Log crashes explicitly instead of pretending they were neutral results.

## Platform notes

{notes}
"""
