# Code Tour

This showcase has a small codebase on purpose.

## `src/autoresearch_showcase/models.py`

Defines the core data structures:

- upstream platform profiles,
- simulated decision scenarios.

These keep the project explicit and easy to inspect.

## `src/autoresearch_showcase/platforms.py`

Stores the grounded facts about the macOS and Unix upstream repos:

- repository URLs,
- commit snapshots,
- hardware assumptions,
- backend differences,
- preflight commands.

The goal is to keep the generated docs and briefs tied to real source details.

## `src/autoresearch_showcase/decision_policy.py`

Encodes the teaching version of the keep/discard logic.

It does not claim to be the only valid policy. It turns the upstream prompt's ideas into a readable rule set so learners can study the tradeoffs directly.

## `src/autoresearch_showcase/agent_brief.py`

Generates launch briefs for:

- Codex on macOS,
- Codex on Unix,
- Claude Code on macOS,
- Claude Code on Unix.

Each brief includes:

- the correct upstream repo,
- setup commands,
- a kickoff prompt,
- non-negotiable workflow rules.

## `src/autoresearch_showcase/reporting.py`

This is the main artifact builder.

It writes:

- platform comparison tables,
- decision scenarios,
- simulated result ledgers,
- agent briefs,
- the manifest used by `make verify`.

## `scripts/run_showcase.py`

Calls the reporting layer and generates the full artifact set.

## `scripts/render_agent_brief.py`

Renders a single brief when you only want one platform or one agent.
