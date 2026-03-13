# Autoresearch Showcase Design

Date: 2026-03-13
Status: Approved for implementation
Scope: Unified showcase project under `projects/autoresearch`

## Summary

This design adds a new educational showcase that translates the ideas behind:

- `miolini/autoresearch-macos`
- `karpathy/autoresearch`

into a student-friendly project that matches the existing structure of this repository.

The showcase is intentionally not a literal port of either upstream repository. The upstream projects optimize for autonomous code editing inside a tiny pretraining codebase. This repository optimizes for readability, reproducibility, documentation quality, and laptop-friendly verification. The project therefore needs to teach the real workflow without requiring this repo's CI to run a 5-minute GPU pretraining loop.

## Primary learning outcome

Help learners understand and operationalize a fixed-budget autonomous research loop for tiny language-model pretraining, including:

- what stays fixed,
- what the agent is allowed to change,
- how keep/discard decisions work,
- how platform differences change the practical workflow,
- how to launch a real Codex or Claude Code run safely on macOS or Unix.

## Target audience

The project should work in layers:

- advanced beginners who know basic ML and need plain-language explanations,
- intermediate learners who understand training loops and want concrete workflow details,
- advanced readers who want to study autonomous experimentation patterns and platform tradeoffs.

## Design choice

Use one unified project:

- `projects/autoresearch`

with two platform tracks:

- macOS track referencing `miolini/autoresearch-macos`
- Unix/NVIDIA track referencing `karpathy/autoresearch`

This keeps the learning flow unified while preserving the important platform differences.

## Why a unified project makes sense

One project is the right boundary because the core idea is the same in both upstream repos:

1. `prepare.py` is fixed.
2. `train.py` is the mutable research surface.
3. `program.md` is the human-authored instruction surface.
4. runs use a fixed 5-minute budget.
5. the score is `val_bpb`.
6. experiments are logged and either kept or discarded.

The main conceptual difference is platform implementation, not the research loop itself.

## Non-goals

- vendoring the full upstream training code into this repo,
- making the repo's CI run real GPU pretraining jobs,
- replacing the upstream repos as the best place to run overnight experiments,
- teaching all of LLM pretraining from scratch.

## Project shape

The project should follow the house structure:

```text
projects/autoresearch/
├── README.md
├── Makefile
├── pyproject.toml
├── uv.lock
├── docs/
├── scripts/
├── src/autoresearch_showcase/
├── tests/
├── artifacts/.gitkeep
├── data/raw/.gitkeep
└── data/processed/.gitkeep
```

## Implementation boundary

The project should contain:

1. a deterministic local lab that generates educational artifacts,
2. a platform comparison between the macOS and Unix upstream repos,
3. a decision policy that teaches when to keep, discard, or reject an experiment,
4. generated launch briefs for Codex and Claude Code on both platforms,
5. documentation that maps concepts to code, artifacts, and real agent actions.

The project should not attempt to hide the fact that the real overnight loop happens in the upstream repos, not in this showcase repo.

## Core artifact set

The project run should generate:

- `artifacts/overview/platform_comparison.csv`
- `artifacts/overview/upstream_snapshot.json`
- `artifacts/overview/research_loop_summary.md`
- `artifacts/analysis/decision_scenarios.csv`
- `artifacts/analysis/simulated_results.tsv`
- `artifacts/analysis/decision_summary.json`
- `artifacts/agent/codex_macos.md`
- `artifacts/agent/codex_unix.md`
- `artifacts/agent/claude_macos.md`
- `artifacts/agent/claude_unix.md`
- `artifacts/summary.md`
- `artifacts/manifest.json`

## Documentation set

The project should include:

- `docs/learning-guide.md`
- `docs/concept-learning-map.md`
- `docs/how-the-loop-works.md`
- `docs/platform-notes.md`
- `docs/agent-workflow.md`
- `docs/checkpoint-answer-key.md`
- `docs/code-tour.md`

## Code layout

The internal package should be small and explicit:

```text
src/autoresearch_showcase/
├── __init__.py
├── models.py
├── platforms.py
├── decision_policy.py
├── agent_brief.py
└── reporting.py
```

## Teaching model

Every major concept should have:

1. a plain-language explanation,
2. a code or artifact anchor,
3. a concrete decision question,
4. a transfer idea for real Codex or Claude usage.

## Main workflow

The recommended learner flow should be:

1. run the showcase locally,
2. inspect the platform comparison and decision artifacts,
3. read the code tour and loop explanation,
4. generate or review the agent launch brief,
5. choose a platform,
6. clone the real upstream repo and run Codex or Claude Code against `program.md`.

This keeps real agent integration in the main path while preserving deterministic local verification.

## Root integration changes

Implementation should also update:

- root `README.md`
- `docs/getting-started.md`
- `docs/learning-path.md`
- `docs/aspect-coverage-matrix.md`
- `docs/showcase-architecture.md`
- `docs/tracks/optimization.md`
- root `Makefile`
- `.github/workflows/ci.yml`
- issue template project dropdowns

## Validation expectations

Project-level validation should include:

- `make smoke`
- `make test`
- `make check`
- `make verify`

The root repo should also know how to run:

- `make sync`
- `make lint`
- `make ty`
- `make test`
- `make smoke`
- conditional `make verify`

for the new project.

## Key design constraints

- Use clear docstrings and small modules.
- Keep the package CPU-friendly and deterministic.
- Be explicit about what is simulated vs what is a real upstream workflow.
- Do not claim direct comparability across platforms.
- Make the Codex and Claude Code path concrete enough that a reader can actually follow it.

## Definition of done

The project is done when:

1. `projects/autoresearch` exists with the standard project structure.
2. The local run produces the required artifacts and manifest.
3. The docs are strong enough for self-study without external explanation.
4. The root docs and CI surfaces know about the new project.
5. The real agent path for macOS and Unix is first-class and easy to follow.
