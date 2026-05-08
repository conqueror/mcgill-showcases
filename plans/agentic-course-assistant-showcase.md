# Agentic Course Assistant Showcase Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `core-executing-plans` to implement this plan task-by-task.

## Goal

Build `projects/agentic-course-assistant-showcase` as a student-friendly, script-first agent framework project that demonstrates routing, tool use, guardrails, traceability, eval rubrics, agent-as-judge, A2A/session/memory concepts, harness evidence, and optional OpenAI Agents SDK / Google ADK code shapes.

## Scope

In scope:

- scaffold a new Python showcase,
- implement deterministic offline agent routing,
- include optional SDK reference modules,
- generate a comprehensive OpenAI Agents SDK vs Google ADK concept atlas,
- include refined questions, learning path, and agent-as-judge rubric artifacts,
- generate stable artifacts,
- write tests and artifact verifier,
- wire root docs, Makefile, CI, and issue templates.

Out of scope:

- requiring live API credentials,
- deploying an agent service,
- adding MCP, A2A, session database, or memory runtime dependencies,
- building a UI.

## Stop Conditions

Stop and re-scope if:

- the default path requires network or API credentials,
- the project expands beyond one coherent course-assistant workflow,
- root integration conflicts with existing dirty worktree changes,
- harness readiness or project quality gates fail for reasons unrelated to this slice.

## Success Criteria

- `cd projects/agentic-course-assistant-showcase && make smoke` passes.
- `cd projects/agentic-course-assistant-showcase && make verify` passes.
- `cd projects/agentic-course-assistant-showcase && make check` passes.
- Concept coverage verifies the user-requested topics: tools, guardrails, tracing, workflows, evals, agent-as-judge, multi-agent orchestration, handoff/triage, A2A, sessions, memory, skills, and harness.
- `bash scripts/dev/harness-cli-preflight.sh` and `python3 scripts/harness_config_lint.py` pass.
- `make docs-check` passes or any doc issue is reported with exact residual risk.

## Tasks

1. Create the project skeleton with `README.md`, `Makefile`, `pyproject.toml`, `src/`, `scripts/`, `tests/`, `docs/`, and `artifacts/manifest.json`.
2. Implement deterministic routing, resource lookup, guardrails, and artifact writing.
3. Add optional OpenAI Agents SDK and Google ADK reference modules.
4. Add a concept atlas module that maps each concept to OpenAI Agents SDK, Google ADK, a student build step, an artifact, an eval prompt, and a risk.
5. Add tests for routing, lookup, guardrails, artifact writing, concept coverage, and artifact verification.
6. Wire the new showcase into root Makefile, README, docs, CI, issue templates, learning path, and coverage matrix.
7. Run project-local gates, harness gates, docs checks, and root verification.
8. Record the harness-lite run ledger under `docs/agents/runs/`.
