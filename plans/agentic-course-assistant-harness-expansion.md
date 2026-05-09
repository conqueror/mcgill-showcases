# Agentic Course Assistant Harness Expansion Plan

> Required orchestration skill: `core-harness-flow`

## Goal

Upgrade `projects/agentic-course-assistant-showcase` into a first-class
agentic systems showcase that covers deterministic ADK-inspired workflows,
optional live OpenAI Agents SDK and Google ADK execution, MkDocs discovery, and
public-safe harness engineering artifacts.

## Task Classification

- Task class: `standard`
- Run mode: `long_running`
- Audit mode: `OFFLINE_EVIDENCE_ONLY`
- Evaluator cadence: `per_phase`
- Applied policy path: `AGENTS.md`
- Harness routing contract: `.codex/harness/role-skill-matrix.toml`
- Operating pack: `docs/agents/oodaris-harness-v2-operating-pack.md`

## Scope

In scope:

- Add MkDocs first-class navigation for the agentic course assistant showcase.
- Add `docs/deep-dives/agentic-course-assistant.md`.
- Add `docs/tracks/agent-frameworks.md`.
- Strengthen `docs/index.md` with a flagship agentic systems card.
- Add deterministic offline examples for sequential, loop, parallel, router,
  and custom policy agent patterns.
- Add optional live OpenAI and Google ADK paths that read `.env` values without
  requiring credentials for default CI.
- Add `.env.example` and ensure `.env` is ignored.
- Add harness lab artifacts and a `scripts/run_harness_lab.py` entrypoint.
- Add Make targets for `run`, `eval`, `trace-check`, `harness-report`, and
  `test`.
- Add tests and artifact-contract validation for the harness lab.
- Update README and project docs for student flow.

Out of scope:

- Requiring live API calls in default CI.
- Copying private harness policy or private infrastructure assumptions.
- Splitting the current showcase into separate projects during this slice.
- Mutating shared or production environments.

## Model Defaults

- OpenAI live default: `gpt-5.4-mini`, selected from the official OpenAI API
  pricing page as the cheapest currently listed GPT-5.4-class model.
- Google live default: `gemini-3.1-flash-lite-preview`, selected from the
  official Gemini pricing and model pages as the cheapest Gemini 3.1 agentic
  text model currently listed.
- Both defaults are configurable through `.env`.

## File Claims

- Docs: `docs/index.md`, `docs/deep-dives/agentic-course-assistant.md`,
  `docs/tracks/agent-frameworks.md`, `mkdocs.yml`
- Showcase docs: `projects/agentic-course-assistant-showcase/README.md`,
  `projects/agentic-course-assistant-showcase/docs/lab-guide.md`,
  `projects/agentic-course-assistant-showcase/docs/concept-map.md`
- Showcase code: `projects/agentic-course-assistant-showcase/src/agentic_course_assistant/**`,
  `projects/agentic-course-assistant-showcase/scripts/**`,
  `projects/agentic-course-assistant-showcase/tests/**`
- Showcase config: `projects/agentic-course-assistant-showcase/Makefile`,
  `projects/agentic-course-assistant-showcase/pyproject.toml`,
  `projects/agentic-course-assistant-showcase/.env.example`,
  `.gitignore`
- Harness artifacts: `projects/agentic-course-assistant-showcase/artifacts/manifest.json`,
  `projects/agentic-course-assistant-showcase/artifacts/harness/**`
- Run evidence: `docs/agents/runs/2026-05-09-agentic-course-assistant-harness-expansion.md`

## Implementation Slices

1. Docs discovery:
   - Add the MkDocs deep dive and agent-frameworks track.
   - Update `mkdocs.yml` nav.
   - Add a flagship card to `docs/index.md`.

2. Deterministic workflow lab:
   - Implement offline sequential, loop, parallel, router, and custom policy
     examples with stable JSON-serializable results.
   - Add tests that verify behavior and trace names.

3. Live SDK path:
   - Add `.env` loading without adding a required dependency.
   - Add OpenAI and Google live runners gated by API key presence.
   - Keep optional SDK imports behind clear runtime errors.

4. Harness lab:
   - Add trace schema, golden eval cases, judge verdicts, failure-injection
     report, run ledger, and harness report generation.
   - Add artifact verifier coverage for these files.

5. Closure:
   - Run project checks, smoke, eval, trace-check, harness-report, verify,
     MkDocs strict build, harness preflight, harness lint, and diff checks.
   - Run independent critic before final status.

## Acceptance Criteria

- Default path runs offline with no API keys.
- `.env.example` documents `OPENAI_API_KEY`, `GEMINI_API_KEY`, OpenAI model, and
  Gemini model defaults.
- `.env` files are ignored.
- Optional live examples are discoverable but skipped unless keys and optional
  SDK extras are available.
- `make eval`, `make trace-check`, and `make harness-report` work locally.
- `make check`, `make smoke`, and `make verify` pass for the showcase.
- `make docs-check`, `make harness-preflight`, and `make harness-lint` pass at
  the repo root.
- Independent critic verdict is recorded before closure.

## Stop Rules

- Stop if default CI would require hosted API credentials.
- Stop if ADK/OpenAI optional examples require importing optional SDKs during
  default tests.
- Stop if MkDocs strict build fails on navigation or broken docs structure.
- Stop if harness artifacts cannot be validated deterministically.
