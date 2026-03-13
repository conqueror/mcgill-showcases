# Public Harness Lite Bootstrap Design

Date: 2026-03-13
Status: Draft for review
Scope: Design only. No implementation is covered by this document.

## Summary

This design proposes a minimal, public-safe Harness V2 bootstrap for `mcgill-showcases` that is just sufficient to run the approved implementation plan at `plans/deep-learning-math-foundations-showcase.md` end-to-end.

The bootstrap is intentionally narrow. It should provide:

- repo-local harness files required by `core-harness-flow`,
- deterministic task classification and role/skill routing,
- lightweight readiness checks,
- public-safe run ledgers and gate reporting,
- enough role coverage to execute a Python showcase implementation workflow.

The bootstrap should not attempt to mirror the broader private OODARIS control plane.

## Problem

The current repository does not contain the repo-local Harness V2 artifacts required by the canonical orchestration flow:

- `.codex/config.toml`
- `.codex/agents/workflow_orchestrator.toml`
- `.codex/harness/role-skill-matrix.toml`
- `docs/agents/oodaris-harness-v2-operating-pack.md`
- `scripts/dev/harness-cli-preflight.sh`
- `scripts/harness_config_lint.py`

As a result, `core-harness-flow` cannot legally classify, route, or admit the showcase implementation task. The harness run stops before implementation starts.

## Goal

Add the smallest public-safe harness surface that allows this repo to:

1. admit and route a standard implementation task,
2. run readiness checks locally,
3. execute the deep-learning math foundations project plan through deterministic phases,
4. emit role and skill activation ledgers,
5. run independent critique before optional commit curation.

## Non-Goals

- Port the private OODARIS harness as-is.
- Import private business language, audit rules, or operating assumptions.
- Add Jira, Tempo, BD, OTel, or other private-infrastructure dependencies.
- Add domain-specific councils for retail, pricing, optimization, or private ML governance.
- Support every future task class in the first bootstrap.
- Implement browser orchestration in the first bootstrap.

## Public Safety Constraint

This repo is public and open source. The bootstrap must be authored as if it originated for this repository.

Allowed reuse from private reference material:

- generic file/folder shape,
- generic role names,
- general ideas such as routing manifests, ledgers, readiness checks, and gate reporting,
- public-safe workflow concepts that are not proprietary by wording or dependency.

Disallowed carryover:

- OODARIS-specific business semantics,
- retail or KPI governance language,
- Jira/Tempo process rules,
- council and audit protocols tied to private operations,
- references to private repositories, services, or infra,
- scripts that assume private CLIs or private environments,
- copied prose that is specific to OODARIS operating policy.

The resulting files must read as repo-native documentation and configuration for `mcgill-showcases`.

## Why a Minimal Bootstrap

The current task is to implement one educational Python showcase project. That does not require a large multi-agent platform.

A minimal bootstrap is preferred because it:

- unblocks the requested canonical harness flow,
- keeps maintenance burden small,
- reduces privacy and provenance risk,
- stays aligned with this repository's actual workflow,
- can be expanded later only if the repo develops real need for more orchestration complexity.

## Proposed Scope

### Files to create

```text
.codex/
├── config.toml
├── agents/
│   ├── workflow_orchestrator.toml
│   ├── requirements_clarifier.toml
│   ├── design_strategist.toml
│   ├── tracking_operator.toml
│   ├── backend_executor.toml
│   ├── quality_gate_runner.toml
│   ├── independent_critic.toml
│   └── commit_curator.toml
└── harness/
    └── role-skill-matrix.toml

docs/
└── agents/
    ├── oodaris-harness-v2-operating-pack.md
    ├── harness-evals/
    │   └── README.md
    └── runs/
        └── .gitkeep

scripts/
├── harness_config_lint.py
└── dev/
    └── harness-cli-preflight.sh
```

### Files not included in v1 bootstrap

- `.beads/`
- browser validator role config
- frontend executor role config
- release evidence role config
- domain council role configs
- Jira/Tempo integration scripts
- OTel or local telemetry wiring
- harness eval catalogs beyond a basic README

## Role Set

The bootstrap supports only the roles required by this repo's current implementation workflow.

### Included roles

1. `workflow_orchestrator`
2. `requirements_clarifier`
3. `design_strategist`
4. `tracking_operator`
5. `backend_executor`
6. `quality_gate_runner`
7. `independent_critic`
8. `commit_curator`

### Excluded roles

- `frontend_executor`
- `browser_validator`
- `release_evidence_operator`
- `agentic_ai_architect`
- `ml_scientist`
- `data_engineer`
- `or_researcher`
- `retail_sme`

These are excluded because the immediate task is a backend-style Python project implementation with no browser flow, no release automation, and no private-domain council requirement.

## Supported Task Classes

The minimal bootstrap supports:

- `standard`
- `review_only`
- `closure_only`
- `harness_change`

The minimal bootstrap explicitly does not support:

- `high_impact`

Reason:

- `high_impact` in the full OODARIS system depends on architecture and domain-council behavior that is not appropriate to copy into this public repository.
- If a future task requests `high_impact`, the harness should stop cleanly and require an explicit repo-local extension.

## Phase Model

The public-safe golden workflow for this repo should be:

1. Intake
2. Clarify
3. Design
4. Plan
5. Execute
6. Test
7. Docs
8. Critique
9. Commit optional

This preserves deterministic sequencing while staying appropriate for a public educational repo.

## Skill Routing

### Mandatory phase skills

| Phase | Skill |
|---|---|
| Intake | `core-qna-synthesis` |
| Clarify | `core-ask-questions-if-underspecified` |
| Design | `core-brainstorming` |
| Plan | `core-writing-plans` |
| Execute | `core-executing-plans` |
| Implementation detail safety | `core-test-driven-development` |
| Commit quality | `eng-conventional-commit-helper` |

### Skill routing rules

- User-requested skills must be honored or explicitly skipped with a reason.
- The most specific applicable skill should be chosen.
- If no domain-specific skill exists, the harness uses the generic phase skill.
- Quality, critique, and readiness are command-driven rather than skill-driven in this bootstrap.

## Routing Manifest Requirements

The file `.codex/harness/role-skill-matrix.toml` must define:

- policy settings,
- supported roles,
- supported task classes,
- phase-to-skill routing,
- role activation triggers,
- required ledger sections.

It should remain intentionally smaller than the private reference manifest.

## Role Activation Rules

### `standard`

Required roles:

- `workflow_orchestrator`
- `requirements_clarifier`
- `design_strategist`
- `tracking_operator`
- `quality_gate_runner`
- `independent_critic`

Conditional roles:

- `backend_executor`
- `commit_curator`

### `review_only`

Required roles:

- `workflow_orchestrator`
- `requirements_clarifier`
- `tracking_operator`
- `quality_gate_runner`
- `independent_critic`

Optional roles:

- `design_strategist`

Forbidden roles:

- `backend_executor`
- `commit_curator`

### `closure_only`

Required roles:

- `workflow_orchestrator`
- `tracking_operator`
- `quality_gate_runner`
- `independent_critic`
- `commit_curator`

Forbidden roles:

- `backend_executor`

### `harness_change`

Required roles:

- `workflow_orchestrator`
- `requirements_clarifier`
- `design_strategist`
- `tracking_operator`
- `quality_gate_runner`
- `independent_critic`

Conditional roles:

- `backend_executor`
- `commit_curator`

## Current Task Routing

For the approved task of implementing `deep-learning-math-foundations-showcase`, the harness should classify the work as:

- `standard`

Activated roles for this task:

- `workflow_orchestrator`
- `requirements_clarifier`
- `design_strategist`
- `tracking_operator`
- `backend_executor`
- `quality_gate_runner`
- `independent_critic`

Optional later:

- `commit_curator` if the user asks for a commit.

## Readiness Checks

The script `scripts/dev/harness-cli-preflight.sh` should run lightweight public-safe checks only.

### Required checks

1. Required harness files exist.
2. Required commands exist:
   - `python3`
   - `uv`
   - `git`
   - `rg`
3. The target plan file exists:
   - `plans/deep-learning-math-foundations-showcase.md`
4. The target design spec exists:
   - `docs/superpowers/specs/2026-03-13-deep-learning-showcase-series-design.md`

### Explicitly excluded checks

- `bd`
- `acli`
- `tempo`
- auth checks
- external service status checks

## Lint Invariants

The script `scripts/harness_config_lint.py` should validate only repo-local invariants.

### Required validations

1. `.codex/config.toml` exists and parses.
2. `.codex/harness/role-skill-matrix.toml` exists and parses.
3. All roles referenced in the routing manifest have matching files under `.codex/agents/`.
4. `docs/agents/oodaris-harness-v2-operating-pack.md` exists.
5. `docs/agents/harness-evals/README.md` exists.
6. Supported task classes are limited to the intended public-safe set.
7. The operating pack includes required public-safe markers:
   - public-safe
   - minimal bootstrap
   - unsupported `high_impact`
   - no private infra dependency
8. No private-only CLI names or private service references appear in:
   - `.codex/config.toml`
   - `.codex/harness/role-skill-matrix.toml`
   - `docs/agents/oodaris-harness-v2-operating-pack.md`
   - `scripts/dev/harness-cli-preflight.sh`

## Operating Pack Content

The file `docs/agents/oodaris-harness-v2-operating-pack.md` should explain:

- that this is a public-safe harness baseline,
- decision precedence for this repo,
- supported task classes,
- supported roles,
- supported skill phases,
- readiness and lint expectations,
- role/skill activation ledger requirements,
- run ledger expectations,
- stop conditions when the bootstrap is too small for a requested task.

The operating pack should make it explicit that:

- advanced OODARIS features are intentionally absent,
- missing private integrations are not bugs,
- future expansion requires repo-local design approval.

## Run Ledger Design

The directory `docs/agents/runs/` should hold lightweight run evidence files.

Each run ledger should capture:

- task instruction,
- task classification,
- file scope,
- role activation ledger,
- skill activation ledger,
- gate status,
- commands run,
- evidence collected,
- critic verdict,
- final status.

This keeps orchestration evidence auditable without introducing private operational tooling.

## Context Pack Requirements

Each major run summary should carry a lightweight context pack containing:

1. objective and success criteria,
2. active plan path,
3. file scope,
4. current working tree status,
5. tests and verification expectations,
6. blockers and stop conditions.

This is sufficient for a single-repo educational workflow.

## Tracking Role Scope

`tracking_operator` in this repo should remain lightweight.

It should:

- maintain the current task summary,
- track file claims inside run ledgers,
- record plan progress and gate state,
- avoid external ticketing dependencies,
- avoid pretending to be a program-management system.

It should not:

- assume BD exists,
- assume Jira exists,
- create external audit artifacts.

## Backend Executor Scope

`backend_executor` in this repo means Python implementation and test execution for project code under `projects/`.

It should:

- implement tasks from approved plans,
- follow TDD when requested by the routing manifest,
- keep changes scoped to claimed files,
- run local verification before handing off to critique.

## Quality Gate Runner Scope

`quality_gate_runner` should verify:

- project tests,
- project lint and formatting checks,
- project type checks,
- project artifact verification scripts,
- any explicit plan acceptance commands.

For the current showcase plan, this should eventually include:

- `uv run pytest`
- `uv run python scripts/run_showcase.py`
- `uv run python scripts/verify_artifacts.py`
- `make quality`

within the target project directory.

## Independent Critic Scope

`independent_critic` should provide a severity-ranked review before closure or commit.

Its job in this public bootstrap is not to emulate a large council. It should:

- review correctness,
- identify regressions or missing tests,
- call out mismatches with the plan,
- block closure on unresolved serious issues.

## Commit Curator Scope

`commit_curator` remains optional and only activates when:

- the user explicitly asks for a commit, or
- the run reaches a deliberate landing step and commit behavior is approved.

This respects the repo-wide rule against committing without explicit user approval.

## Acceptance Criteria

The bootstrap design is complete when implementation can deliver:

1. The required harness file set exists in this repo.
2. Preflight passes without private CLI dependencies.
3. Harness lint passes with repo-local invariants only.
4. A `standard` task can be classified and routed deterministically.
5. The current showcase plan can be executed through the supported role set.
6. Role and skill activation ledgers are produced.
7. The bootstrap stops cleanly for unsupported task classes instead of improvising.

## Risks and Mitigations

### Risk: accidental private-policy leakage

Mitigation:

- write all public bootstrap files from scratch,
- keep language repo-native,
- lint for private CLI and private reference names.

### Risk: bootstrap is too small for future tasks

Mitigation:

- document unsupported cases explicitly,
- fail clearly on unsupported task classes,
- expand only through a new public design review.

### Risk: harness overhead exceeds repo value

Mitigation:

- keep the role set small,
- avoid external systems,
- keep evidence lightweight and local.

### Risk: role names imply capabilities the repo does not really have

Mitigation:

- document each role's narrow repo-specific scope,
- avoid aspirational private-platform wording.

## Suggested Delivery Order

### Phase 1

Create repo-local harness directories and configs:

- `.codex/config.toml`
- `.codex/agents/*.toml`
- `.codex/harness/role-skill-matrix.toml`

### Phase 2

Add operating docs and lightweight eval/run directories:

- `docs/agents/oodaris-harness-v2-operating-pack.md`
- `docs/agents/harness-evals/README.md`
- `docs/agents/runs/.gitkeep`

### Phase 3

Add readiness and lint scripts:

- `scripts/dev/harness-cli-preflight.sh`
- `scripts/harness_config_lint.py`

### Phase 4

Run the bootstrap through preflight and lint, then resume the showcase task under the new harness.

## Open Review Questions

These should be confirmed before implementation starts:

1. Should the public docs keep the name `oodaris-harness-v2-operating-pack.md` for compatibility, or should that file explain the compatibility reason and the public-safe differences explicitly?
2. Do you want run ledgers committed in-repo under `docs/agents/runs/`, or only generated locally and ignored by default?
3. Should the minimal bootstrap add a root-level `AGENTS.md` for repo-local policy clarity, or remain driven only by `.codex/` and docs for now?

## Recommended Next Step

After this spec is approved, create a focused implementation plan for the harness bootstrap itself before touching the showcase project code.
