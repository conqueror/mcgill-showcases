# OODARIS Harness V2 Operating Pack

Status: public-safe minimal bootstrap
Scope: `mcgill-showcases`

## Purpose

This document defines the repo-local harness-lite workflow used in `mcgill-showcases`.

It exists for compatibility with the canonical `core-harness-flow` skill while staying appropriate for a public educational repository.

This operating pack is:

- public-safe,
- a minimal bootstrap,
- intentionally smaller than a private multi-agent control plane,
- free of private infra dependencies.

## Decision Precedence

Use this order during harness-lite runs:

1. Closest `AGENTS.md` by path.
2. Approved design specs under `docs/superpowers/specs/`.
3. Approved implementation plans under `plans/`.
4. `.codex/harness/role-skill-matrix.toml`.
5. Repo-local verification commands and documented quality gates.

## Supported Task Classes

This bootstrap supports:

- `standard`
- `review_only`
- `closure_only`
- `harness_change`

This bootstrap does not support:

- unsupported `high_impact`

If a task requires unsupported `high_impact` behavior, stop and design a repo-local extension first.

## Supported Roles

The active public-safe role set is:

1. `workflow_orchestrator`
2. `requirements_clarifier`
3. `design_strategist`
4. `tracking_operator`
5. `backend_executor`
6. `quality_gate_runner`
7. `independent_critic`
8. `commit_curator`

## Golden Workflow

The harness-lite golden workflow for this repo is:

1. Intake
2. Clarify
3. Design
4. Plan
5. Execute
6. Test
7. Docs
8. Critique
9. Commit optional

## Skill Phases

Mandatory phase skills:

- Intake: `core-qna-synthesis`
- Clarify: `core-ask-questions-if-underspecified`
- Design: `core-brainstorming`
- Plan: `core-writing-plans`
- Execute: `core-executing-plans`
- Implementation detail safety: `core-test-driven-development`
- Commit quality: `eng-conventional-commit-helper`

## Readiness Checks

Run these before a harness-lite implementation run:

1. `bash scripts/dev/harness-cli-preflight.sh`
2. `python3 scripts/harness_config_lint.py`

The bootstrap intentionally does not depend on:

- private CLIs,
- private auth checks,
- external issue trackers,
- external audit services.

## Run Ledgers

Harness-lite runs should write local evidence to `docs/agents/runs/`.

Each run ledger should record:

- task instruction,
- task classification,
- file scope,
- gate status,
- commands run,
- evidence collected,
- final verdict.

## Activation Ledgers

Every major harness summary must include:

- Role Activation Ledger
- Skill Activation Ledger

Role Activation Ledger fields:

- `roles_considered`
- `roles_activated`
- `roles_skipped_with_reason`

Skill Activation Ledger fields:

- `user_requested_skills`
- `phase_required_skills`
- `domain_skills_considered`
- `skills_invoked`
- `skills_skipped_with_reason`

## Tracking Scope

Tracking in this repo is local only.

The harness-lite tracking role should:

- track plan progress,
- maintain file claims,
- record gate state,
- write local run evidence.

The harness-lite tracking role should not:

- assume a private task tracker exists,
- assume an external audit system exists,
- create external audit artifacts,
- depend on non-repo execution databases.

## Quality Gates

Closure requires repo-local evidence only.

Expected evidence includes:

- tests pass,
- project run commands pass,
- artifact verification passes,
- docs referenced by the plan exist,
- critique has been completed.

## Independent Critique

The `independent_critic` role is mandatory before marking an implementation run complete.

The critic should:

- identify correctness issues,
- identify regressions,
- identify missing tests or documentation,
- block closure on unresolved serious findings.

## Commit Policy

`commit_curator` is optional.

It activates only when:

- the user explicitly requests a commit, or
- the landing step is approved.

This repo does not permit commits by default during harness-lite runs.

## Stop Conditions

Stop the run when:

- required harness files are missing,
- readiness checks fail,
- the task class is unsupported,
- required ledgers are missing,
- user-requested skills are skipped without a reason,
- critique evidence is missing before closure.

## Expansion Rule

This bootstrap is intentionally small.

Advanced OODARIS features are intentionally absent.
Missing private integrations are not bugs.
Future expansion requires repo-local design approval before new harness behavior is added.
