# RL Showcase Review Remediation

## Intake

- Primary instruction: review the showcase in detail with `code-review`, `eng-algorithm-review`, and `eng-code-review-playbook`, then use `core-harness-flow` with multiple subagents to make confirmed improvements.
- Active repo policy path: `AGENTS.md`
- Harness-lite policy sources:
  - `.codex/harness/role-skill-matrix.toml`
  - `docs/agents/oodaris-harness-v2-operating-pack.md`

## Routing Manifest and Task Classification

- Task classification: `standard`
- Run mode: `single_session`
- Rationale:
  - educational repo scope,
  - repo-local code and docs only,
  - no production mutation,
  - multiple verified findings across two showcase slices.

## Role Activation Ledger

- `roles_considered`:
  - `workflow_orchestrator`
  - `requirements_clarifier`
  - `design_strategist`
  - `tracking_operator`
  - `backend_executor`
  - `quality_gate_runner`
  - `independent_critic`
  - `commit_curator`
- `roles_activated`:
  - `workflow_orchestrator`
  - `tracking_operator`
  - `backend_executor`
  - `quality_gate_runner`
  - `independent_critic`
- `roles_skipped_with_reason`:
  - `requirements_clarifier`: repo context and confirmed findings were already sufficient
  - `design_strategist`: the remediation scope was narrow and evidence-driven
  - `commit_curator`: no commit was requested

## Skill Activation Ledger

- `user_requested_skills`:
  - `code-review`
  - `eng-algorithm-review`
  - `eng-code-review-playbook`
  - `core-harness-flow`
- `phase_required_skills`:
  - `core-qna-synthesis`
  - `core-writing-plans`
  - `core-executing-plans`
  - `core-test-driven-development`
- `domain_skills_considered`:
  - `eng-python-engineer`
  - `humanizer`
- `skills_invoked`:
  - `code-review`
  - `eng-algorithm-review`
  - `eng-code-review-playbook`
  - `core-harness-flow`
  - `core-executing-plans`
  - `core-test-driven-development`
- `skills_skipped_with_reason`:
  - `core-ask-questions-if-underspecified`: findings were already concrete and locally reproducible
  - `eng-conventional-commit-helper`: no commit work was requested
  - `humanizer`: the touched docs were already readable; changes stayed focused on correctness and contract honesty

## Review Findings That Triggered Remediation

- Adaptive RL showcase:
  - false resolution path for `assign_targeted_practice` while intent remained uncertain,
  - repeated deterministic rollouts being presented without enough honesty in the evaluation story,
  - comparison/export artifacts needing stronger boundary language.
- Learning-agents showcase:
  - manifest tracking gap,
  - root integration gap,
  - project-level readiness/docs gap.

## Commands And Evidence

Readiness:

- `bash scripts/dev/harness-cli-preflight.sh`
- `python3 scripts/harness_config_lint.py`

Review verification:

- `cd projects/adaptive-course-assistant-rl-showcase && uv run pytest`
- `cd projects/adaptive-course-assistant-rl-showcase && uv run python scripts/run_showcase.py --quick && uv run python scripts/verify_artifacts.py`
- `cd projects/adaptive-course-assistant-rl-showcase && uv run mypy src tests scripts && uv run ruff check src tests scripts`
- `cd projects/agentic-course-assistant-showcase && uv run pytest`
- `cd projects/learning-agents-showcase && uv run pytest`
- `cd projects/learning-agents-showcase && uv run ruff check src tests scripts`

Post-remediation verification:

- `cd projects/adaptive-course-assistant-rl-showcase && uv run pytest`
- `cd projects/adaptive-course-assistant-rl-showcase && uv run mypy src tests scripts`
- `cd projects/adaptive-course-assistant-rl-showcase && uv run ruff check src tests scripts`
- `cd projects/adaptive-course-assistant-rl-showcase && uv run python scripts/run_showcase.py --quick`
- `cd projects/adaptive-course-assistant-rl-showcase && uv run python scripts/verify_artifacts.py`
- `cd projects/learning-agents-showcase && uv run pytest -q`
- `cd projects/learning-agents-showcase && uv run ruff check src tests scripts`
- `cd projects/learning-agents-showcase && uv run mypy src tests scripts`
- `cd projects/learning-agents-showcase && make smoke`
- `cd projects/learning-agents-showcase && make verify`
- `git check-ignore -v projects/learning-agents-showcase/artifacts/manifest.json`

## Final Status

- Adaptive RL showcase: corrected the false-resolution path and tightened the comparison/export honesty surface.
- Learning-agents showcase: repo-readiness surfaces are now present in this checkout, including docs, trackable manifest support, and root integration.
- Critic re-review still required before closure in this remediation wave.
