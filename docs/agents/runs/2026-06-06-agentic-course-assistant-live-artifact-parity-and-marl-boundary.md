# Agentic Course Assistant Live Artifact Parity And MARL Boundary Wave

## 1) Intake Synthesis

- Primary instruction: use `core-harness-flow` with multiple subagents to:
  - add live artifact parity for the OpenAI Agents SDK path so a hosted run writes trace, resource, and eval surfaces comparable to the offline path,
  - add one explicit "why MARL is deferred" note to the design/spec layer,
  - keep MARL positioned as a separate showcase rather than stretching the adaptive RL showcase past its teaching boundary.
- Delivery intent: improve teaching honesty, preserve the offline-first classroom contract, and keep the repo's RL showcase boundaries conceptually clean.

## 2) AGENTS Applied

- Applied policy path: `AGENTS.md`
- Harness policy sources:
  - `.codex/config.toml`
  - `.codex/harness/role-skill-matrix.toml`
  - `docs/agents/oodaris-harness-v2-operating-pack.md`
- Additional source-of-truth docs inspected:
  - `docs/superpowers/specs/2026-06-06-adaptive-course-assistant-rl-showcase-design.md`
  - `docs/superpowers/specs/2026-06-06-learning-agents-showcase-design.md`
  - `plans/adaptive-course-assistant-rl-showcase.md`
  - `plans/learning-agents-showcase.md`

## 3) Routing Manifest And Task Classification

- Task classification: `standard`
- Run mode: `long_running`
- Evaluator cadence: `per_phase`
- Closure intent: `OFFLINE_EVIDENCE_ONLY`
- Repo-inferred triggers:
  - `python_project_work`
  - `docs_only_or_docs_plus_small_code`
  - `tests_required`
  - `optional_live_sdk_path`

## 4) Role Activation Ledger

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
  - `requirements_clarifier`
  - `design_strategist`
  - `tracking_operator`
  - `backend_executor`
  - `independent_critic`
- `roles_skipped_with_reason`:
  - `quality_gate_runner`: the orchestrator ran the required local gates directly for this scoped wave
  - `commit_curator`: skipped because no commit was requested

## 5) Skill Activation Ledger

- `user_requested_skills`:
  - `core-harness-flow`
- `phase_required_skills`:
  - `core-qna-synthesis`
  - `core-writing-plans`
  - `core-executing-plans`
  - `core-test-driven-development`
- `domain_skills_considered`:
  - `core-qna-synthesis`
  - `code-review`
- `skills_invoked`:
  - `core-harness-flow`
  - `core-qna-synthesis`
  - `core-writing-plans`
  - `core-executing-plans`
  - `core-test-driven-development`
- `skills_skipped_with_reason`:
  - `core-ask-questions-if-underspecified`: repo discovery plus the user prompt were sufficient for safe defaults
  - `eng-conventional-commit-helper`: skipped because no commit handling was requested

## 6) Subagent Contributions

- `requirements_clarifier`:
  - defined the minimum acceptable parity surface for the hosted OpenAI path,
  - pushed for a separate output namespace so hosted artifacts would not overwrite the offline bundle,
  - reinforced that this change must not blur the adaptive RL learning-agent boundary.
- `design_strategist`:
  - recommended fixing the parity gap inside `agentic-course-assistant-showcase`,
  - recommended app-owned artifact writing rather than pretending to recreate OpenAI platform traces,
  - recommended positioning MARL in the separate `learning-agents-showcase`.
- `tracking_operator`:
  - classified the wave as `standard`, `long_running`, `per_phase`,
  - identified the separate-capstone wording drift in `learning-agents-showcase`,
  - recommended a critic checkpoint before closure.
- `independent_critic`:
  - first pass returned `no` because the initial live trace implied hosted triage/tool steps that were actually synthesized locally,
  - second pass returned `yes` after the trace, verifier, tests, and docs were updated to make the teaching-adapter boundary explicit.

## 7) File Claims

- Hosted OpenAI artifact-parity surfaces:
  - `projects/agentic-course-assistant-showcase/src/agentic_course_assistant/artifacts.py`
  - `projects/agentic-course-assistant-showcase/src/agentic_course_assistant/artifact_contract.py`
  - `projects/agentic-course-assistant-showcase/src/agentic_course_assistant/openai_agents_example.py`
  - `projects/agentic-course-assistant-showcase/src/agentic_course_assistant/openai_live_artifacts.py`
  - `projects/agentic-course-assistant-showcase/scripts/run_openai_showcase.py`
  - `projects/agentic-course-assistant-showcase/scripts/verify_artifacts.py`
  - `projects/agentic-course-assistant-showcase/tests/test_openai_live_artifacts.py`
  - `projects/agentic-course-assistant-showcase/Makefile`
  - `projects/agentic-course-assistant-showcase/README.md`
  - `projects/agentic-course-assistant-showcase/docs/lab-guide.md`
  - `projects/agentic-course-assistant-showcase/docs/sdk-comparison.md`
- Design/spec boundary surfaces:
  - `docs/superpowers/specs/2026-06-06-adaptive-course-assistant-rl-showcase-design.md`
  - `plans/adaptive-course-assistant-rl-showcase.md`
  - `docs/superpowers/specs/2026-06-06-learning-agents-showcase-design.md`
  - `plans/learning-agents-showcase.md`
  - `projects/learning-agents-showcase/README.md`

## 8) Implementation Summary

Implemented changes:

- Added an opt-in hosted OpenAI artifact-parity bundle:
  - new writer module `src/agentic_course_assistant/openai_live_artifacts.py`
  - new script `scripts/run_openai_showcase.py`
  - new Make targets `run-openai` and `verify-openai`
- Kept the raw OpenAI Agents SDK example available while making SDK imports lazy:
  - `openai_agents_example.py` no longer forces the optional dependency at import time
  - hosted specialist execution is separated from the offline-first default path
- Reused the offline teaching contract for the hosted bundle while adding an explicit honesty boundary:
  - the live bundle writes a separate namespace at `artifacts/live_openai/`
  - the emitted trace and markdown now say that a local teaching adapter performs intent selection and course-catalog grounding before the hosted specialist call
  - the artifact verifier now enforces the live-trace honesty note when `trace_source=hosted_response_with_local_teaching_adapter`
- Added design/spec-layer MARL deferral guidance:
  - adaptive RL spec now includes an explicit "Why MARL Is Deferred" section
  - adaptive RL plan now treats multi-agent drift as a re-scope trigger
  - `learning-agents-showcase` docs now position MARL as a future lane in the separate capstone rather than an implicit dependency of the current adaptive learning path

## 9) Gate Matrix

| Gate | Status | Evidence |
|---|---|---|
| Harness preflight | Verified | `bash scripts/dev/harness-cli-preflight.sh` passed |
| Harness config lint | Verified | `python3 scripts/harness_config_lint.py` passed |
| Agentic project lint | Verified | `uv run ruff check src tests scripts` passed |
| Agentic project types | Verified | `uv run mypy src tests scripts` passed |
| Agentic project tests | Verified | `uv run pytest` passed with `33` tests |
| Agentic offline smoke | Verified | `make smoke` passed |
| Agentic offline artifact verify | Verified | `make verify` passed |
| Hosted negative-path behavior | Verified | `uv run python scripts/run_openai_showcase.py ...` failed clearly with missing `OPENAI_API_KEY` |
| Learning-agents scaffold checks | Verified | `cd projects/learning-agents-showcase && make check` passed |
| Root docs build | Verified | `make docs-check` passed with existing non-blocking nav warnings plus the upstream Material warning |
| Root artifact sweep | Verified | `make verify` passed |
| Diff hygiene | Verified | `git diff --check` was clean |
| Independent critique | Verified after remediation | first pass `no`, second pass `yes` after teaching-adapter honesty fixes |

## 10) Verification Evidence

Verified commands:

- `bash scripts/dev/harness-cli-preflight.sh`
- `python3 scripts/harness_config_lint.py`
- `cd projects/agentic-course-assistant-showcase && uv run ruff check src tests scripts`
- `cd projects/agentic-course-assistant-showcase && uv run mypy src tests scripts`
- `cd projects/agentic-course-assistant-showcase && uv run pytest`
- `cd projects/agentic-course-assistant-showcase && make smoke`
- `cd projects/agentic-course-assistant-showcase && make verify`
- `cd projects/agentic-course-assistant-showcase && uv run python scripts/run_openai_showcase.py --question "How should I debug a suspicious validation score?"`
- `cd projects/learning-agents-showcase && make check`
- `cd /Users/fatih/dev/mcgill-showcases && make docs-check`
- `cd /Users/fatih/dev/mcgill-showcases && make verify`
- `cd /Users/fatih/dev/mcgill-showcases && git diff --check`

Observed outcomes:

- All repo-local required gates passed.
- The hosted OpenAI command still requires credentials, but it now fails clearly and predictably without affecting the offline lane.
- The initial critic blocker about misrepresenting hosted runtime behavior was fixed and re-reviewed successfully.

## 11) Residual Risks

- A real hosted end-to-end OpenAI run was not executed because the current local environment does not provide `OPENAI_API_KEY`; the hosted lane is verified through lazy-import tests, contract validation, and negative-path behavior only.
- The artifact verifier enforces the live trace honesty boundary in JSON, while the markdown runtime note is enforced by regression test rather than the shared verifier itself.
- The broader untracked `projects/learning-agents-showcase/**` scaffold remains a future lane and was not expanded beyond the doc-level boundary updates in this wave.

## 12) Final Verdict

- Verdict: `GO`
- Scope closure: the hosted OpenAI path now has an opt-in artifact bundle that is comparable to the offline teaching contract without pretending to be a raw OpenAI trace, and the MARL deferral boundary is explicit in the design/spec layer as well as the student-facing docs.
