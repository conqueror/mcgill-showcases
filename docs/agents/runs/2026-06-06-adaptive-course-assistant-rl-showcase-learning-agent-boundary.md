# Adaptive Course Assistant RL Showcase Learning-Agent Boundary Wave

## 1) Intake Synthesis

- Primary instruction: use `core-qna-synthesis` deep to refine and expand a question set about whether the OpenAI Agents SDK example is a learning agent, whether the adaptive RL showcase is a standalone DRL learning-agent project, whether PPO and related algorithms are used, and what changes would make the learning story more complete.
- Execution instruction: based on those answers, use `core-harness-flow` with subagents to plan, implement, review, and test the resulting changes.
- Delivery intent: improve conceptual honesty, student understanding, and bridge clarity between the agent-framework showcase and the adaptive RL showcase.

## 2) AGENTS Applied

- Applied policy path: `AGENTS.md`
- Harness policy sources:
  - `.codex/config.toml`
  - `.codex/harness/role-skill-matrix.toml`
  - `docs/agents/oodaris-harness-v2-operating-pack.md`
- Showcase playbook:
  - `docs/new-showcase-playbook.md`

## 3) Routing Manifest and Task Classification

- Task classification: `standard`
- Run mode: `single_session`
- Repo-inferred triggers:
  - `python_project_work`
  - `docs_only_or_docs_plus_small_code`
  - `tests_required`
- Boundary decision:
  - keep the relationship between `agentic-course-assistant-showcase` and `adaptive-course-assistant-rl-showcase` conceptual and artifact-driven
  - do not collapse them into a new broader capstone in this wave

## 4) Deep QNA Summary

Refined question set answered in this wave:

1. Is the optional OpenAI Agents SDK example already a learning agent?
2. Is the adaptive RL showcase already a standalone learning-agent project?
3. Does the current repo justify a full “learning agent with DRL” claim?
4. Where do DQN and PPO fit, and how central should they be in the student path?
5. Should MARL be added now, deferred, or explicitly excluded?
6. What is the cleanest way to explain the seam between agent orchestration and learned policy control?
7. Which minimal code or artifact addition would improve the learning experience without turning the repo into a research project?

High-level answers:

- The OpenAI Agents SDK example is an orchestration example, not a learning agent.
- The adaptive RL showcase is standalone within its bounded scope: learned intervention choice around a deterministic assistant.
- The current adaptive project supports a careful “bounded learning-agent story with optional DRL bridge” claim, not a full end-to-end DRL agent claim.
- PPO belongs in the optional DRL comparison lane, not at the center of the first learning path.
- MARL should be explicitly deferred because it changes the teaching problem too much for this showcase.
- The best improvement is a thin bridge layer, not a third large showcase.

## 5) Role Activation Ledger

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
  - `quality_gate_runner`: not separately activated because the orchestrator ran the required local gates directly
  - `commit_curator`: skipped because no commit was requested

## 6) Skill Activation Ledger

- `user_requested_skills`:
  - `core-qna-synthesis`
  - `core-harness-flow`
- `phase_required_skills`:
  - `core-qna-synthesis`
  - `core-writing-plans`
  - `core-executing-plans`
  - `core-test-driven-development`
  - `eng-python-engineer`
- `skills_invoked`:
  - `core-qna-synthesis`
  - `core-harness-flow`
  - `core-writing-plans`
  - `core-executing-plans`
  - `core-test-driven-development`
  - `eng-python-engineer`
- `skills_skipped_with_reason`:
  - `eng-conventional-commit-helper`: skipped because no commit handling was requested
  - `core-ask-questions-if-underspecified`: skipped because repo discovery plus the user prompt gave enough context for a safe default

## 7) File Claims

- Adaptive showcase implementation and docs:
  - `projects/adaptive-course-assistant-rl-showcase/src/**`
  - `projects/adaptive-course-assistant-rl-showcase/scripts/**`
  - `projects/adaptive-course-assistant-rl-showcase/tests/**`
  - `projects/adaptive-course-assistant-rl-showcase/docs/**`
  - `projects/adaptive-course-assistant-rl-showcase/README.md`
  - `projects/adaptive-course-assistant-rl-showcase/Makefile`
  - `projects/adaptive-course-assistant-rl-showcase/artifacts/manifest.json`
- Agentic showcase boundary docs:
  - `projects/agentic-course-assistant-showcase/README.md`
  - `projects/agentic-course-assistant-showcase/docs/sdk-comparison.md`
- Root docs:
  - `README.md`
  - `docs/learning-path.md`
  - `docs/aspect-coverage-matrix.md`
- Run ledger:
  - `docs/agents/runs/2026-06-06-adaptive-course-assistant-rl-showcase-learning-agent-boundary.md`

## 8) Implementation Summary

Implemented changes:

- Added a canonical student-facing bridge artifact:
  - `projects/adaptive-course-assistant-rl-showcase/scripts/run_learning_agent_story.py`
  - `projects/adaptive-course-assistant-rl-showcase/src/adaptive_course_assistant_rl/learning_agent_story.py`
  - generated artifact: `artifacts/bridge/learning_agent_story.md`
- Wired the bridge artifact into the core showcase path and artifact contract.
- Expanded adaptive-project docs to say plainly:
  - the learned part is intervention choice, not answer generation
  - DQN and PPO are optional DRL bridge algorithms
  - MARL is out of scope for this showcase
  - simulator limits matter
- Expanded the agentic-course-assistant docs to say plainly:
  - the optional OpenAI Agents SDK example is not itself a learning agent
  - the adaptive RL showcase is the next project for the policy-learning layer
- Tightened root learning-path and coverage docs to make the two-showcase bridge explicit.

## 9) Gate Matrix

| Gate | Status | Evidence |
|---|---|---|
| Intake | Verified | repo files plus official OpenAI docs were reviewed |
| QNA synthesis | Verified | refined question set and bounded answers derived from live repo state |
| Implementation | Verified | bridge artifact, doc clarifications, and root learning-path updates landed |
| Tests | Verified | adaptive project Ruff, mypy, pytest, smoke, verify, verify-full, sibling smoke/verify, root docs-check, root verify |
| Critique | Verified after remediation | initial independent critic found quickstart and verifier-contract issues; follow-up critic returned `GO` with no blocking findings |
| Commit | Not run | no commit requested |

## 10) Verification Evidence

Verified commands:

- `bash scripts/dev/harness-cli-preflight.sh`
- `python3 scripts/harness_config_lint.py`
- `cd projects/adaptive-course-assistant-rl-showcase && uv run ruff check src tests scripts`
- `cd projects/adaptive-course-assistant-rl-showcase && uv run mypy src tests scripts`
- `cd projects/adaptive-course-assistant-rl-showcase && uv run pytest`
- `cd projects/adaptive-course-assistant-rl-showcase && make smoke`
- `cd projects/adaptive-course-assistant-rl-showcase && make run-drl-optional`
- `cd projects/adaptive-course-assistant-rl-showcase && make verify-core`
- `cd projects/adaptive-course-assistant-rl-showcase && make verify`
- `cd projects/adaptive-course-assistant-rl-showcase && make verify-full`
- `cd projects/agentic-course-assistant-showcase && make smoke`
- `cd projects/agentic-course-assistant-showcase && make verify`
- `cd /Users/fatih/dev/mcgill-showcases && make docs-check`
- `cd /Users/fatih/dev/mcgill-showcases && make verify`
- `cd /Users/fatih/dev/mcgill-showcases && git diff --check`

Observed outcomes:

- Harness-lite preflight passed.
- Harness-lite config lint passed.
- Adaptive project Ruff passed.
- Adaptive project mypy passed.
- Adaptive project pytest passed: `20` tests.
- Adaptive `make smoke` passed and generated `artifacts/bridge/learning_agent_story.md`.
- Adaptive `make verify`, `make verify-core`, and `make verify-full` passed.
- Agentic-course-assistant smoke and verify passed.
- Root strict docs build passed with existing informational nav warnings plus the upstream Material warning.
- Root `make verify` passed and selected the stricter adaptive verifier when optional DRL artifacts were present.
- `git diff --check` was clean.

## 11) Subagent Contributions

- `requirements_clarifier`:
  - confirmed that the OpenAI Agents SDK example is not a learning agent and identified the top student-facing conceptual gaps.
- `design_strategist`:
  - recommended a thin bridge layer instead of a broader learning-agents capstone, with `learning_agent_story.md` as the highest-value new artifact.
- `tracking_operator`:
  - recommended a narrow-claim, docs-plus-small-code wave with explicit cross-showcase boundaries and mandatory critic closure.

## 12) Residual Risk Register

- The repo still does not provide a live OpenAI Agents SDK run that writes the same artifact/eval surfaces as the offline path. This was left out of scope for this wave.
- The bridge remains conceptual and artifact-driven rather than a runtime integration that imports the sibling showcase. That is intentional for maintainability and scope control.
- MARL remains deferred by design. That is a clarity choice, not a missing implementation bug.
- Final independent critic posture: `GO`; no blocking findings remained after the quickstart/verify and verifier-path fixes.
