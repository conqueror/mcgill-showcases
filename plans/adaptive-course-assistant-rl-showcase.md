# Adaptive Course Assistant RL Showcase Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `core-executing-plans` to implement this plan task-by-task.

## Goal

Build `projects/adaptive-course-assistant-rl-showcase` as a script-first public showcase about
learned pedagogical intervention in an agentic tutoring workflow. The bridge runs from a
deterministic assistant front-end to contextual bandits, tabular RL, and optional DRL.

## Design Reference

- `docs/superpowers/specs/2026-06-06-adaptive-course-assistant-rl-showcase-design.md`

## Scope

In scope:

- scaffold the new showcase project,
- implement a synthetic tutoring environment with bounded episodes,
- implement heuristic intervention policies plus a real contextual bandit,
- implement tabular Q-learning, SARSA, and a small REINFORCE path,
- add optional DQN and PPO bridge paths,
- export a learned policy into deterministic intervention artifacts,
- generate stable artifacts and student-facing docs,
- wire root-level docs, CI, Makefile, and issue-template integration.

Out of scope:

- live student data,
- online learning,
- production deployment,
- mandatory hosted-agent execution,
- full MARL or RLHF implementation.

## Assumptions

- The repo wants this as a new sibling showcase, not a retrofit of an existing one.
- The core path must run without API keys or network dependency.
- The optional DRL path may use extra dependencies, but the default path must remain laptop-friendly.
- The environment should be small enough to keep tabular methods readable.
- The deterministic assistant is fixed content infrastructure; the learned layer chooses intervention type or pacing.
- If MARL is ever explored, it should move to a separate showcase rather than widening this plan.

## Stop Conditions

Stop and re-scope if:

- the environment state grows beyond a compact teaching example,
- multiple co-adapting learned agents become necessary to explain the lesson,
- the core path requires optional DRL dependencies,
- the project cannot stay distinct from the existing agentic-course-assistant and student-support RL showcases,
- artifact requirements become unstable or too heavy for a deterministic verifier,
- root integration conflicts with unrelated in-flight changes.

## Success Criteria

- `cd projects/adaptive-course-assistant-rl-showcase && make smoke` passes once implemented.
- `cd projects/adaptive-course-assistant-rl-showcase && make run` passes once implemented.
- `cd projects/adaptive-course-assistant-rl-showcase && make test` passes once implemented.
- `cd projects/adaptive-course-assistant-rl-showcase && make check` passes once implemented.
- `cd projects/adaptive-course-assistant-rl-showcase && make verify` passes once implemented.
- Core artifacts prove the contextual-bandit, intervention-policy MDP, tabular-RL, reward-audit, evaluation, and policy-export lanes.
- Optional artifacts prove DQN and PPO on the same environment family.
- Root docs and CI integrate the project cleanly.

## Phase Plan

### Phase 1: Project Skeleton

1. Create `README.md`, `Makefile`, `pyproject.toml`, `docs/`, `scripts/`, `src/`, `tests/`, `artifacts/manifest.json`, and placeholders.
2. Add project-local quality commands and verifier wiring.
3. Write the initial README sections required by `docs/new-showcase-playbook.md`.

### Phase 2: Environment And Heuristic Baselines

4. Implement the tutoring episode schema, transition logic, and heuristic intervention policies.
5. Add MDP description artifacts and sample episodes.
6. Add tests for transitions, terminal rules, and deterministic seeds.

### Phase 3: Contextual Bandit

7. Implement the one-step contextual bandit over first-turn intervention choice.
8. Add regret, action mix, and first-turn performance artifacts.
9. Add tests proving the bandit is contextual and one-step.

### Phase 4: Tabular RL

10. Implement Q-learning and SARSA.
11. Add training curves, Q-table outputs, and scenario evaluation.
12. Add tests for convergence direction, shape, and artifact contract expectations.

### Phase 5: Policy-Gradient And DRL Bridge

13. Implement REINFORCE.
14. Add a Gymnasium adapter for the same environment family.
15. Implement optional DQN and PPO runners with Stable-Baselines3.
16. Add cross-family comparison artifacts and tests.

### Phase 6: Agent Bridge And Governance

17. Export the learned policy into deterministic assistant-intervention artifacts.
18. Add reward-audit outputs, offline evaluation outputs, and deployment recommendation docs.
19. Add student-facing bridge docs that explicitly connect Q-learning, DQN, policy gradients, actor-critic, and PPO.

### Phase 7: Root Integration And Final Verification

20. Update root `README.md`, `docs/getting-started.md`, `docs/learning-path.md`, `docs/aspect-coverage-matrix.md`, root `Makefile`, `.github/workflows/ci.yml`, and issue templates.
21. Run project-local and root-level checks.
22. Request an independent critic review before closure.

## Verification Strategy

Project-local checks after implementation:

- `make sync`
- `make smoke`
- `make test`
- `make verify`
- `make check`
- `make sync-drl && make run-drl-optional`

Repo-level checks after integration:

- root CI matrix coverage for the new project
- relevant docs or catalog verification commands already used in this repo

## Residual Risks To Watch

- overlap with existing showcases causing student confusion,
- optional DRL path taking too long on laptops,
- state explosion undermining tabular readability,
- policy export being too abstract if not tied to a concrete intervention artifact,
- documentation drift between core and optional artifact expectations.
