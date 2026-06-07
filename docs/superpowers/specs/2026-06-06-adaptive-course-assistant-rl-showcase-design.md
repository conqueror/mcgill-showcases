# Adaptive Course Assistant RL Showcase Design

## Scope

Build a new showcase project under `projects/adaptive-course-assistant-rl-showcase` that teaches how an agentic tutoring workflow can move from fixed pedagogical intervention heuristics to learned decision policies. The project should stay offline-first and laptop-friendly while making the bridge from contextual bandits and tabular RL to DQN, policy gradients, actor-critic ideas, PPO, and optional agent-runtime integration explicit.

## Primary Learning Outcome

Learners should be able to explain when an agentic course assistant’s intervention policy can be framed as a contextual bandit versus a sequential MDP, compare heuristic and learned policies, inspect the state-action-reward design directly, and understand how a learned controller can be exported back into a deterministic agent workflow.

## Target Audience And Runtime Contract

- Audience level: intermediate
- Primary audience: students who already completed `projects/agentic-course-assistant-showcase` or `projects/student-support-rl-showcase`
- Expected laptop runtime:
  - `make smoke`: under 60 seconds
  - `make run`: under 10 minutes
  - `make run-drl-optional`: allowed to be slower, but still CPU-friendly and reproducible
- Prerequisites:
  - Python 3.11+
  - `uv`
  - basic Python and CSV literacy
  - prior exposure to agent workflows or the existing RL showcase

## Deep QNA Synthesis

### Original Questions

- Can we create a showcase where the course assistant learns with DRL?
- Should that be a new showcase instead of modifying an existing one?
- How would OpenAI Agents SDK or a broader agentic runtime connect to RL?
- What would make the project pedagogically coherent instead of a mash-up?

### Refined Questions

- What is the smallest new showcase that uniquely fills the gap between the deterministic course assistant and the generic student-support RL project?
- Which parts of the agent loop should become the RL decision problem, and which parts should stay deterministic?
- What learning ladder best connects contextual bandits, tabular control, DQN, policy gradients, actor-critic, and PPO in one teaching arc?
- Which artifact surfaces make the “RL for agentic systems” claim inspectable instead of implied?
- Which pieces belong in the required default path, and which should remain optional extensions?

### Pass A: Whole To Parts

This is not a production tutor, and it is not a full multi-agent RL system. The point is narrower:
show how part of an agent workflow becomes a decision problem. That means the project should focus
on one slice of agent behavior, such as hint strategy, retrieval timing, clarification timing,
pacing, or escalation choice, instead of trying to learn everything at once.

### Pass B: Parts To Whole

Because the repo already has:

- `projects/agentic-course-assistant-showcase` for deterministic routing, tools, guardrails, traces, and optional SDK examples,
- `projects/student-support-rl-showcase` for the general RL and DRL ladder in a student-support domain,

this new project must be narrower and more explicit. It should teach policy learning inside an
agent loop. The clearest frame is "learned pedagogical intervention inside a course assistant," not
"another RL course assistant" and not "another generic RL domain."

### Confirmed Facts

- The repo playbook requires new showcases to be script-first, laptop-friendly, and structured around `README.md`, `Makefile`, `pyproject.toml`, `src/`, `scripts/`, `tests/`, `docs/`, and `artifacts/manifest.json`.
- The repo already contains a deterministic agentic course assistant showcase.
- The repo already contains a student-support RL showcase covering contextual bandits, tabular control, and optional DQN/PPO comparison.
- The aspect matrix and learning path already treat those two projects as separate tracks.

### Recommendation

- Create `projects/adaptive-course-assistant-rl-showcase` as a new sibling project.
- Keep the environment synthetic, deterministic by seed, and offline-first.
- Make the policy-learning target explicit: pedagogical intervention choices inside a bounded tutoring episode after the assistant already understands the request.
- Treat OpenAI Agents SDK as an optional export or runtime bridge, not as the training engine.

## Positioning

### Why This Should Be A New Showcase

This project fills a teaching gap between two existing showcases:

- `agentic-course-assistant-showcase` teaches how an agent workflow is structured.
- `student-support-rl-showcase` teaches the RL algorithm ladder in a general student-support domain.

This new project teaches how those two worlds meet. It replaces fixed pedagogical intervention
heuristics with a learned policy while keeping the governance and offline evaluation easy to
inspect.

### What It Must Uniquely Teach

- The conversion from an agent workflow into an RL problem definition.
- The distinction between one-shot intervention decisions and multi-turn tutoring control.
- The mapping from learned policy outputs back into deterministic assistant actions.
- The tradeoff between pedagogical helpfulness, grounding quality, escalation, and turn efficiency.

### Duplication Boundary

This project should not re-own the following:

- agent routing, tool design, traces, guardrails, and SDK concept coverage already taught in `projects/agentic-course-assistant-showcase`,
- the broad RL survey and generic student-support algorithm ladder already taught in `projects/student-support-rl-showcase`.

Instead, it should assume a deterministic assistant front-end already exists and teach only the intervention-policy layer that sits around that assistant.

### Non-Goals

- No live student data.
- No online learning from real users.
- No true multi-agent reinforcement learning in v1.
- No RLHF implementation.
- No API-key-dependent default path.
- No production deployment claim.

### Why MARL Is Deferred

MARL is deferred on purpose because it changes the teaching problem.

This showcase is about one learned intervention controller wrapped around a deterministic assistant.
Once multiple learning policies co-adapt, students have to reason about non-stationarity, credit
assignment across agents, coordination protocols, and a different state-action-reward contract.
That is worth teaching, but it is not the bounded lesson this project is built for.

If the repo later wants MARL, it should live in a separate showcase rather than stretching this
project past its teaching boundary.

## Learning Outcomes

By the end of this project, students should be able to:

- explain why a first intervention choice can be modeled as a contextual bandit,
- explain why multi-turn tutoring requires an MDP,
- define a usable state, action, reward, and horizon for a tutoring assistant,
- compare heuristic policies with contextual-bandit, tabular, and deep-RL policies,
- explain the ladder from Q-learning to DQN to policy gradients to actor-critic to PPO,
- evaluate a learned agent policy beyond total reward alone,
- identify reward-design failure modes specific to agentic tutoring,
- export a learned policy into a deterministic intervention-policy contract.

## Proposed Project Structure

```text
projects/adaptive-course-assistant-rl-showcase/
├── README.md
├── Makefile
├── pyproject.toml
├── docs/
│   ├── 00-start-here.md
│   ├── learning-guide.md
│   ├── system-boundary.md
│   ├── algorithm-ladder.md
│   ├── state-action-reward.md
│   ├── policy-export-and-agent-bridge.md
│   ├── evaluation-and-governance.md
│   ├── anti-copy-policy.md
│   └── assignment-transfer-guide.md
├── scripts/
│   ├── run_showcase.py
│   ├── run_rule_policy.py
│   ├── run_contextual_bandit.py
│   ├── run_mdp_policy.py
│   ├── run_q_learning.py
│   ├── run_sarsa.py
│   ├── run_policy_gradient.py
│   ├── run_dqn_optional.py
│   ├── run_ppo_optional.py
│   ├── run_rl_family_comparison.py
│   ├── run_reward_audit.py
│   ├── run_policy_export.py
│   └── verify_artifacts.py
├── src/adaptive_course_assistant_rl/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── environment.py
│   ├── contextual_bandit.py
│   ├── heuristic_policy.py
│   ├── q_learning.py
│   ├── sarsa.py
│   ├── policy_gradient.py
│   ├── drl.py
│   ├── reward_design.py
│   ├── evaluation.py
│   ├── agent_bridge.py
│   ├── reporting.py
│   └── artifact_manifest.py
├── tests/
│   ├── test_environment.py
│   ├── test_contextual_bandit.py
│   ├── test_q_learning.py
│   ├── test_sarsa.py
│   ├── test_policy_gradient.py
│   ├── test_drl.py
│   ├── test_agent_bridge.py
│   └── test_artifact_contract.py
└── artifacts/
    ├── .gitkeep
    └── manifest.json
```

## Concept And Algorithm Ladder

The teaching sequence should be explicit and artifact-backed:

1. Deterministic course-assistant intervention baseline
2. Contextual bandit for first intervention choice
3. Small custom MDP for bounded multi-turn tutoring
4. Tabular Q-learning for value-based control
5. SARSA for on-policy TD contrast
6. REINFORCE for policy-gradient intuition
7. DQN on the same environment family through a Gymnasium adapter
8. PPO on the same environment family
9. Policy export into an agent-intervention bridge

This project should not pretend those are all the same method. The docs must explicitly show:

- Q-learning uses tabular state-action values,
- DQN approximates those values with a network,
- policy gradients learn the policy directly,
- actor-critic introduces a learned critic,
- PPO is a stabilized actor-critic method.

## State, Action, Reward, And Transition Schema

### System Boundary

The environment should assume a deterministic assistant front-end has already:

- classified the student request,
- retrieved any initial relevant course resources,
- constructed a compact tutoring context.

The learned policy should decide only the next pedagogical intervention, not the content-generation stack itself.

### Environment Frame

The environment represents a bounded tutoring episode with horizon `H = 4` to `6` turns. Each episode simulates a student request plus hidden learning dynamics. The intervention policy decides what to do next, receives a shaped reward, and either resolves the issue, escalates, or runs out of turns.

### State Schema

Core state fields for the MDP:

| Field | Type | Example values | Why it matters |
|---|---|---|---|
| `intent_type` | categorical | `concept_help`, `debug_help`, `study_plan`, `exam_review` | Deterministic assistant summary of the student request |
| `difficulty_level` | categorical | `intro`, `intermediate`, `advanced` | Helps calibrate hint depth and example choice |
| `confidence_level` | categorical | `low`, `medium`, `high` | Tracks whether the student needs reassurance or challenge |
| `misconception_type` | categorical | `none`, `notation`, `conceptual`, `procedural` | Controls whether hints or worked examples are effective |
| `retrieval_quality` | categorical | `poor`, `partial`, `strong` | Indicates how well the assistant is grounded in course material |
| `intent_uncertainty` | categorical | `low`, `medium`, `high` | Makes clarifying questions sometimes optimal |
| `cognitive_load` | categorical | `low`, `medium`, `high` | Helps distinguish “push forward” from “slow down” |
| `turn_index` | integer | `0..5` | Supports turn budget and delayed effects |
| `attempt_count` | integer | `0..3` | Differentiates first help from repeated failure |
| `last_action` | categorical | action id or `none` | Makes repeated ineffective moves visible |
| `safety_risk` | categorical | `low`, `medium`, `high` | Makes escalation and guardrails part of the policy |
| `resolved_flag` | binary | `0`, `1` | Terminal success signal |

Contextual-bandit view:

- use only the first-turn slice of state,
- choose one initial intervention action,
- end the episode immediately after the first reward.

This preserves a real contextual-bandit formulation rather than smuggling in future-state effects.

### Action Schema

Core MDP action space:

| Action | Meaning | Typical use |
|---|---|---|
| `ask_clarifying_question` | ask for missing context before helping | high uncertainty, ambiguous request |
| `retrieve_course_note` | ground the next step in course material | weak retrieval quality |
| `give_hint` | provide a small next step | medium uncertainty, recoverable struggle |
| `give_worked_example` | provide a fuller solution path | high difficulty or strong misconception |
| `assign_targeted_practice` | reinforce with a short exercise | low risk, low confidence, not yet resolved |
| `check_understanding` | verify whether the student is now resolved | later turns, before termination |
| `slow_down_and_rephrase` | reduce overload and restate more clearly | high cognitive load |
| `escalate_to_human` | stop and hand off | high risk or repeated failure |

Contextual-bandit action subset:

- `ask_clarifying_question`
- `retrieve_course_note`
- `give_hint`
- `give_worked_example`
- `slow_down_and_rephrase`

### Reward Schema

Recommended good reward components:

| Component | Reward |
|---|---:|
| student resolved with grounded help | `+8` |
| confidence improves without unsafe shortcut | `+3` |
| retrieval quality improves before final answer | `+1` |
| appropriate escalation on high-risk case | `+2` |
| each additional turn | `-1` |
| unnecessary intervention switch | `-2` |
| ungrounded answer | `-4` |
| unresolved episode at horizon | `-5` |
| unsafe failure to escalate | `-7` |

Required bad reward for teaching contrast:

- over-reward short-term confidence gain,
- under-penalize ungrounded answers,
- ignore later confusion and escalation cost.

This bad reward should make “fast but shallow help” look artificially attractive and create a visible reward-hacking discussion.

### Transition Notes

The transition model should be simple and seeded:

- `ask_clarifying_question` reduces `intent_uncertainty` but consumes a turn,
- `retrieve_course_note` improves `retrieval_quality`,
- `slow_down_and_rephrase` reduces `cognitive_load`,
- `assign_targeted_practice` can improve long-term resolution probability while costing turns,
- repeated ineffective actions increase the chance of non-resolution,
- `check_understanding` can terminate when the episode is likely solved,
- `escalate_to_human` terminates immediately.

The implementation should keep the transition logic readable enough that students can trace one episode by hand.

## Script Surface

Required scripts and their responsibility:

| Script | Responsibility | Primary outputs |
|---|---|---|
| `scripts/run_showcase.py` | orchestrate the default artifact path | all core artifacts |
| `scripts/run_rule_policy.py` | run deterministic pedagogical baselines | `artifacts/policy/rule_policy_summary.csv` |
| `scripts/run_contextual_bandit.py` | compare first-turn intervention baselines | `artifacts/bandit/*` |
| `scripts/run_mdp_policy.py` | write episode examples and intervention-policy MDP outputs | `artifacts/mdp/*`, `artifacts/concepts/mdp_spec.md` |
| `scripts/run_q_learning.py` | tabular Q-learning training and reporting | `artifacts/q_learning/*` |
| `scripts/run_sarsa.py` | on-policy TD comparison | `artifacts/sarsa/*` |
| `scripts/run_policy_gradient.py` | REINFORCE training curve and notes | `artifacts/policy_gradient/*` |
| `scripts/run_dqn_optional.py` | DQN-focused entry point that runs the shared optional DQN/PPO comparison bundle | `artifacts/drl_optional/dqn_*` plus shared comparison artifacts |
| `scripts/run_ppo_optional.py` | PPO-focused entry point that runs the shared optional DQN/PPO comparison bundle | `artifacts/drl_optional/ppo_*` plus shared comparison artifacts |
| `scripts/run_rl_family_comparison.py` | compare heuristic, bandit, Q-learning, DQN, and PPO | `artifacts/comparison/*` |
| `scripts/run_reward_audit.py` | bad-vs-good reward comparison | `artifacts/reward/*` |
| `scripts/run_policy_export.py` | map learned policy into assistant-intervention artifacts | `artifacts/bridge/*` |
| `scripts/verify_artifacts.py` | verify the artifact contract | verifier output only |

## Makefile Contract

The project should expose at minimum:

- `sync`
- `ruff`
- `ty`
- `test`
- `check`
- `smoke`
- `run`
- `verify`
- `sync-drl`
- `run-drl-optional`

Suggested command mapping:

- `make smoke`: quick contextual bandit + one MDP sample + short Q-learning + reward audit + verify
- `make run`: full core learning path with deterministic seeds
- `make run-drl-optional`: DQN + PPO bridge and comparison artifacts

## Artifact Contract

### Core Required Files

These should be produced by `make run` and enforced by `artifacts/manifest.json`:

- `artifacts/manifest.json`
- `artifacts/concepts/mdp_spec.md`
- `artifacts/concepts/algorithm_progression.md`
- `artifacts/concepts/state_action_reward_schema.csv`
- `artifacts/concepts/agentic_rl_bridge.md`
- `artifacts/concepts/interpretation_prompts.md`
- `artifacts/assistant/episode_trace.json`
- `artifacts/assistant/resource_matches.csv`
- `artifacts/bandit/contextual_policy_metrics.csv`
- `artifacts/bandit/regret_trace.csv`
- `artifacts/bandit/action_breakdown.csv`
- `artifacts/mdp/sample_episodes.csv`
- `artifacts/policy/rule_policy_summary.csv`
- `artifacts/policy/intervention_decisions.csv`
- `artifacts/q_learning/training_curve.csv`
- `artifacts/q_learning/q_table.csv`
- `artifacts/sarsa/training_curve.csv`
- `artifacts/policy_gradient/training_curve.csv`
- `artifacts/reward/reward_hacking_report.md`
- `artifacts/reward/reward_spec_bad.md`
- `artifacts/reward/reward_spec_good.md`
- `artifacts/eval/offline_policy_eval.csv`
- `artifacts/eval/scenario_rollups.csv`
- `artifacts/eval/safety_summary.md`
- `artifacts/bridge/policy_router.json`
- `artifacts/bridge/action_mapping.md`
- `artifacts/business/deployment_recommendation.md`

### Optional DRL Files

These should be documented in the README and verified by a DRL-aware mode of `scripts/verify_artifacts.py` when `make run-drl-optional` is executed:

- `artifacts/drl_optional/dqn_training_summary.csv`
- `artifacts/drl_optional/ppo_training_summary.csv`
- `artifacts/drl_optional/rl_family_comparison.csv`
- `artifacts/drl_optional/scenario_rollups.csv`
- `artifacts/drl_optional/policy_gradient_bridge_notes.md`

### Contract Rules

- Artifact paths must be deterministic and stable.
- The verifier should distinguish missing optional DRL artifacts from missing core artifacts.
- The README must explain how each artifact should be interpreted.

## Student-Facing Documentation Plan

Required docs emphasis:

- `docs/00-start-here.md`: reading order and concept-to-code map
- `docs/system-boundary.md`: deterministic assistant responsibilities versus learned policy responsibilities
- `docs/algorithm-ladder.md`: Q-learning -> DQN -> policy gradients -> actor-critic -> PPO
- `docs/state-action-reward.md`: domain modeling and reward tradeoffs
- `docs/policy-export-and-agent-bridge.md`: how learned policy maps back to agent actions
- `docs/evaluation-and-governance.md`: offline evaluation, safety, and deploy-shadow-reject reasoning
- `docs/anti-copy-policy.md` and `docs/assignment-transfer-guide.md`: student-safe transfer guidance

## Root Integration Expectations

If the project is implemented, the repo should also update:

- root `README.md`
- `docs/getting-started.md`
- `docs/learning-path.md`
- `docs/aspect-coverage-matrix.md`
- track pages where agentic systems or RL are discussed
- root `Makefile`
- `.github/workflows/ci.yml`
- issue-template project lists

## Staged Implementation Plan

### Stage 0: Design And Scaffolding

- create the project skeleton,
- define the artifact manifest,
- write the README and docs shells,
- wire project-local Make targets,
- add a small deterministic config module.

### Stage 1: Deterministic Agent-Loop Simulator

- implement the tutoring episode state schema around a deterministic assistant boundary,
- implement heuristic intervention policies,
- generate `mdp_spec.md`, sample episodes, and concept artifacts,
- add unit tests for transitions and terminal logic.

### Stage 2: Real Contextual Bandit

- implement a one-step contextual bandit over first-turn intervention choice,
- add regret and action-breakdown artifacts,
- verify the bandit uses first-turn context only,
- document why this is not yet a sequential controller.

### Stage 3: Tabular Control

- implement Q-learning and SARSA,
- write training curves and Q-table artifacts,
- add scenario-based evaluation against heuristics,
- connect the explanation to Bellman intuition and delayed reward.

### Stage 4: Policy Gradients And DRL Bridge

- implement REINFORCE from scratch,
- add a Gymnasium adapter for the same environment family,
- implement optional DQN and PPO runs with Stable-Baselines3,
- write comparison artifacts across tabular Q-learning, DQN, and PPO.

### Stage 5: Agent Bridge

- export a learned policy into a deterministic intervention contract,
- add `policy_router.json` and `action_mapping.md`,
- write docs showing how the learned controller could sit behind an OpenAI Agents SDK or similar runtime without making the SDK part of the core path.

### Stage 6: Quality Gates And Repo Integration

- add focused tests, smoke paths, and verifier coverage,
- add root integration surfaces,
- update learning-path and aspect-coverage docs,
- run project-local and root-level checks.

## Success Criteria

- The project has a unique learning contract distinct from the existing two neighbor showcases.
- `make smoke`, `make run`, `make test`, `make check`, and `make verify` pass locally once implemented.
- Core artifacts prove contextual bandit, MDP, tabular control, reward auditing, and policy export.
- Optional DRL artifacts prove DQN and PPO on the same environment family.
- Student-facing docs explicitly connect Q-learning, DQN, policy gradients, actor-critic, and PPO.

## Assumption Log

| Assumption | Risk if wrong | Validation path |
|---|---|---|
| Students will arrive after one of the two precursor showcases | The project may feel too advanced or repetitive | confirm intended placement in `docs/learning-path.md` during implementation |
| A bounded tutoring episode can stay small enough for tabular control | The state space may become too large | cap categorical values and keep horizon short |
| Optional DRL extras can remain CPU-friendly | project may exceed laptop expectations | enforce quick-mode hyperparameters and optional extras |
| Policy export is enough for the agent bridge in v1 | users may expect live SDK execution | document live runtime as future extension, not default scope |

## Evidence Checklist

Files reviewed for this design:

- `AGENTS.md`
- `docs/new-showcase-playbook.md`
- `docs/aspect-coverage-matrix.md`
- `docs/learning-path.md`
- `docs/superpowers/specs/2026-05-08-agentic-course-assistant-showcase-design.md`
- `docs/superpowers/specs/2026-06-06-student-support-rl-showcase-design.md`
- `projects/agentic-course-assistant-showcase/README.md`
- `projects/student-support-rl-showcase/README.md`

## Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Project duplicates too much of `student-support-rl-showcase` | high | focus the new project on learned orchestration and policy export, not generic RL coverage |
| Project duplicates too much of `agentic-course-assistant-showcase` | high | keep deterministic agent workflow recap short and use it only as the control baseline |
| DRL bridge becomes too heavy for classroom use | medium | keep DQN and PPO optional and provide quick deterministic settings |
| State design grows too complex for tabular methods | medium | keep a compressed categorical state for core runs |
| Students confuse policy quality with reward alone | high | make offline evaluation and reward auditing first-class artifacts |

## Completeness Check

This design spec now concretely defines:

- project structure,
- learning outcomes,
- state/action/reward schema,
- scripts,
- artifact contract,
- staged implementation plan,
- assumptions, evidence, and risks for deep-mode review.
