# Student Support RL Showcase Design

## Scope

Build a new showcase project under `projects/student-support-rl-showcase` that teaches Assignment 2 concepts for deep reinforcement learning through a synthetic student-support intervention policy.

## Primary Learning Outcome

Learners should be able to frame a bounded sequential decision problem as an MDP, compare baseline and learned policies, explain Bellman and Q-learning intuition, identify reward hacking, and make a deploy-shadow-reject recommendation with governance constraints.

## Source Alignment

Primary course sources:

- `/Users/fatih/dev/McGill/agentic-ai/assignment-2-drl-showcase-agent-instructions.md`
- `/Users/fatih/dev/McGill/agentic-ai/deep-reinforcement-learning-agentic-ai-deckset.md`
- `/Users/fatih/dev/McGill/agentic-ai/mgsc-695-agentic-ai-assignments.docx`

Concepts the design must cover:

- MDP framing: state, action, reward, transition, horizon, policy
- Bellman and Q-learning intuition
- exploration vs exploitation
- baseline comparison
- reward hacking and better reward redesign
- offline evaluation beyond reward alone
- DQN and PPO as a bridge, not the required default
- governance, safety constraints, and human escalation
- business communication through a deploy-shadow-reject memo

## Design Decisions

### 1. Student support instead of an assignment-ready industry domain

The showcase uses a synthetic student-support domain so it stays familiar, low-risk, and clearly different from common pricing, inventory, recommendation, or workflow-assistant assignment domains.

### 2. Small readable MDP before deep RL tooling

The core learning path uses a deterministic-by-seed custom environment and tabular Q-learning so students can inspect the full state-action loop in one sitting. The optional DRL path exists only as a bridge to Gymnasium and Stable-Baselines3 concepts.

### 3. Reward design as the central teaching lever

The showcase will intentionally include a bad reward and a better reward so students see why local proxy improvement can still fail the true objective.

### 4. Offline evaluation as a first-class artifact

The evaluation phase must compare random, heuristic, and learned policies across fixed student scenarios and report risk, cost, over-intervention, escalation, and questionable actions instead of treating reward as sufficient.

### 5. Anti-copy controls must be visible

The README and docs must explicitly prohibit reusing the showcase domain, environment, state variables, action set, reward function, or evaluation report structure verbatim for Assignment 2 submissions.

## Proposed Project Structure

```text
projects/student-support-rl-showcase/
├── README.md
├── Makefile
├── pyproject.toml
├── docs/
│   ├── learning-guide.md
│   ├── method-notes.md
│   ├── domain-use-cases.md
│   ├── anti-copy-policy.md
│   └── assignment-transfer-guide.md
├── scripts/
│   ├── run_bandit.py
│   ├── run_mdp_simulation.py
│   ├── run_q_learning.py
│   ├── run_reward_hacking_check.py
│   ├── run_policy_evaluation.py
│   ├── run_drl_optional.py
│   ├── write_business_memo.py
│   └── verify_artifacts.py
├── src/student_support_rl/
├── tests/
└── artifacts/
```

## Artifact Contract

Required files:

- `artifacts/concepts/mdp_spec.md`
- `artifacts/concepts/concept_map.csv`
- `artifacts/concepts/algorithm_progression.md`
- `artifacts/bandit/reward_trace.csv`
- `artifacts/bandit/regret_trace.csv`
- `artifacts/mdp/sample_episodes.csv`
- `artifacts/q_learning/training_curve.csv`
- `artifacts/q_learning/q_table.csv`
- `artifacts/eval/policy_comparison.csv`
- `artifacts/eval/scenario_results.csv`
- `artifacts/reward/reward_hacking_report.md`
- `artifacts/reward/reward_spec_good.md`
- `artifacts/reward/reward_spec_bad.md`
- `artifacts/governance/safety_controls.md`
- `artifacts/governance/offline_eval_plan.md`
- `artifacts/business/deploy_shadow_reject_memo.md`
- `artifacts/manifest.json`

## Runtime Strategy

- `make smoke` must finish in under 60 seconds using quick episode counts and reduced training steps.
- `make run` must finish in under 10 minutes on a normal laptop using small deterministic simulations.
- `make run-drl-optional` may install or use optional DRL extras, but it must not block the core learning path or core verification.

## Success Criteria

- The project follows the standard showcase structure and root integration rules.
- `make smoke`, `make run`, `make test`, `make check`, and `make verify` pass in the project directory.
- Root `Makefile`, CI, docs pages, and issue templates include the new showcase.
- The README includes the anti-copy rule and Domain Delta Statement guidance.
- Artifact names are stable and the verifier enforces the contract.

## Expansion Wave: RL and DRL Coverage Completion

The first implementation wave shipped a small MDP, tabular Q-learning, and an optional PPO bridge. The next wave should close the remaining coverage gaps called out in review.

Additional required outcomes for the expansion wave:

- Replace the current stationary bandit warm-up with a real contextual bandit, or explicitly rename and rescope the docs if a contextual formulation is not implemented.
- Add an executable DQN path that uses Gymnasium and Stable-Baselines3 on the same student-support environment family.
- Strengthen the optional DRL lane so students can compare tabular Q-learning, DQN, and PPO on aligned scenarios instead of treating PPO as a standalone curiosity.
- Expand student-facing docs so they explicitly connect:
  - Q-learning
  - DQN
  - policy gradients
  - actor-critic
  - PPO
- Add comparison artifacts that make the DRL bridge inspectable rather than implied.

Recommended artifact additions for the expansion wave:

- `artifacts/drl_optional/bridge_report.md`
- `artifacts/drl_optional/rl_family_comparison.csv`
- `artifacts/drl_optional/scenario_rollups.csv`
- `artifacts/drl_optional/training_summary.csv`
- `artifacts/drl_optional/policy_gradient_notes.md`

The expansion must remain laptop-friendly:

- quick mode should stay under the existing smoke target,
- the non-optional path should stay under the existing full-run target,
- DQN and PPO can remain optional but their outputs should be reproducible and documented.
