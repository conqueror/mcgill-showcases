# Learning Agents Showcase

> Status: the core runnable path is ready, and all four locus-of-learning lanes run locally today.
> The orchestration-policy ladder, offline RL and off-policy evaluation, the cost-aware cascade,
> the OpenAI Agents SDK bridge (live SDK gated behind an optional extra), the toy RLHF/DPO/GRPO/RLVR
> loop, and the simulated multi-agent lane all ship with deterministic artifacts, tests, and an
> artifact verifier.

A **self-contained** capstone about **where learning lives in an agentic system** and how to make
an agent *learn its decisions* with RL/DRL by learning the **orchestration policy**: routing, tool
choice, escalation, and when to stop. It does not retrain the LLM.

It has no dependency on the other showcases. It vendors its own RL algorithm library and its own
judge-rubric reward. The deterministic core is laptop- and CI-friendly. Beyond that core it also
ships three locus-of-learning lanes: an OpenAI Agents SDK bridge, RLHF/DPO/GRPO/RLVR concepts at
toy scale, and a simulated multi-agent RL lane.

This is the separate capstone where those extensions belong. They are intentionally not folded
back into `adaptive-course-assistant-rl-showcase`, which stays focused on one learned controller
around a deterministic assistant.

## Start Here

Open [docs/00-start-here.md](docs/00-start-here.md) first. It gives the reading order, the
concept-to-artifact map, and the honest boundary between this capstone and its sibling
showcases.

## The central idea: where does learning live in an agent?

| Locus | What is optimized | Methods | Here |
|---|---|---|---|
| Orchestration policy | the agent's discrete decisions | bandits, Q-learning/SARSA, DP, REINFORCE, offline RL, DQN/PPO | core, built ([ladder](docs/rl-ladder.md)) |
| LLM weights | the token policy | RLHF, DPO, GRPO, RLVR | Lane B, toy loop ([built](docs/lane-b-preference-optimization.md)) |
| Multi-agent coordination | sub-agent policies that co-adapt | MARL (IQL, joint-action learning) | Lane C, simulated ([built](docs/lane-c-marl.md)) |

## Quickstart

```bash
cd projects/learning-agents-showcase
make sync
make smoke
make verify
make check
```

`make smoke` is the fastest honest path from a clean checkout to the checked artifact set.
`make check` is the code-quality gate; it does not generate artifacts on its own.

If you want the longer per-concept run after the quick path passes:

```bash
make run
```

## What Ships Today

- Core orchestration-policy ladder: bandit warm-up, MDP framing, Q-learning, SARSA, dynamic
  programming, and REINFORCE.
- Offline RL from a logged dataset (Fitted-Q Iteration) and off-policy evaluation (importance
  sampling, weighted IS, direct method, doubly robust).
- A cost-aware effort cascade with a cost/quality Pareto frontier.
- Reward-hacking audit, offline policy comparison, and the deploy/shadow/reject governance memo.
- Lane A: an OpenAI Agents SDK bridge (offline by default; live SDK gated behind `make sync-sdk`).
- Lane B: a toy preference-optimization loop covering RLHF, DPO, GRPO, and RLVR.
- Lane C: a simulated multi-agent coordination lane (independent vs joint-action learning).
- Optional deep-RL lane: from-scratch NumPy DQN and PPO (no torch), via `make run-drl`.
- Local quality gates: `make check` and `make verify`.
- Stable artifact contract: `artifacts/manifest.json` plus `scripts/verify_artifacts.py`.
- A full set of concept guides under `docs/` (start with `docs/00-start-here.md`).

## Documentation

Concept guides live in `docs/` (start with `docs/00-start-here.md`):

- [Where learning lives in an agentic system](docs/locus-of-learning.md) — the centerpiece taxonomy.
- [Showcase architecture](docs/showcase-architecture.md) and [the RL ladder](docs/rl-ladder.md).
- [Deep RL: DQN and PPO](docs/deep-rl.md) — the optional function-approximation lane.
- [Offline RL and off-policy evaluation](docs/offline-rl-and-ope.md) and [cost-aware cascades](docs/cost-aware-cascade.md).
- [Reward design and hacking](docs/reward-design-and-hacking.md) and [evaluation and governance](docs/evaluation-and-governance.md).
- Lanes: [A — agent frameworks](docs/lane-a-agent-frameworks.md), [B — preference optimization](docs/lane-b-preference-optimization.md), [C — MARL](docs/lane-c-marl.md).
- Reference: [glossary](docs/glossary.md), [math notes](docs/math-notes.md), [exercises](docs/exercises.md).
- [Results dashboard](docs/results-dashboard.md) — a visual scorecard across all lanes.

## Key Artifacts

After `make smoke` or `make run`, inspect:

- `artifacts/concepts/mdp_spec.md`
- `artifacts/concepts/concept_map.csv`
- `artifacts/concepts/algorithm_progression.md`
- `artifacts/bandit/reward_trace.csv`
- `artifacts/bandit/regret_trace.csv`
- `artifacts/q_learning/training_curve.csv`
- `artifacts/q_learning/q_table.csv`
- `artifacts/offline_rl/dataset_summary.csv`
- `artifacts/ope/estimator_comparison.csv`
- `artifacts/cost_cascade/cost_quality_curve.csv`
- `artifacts/eval/policy_comparison.csv`
- `artifacts/eval/scenario_results.csv`
- `artifacts/sdk_bridge/bridge_report.md`
- `artifacts/preference/method_comparison.csv`
- `artifacts/marl/coordination_comparison.csv`
- `artifacts/reward/reward_hacking_report.md`
- `artifacts/business/deploy_shadow_reject_memo.md`
- `artifacts/manifest.json`

## Suggested Next Projects

- `../agentic-course-assistant-showcase/README.md`
- `../adaptive-course-assistant-rl-showcase/README.md`
- `../student-support-rl-showcase/README.md`

## Project Structure

```text
learning-agents-showcase/
├── README.md
├── Makefile
├── pyproject.toml
├── docs/
├── scripts/
├── src/learning_agents/
├── tests/
└── artifacts/
```
