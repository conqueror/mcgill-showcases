# Learning Agents Showcase: Start Here

## Status

The core runnable path is ready today:

- `make smoke` generates the deterministic artifact set.
- `make verify` checks the artifact contract from `artifacts/manifest.json`.
- `make check` runs lint, type checks, and tests.

All four locus-of-learning lanes now ship and run locally: the orchestration-policy ladder
(Lane 1), the OpenAI Agents SDK bridge (Lane A, with the live SDK gated behind an optional
extra), the toy preference-optimization loop (Lane B: RLHF/DPO/GRPO/RLVR), and the simulated
multi-agent coordination lane (Lane C: independent vs joint-action learning). Each has a
concept guide under `docs/` and deterministic artifacts under `artifacts/`. An optional deep-RL
lane (DQN vs PPO, NumPy-only) extends the Lane 1 ladder via `make run-drl`.

## What This Showcase Is Teaching

This project is about **learning an agent's orchestration policy**:

- which tool or action to take next,
- when to escalate,
- when to stop,
- how to compare learned decisions against rule-based baselines.

It is **not** retraining the LLM weights. The learned object is the smaller decision policy around
the agent.

## Suggested Reading Order

A guided path through the showcase. Set up once with `make sync`, then generate the evidence with
`make smoke` (fast) or `make run` (full per-concept); read each guide, then open the artifact it
cites to see the numbers for yourself.

1. **Orient.** Skim `README.md` and the status note above, then run `make smoke`.
2. **The big idea.** [Where learning lives in an agentic system](locus-of-learning.md) — the
   centerpiece taxonomy — then [showcase architecture](showcase-architecture.md) for how the
   environment, reward, RL library, lanes, and artifact contract fit together.
3. **Foundations.** [Exploration and the contextual bandit warm-up](exploration-and-bandits.md),
   then [the RL ladder](rl-ladder.md): bandit → MDP → Q-learning → SARSA → dynamic programming →
   REINFORCE.
4. **Scaling value learning.** [Offline RL and off-policy evaluation](offline-rl-and-ope.md), plus
   the optional [deep RL: DQN and PPO](deep-rl.md) lane.
5. **Real-world trade-offs.** [Cost-aware effort cascades](cost-aware-cascade.md).
6. **Getting the objective right.** [Reward design and reward hacking](reward-design-and-hacking.md),
   then [evaluation and governance](evaluation-and-governance.md) — the deploy / shadow / reject call.
7. **The other loci of learning.** [Lane A: the agent framework as environment](lane-a-agent-frameworks.md),
   [Lane B: learning the LLM weights](lane-b-preference-optimization.md), and
   [Lane C: multi-agent coordination](lane-c-marl.md).
8. **Consolidate.** The [results dashboard](results-dashboard.md) shows every lane's headline at a
   glance; keep the [glossary](glossary.md) and [mathematical notes](math-notes.md) handy, and test
   yourself with the [exercises](exercises.md).

## Quick Commands

| Goal | Command | What To Expect |
|---|---|---|
| Install dependencies | `make sync` | Creates the local `uv` environment with dev tools |
| Fast runnable path | `make smoke` | Generates the core artifacts on a normal laptop |
| Artifact gate | `make verify` | Confirms required files exist and match the contract |
| Code-quality gate | `make check` | Runs ruff, mypy, and pytest |
| Full per-concept run | `make run` | Replays each concept runner in sequence |

## Concept To Artifact Map

| Concept | Command | Main artifact(s) |
|---|---|---|
| Orchestration policy framing | `make run-mdp` | `artifacts/concepts/mdp_spec.md`, `artifacts/mdp/sample_episodes.csv` |
| Contextual exploration warm-up | `make run-bandit` | `artifacts/bandit/reward_trace.csv`, `artifacts/bandit/regret_trace.csv` |
| Off-policy control | `make run-q-learning` | `artifacts/q_learning/training_curve.csv`, `artifacts/q_learning/q_table.csv` |
| Planning vs learning | `make run-dp` | `artifacts/dp/optimal_action_values.csv`, `artifacts/dp/q_learning_gap.csv` |
| On-policy contrast | `make run-sarsa` | `artifacts/sarsa/training_curve.csv`, `artifacts/sarsa/q_table.csv` |
| Policy gradients | `make run-reinforce` | `artifacts/policy_gradient/training_curve.csv` |
| Offline RL from a log | `make run-offline` | `artifacts/offline_rl/dataset_summary.csv`, `artifacts/offline_rl/training_curve.csv` |
| Off-policy evaluation | `make run-ope` | `artifacts/ope/estimator_comparison.csv` |
| Cost-aware effort cascade | `make run-cascade` | `artifacts/cost_cascade/cost_quality_curve.csv` |
| Reward design audit | `make run-reward-check` | `artifacts/reward/reward_hacking_report.md` |
| Offline evaluation and governance | `make run-eval` and `make run-business` | `artifacts/eval/policy_comparison.csv`, `artifacts/business/deploy_shadow_reject_memo.md` |
| Lane A: agent framework bridge | `make run-sdk-bridge` | `artifacts/sdk_bridge/bridge_report.md`, `artifacts/sdk_bridge/orchestration_trace.csv` |
| Lane B: preference optimization | `make run-preference` | `artifacts/preference/method_comparison.csv`, `artifacts/preference/training_curves.csv` |
| Lane C: multi-agent coordination | `make run-marl` | `artifacts/marl/coordination_comparison.csv`, `artifacts/marl/training_curves.csv` |
| Deep RL (optional): DQN vs PPO | `make run-drl` | `artifacts/drl_optional/rl_family_comparison.csv`, `artifacts/drl_optional/policy_gradient_notes.md` |

## The Documentation Set

Conceptual guides live alongside this file in `docs/`:

- [Where learning lives in an agentic system](locus-of-learning.md) — the centerpiece taxonomy of the three loci of learning.
- [Showcase architecture](showcase-architecture.md) — how the environment, reward, RL library, lanes, scripts, and artifact contract fit together.
- [Exploration and the contextual bandit warm-up](exploration-and-bandits.md) — exploration vs exploitation and regret.
- [The RL ladder](rl-ladder.md) — bandit to MDP to Q-learning to SARSA to dynamic programming to REINFORCE.
- [Deep RL: DQN and PPO](deep-rl.md) — the optional function-approximation lane (NumPy, no torch).
- [Offline RL and off-policy evaluation](offline-rl-and-ope.md) — learning and grading policies from a fixed log.
- [Cost-aware effort cascades](cost-aware-cascade.md) — trading quality against money and latency on a Pareto frontier.
- [Reward design and reward hacking](reward-design-and-hacking.md) — why the reward is the specification.
- [Evaluation and governance](evaluation-and-governance.md) — the deploy / shadow / reject decision.
- [Lane A: the agent framework as environment](lane-a-agent-frameworks.md) — the OpenAI Agents SDK bridge.
- [Lane B: learning the LLM weights](lane-b-preference-optimization.md) — RLHF, DPO, GRPO, and RLVR at toy scale.
- [Lane C: multi-agent coordination](lane-c-marl.md) — independent vs joint-action learning.
- Reference: [glossary](glossary.md), [mathematical notes](math-notes.md), and [exercises with solutions](exercises.md).
- [Results dashboard](results-dashboard.md) — a visual, at-a-glance scorecard of every lane's result.

## Honest Boundary

Use this showcase when you want the standalone capstone version of the idea:
"what exactly is being learned in an agentic system?"

Use the sibling showcases when you want adjacent stories:

- `../agentic-course-assistant-showcase/README.md` for agent runtime and framework concepts.
- `../adaptive-course-assistant-rl-showcase/README.md` for a learned tutoring controller around a
  deterministic assistant.
- `../student-support-rl-showcase/README.md` for the broader RL ladder and optional DRL bridge.
