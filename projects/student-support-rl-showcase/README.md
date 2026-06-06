# Student Support RL Showcase

Learn reinforcement learning and deep reinforcement learning through a synthetic student-support intervention policy that is small enough to inspect and rich enough to discuss contextual bandits, MDPs, dynamic programming, Q-learning, SARSA, DQN, policy gradients, actor-critic methods, reward design, governance, and deployment caution.

## Learning Outcomes

By the end of this showcase, you should be able to:

- frame a sequential decision problem as an MDP,
- explain what changes when you move from a contextual bandit to a sequential MDP,
- compare random, heuristic, and learned policies,
- explain the ladder from tabular Q-learning to DQN to policy gradients to actor-critic to PPO,
- compute exact optimal action values with dynamic programming and watch tabular Q-learning converge toward that ground truth,
- contrast on-policy (SARSA) with off-policy (Q-learning) temporal-difference control,
- read and explain a from-scratch policy-gradient (REINFORCE) update,
- identify reward hacking from a bad proxy reward,
- evaluate policy behavior beyond reward alone,
- write a deploy-shadow-reject recommendation with safety controls.

## Concepts and Guides

New here? Open **[docs/00-start-here.md](docs/00-start-here.md)** — it has the repository map, a
concept → guide → code → artifact table, and the suggested reading order. Each concept has a guide
with full equations and a diagram:

- [Algorithm ladder](docs/algorithm-ladder.md) — the narrative arc
- [MDP and environment](docs/mdp-and-environment.md) · [Exploration and bandits](docs/exploration-and-bandits.md)
- [Value-based learning (DP, Q-learning, SARSA)](docs/value-based-learning.md) · [Deep RL (DQN)](docs/deep-rl.md)
- [Policy gradients and actor-critic](docs/policy-gradient-and-actor-critic.md)
- [Reward design and hacking](docs/reward-design-and-hacking.md) · [Evaluation and governance](docs/evaluation-and-governance.md)
- Reference: [glossary](docs/glossary.md) · [math notes](docs/math-notes.md) · [exercises (with solutions)](docs/exercises.md)

## Prerequisites

- Python 3.11+
- `uv`
- Basic Python functions and loops
- Comfort reading CSV and Markdown artifacts

## Quickstart

```bash
cd projects/student-support-rl-showcase
make sync
make smoke
make verify
```

Install the optional DRL extras and run the DQN/PPO bridge:

```bash
make sync-drl
make run-drl-optional
```

Run the full learning path (one script per concept):

```bash
make run
```

> `make run` and `make smoke` also invoke the optional DQN/PPO bridge, which writes a short fallback
> note unless you first install the extras with `make sync-drl`. The core path — bandit, MDP, dynamic
> programming, Q-learning, SARSA, REINFORCE, evaluation, governance — does not depend on it.

Run project checks:

```bash
make check
```

## Key Outputs

After `make run`, inspect:

- `artifacts/concepts/mdp_spec.md`
- `artifacts/concepts/concept_map.csv`
- `artifacts/concepts/algorithm_progression.md`
- `artifacts/bandit/reward_trace.csv`
- `artifacts/bandit/regret_trace.csv`
- `artifacts/mdp/sample_episodes.csv`
- `artifacts/q_learning/training_curve.csv`
- `artifacts/q_learning/q_table.csv`
- `artifacts/dp/optimal_action_values.csv`
- `artifacts/dp/q_learning_gap.csv`
- `artifacts/sarsa/training_curve.csv`
- `artifacts/sarsa/q_table.csv`
- `artifacts/policy_gradient/training_curve.csv`
- `artifacts/eval/policy_comparison.csv`
- `artifacts/eval/scenario_results.csv`
- `artifacts/reward/reward_hacking_report.md`
- `artifacts/reward/reward_spec_good.md`
- `artifacts/reward/reward_spec_bad.md`
- `artifacts/governance/safety_controls.md`
- `artifacts/business/deploy_shadow_reject_memo.md`

After `make run-drl-optional`, inspect:

- `artifacts/drl_optional/bridge_report.md`
- `artifacts/drl_optional/rl_family_comparison.csv`
- `artifacts/drl_optional/scenario_rollups.csv`
- `artifacts/drl_optional/training_summary.csv`
- `artifacts/drl_optional/policy_gradient_notes.md`

## Anti-Copy Rule

> Assignment 2 submissions may not reuse the showcase domain, environment class, state variables, action set, reward function, or evaluation report structure verbatim. Students must choose a materially different domain and define their own MDP, baseline, reward specification, and evaluation plan.

For Assignment 2, include this Domain Delta Statement:

1. My assignment domain is different from the showcase because...
2. My state variables are different because...
3. My action space is different because...
4. My reward function is different because...
5. My evaluation metrics are different because...

Read the full policy in [docs/anti-copy-policy.md](docs/anti-copy-policy.md) and the transfer guide in [docs/assignment-transfer-guide.md](docs/assignment-transfer-guide.md).

## Student Path

1. Start with `make smoke`, then read [docs/00-start-here.md](docs/00-start-here.md) and `artifacts/concepts/mdp_spec.md`.
2. Read [docs/algorithm-ladder.md](docs/algorithm-ladder.md) and `artifacts/concepts/algorithm_progression.md`.
3. Inspect the contextual bandit in `artifacts/bandit/reward_trace.csv` and `artifacts/bandit/regret_trace.csv`.
4. Read one sample episode in `artifacts/mdp/sample_episodes.csv`.
5. Run `make run-q-learning` and inspect the training curve and Q-table.
6. Run `make run-dp` and open `artifacts/dp/q_learning_gap.csv` to see where Q-learning has — and has not — matched the exact optimum.
7. Run `make run-sarsa` and `make run-reinforce` to compare on-policy SARSA and a from-scratch policy gradient.
8. Run `make sync-drl && make run-drl-optional` to compare tabular Q-learning, DQN, and PPO on the same environment.
9. Run `make run-reward-check` and study the reward hacking report.
10. Read the final memo and decide whether the policy should deploy, shadow, or be rejected.

## Runtime Expectations

- `make smoke` should finish in under 60 seconds on a normal laptop.
- `make run` should finish in under 10 minutes on a normal laptop.
- `make run-drl-optional` is a bridge to Gymnasium and Stable-Baselines3 and is not required for the core learning path.
- `make run-drl-optional` produces a fallback bridge report even when the optional DRL extras are not installed.

## Common Failure Modes

- Treating reward as the same thing as student success.
- Looking only at average reward and ignoring escalation cost or unsafe actions.
- Reusing the showcase domain instead of transferring the design pattern.
- Training too long before validating whether the reward function itself is flawed.
- Forgetting that human approval and rollout limits are part of the policy design.

## Suggested Next Projects

- `../rl-bandits-policy-showcase/README.md`
- `../agentic-course-assistant-showcase/README.md`
- `../model-release-rollout-showcase/README.md`

## Project Structure

```text
student-support-rl-showcase/
├── README.md
├── Makefile
├── pyproject.toml
├── docs/
├── scripts/
├── src/student_support_rl/
├── tests/
└── artifacts/
```
