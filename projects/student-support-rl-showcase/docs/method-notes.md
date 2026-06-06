# Method Notes

## Why this project starts with tabular RL

The first goal is interpretability. A small environment plus tabular Q-learning makes it easy to inspect the state, action, reward, and update loop before introducing neural approximators.

## Core methods

- Real contextual bandit warm-up for one-step exploration, context-dependent action quality, and regret.
- Small custom MDP for sequential state transitions.
- Exact dynamic programming (backward induction) for the ground-truth optimal `Q*`, so model-free learning can be measured against truth.
- Tabular Q-learning (off-policy) and SARSA (on-policy) for temporal-difference value learning.
- Tabular REINFORCE for a from-scratch policy-gradient baseline.
- Fixed-scenario offline evaluation for policy comparison.
- Reward-design comparison to expose reward hacking.

## DRL bridge

The optional DRL path is intentionally separate from the core grading path. It exists to connect this small showcase to DQN, policy-gradient, actor-critic, and PPO concepts from the course deck without turning the project into a large benchmarking exercise.

The bridge keeps one canonical student-support environment family and compares:

- tabular Q-learning as the small, inspectable value-learning baseline,
- DQN as the neural value-based extension, and
- PPO as the actor-critic, policy-gradient reference point.

That same-environment comparison is the important teaching choice. Students can change the learning algorithm without also changing the domain, the scenario set, or the evaluation metrics.

## What to pay attention to

- Which parts of the reward are proxies rather than true success.
- Whether a policy reduces risk by improving outcomes or by overusing escalations.
- Whether a policy generalizes across scenario types instead of optimizing a narrow corner of the simulator.
- Whether DQN and PPO improve on tabular Q-learning because of richer function approximation or just because they are harder to inspect and tune.
