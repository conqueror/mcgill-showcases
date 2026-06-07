# Algorithm Ladder

Here is the simplest way to read the ladder in this project.

| Method | Learns | Best artifact | Key lesson |
|---|---|---|---|
| Rule-based policy | Nothing; it is hand-written | `artifacts/policy/rule_policy_summary.csv` | Always compare learned policies to a simple baseline. |
| Contextual bandit | The best first action for a context | `artifacts/bandit/contextual_policy_metrics.csv` | Bandits do not plan through future states. |
| Q-learning | Tabular action values | `artifacts/q_learning/q_table.csv` | Value learning estimates how good each action is in each state. |
| DQN | Neural action values | `artifacts/drl_optional/dqn_training_summary.csv` | DQN is Q-learning with function approximation. |
| Policy gradients | Action probabilities | `artifacts/policy_gradient/training_curve.csv` | A policy can be optimized directly from returns. |
| Actor-critic / PPO | A policy plus a critic | `artifacts/drl_optional/ppo_training_summary.csv` | PPO is the actor-critic anchor in this showcase. |

## Rule-Based Policy

Start here. It gives you a human-written baseline and keeps the control problem concrete.

## Contextual Bandit

Use this when the question is only:

> what should the assistant do first?

There is no future state to model. You act once and take the immediate reward.

## Q-Learning

Use this when later turns matter.

Q-learning learns action values and then acts greedily from them.

## DQN

DQN keeps the Q-learning idea, but replaces the table with a neural approximation.

That is the cleanest mental bridge from tabular RL to deep RL in this project.

## Policy Gradients

REINFORCE does not learn action values first. It pushes the policy directly toward better action probabilities.

## Actor-Critic

Actor-critic methods mix the two ideas:

- an actor chooses actions,
- a critic helps judge them.

## PPO

PPO is the practical actor-critic anchor in this showcase. It is the optional deep-policy baseline that sits opposite DQN in the DRL bridge.
