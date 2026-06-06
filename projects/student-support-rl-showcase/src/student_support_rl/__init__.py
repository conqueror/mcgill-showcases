"""Student-support reinforcement-learning teaching showcase.

This package implements a small, fully inspectable RL/DRL curriculum around a synthetic
student-support intervention MDP. It walks the algorithm ladder one idea at a time:

    contextual bandit -> MDP -> Q-learning -> DQN -> policy gradient -> actor-critic -> PPO

Key entry points by concept (see ``docs/00-start-here.md`` for the reading order):

- ``environment``: the MDP itself (state, actions, transition, reward, horizon).
- ``bandit``: contextual (ridge-regression, epsilon-greedy) bandit warm-up.
- ``q_learning`` / ``sarsa``: off-policy and on-policy tabular TD control.
- ``dynamic_programming``: exact optimal Q* by backward induction -- the ground truth that
  model-free Q-learning is measured against.
- ``policy_gradient``: tabular softmax REINFORCE (the policy is optimized directly).
- ``drl``: optional Gymnasium + Stable-Baselines3 DQN/PPO bridge.
- ``evaluation`` / ``reward_design`` / ``reporting``: offline policy comparison, the
  reward-hacking demonstration, and artifact generation.
"""

from student_support_rl.dynamic_programming import optimal_action_values, q_learning_gap
from student_support_rl.environment import StudentState, StudentSupportEnvironment
from student_support_rl.policies import AdvisorHeavyPolicy, HeuristicPolicy, RandomPolicy
from student_support_rl.policy_gradient import ReinforcePolicy, ReinforceResult, train_reinforce
from student_support_rl.q_learning import QLearningResult, train_q_learning
from student_support_rl.sarsa import SarsaResult, train_sarsa

__all__ = [
    "AdvisorHeavyPolicy",
    "HeuristicPolicy",
    "QLearningResult",
    "RandomPolicy",
    "ReinforcePolicy",
    "ReinforceResult",
    "SarsaResult",
    "StudentState",
    "StudentSupportEnvironment",
    "optimal_action_values",
    "q_learning_gap",
    "train_q_learning",
    "train_reinforce",
    "train_sarsa",
]
