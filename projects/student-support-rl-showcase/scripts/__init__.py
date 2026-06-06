"""Thin CLI runners that drive the student-support RL showcase end to end.

Each module in this package is a small command-line entry point that wires together one rung of
the RL ladder (contextual bandit -> MDP -> Q-learning -> DQN -> policy gradient -> actor-critic
-> PPO) with the reporting layer, then writes the corresponding artifact files under
``artifacts/``. The scripts deliberately hold no algorithm logic of their own; they import the
implementations from ``student_support_rl`` so the teaching code lives in one place.

RL concept:
    Reproducible experiment harness. See docs/evaluation-and-governance.md for how the written
    artifacts feed the deploy/shadow/reject decision, and docs/showcase-architecture.md for the
    full runner-to-artifact map.
"""
