# Learning Guide

## Beginner Path

1. Run `make smoke`.
2. Read `artifacts/concepts/mdp_spec.md`.
3. Inspect one row group in `artifacts/mdp/sample_episodes.csv`.
4. Compare how random and heuristic actions differ.

## Core Path

1. Run `make run`.
2. Read `artifacts/concepts/algorithm_progression.md` and [algorithm-ladder.md](algorithm-ladder.md).
3. Inspect the contextual bandit: `artifacts/bandit/reward_trace.csv` and `artifacts/bandit/regret_trace.csv` ([exploration-and-bandits.md](exploration-and-bandits.md)).
4. Read `artifacts/q_learning/training_curve.csv` and `artifacts/q_learning/q_table.csv` ([value-based-learning.md](value-based-learning.md)).
5. Compare the learned Q-table against the exact optimum in `artifacts/dp/q_learning_gap.csv` to see where Q-learning has and has not converged.
6. Inspect on-policy SARSA (`artifacts/sarsa/training_curve.csv`) beside off-policy Q-learning, and the from-scratch policy gradient (`artifacts/policy_gradient/training_curve.csv`, [policy-gradient-and-actor-critic.md](policy-gradient-and-actor-critic.md)).
7. Compare policy behavior in `artifacts/eval/policy_comparison.csv` ([evaluation-and-governance.md](evaluation-and-governance.md)).
8. Read `artifacts/reward/reward_hacking_report.md` ([reward-design-and-hacking.md](reward-design-and-hacking.md)).

## Advanced Path

1. Read [algorithm-ladder.md](algorithm-ladder.md).
2. Run `make sync-drl`.
3. Run `make run-drl-optional`.
4. Compare `artifacts/drl_optional/rl_family_comparison.csv` for tabular Q-learning, DQN, and PPO on the same environment family.
5. Use `artifacts/drl_optional/scenario_rollups.csv` to see whether the ranking changes by scenario.
6. Read `artifacts/drl_optional/policy_gradient_notes.md` to connect policy gradients, actor-critic, and PPO.
7. Discuss why stability, sample efficiency, and reproducibility become harder in DRL.

## Business Path

1. Read `artifacts/eval/policy_comparison.csv`.
2. Read `artifacts/governance/offline_eval_plan.md`.
3. Read `artifacts/business/deploy_shadow_reject_memo.md`.
4. Decide what evidence would still be needed before any real-world rollout.
