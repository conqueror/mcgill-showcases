# RL Bandits Policy Showcase

Learn exploration vs exploitation through multi-armed bandit simulations and policy benchmarking.

## Learning outcomes
- Implement epsilon-greedy, UCB1, and Thompson sampling.
- Track cumulative reward and pseudo-regret over time: add the best arm's mean minus the chosen arm's mean at each round.
- Rank policies within one seeded simulation; the ranking does not establish general superiority.

## Quickstart
```bash
cd projects/rl-bandits-policy-showcase
make sync
make run
make run-compare
make verify
```

## Key outputs
- `artifacts/sim/reward_trace.csv`
- `artifacts/sim/regret_trace.csv`
- `artifacts/sim/policy_comparison.csv`
- `artifacts/sim/policy_recommendation.md`

