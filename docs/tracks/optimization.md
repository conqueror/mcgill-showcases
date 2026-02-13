# Optimization Track

This track focuses on improving model and policy quality under resource and feedback constraints.

## Recommended Sequence

1. `projects/automl-hpo-showcase`
2. `projects/rl-bandits-policy-showcase`
3. `projects/sota-supervised-learning-showcase` (benchmark extension)

## Core Skills Covered

- Hyperparameter search strategy tradeoffs.
- Experiment logging and comparability.
- Bandit policy reward/regret analysis.
- Choosing optimization objectives aligned with business constraints.

## Evidence Artifacts To Inspect

- `artifacts/hpo/trials.csv`
- `artifacts/hpo/strategy_comparison.csv`
- reward/regret outputs in RL bandits artifacts
- experiment logs in `artifacts/experiments/`

## Suggested Reflection Prompts

- Which search strategy wins under strict runtime budget?
- How sensitive are recommendations to objective choice?
- What offline checks are required before online policy rollout?
