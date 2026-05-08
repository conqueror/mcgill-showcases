# Optimization Track

This track focuses on improving model and policy quality under resource and feedback constraints.

## Recommended Sequence

1. `projects/automl-hpo-showcase`
2. `projects/autoresearch`
3. `projects/agentic-course-assistant-showcase`
4. `projects/rl-bandits-policy-showcase`
5. `projects/sota-supervised-learning-showcase` (benchmark extension)

## Core Skills Covered

- Hyperparameter search strategy tradeoffs.
- Fixed-budget experiment governance for agent-authored changes.
- Prompt and platform tradeoffs for autonomous research workflows.
- Agent routing, tool use, guardrails, and trace interpretation.
- Experiment logging and comparability.
- Bandit policy reward/regret analysis.
- Choosing optimization objectives aligned with business constraints.

## Evidence Artifacts To Inspect

- `artifacts/hpo/trials.csv`
- `artifacts/hpo/strategy_comparison.csv`
- `projects/autoresearch/artifacts/overview/platform_comparison.csv`
- `projects/autoresearch/artifacts/analysis/decision_scenarios.csv`
- `projects/agentic-course-assistant-showcase/artifacts/agent_trace.json`
- reward/regret outputs in RL bandits artifacts
- experiment logs in `artifacts/experiments/`

## Suggested Reflection Prompts

- Which search strategy wins under strict runtime budget?
- When is a tiny improvement too expensive in code complexity?
- How should the same research loop change between Apple Silicon and CUDA?
- Which agent decisions should be deterministic before a hosted model is introduced?
- How sensitive are recommendations to objective choice?
- What offline checks are required before online policy rollout?
