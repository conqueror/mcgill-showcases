# Optimization Track

This track focuses on improving model and policy quality under resource and feedback constraints.

## Recommended Sequence

1. `projects/automl-hpo-showcase`
2. `projects/autoresearch`
3. `projects/agentic-course-assistant-showcase`
4. `projects/rl-bandits-policy-showcase`
5. `projects/student-support-rl-showcase`
6. `projects/sota-supervised-learning-showcase` (benchmark extension)

## Core Skills Covered

- Hyperparameter search strategy tradeoffs.
- Fixed-budget experiment governance for agent-authored changes.
- Prompt and platform tradeoffs for autonomous research workflows.
- Agent routing, tool use, guardrails, and trace interpretation.
- Experiment logging and comparability.
- Bandit policy reward/regret analysis.
- Contextual bandits, MDP design, Bellman intuition, tabular Q-learning, and the DQN/PPO bridge.
- Reward hacking diagnosis and offline policy-governance tradeoffs.
- Choosing optimization objectives aligned with business constraints.

## Evidence Artifacts To Inspect

- `artifacts/hpo/trials.csv`
- `artifacts/hpo/strategy_comparison.csv`
- `projects/autoresearch/artifacts/overview/platform_comparison.csv`
- `projects/autoresearch/artifacts/analysis/decision_scenarios.csv`
- `projects/agentic-course-assistant-showcase/artifacts/agent_trace.json`
- `projects/student-support-rl-showcase/artifacts/q_learning/training_curve.csv`
- `projects/student-support-rl-showcase/artifacts/eval/policy_comparison.csv`
- reward/regret outputs in RL bandits artifacts
- experiment logs in `artifacts/experiments/`

## Suggested Reflection Prompts

- Which search strategy wins under strict runtime budget?
- When is a tiny improvement too expensive in code complexity?
- How should the same research loop change between Apple Silicon and CUDA?
- Which agent decisions should be deterministic before a hosted model is introduced?
- How sensitive are recommendations to objective choice?
- What offline checks are required before online policy rollout?
- Which reward terms create the biggest mismatch between local reward and true objective?
