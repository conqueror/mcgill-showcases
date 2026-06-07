# Learning Path

## Path A: New to Applied ML
1. `projects/deep-learning-math-foundations-showcase`
2. `projects/sota-supervised-learning-showcase`
3. `projects/sota-unsupervised-semisup-showcase`
4. `projects/causalml-kaggle-showcase`

## Path B: Deep Learning Foundations
1. `projects/deep-learning-math-foundations-showcase`
2. `projects/neural-network-foundations-showcase`
3. `projects/pytorch-training-regularization-showcase`
4. `projects/sota-supervised-learning-showcase`

## Path C: Decision Science / Causal Focus
1. `projects/sota-supervised-learning-showcase`
2. `projects/causalml-kaggle-showcase`
3. `projects/xai-fairness-audit-showcase`
4. `projects/mlops-drift-production-showcase`

## Path D: ML in Production Focus
1. `projects/sota-supervised-learning-showcase`
2. `projects/mlops-drift-production-showcase`
3. `projects/batch-vs-stream-ml-systems-showcase`
4. `projects/model-release-rollout-showcase`

## Path E: Modeling Optimization Focus
1. `projects/sota-supervised-learning-showcase`
2. `projects/automl-hpo-showcase`
3. `projects/autoresearch`
4. `projects/rl-bandits-policy-showcase`
5. `projects/student-support-rl-showcase`

## Path F: Feature and Representation Focus
1. `projects/sota-supervised-learning-showcase`
2. `projects/feature-engineering-dimred-showcase`
3. `projects/sota-unsupervised-semisup-showcase`

## Path G: Short Course (Two Weeks)
- Day 1-2: `deep-learning-math-foundations-showcase`
- Day 3-4: `neural-network-foundations-showcase`
- Day 5-6: `pytorch-training-regularization-showcase`
- Day 7-8: `sota-supervised-learning-showcase`
- Day 9-10: `feature-engineering-dimred-showcase`
- Day 11-12: `xai-fairness-audit-showcase`
- Day 13-14: `mlops-drift-production-showcase`

## Path M: Deep Learning Mini-Series
1. `projects/deep-learning-math-foundations-showcase`
2. `projects/neural-network-foundations-showcase`
3. `projects/pytorch-training-regularization-showcase`
4. `projects/sota-unsupervised-semisup-showcase`

## Path H: Contract-First Supervised Workflow
1. `projects/eda-leakage-profiling-showcase`
2. `projects/feature-engineering-dimred-showcase`
3. `projects/automl-hpo-showcase`
4. `projects/xai-fairness-audit-showcase`

## Path I: Credit Risk Capstone Workflow
1. `projects/eda-leakage-profiling-showcase`
2. `projects/feature-engineering-dimred-showcase`
3. `projects/credit-risk-classification-capstone-showcase`
4. `projects/xai-fairness-audit-showcase`

## Path J: Ranking and Serving Workflow
1. `projects/learning-to-rank-foundations-showcase`
2. `projects/ranking-api-productization-showcase`
3. `projects/model-release-rollout-showcase`

## Path N: NLP Systems Workflow
1. `projects/pytorch-training-regularization-showcase`
2. `projects/modern-nlp-pipeline-showcase`
3. `projects/learning-to-rank-foundations-showcase`
4. `projects/ranking-api-productization-showcase`

## Path K: Forecasting and Observability Workflow
1. `projects/nyc-demand-forecasting-foundations-showcase`
2. `projects/demand-api-observability-showcase`
3. `projects/mlops-drift-production-showcase`

## Path L: Agentic Research Workflow
1. `projects/automl-hpo-showcase`
2. `projects/autoresearch`
3. `projects/agentic-course-assistant-showcase`
4. `projects/adaptive-course-assistant-rl-showcase`
5. `projects/model-release-rollout-showcase`

## Path O: Agent Frameworks Workflow
1. `projects/autoresearch`
2. `projects/agentic-course-assistant-showcase`
3. `projects/adaptive-course-assistant-rl-showcase`
4. Read `projects/adaptive-course-assistant-rl-showcase/artifacts/bridge/learning_agent_story.md` after running that project once
5. `projects/modern-nlp-pipeline-showcase`
6. `projects/model-release-rollout-showcase`

## Path P: Reinforcement Learning Decision Workflow
1. `projects/rl-bandits-policy-showcase`
2. `projects/student-support-rl-showcase`
3. `projects/adaptive-course-assistant-rl-showcase`
4. Re-run `projects/adaptive-course-assistant-rl-showcase` with `make sync-drl && make run-drl-optional` to study the DQN/PPO bridge around an agentic tutoring workflow
5. Re-run `projects/student-support-rl-showcase` with `make sync-drl && make run-drl-optional` to compare that broader RL ladder against the more focused agentic-tutoring bridge
6. `projects/model-release-rollout-showcase`

## Path Q: Learning-Agent Bridge Workflow
1. `projects/agentic-course-assistant-showcase`
2. `projects/adaptive-course-assistant-rl-showcase`
3. `projects/learning-agents-showcase`
4. Re-run `projects/adaptive-course-assistant-rl-showcase` with `make sync-drl && make run-drl-optional`
5. If you have not already run it, run `projects/student-support-rl-showcase`; otherwise re-run it with `make sync-drl && make run-drl-optional`

This path is the cleanest answer to: "how do agent frameworks and learned intervention policies fit together without overclaiming?"

`projects/learning-agents-showcase` is the standalone capstone in that bridge. Its deterministic
core path is runnable today, and the OpenAI Agents SDK bridge, RLHF/DPO/GRPO/RLVR, MARL, and an
optional NumPy DQN/PPO deep-RL lane now ship as well.

## How To Know You Are Progressing
- You can explain outputs in plain language.
- You can justify model choices with evidence.
- You can describe one limitation or risk per method.
- You can propose a production or governance guardrail for each modeling workflow.

## Coverage Cross-Reference
Use `docs/aspect-coverage-matrix.md` to confirm which project demonstrates each method (splits, imbalance handling, explainability, HPO, tracking, and productionization).

## Track Pages
For track-level documentation with artifact-focused guidance:

- `docs/tracks/foundations.md`
- `docs/tracks/production.md`
- `docs/tracks/ranking.md`
- `docs/tracks/forecasting.md`
- `docs/tracks/responsible-ai.md`
- `docs/tracks/optimization.md`
- `docs/tracks/reinforcement-learning.md`
- `docs/tracks/agent-frameworks.md`
- `docs/tracks/agentic-rl.md`
