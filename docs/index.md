# McGill ML Showcases Documentation

Welcome to the documentation site for `mcgill-showcases`.

This repo is organized as learning-by-doing showcase projects with reproducible scripts, clear artifacts, and track-based progression paths.

## Flagship Agentic Systems Showcase

!!! tip "Agentic Course Assistant"
    Start here when you want to learn agent frameworks without making your first run depend on hosted credentials. The [Agentic Course Assistant deep dive](deep-dives/agentic-course-assistant.md) teaches routing, tools, guardrails, trace evidence, eval rubrics, and framework comparison through a deterministic offline harness first, then shows how the same design maps to optional OpenAI Agents SDK and Google ADK usage.

    - Track: [Agent Frameworks](tracks/agent-frameworks.md)
    - Project: `projects/agentic-course-assistant-showcase`
    - Default path: local, deterministic, and CI-safe with no API keys
    - Optional extension: live OpenAI or Google ADK runs after students install SDK extras and configure credentials

## Start Here

1. Run local setup from [Getting Started](getting-started.md) (or the repo [README.md](https://github.com/conqueror/mcgill-showcases/blob/main/README.md)).
2. Pick a project from the [catalog](#project-catalog) — or a curated [track guide](#curated-track-guides) — based on your goal.
3. Run one showcase end-to-end.
4. Interpret generated artifacts with the [Coverage Matrix](aspect-coverage-matrix.md).

## Project Catalog

All 25 showcases, grouped by area and ordered so each builds on the last. Project links open the
project's `README.md` on GitHub (the source of truth for setup and commands). For curated, guided
tours of related projects see [Curated Track Guides](#curated-track-guides); for end-to-end
sequences see [Learning Paths](#learning-paths).

**Browse by area:** [Deep Learning Foundations](#deep-learning-foundations) ·
[Data Foundations](#data-foundations) ·
[Supervised Learning and Applications](#supervised-learning-and-applications) ·
[Unsupervised, Causal, and NLP](#unsupervised-causal-and-nlp) ·
[Responsible AI](#responsible-ai) ·
[Optimization and Autonomous Research](#optimization-and-autonomous-research) ·
[Reinforcement Learning](#reinforcement-learning) ·
[Agents and Agentic RL](#agents-and-agentic-rl) ·
[MLOps and Production Systems](#mlops-and-production-systems) ·
[Serving APIs and Observability](#serving-apis-and-observability)

### Deep Learning Foundations
Build the neural-network toolkit from the math up.

- [`deep-learning-math-foundations-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/deep-learning-math-foundations-showcase/README.md) — essential math for deep learning: vectors, derivatives, entropy, and gradient descent.
- [`neural-network-foundations-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/neural-network-foundations-showcase/README.md) — perceptrons, activations, backprop intuition, initialization, and decision boundaries.
- [`pytorch-training-regularization-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/pytorch-training-regularization-showcase/README.md) — PyTorch training loops, optimizers, schedulers, dropout, batch norm, and regularization.

### Data Foundations
Understand and prepare data before you model it.

- [`eda-leakage-profiling-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/eda-leakage-profiling-showcase/README.md) — data profiling, missingness diagnostics, leakage analysis, and split-strategy comparison.
- [`feature-engineering-dimred-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/feature-engineering-dimred-showcase/README.md) — encoding, feature selection, and PCA/t-SNE/UMAP comparison.

### Supervised Learning and Applications
Core supervised modeling and applied case studies.

- [`sota-supervised-learning-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/sota-supervised-learning-showcase/README.md) — supervised learning foundations plus SOTA-style evaluation.
- [`credit-risk-classification-capstone-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/credit-risk-classification-capstone-showcase/README.md) — credit-default capstone: EDA, imbalance handling, and threshold decisions.
- [`nyc-demand-forecasting-foundations-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/nyc-demand-forecasting-foundations-showcase/README.md) — time-aware demand forecasting with explicit train/val/test splits.
- [`learning-to-rank-foundations-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/learning-to-rank-foundations-showcase/README.md) — learning-to-rank foundations with grouped splits and NDCG.

### Unsupervised, Causal, and NLP
Methods that go beyond labeled supervised learning.

- [`sota-unsupervised-semisup-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/sota-unsupervised-semisup-showcase/README.md) — unsupervised, semi-supervised, self-supervised, and active learning.
- [`causalml-kaggle-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/causalml-kaggle-showcase/README.md) — causal inference, uplift modeling, and policy simulation.
- [`modern-nlp-pipeline-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/modern-nlp-pipeline-showcase/README.md) — shared text pipeline for classification, retrieval, QA, and summarization.

### Responsible AI
Explain, audit, and make models fair.

- [`xai-fairness-audit-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/xai-fairness-audit-showcase/README.md) — explainability, subgroup fairness metrics, and mitigation tradeoffs.

### Optimization and Autonomous Research
Tune models and run autonomous, budgeted research loops.

- [`automl-hpo-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/automl-hpo-showcase/README.md) — hyperparameter-optimization strategy benchmarking (grid/random/TPE).
- [`autoresearch`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/autoresearch/README.md) — fixed-budget autonomous research loops with Codex/Claude launch briefs.

### Reinforcement Learning
Sequential decision-making from bandits to policy gradients.

- [`rl-bandits-policy-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/rl-bandits-policy-showcase/README.md) — multi-armed bandits, reward/regret analysis, and policy recommendation.
- [`student-support-rl-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/student-support-rl-showcase/README.md) — contextual bandits, MDPs, dynamic programming (exact Q*), Q-learning/SARSA, REINFORCE, optional DQN/PPO, reward hacking, and offline evaluation.

### Agents and Agentic RL
Build agents, then learn the policies that drive them.

- [`agentic-course-assistant-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/agentic-course-assistant-showcase/README.md) — agent routing, tools, guardrails, traces, eval rubrics, and optional OpenAI Agents SDK / Google ADK examples.
- [`adaptive-course-assistant-rl-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/adaptive-course-assistant-rl-showcase/README.md) — learned pedagogical intervention around a deterministic assistant: bandit, tutoring MDP, Q-learning/SARSA/REINFORCE, optional DQN/PPO bridge, and governance.
- [`learning-agents-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/learning-agents-showcase/README.md) — capstone on where learning lives in an agent: orchestration-policy RL, offline RL and off-policy evaluation, cost-aware cascades, governance, plus an OpenAI Agents SDK bridge, RLHF/DPO/GRPO/RLVR, MARL, and an optional NumPy DQN/PPO deep-RL lane.

### MLOps and Production Systems
Operate, monitor, and release models safely.

- [`mlops-drift-production-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/mlops-drift-production-showcase/README.md) — MLOps lifecycle, drift detection, retraining decisions, and local API serving.
- [`batch-vs-stream-ml-systems-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/batch-vs-stream-ml-systems-showcase/README.md) — batch vs stream KPI pipelines with parity and latency analysis.
- [`model-release-rollout-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/model-release-rollout-showcase/README.md) — canary rollout, promote/hold/rollback decisions, and registry artifacts.

### Serving APIs and Observability
Ship models behind real APIs with metrics and tracing.

- [`ranking-api-productization-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/ranking-api-productization-showcase/README.md) — FastAPI ranking service, schema contracts, structured logging, and OpenAPI.
- [`demand-api-observability-showcase`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/demand-api-observability-showcase/README.md) — demand-prediction API with Prometheus metrics and optional OTel tracing.

## Curated Track Guides

Hand-picked, artifact-focused tours through related projects (each is a full page in this site):

- [Foundations](tracks/foundations.md) — math foundations, neural-network mechanics, PyTorch training, core supervised and unsupervised workflows, EDA, and feature engineering.
- [Production](tracks/production.md) — serving, drift monitoring, rollout decisions, and system patterns.
- [Ranking](tracks/ranking.md) — grouped ranking modeling and API productization.
- [Forecasting](tracks/forecasting.md) — time-aware demand modeling and observability-ready APIs.
- [Responsible AI](tracks/responsible-ai.md) — fairness, explainability, and causal decision support.
- [Optimization](tracks/optimization.md) — HPO, agentic workflows, policy optimization, reward design, and offline policy evaluation.
- [Agent Frameworks](tracks/agent-frameworks.md) — deterministic agent workflows, course-assistant tools, guardrails, traces, eval rubrics, and OpenAI Agents SDK / Google ADK concepts.

## Learning Paths

For ordered, end-to-end sequences across tracks (e.g. "New to Applied ML", "Deep Learning
Foundations", "ML in Production", "Learning-Agent Bridge"), see the [Learning Path](learning-path.md)
page. Use the [Coverage Matrix](aspect-coverage-matrix.md) to map course topics to concrete
commands and artifacts.

## Project Deep Dives

- [Deep dive overview](deep-dives/index.md)
- [Supervised learning deep dive](deep-dives/sota-supervised.md)
- [Causal inference deep dive](deep-dives/causal-inference.md)
- [MLOps drift deep dive](deep-dives/mlops-drift.md)
- [Ranking track deep dive](deep-dives/ranking-track.md)
- [Forecasting track deep dive](deep-dives/forecasting-track.md)
- [Agentic course assistant deep dive](deep-dives/agentic-course-assistant.md)

## Contributor Entry

- [Showcase architecture map](showcase-architecture.md)
- [New project checklist](new-showcase-playbook.md)
- [Troubleshooting](troubleshooting.md) and [FAQ](faq.md)

## API Reference

- [API overview](api/index.md)
- [Ranking API docs](api/ranking-api.md)
- [Demand API docs](api/demand-api.md)
