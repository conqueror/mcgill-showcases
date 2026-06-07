# McGill ML Showcases

Public, student-friendly machine learning showcase projects for learning by doing.

This repository contains tutorial-style projects with reproducible tooling (`uv` + `make`), clear learning flows, and practical artifacts.

[![CI](https://github.com/conqueror/mcgill-showcases/actions/workflows/ci.yml/badge.svg)](https://github.com/conqueror/mcgill-showcases/actions/workflows/ci.yml)
[![Markdown Links](https://github.com/conqueror/mcgill-showcases/actions/workflows/markdown-links.yml/badge.svg)](https://github.com/conqueror/mcgill-showcases/actions/workflows/markdown-links.yml)
[![Notebook Smoke](https://github.com/conqueror/mcgill-showcases/actions/workflows/notebooks-smoke.yml/badge.svg)](https://github.com/conqueror/mcgill-showcases/actions/workflows/notebooks-smoke.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Table of Contents
- [Start Here](#start-here)
- [Project Catalog](#project-catalog)
- [Clean-Checkout Data And Artifacts](#clean-checkout-data-and-artifacts)
- [Repository Commands](#repository-commands)
- [Documentation Site](#documentation-site)
- [Learning Path](#learning-path)
- [Coverage Matrix](#coverage-matrix)
- [How to Get Help](#how-to-get-help)
- [Contributing](#contributing)
- [License](#license)

## Start Here
1. Install Python 3.11+ and `uv`.
2. Run:
```bash
make sync
```
3. Pick one project from the catalog below.
4. Enter that project and follow its `README.md`.

If this is your first time, start with `sota-supervised-learning-showcase`.
If you want the deep-learning sequence specifically, start with `deep-learning-math-foundations-showcase`.

## Project Catalog

| Project | Topic | Difficulty | Estimated Time | Prerequisites | Start Link |
|---|---|---|---|---|---|
| `deep-learning-math-foundations-showcase` | Essential math for deep learning: vectors, derivatives, entropy, and gradient descent | Beginner | 1.0-1.5 hours | Python basics, high-school algebra | [`projects/deep-learning-math-foundations-showcase/README.md`](projects/deep-learning-math-foundations-showcase/README.md) |
| `neural-network-foundations-showcase` | Perceptrons, activations, backprop intuition, initialization, and decision boundaries | Beginner | 1.0-1.5 hours | Python, basic algebra, deep-learning math foundations recommended | [`projects/neural-network-foundations-showcase/README.md`](projects/neural-network-foundations-showcase/README.md) |
| `pytorch-training-regularization-showcase` | PyTorch training loops, optimizers, schedulers, dropout, batch norm, and regularization experiments | Beginner-Intermediate | 1.5-2.0 hours | Python, neural-network foundations recommended | [`projects/pytorch-training-regularization-showcase/README.md`](projects/pytorch-training-regularization-showcase/README.md) |
| `sota-supervised-learning-showcase` | Supervised learning foundations + SOTA-style evaluation | Beginner-Intermediate | 1.5-2.5 hours | Python, basic classification/regression | [`projects/sota-supervised-learning-showcase/README.md`](projects/sota-supervised-learning-showcase/README.md) |
| `credit-risk-classification-capstone-showcase` | Credit default capstone (EDA, imbalance handling, threshold decisions) | Intermediate | 2-3 hours | Supervised ML basics, tabular data prep | [`projects/credit-risk-classification-capstone-showcase/README.md`](projects/credit-risk-classification-capstone-showcase/README.md) |
| `nyc-demand-forecasting-foundations-showcase` | Time-aware demand forecasting with explicit train/val/test splits | Intermediate | 1.5-2.5 hours | Python, regression basics, time-based validation intuition | [`projects/nyc-demand-forecasting-foundations-showcase/README.md`](projects/nyc-demand-forecasting-foundations-showcase/README.md) |
| `sota-unsupervised-semisup-showcase` | Unsupervised, semi-supervised, self-supervised, active learning | Intermediate | 2-3 hours | Python, basic ML intuition | [`projects/sota-unsupervised-semisup-showcase/README.md`](projects/sota-unsupervised-semisup-showcase/README.md) |
| `causalml-kaggle-showcase` | Causal inference, uplift modeling, policy simulation | Intermediate | 2-3 hours | Python, basic ML, Kaggle token | [`projects/causalml-kaggle-showcase/README.md`](projects/causalml-kaggle-showcase/README.md) |
| `mlops-drift-production-showcase` | MLOps lifecycle, drift detection, retraining decisions, local API serving | Intermediate | 2-3 hours | Python, ML basics, API basics | [`projects/mlops-drift-production-showcase/README.md`](projects/mlops-drift-production-showcase/README.md) |
| `xai-fairness-audit-showcase` | Explainability, subgroup fairness metrics, mitigation tradeoffs | Intermediate | 2-3 hours | Python, classification metrics | [`projects/xai-fairness-audit-showcase/README.md`](projects/xai-fairness-audit-showcase/README.md) |
| `automl-hpo-showcase` | Hyperparameter optimization strategy benchmarking (grid/random/TPE) | Intermediate | 1.5-2.5 hours | Python, model tuning basics | [`projects/automl-hpo-showcase/README.md`](projects/automl-hpo-showcase/README.md) |
| `autoresearch` | Fixed-budget autonomous research loops with Codex/Claude launch briefs for macOS and Unix | Intermediate-Advanced | 2-3 hours | Python, basic ML, Git, access to Apple Silicon or an NVIDIA GPU for the real upstream path | [`projects/autoresearch/README.md`](projects/autoresearch/README.md) |
| `agentic-course-assistant-showcase` | Agent routing, tools, guardrails, traces, eval rubrics, A2A/session/memory concepts, and optional OpenAI Agents SDK / Google ADK examples | Intermediate | 1-1.5 hours | Python, basic ML workflow vocabulary | [`projects/agentic-course-assistant-showcase/README.md`](projects/agentic-course-assistant-showcase/README.md) |
| `adaptive-course-assistant-rl-showcase` | Learned pedagogical intervention around a deterministic course assistant: contextual bandit, tutoring MDP, Q-learning, SARSA, REINFORCE, optional DQN/PPO bridge, policy export, and governance | Intermediate | 1.5-2.5 hours | Python, basic ML workflow vocabulary, `agentic-course-assistant-showcase` or `student-support-rl-showcase` helpful | [`projects/adaptive-course-assistant-rl-showcase/README.md`](projects/adaptive-course-assistant-rl-showcase/README.md) |
| `learning-agents-showcase` | Standalone capstone on learning an agent's orchestration policy with contextual bandits, tabular RL, offline evaluation, and governance artifacts; optional SDK/RLHF/MARL lanes remain scaffolded | Intermediate | 1.5-2.5 hours | Python, basic ML workflow vocabulary, `agentic-course-assistant-showcase` or `student-support-rl-showcase` helpful | [`projects/learning-agents-showcase/README.md`](projects/learning-agents-showcase/README.md) |
| `eda-leakage-profiling-showcase` | Data profiling, missingness diagnostics, leakage analysis, split strategy comparison | Beginner-Intermediate | 1.5-2.0 hours | Python, pandas basics | [`projects/eda-leakage-profiling-showcase/README.md`](projects/eda-leakage-profiling-showcase/README.md) |
| `feature-engineering-dimred-showcase` | Encoding, feature selection, PCA/t-SNE/UMAP comparison | Beginner-Intermediate | 1.5-2.5 hours | Python, preprocessing basics | [`projects/feature-engineering-dimred-showcase/README.md`](projects/feature-engineering-dimred-showcase/README.md) |
| `modern-nlp-pipeline-showcase` | Shared text pipeline for classification, retrieval, QA, and summarization on research abstracts | Intermediate | 2-3 hours | Python, basic ML, interest in NLP systems | [`projects/modern-nlp-pipeline-showcase/README.md`](projects/modern-nlp-pipeline-showcase/README.md) |
| `rl-bandits-policy-showcase` | Multi-armed bandits, reward/regret analysis, policy recommendation | Intermediate | 1.5-2.5 hours | Python, probability basics | [`projects/rl-bandits-policy-showcase/README.md`](projects/rl-bandits-policy-showcase/README.md) |
| `student-support-rl-showcase` | Contextual bandits, MDPs, dynamic programming (exact Q*), tabular Q-learning and SARSA, REINFORCE policy gradients, optional DQN/PPO comparison, reward hacking, offline evaluation, and deployment caution | Intermediate | 1.5-2.5 hours | Python, probability basics, `rl-bandits-policy-showcase` helpful | [`projects/student-support-rl-showcase/README.md`](projects/student-support-rl-showcase/README.md) |
| `batch-vs-stream-ml-systems-showcase` | Batch vs stream KPI pipelines, parity and latency analysis | Intermediate | 2-3 hours | Python, data systems basics | [`projects/batch-vs-stream-ml-systems-showcase/README.md`](projects/batch-vs-stream-ml-systems-showcase/README.md) |
| `model-release-rollout-showcase` | Canary rollout, promote/hold/rollback decisions, registry artifacts | Intermediate | 1.5-2.0 hours | Python, model metrics basics | [`projects/model-release-rollout-showcase/README.md`](projects/model-release-rollout-showcase/README.md) |
| `learning-to-rank-foundations-showcase` | Learning-to-rank foundations with grouped splits and NDCG | Intermediate | 1.5-2.5 hours | Python, ranking/recommendation basics | [`projects/learning-to-rank-foundations-showcase/README.md`](projects/learning-to-rank-foundations-showcase/README.md) |
| `ranking-api-productization-showcase` | FastAPI ranking service, schema contracts, structured logging, OpenAPI | Intermediate | 1.5-2.5 hours | Python, API basics, model serving basics | [`projects/ranking-api-productization-showcase/README.md`](projects/ranking-api-productization-showcase/README.md) |
| `demand-api-observability-showcase` | Demand prediction API with Prometheus metrics and optional OTel tracing | Intermediate | 1.5-2.5 hours | Python, API basics, observability basics | [`projects/demand-api-observability-showcase/README.md`](projects/demand-api-observability-showcase/README.md) |

## Clean-Checkout Data And Artifacts

This repo keeps generated outputs out of git so students can reproduce them locally.
Most projects write files under `artifacts/` only after `make run`, `make smoke`, or a
similar project command. A clean checkout may therefore contain only placeholders such
as `.gitkeep`.

Raw local inputs are also kept out of git by default. Projects that need starter data
either generate it in code or ship a small bundled sample dataset inside `src/` so
tests and smoke runs work on a normal laptop without private files.

Use each project README as the source of truth, but the usual flow is:

```bash
make sync
make smoke  # or make run
make verify
make test
```

If `make verify` reports missing artifacts before a run, generate the artifacts first.
The verifier is checking the stable contract for what the project is expected to
produce, not requiring generated outputs to be committed.

## Repository Commands

Use root commands to run quality gates across all projects:

```bash
make help
make sync
make lint
make ty
make test
make check
make check-contracts
make verify
make smoke
make docs-build
make docs-serve
make docs-check
make harness-preflight
make harness-lint
```

Project-specific runs should be started from each project folder.

Contract note:
- `make check-contracts` bootstraps missing supervised artifacts in quick mode, then validates split/EDA/leakage/eval/experiment contracts.
- `make harness-preflight` and `make harness-lint` validate the repo-local public harness-lite bootstrap.

## Documentation Site
- MkDocs Material config: `mkdocs.yml`
- Docs dependency set: `docs/requirements-mkdocs.txt`
- Public URL: https://conqueror.github.io/mcgill-showcases/
- Live docs quick links:
  - [Home](https://conqueror.github.io/mcgill-showcases/)
  - [Getting Started](https://conqueror.github.io/mcgill-showcases/getting-started/)
  - [Showcase Architecture](https://conqueror.github.io/mcgill-showcases/showcase-architecture/)
  - [API Overview](https://conqueror.github.io/mcgill-showcases/api/)
  - [Ranking API](https://conqueror.github.io/mcgill-showcases/api/ranking-api/)
  - [Demand API](https://conqueror.github.io/mcgill-showcases/api/demand-api/)
  - [Glossary](https://conqueror.github.io/mcgill-showcases/glossary/)
- Local docs server:
```bash
make docs-serve
```
- Strict docs build check:
```bash
make docs-check
```
- API docs note:
  - GitHub Pages hosts static API reference pages and embedded ReDoc viewers backed by versioned OpenAPI JSON assets.
  - Interactive Swagger UI (`/docs`) is available when running each FastAPI showcase locally with `make dev`.
- Main docs entry points:
  - `docs/index.md`
  - `docs/showcase-architecture.md`
  - `docs/new-showcase-playbook.md`
  - `docs/api/index.md`
  - `docs/api/ranking-api.md`
  - `docs/api/demand-api.md`

## Learning Path
- Deep learning foundations path: deep-learning math foundations -> neural-network foundations -> pytorch training regularization -> supervised or unsupervised next.
- Core ML path: supervised -> unsupervised/semisup -> causal.
- Production path: supervised -> mlops drift -> batch vs stream.
- Forecasting path: nyc-demand forecasting foundations -> demand API observability -> model rollout.
- Ranking path: learning-to-rank foundations -> ranking API productization -> model rollout.
- NLP systems path: pytorch training regularization -> modern NLP pipeline -> learning to rank -> ranking API productization.
- Release path: mlops drift -> batch vs stream -> model rollout.
- Responsible AI path: supervised -> xai fairness -> causal.
- Optimization path: supervised -> automl hpo -> autoresearch -> rl bandits -> student support rl.
- Agent frameworks path: automl hpo -> autoresearch -> agentic course assistant -> model rollout.
- Agent-plus-RL bridge path: autoresearch -> agentic course assistant -> adaptive course assistant RL -> rerun adaptive DRL bridge -> student support RL.
- Learning-agent capstone path: autoresearch -> agentic course assistant -> adaptive course assistant RL -> learning agents showcase.
- Data quality path: eda leakage profiling -> feature engineering -> supervised contract artifacts.
- See detailed guidance in `docs/learning-path.md`.

## Coverage Matrix
- Full aspect mapping is available in `docs/aspect-coverage-matrix.md`.
- Use this matrix to match course topics to concrete commands and artifacts.

## How to Get Help
- Read `docs/faq.md` and `docs/troubleshooting.md` first.
- Ask learning questions using GitHub Issues template: "Learning Question".
- Open bug reports with reproducible steps and command output.

## Contributing
See `CONTRIBUTING.md` for setup, standards, and pull request workflow.

## Harness Lite
- Repo-local harness config: `.codex/config.toml`
- Routing manifest: `.codex/harness/role-skill-matrix.toml`
- Operating pack: `docs/agents/oodaris-harness-v2-operating-pack.md`

## License
MIT License. See `LICENSE`.
