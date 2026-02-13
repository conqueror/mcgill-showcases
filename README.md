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

## Project Catalog

| Project | Topic | Difficulty | Estimated Time | Prerequisites | Start Link |
|---|---|---|---|---|---|
| `sota-supervised-learning-showcase` | Supervised learning foundations + SOTA-style evaluation | Beginner-Intermediate | 1.5-2.5 hours | Python, basic classification/regression | [`projects/sota-supervised-learning-showcase/README.md`](projects/sota-supervised-learning-showcase/README.md) |
| `credit-risk-classification-capstone-showcase` | Credit default capstone (EDA, imbalance handling, threshold decisions) | Intermediate | 2-3 hours | Supervised ML basics, tabular data prep | [`projects/credit-risk-classification-capstone-showcase/README.md`](projects/credit-risk-classification-capstone-showcase/README.md) |
| `nyc-demand-forecasting-foundations-showcase` | Time-aware demand forecasting with explicit train/val/test splits | Intermediate | 1.5-2.5 hours | Python, regression basics, time-based validation intuition | [`projects/nyc-demand-forecasting-foundations-showcase/README.md`](projects/nyc-demand-forecasting-foundations-showcase/README.md) |
| `sota-unsupervised-semisup-showcase` | Unsupervised, semi-supervised, self-supervised, active learning | Intermediate | 2-3 hours | Python, basic ML intuition | [`projects/sota-unsupervised-semisup-showcase/README.md`](projects/sota-unsupervised-semisup-showcase/README.md) |
| `causalml-kaggle-showcase` | Causal inference, uplift modeling, policy simulation | Intermediate | 2-3 hours | Python, basic ML, Kaggle token | [`projects/causalml-kaggle-showcase/README.md`](projects/causalml-kaggle-showcase/README.md) |
| `mlops-drift-production-showcase` | MLOps lifecycle, drift detection, retraining decisions, local API serving | Intermediate | 2-3 hours | Python, ML basics, API basics | [`projects/mlops-drift-production-showcase/README.md`](projects/mlops-drift-production-showcase/README.md) |
| `xai-fairness-audit-showcase` | Explainability, subgroup fairness metrics, mitigation tradeoffs | Intermediate | 2-3 hours | Python, classification metrics | [`projects/xai-fairness-audit-showcase/README.md`](projects/xai-fairness-audit-showcase/README.md) |
| `automl-hpo-showcase` | Hyperparameter optimization strategy benchmarking (grid/random/TPE) | Intermediate | 1.5-2.5 hours | Python, model tuning basics | [`projects/automl-hpo-showcase/README.md`](projects/automl-hpo-showcase/README.md) |
| `eda-leakage-profiling-showcase` | Data profiling, missingness diagnostics, leakage analysis, split strategy comparison | Beginner-Intermediate | 1.5-2.0 hours | Python, pandas basics | [`projects/eda-leakage-profiling-showcase/README.md`](projects/eda-leakage-profiling-showcase/README.md) |
| `feature-engineering-dimred-showcase` | Encoding, feature selection, PCA/t-SNE/UMAP comparison | Beginner-Intermediate | 1.5-2.5 hours | Python, preprocessing basics | [`projects/feature-engineering-dimred-showcase/README.md`](projects/feature-engineering-dimred-showcase/README.md) |
| `rl-bandits-policy-showcase` | Multi-armed bandits, reward/regret analysis, policy recommendation | Intermediate | 1.5-2.5 hours | Python, probability basics | [`projects/rl-bandits-policy-showcase/README.md`](projects/rl-bandits-policy-showcase/README.md) |
| `batch-vs-stream-ml-systems-showcase` | Batch vs stream KPI pipelines, parity and latency analysis | Intermediate | 2-3 hours | Python, data systems basics | [`projects/batch-vs-stream-ml-systems-showcase/README.md`](projects/batch-vs-stream-ml-systems-showcase/README.md) |
| `model-release-rollout-showcase` | Canary rollout, promote/hold/rollback decisions, registry artifacts | Intermediate | 1.5-2.0 hours | Python, model metrics basics | [`projects/model-release-rollout-showcase/README.md`](projects/model-release-rollout-showcase/README.md) |
| `learning-to-rank-foundations-showcase` | Learning-to-rank foundations with grouped splits and NDCG | Intermediate | 1.5-2.5 hours | Python, ranking/recommendation basics | [`projects/learning-to-rank-foundations-showcase/README.md`](projects/learning-to-rank-foundations-showcase/README.md) |
| `ranking-api-productization-showcase` | FastAPI ranking service, schema contracts, structured logging, OpenAPI | Intermediate | 1.5-2.5 hours | Python, API basics, model serving basics | [`projects/ranking-api-productization-showcase/README.md`](projects/ranking-api-productization-showcase/README.md) |
| `demand-api-observability-showcase` | Demand prediction API with Prometheus metrics and optional OTel tracing | Intermediate | 1.5-2.5 hours | Python, API basics, observability basics | [`projects/demand-api-observability-showcase/README.md`](projects/demand-api-observability-showcase/README.md) |

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
```

Project-specific runs should be started from each project folder.

Contract note:
- `make check-contracts` bootstraps missing supervised artifacts in quick mode, then validates split/EDA/leakage/eval/experiment contracts.

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
- Core ML path: supervised -> unsupervised/semisup -> causal.
- Production path: supervised -> mlops drift -> batch vs stream.
- Forecasting path: nyc-demand forecasting foundations -> demand API observability -> model rollout.
- Ranking path: learning-to-rank foundations -> ranking API productization -> model rollout.
- Release path: mlops drift -> batch vs stream -> model rollout.
- Responsible AI path: supervised -> xai fairness -> causal.
- Optimization path: supervised -> automl hpo -> rl bandits.
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

## License
MIT License. See `LICENSE`.
