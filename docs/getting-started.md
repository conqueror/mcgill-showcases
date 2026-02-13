# Getting Started

## Prerequisites
- Python 3.11+
- `uv` installed
- Git

## 5-Minute Setup
```bash
git clone git@github.com:conqueror/mcgill-showcases.git
cd mcgill-showcases
make sync
```

## Pick Your First Project
- Easiest start: `projects/sota-supervised-learning-showcase`
- Causal decisioning: `projects/causalml-kaggle-showcase`
- Modern unlabeled-data workflows: `projects/sota-unsupervised-semisup-showcase`
- Production monitoring and serving: `projects/mlops-drift-production-showcase`
- Credit-risk capstone from course notebooks: `projects/credit-risk-classification-capstone-showcase`
- Responsible AI auditing: `projects/xai-fairness-audit-showcase`
- Rollout and rollback simulation: `projects/model-release-rollout-showcase`
- Data diagnostics and leakage checks: `projects/eda-leakage-profiling-showcase`
- Ranking model training fundamentals: `projects/learning-to-rank-foundations-showcase`
- Ranking API serving and contracts: `projects/ranking-api-productization-showcase`
- Time-aware demand forecasting foundations: `projects/nyc-demand-forecasting-foundations-showcase`
- Demand API with metrics and tracing hooks: `projects/demand-api-observability-showcase`

## First Run Pattern
```bash
cd projects/<project-name>
make help
```

Then run the recommended quickstart in that project's README.

## Recommended Root Checks
After your first project run, validate the repository-level workflow:

```bash
make check-contracts
make check
make verify
make docs-check
```

- `make check-contracts` regenerates missing supervised artifacts in quick mode and validates contract files.
- `make check` runs lint, type checks, tests, and contract verification across projects.
- `make verify` validates per-project artifact manifests where available.
- `make docs-check` runs a strict MkDocs Material build for docs consistency.

## Docs Site
Run local docs with:

```bash
make docs-serve
```

Build static docs output with:

```bash
make docs-build
```

## Topic Coverage Guide
For a direct mapping from course topics to projects, commands, and artifacts, use:

- `docs/aspect-coverage-matrix.md`
