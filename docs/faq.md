# FAQ

## Do I need all projects to run?
No. Start with one project and expand gradually.

## Do I need Kaggle credentials?
Only for `causalml-kaggle-showcase`.

## Do I need cloud services for new projects?
No. All new showcase projects are laptop-first. Some projects have optional advanced extras.

## Why are artifact folders mostly empty?
Some large generated artifacts are excluded from git to keep the repo lightweight. Core example artifacts and manifest contracts are kept for reproducibility.

## What does `make check` do at root?
It runs lint, type checks, tests, and supervised contract checks across projects.

## What does `make check-contracts` do?
It bootstraps missing supervised artifacts in quick mode and then validates required train/val/test split manifests, EDA outputs, leakage reports, and experiment logs.

## Do supervised showcases always enforce train/val/test instead of train/test?
Yes. Supervised showcase contracts enforce explicit `train_rows`, `val_rows`, and `test_rows` in split manifests.

## Where can I see which project covers each advanced topic?
Use `docs/aspect-coverage-matrix.md` for a direct mapping of topics to commands and output artifacts.

## Is there a generated docs site for this repo?
Yes. This repo uses MkDocs Material.

- Run locally: `make docs-serve`
- Build static site: `make docs-build`
- Run strict docs validation: `make docs-check`
- Public URL (after Pages enablement): `https://conqueror.github.io/mcgill-showcases/`

## Do we host API docs too?
Yes. API reference pages are part of the MkDocs site:

- `docs/api/ranking-api.md`
- `docs/api/demand-api.md`

Their schemas are sourced from:
- `docs/api/assets/openapi/ranking-api.json`
- `docs/api/assets/openapi/demand-api.json`

## Why are there project-level Makefiles and a root Makefile?
- Root Makefile: orchestrates all projects.
- Project Makefile: detailed workflows specific to that project.

## Which project should I do after supervised basics?
Choose by goal:
- Production focus: `mlops-drift-production-showcase`.
- Responsible AI focus: `xai-fairness-audit-showcase`.
- Optimization focus: `automl-hpo-showcase`.

## Where is the ranking workflow?
Use this two-project sequence:
- `projects/learning-to-rank-foundations-showcase` for grouped ranking model training and NDCG evaluation.
- `projects/ranking-api-productization-showcase` for FastAPI serving, schema contracts, structured logs, and OpenAPI export.

## Where is the forecasting and observability workflow?
Use this two-project sequence:
- `projects/nyc-demand-forecasting-foundations-showcase` for time-ordered demand forecasting with train/val/test.
- `projects/demand-api-observability-showcase` for demand serving APIs with Prometheus metrics and optional OTel instrumentation.

## Is there a direct showcase architecture map?
Yes. See `docs/showcase-architecture.md` for track-level mapping across projects.

## Where is the contributor checklist for adding a new showcase?
Use `docs/new-showcase-playbook.md`. It defines required project structure, artifact contracts, tests, CI integration, and documentation updates.
