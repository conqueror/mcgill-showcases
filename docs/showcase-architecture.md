# Showcase Architecture

This note maps related showcases into cohesive, in-repo learning tracks.

## Why this architecture

- Keep each showcase focused on one learning outcome.
- Preserve reproducibility and short demo runtime.
- Avoid monolithic project structure for students.

## Ranking Track

1. `projects/learning-to-rank-foundations-showcase`
- Grouped ranking data preparation and relevance labeling.
- LambdaRank model training.
- NDCG-focused evaluation and split artifacts.

2. `projects/ranking-api-productization-showcase`
- FastAPI ranking endpoints (`/health`, `/model/schema`, `/score`, `/rank`).
- Model artifact loading and schema-safe scoring.
- Structured request logging and OpenAPI export workflow.

## Forecasting And Observability Track

1. `projects/nyc-demand-forecasting-foundations-showcase`
- TLC-style hourly aggregation and time feature engineering.
- Explicit time-ordered train/val/test split.
- Demand forecasting metrics (`MAE`, `RMSE`, `sMAPE`).
- Optional real TLC download path with synthetic default mode.

2. `projects/demand-api-observability-showcase`
- FastAPI demand serving endpoint (`/predict`) and health checks.
- Prometheus metrics endpoint (`/metrics`) and request latency counters.
- Optional OpenTelemetry instrumentation hooks.
- OpenAPI export/check and API behavior tests.

## Intentional Scope Boundaries

- Full-size raw datasets are excluded to keep clone and run workflows lightweight.
- Large generated caches are excluded from version control.
- Each showcase keeps only teaching-critical components and artifacts.
