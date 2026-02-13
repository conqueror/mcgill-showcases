# Production Track

This track focuses on operational ML: model serving, drift detection, release decisions, and system-level tradeoffs.

## Recommended Sequence

1. `projects/mlops-drift-production-showcase`
2. `projects/batch-vs-stream-ml-systems-showcase`
3. `projects/model-release-rollout-showcase`
4. `projects/demand-api-observability-showcase`

## Core Skills Covered

- Monitoring feature and prediction drift.
- Serving models with contract-aware API endpoints.
- Canary release decisions and rollback criteria.
- Batch vs stream pipeline tradeoffs.
- Observability with structured logs, metrics, and traces.

## Evidence Artifacts To Inspect

- `artifacts/drift/` outputs in MLOps showcase
- `openapi.json` in API showcases
- `artifacts/registry/model_versions.json`
- rollout decision logs and simulation outputs

## Suggested Reflection Prompts

- Which KPI would trigger a rollback first and why?
- What drift signal is most actionable for retraining decisions?
- Where should alert thresholds differ between batch and online systems?
