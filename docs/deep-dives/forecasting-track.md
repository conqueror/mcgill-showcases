# Forecasting Track Deep Dive

Projects:

- `projects/nyc-demand-forecasting-foundations-showcase`
- `projects/demand-api-observability-showcase`

## Why This Deep Dive

Use this track when you want a time-aware forecasting workflow that extends into API serving and observability:

1. Train and evaluate forecasting models with chronological split discipline.
2. Serve predictions through an API with metrics and tracing hooks.

## Phase 1: Forecasting Foundations

```bash
cd projects/nyc-demand-forecasting-foundations-showcase
make sync
make run
make verify
```

Quick demo mode:

```bash
make smoke
```

Key outputs:

- `artifacts/eval/metrics_summary.csv`
- `artifacts/eval/prediction_examples.csv`
- `artifacts/splits/time_split_manifest.json`

## Phase 2: Demand API Observability

```bash
cd projects/demand-api-observability-showcase
make sync
make train-demo
make test
make export-openapi
make verify
make dev
```

Key outputs:

- `artifacts/model.joblib`
- `artifacts/metrics.json`
- `openapi.json`

## Example: Inspect Forecast Metrics

```bash
cd projects/nyc-demand-forecasting-foundations-showcase
python - <<'PY'
import pandas as pd
df = pd.read_csv("artifacts/eval/metrics_summary.csv")
print(df.to_string(index=False))
PY
```

## Example: Demand API Smoke Checks

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "content-type: application/json" \
  -d '{"pickup_zone_id":132,"pickup_datetime":"2026-02-13T09:00:00Z"}'
curl -s http://127.0.0.1:8000/metrics | head -n 10
```

For complete demand API request and response examples, see [Demand API docs](../api/demand-api.md).

## How To Interpret Outputs

1. `time_split_manifest.json` should confirm strict chronological separation.
2. Evaluate forecast quality with multiple metrics (`MAE`, `RMSE`, `sMAPE`) rather than one number.
3. Prediction API behavior should be checked alongside `/metrics` telemetry for operational readiness.
4. Contract checks should keep OpenAPI and runtime endpoint behavior aligned.

## Next Step

Use [Coverage Matrix](../aspect-coverage-matrix.md) to map this track to adjacent topics like drift monitoring, rollout decisions, and experiment tracking.
