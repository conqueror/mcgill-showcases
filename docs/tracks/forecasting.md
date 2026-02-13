# Forecasting Track

This track focuses on time-aware demand forecasting and API observability for forecast consumption.

## Recommended Sequence

1. `projects/nyc-demand-forecasting-foundations-showcase`
2. `projects/demand-api-observability-showcase`
3. `projects/mlops-drift-production-showcase` (optional extension)

## Core Skills Covered

- Chronological train/val/test splitting.
- Time feature engineering for demand prediction.
- Forecast metrics interpretation (`MAE`, `RMSE`, `sMAPE`).
- Exposing forecasting models through monitored APIs.

## Evidence Artifacts To Inspect

- `artifacts/splits/time_split_manifest.json`
- `artifacts/eval/metrics_summary.csv`
- `artifacts/eval/prediction_examples.csv`
- `artifacts/metrics.json` in API observability showcase
- `/metrics` endpoint counters and latency histograms

## Suggested Reflection Prompts

- Which forecast error metric best reflects product risk?
- How would you detect seasonal drift in near real time?
- Which API metrics should block rollout to production?
