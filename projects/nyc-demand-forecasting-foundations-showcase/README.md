# NYC Demand Forecasting Foundations Showcase

Course-friendly forecasting showcase focused on time-aware demand modeling fundamentals.

## Learning outcomes
- Build hourly zone-level taxi demand data from trip-style events.
- Enforce explicit time-based `train/val/test` splitting.
- Train a LightGBM Poisson regressor for demand prediction.
- Evaluate forecasts with MAE, RMSE, and sMAPE.

## Quickstart
```bash
cd projects/nyc-demand-forecasting-foundations-showcase
make sync
make run
make verify
```

Quick run for demos:
```bash
make smoke
```

Optional real TLC mode:
```bash
make download-data
make run-real
```

## Key outputs
- `artifacts/data/training_dataset_sample.csv`
- `artifacts/model/model.joblib`
- `artifacts/model/model_meta.json`
- `artifacts/eval/metrics_summary.csv`
- `artifacts/eval/prediction_examples.csv`
- `artifacts/splits/time_split_manifest.json`
- `artifacts/manifest.json`
