# MLOps Drift Production Showcase

Hands-on student project for learning how models are trained, monitored, and served in production-like workflows.

## Learning outcomes
- Train and evaluate a baseline classifier.
- Track experiments and model metrics (CSV + optional MLflow).
- Detect feature drift with KS + PSI checks.
- Apply a retrain-vs-monitor decision policy.
- Serve predictions through a local FastAPI endpoint.

## Quickstart
```bash
cd projects/mlops-drift-production-showcase
make sync
make run
make run-drift
make verify
```

Advanced optional run:
```bash
make sync-tracking
make run-tracking
```

## API quick check
```bash
make serve
# In another terminal:
curl -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{"features":[0.2,0.1,0.4,0.0,0.7,0.3,0.1,0.9]}'
```

## Artifact map
- `artifacts/metrics/train_eval_summary.csv`
- `artifacts/tracking/runs.csv`
- `artifacts/tracking/mlflow_status.txt`
- `artifacts/drift/drift_report.csv`
- `artifacts/policy/retrain_recommendation.json`
- `artifacts/manifest.json`

See `docs/learning-guide.md` for a guided lab flow.
