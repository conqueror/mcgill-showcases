# MLOps Drift Deep Dive

Project: `projects/mlops-drift-production-showcase`

## Why This Deep Dive

Use this project to practice the operational loop after model training:

- baseline train/eval,
- drift detection,
- retrain-vs-monitor decisioning,
- local API serving with reproducible artifacts.

## Quickstart

```bash
cd projects/mlops-drift-production-showcase
make sync
make run
make run-drift
make verify
```

Optional tracking path:

```bash
make sync-tracking
make run-tracking
```

## API Smoke Test

```bash
cd projects/mlops-drift-production-showcase
make serve
```

In another terminal:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "content-type: application/json" \
  -d '{"features":[0.2,0.1,0.4,0.0,0.7,0.3,0.1,0.9]}'
```

## Key Operational Artifacts

| Artifact | Operational decision it supports |
|---|---|
| `artifacts/metrics/train_eval_summary.csv` | is baseline quality good enough to serve |
| `artifacts/drift/drift_report.csv` | has feature distribution shifted |
| `artifacts/policy/retrain_recommendation.json` | retrain now vs monitor |
| `artifacts/tracking/runs.csv` | experiment traceability |
| `artifacts/manifest.json` | reproducibility and artifact completeness |

## Example: Read Retrain Recommendation

```bash
cd projects/mlops-drift-production-showcase
python - <<'PY'
import json
from pathlib import Path
path = Path("artifacts/policy/retrain_recommendation.json")
print(json.dumps(json.loads(path.read_text()), indent=2))
PY
```

## How To Interpret Outputs

1. Drift alert without large quality drop may justify monitor-only action.
2. Drift plus quality degradation supports retrain recommendation.
3. Version and run tracking should always tie back to model artifacts used in serving.
4. Keep API health and prediction behavior checks in the same runbook as model checks.

## Next Step

Combine this with [Ranking API docs](../api/ranking-api.md) and [Demand API docs](../api/demand-api.md) to practice contract-first serving patterns.
