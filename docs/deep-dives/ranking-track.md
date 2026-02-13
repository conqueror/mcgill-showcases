# Ranking Track Deep Dive

Projects:

- `projects/learning-to-rank-foundations-showcase`
- `projects/ranking-api-productization-showcase`

## Why This Deep Dive

Use this track when you want to go from ranking model training to production-style ranking inference APIs:

1. Learn grouped ranking data and NDCG-focused evaluation.
2. Productize ranked inference with contract-first FastAPI endpoints.

## Phase 1: Ranking Foundations

```bash
cd projects/learning-to-rank-foundations-showcase
make sync
make run
make verify
```

Key outputs:

- `artifacts/eval/ranking_metrics.json`
- `artifacts/eval/test_rankings_top10.csv`
- `artifacts/splits/group_split_manifest.json`

## Phase 2: Ranking API Productization

```bash
cd projects/ranking-api-productization-showcase
make sync
make train-demo
make test
make export-openapi
make dev
```

Key outputs:

- `artifacts/model.txt`
- `artifacts/feature_names.json`
- `artifacts/model_meta.json`
- `openapi.json`

## Example: Inspect NDCG Metrics

```bash
cd projects/learning-to-rank-foundations-showcase
python - <<'PY'
import json
from pathlib import Path
path = Path("artifacts/eval/ranking_metrics.json")
print(json.dumps(json.loads(path.read_text()), indent=2))
PY
```

## Example: API Smoke Checks

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/model/schema
```

For complete ranking API request and response examples, see [Ranking API docs](../api/ranking-api.md).

## How To Interpret Outputs

1. `group_split_manifest.json` should show strict group isolation across train/val/test.
2. NDCG gains are meaningful only if evaluation is group-correct and leakage-safe.
3. API request schema and model feature schema should remain aligned across training and serving.
4. Exported OpenAPI should be kept in sync with docs assets to avoid contract drift.

## Next Step

Continue with [Forecasting Track Deep Dive](forecasting-track.md) for a time-aware prediction + observability pipeline pattern.
