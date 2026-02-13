# Ranking API

Project: `projects/ranking-api-productization-showcase`

## Summary

This API serves ranking-related inference for grouped candidate items.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service health check |
| `GET` | `/model/schema` | Model feature/schema metadata |
| `POST` | `/score` | Score individual candidate rows |
| `POST` | `/rank` | Rank a batch of candidates |
| `POST` | `/predict` | Compatibility alias for ranked predictions |

## OpenAPI

- Download schema: <a id="ranking-openapi-link" href="../assets/openapi/ranking-api.json">ranking-api.json</a>
- Source schema in repo: `docs/api/assets/openapi/ranking-api.json`

## Local Commands

```bash
cd projects/ranking-api-productization-showcase
make export-openapi
make dev
```

## Example Requests And Responses

Base URL for local runs:

```text
http://127.0.0.1:8000
```

### `GET /health`

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `GET /model/schema`

```bash
curl -s http://127.0.0.1:8000/model/schema
```

```json
{
  "feature_names": [
    "experience_years",
    "avg_match_score",
    "response_rate",
    "recency_days"
  ],
  "meta": {
    "model_type": "lightgbm",
    "trained_at": "2026-02-13T00:00:00Z"
  }
}
```

### `POST /score`

```bash
curl -s -X POST http://127.0.0.1:8000/score \
  -H "content-type: application/json" \
  -d '{
    "records": [
      {
        "player_id": "candidate-001",
        "features": {
          "experience_years": 5,
          "avg_match_score": 0.81,
          "response_rate": 0.67,
          "recency_days": 12
        }
      },
      {
        "player_id": "candidate-002",
        "features": {
          "experience_years": 2,
          "avg_match_score": 0.76,
          "response_rate": 0.51,
          "recency_days": 30
        }
      }
    ]
  }'
```

```json
{
  "scores": [
    {
      "player_id": "candidate-001",
      "score": 0.742
    },
    {
      "player_id": "candidate-002",
      "score": 0.611
    }
  ]
}
```

### `POST /rank`

```bash
curl -s -X POST http://127.0.0.1:8000/rank \
  -H "content-type: application/json" \
  -d '{
    "records": [
      {
        "player_id": "candidate-001",
        "features": {
          "experience_years": 5,
          "avg_match_score": 0.81,
          "response_rate": 0.67,
          "recency_days": 12
        }
      },
      {
        "player_id": "candidate-002",
        "features": {
          "experience_years": 2,
          "avg_match_score": 0.76,
          "response_rate": 0.51,
          "recency_days": 30
        }
      }
    ]
  }'
```

```json
{
  "rankings": [
    {
      "player_id": "candidate-001",
      "score": 0.742,
      "rank": 1
    },
    {
      "player_id": "candidate-002",
      "score": 0.611,
      "rank": 2
    }
  ]
}
```

### `POST /predict` (compatibility alias)

`/predict` accepts the same request schema as `/score` and returns the same `scores` payload shape.

## ReDoc Viewer

<div id="redoc-ranking"></div>
<script src="https://cdn.jsdelivr.net/npm/redoc@2.1.3/bundles/redoc.standalone.js"></script>
<script>
  (function () {
    const target = document.getElementById("redoc-ranking");
    const specAnchor = document.getElementById("ranking-openapi-link");

    if (!target || !specAnchor) {
      return;
    }
    if (!window.Redoc || typeof window.Redoc.init !== "function") {
      target.innerHTML =
        "<p><strong>Unable to load embedded ReDoc.</strong> Use the OpenAPI JSON download link above.</p>";
      return;
    }

    try {
      window.Redoc.init(specAnchor.href, { hideDownloadButton: false }, target);
    } catch (_error) {
      target.innerHTML =
        "<p><strong>Unable to load embedded ReDoc.</strong> Use the OpenAPI JSON download link above.</p>";
    }
  })();
</script>
