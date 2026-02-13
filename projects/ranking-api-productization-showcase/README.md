# Ranking API Productization Showcase

Portable API-serving showcase focused on production-style ranking service patterns.

## Learning outcomes
- Serve a ranking model with FastAPI endpoints (`/health`, `/model/schema`, `/score`, `/rank`).
- Enforce request/response schema contracts with Pydantic.
- Use structured JSON request logs with trace IDs.
- Export OpenAPI for client and documentation workflows.

## Quickstart
```bash
cd projects/ranking-api-productization-showcase
make sync
make train-demo
make test
make export-openapi
```

Run local API:
```bash
make dev
```

## Key outputs
- `artifacts/model.txt`
- `artifacts/feature_names.json`
- `artifacts/model_meta.json`
- `openapi.json`
