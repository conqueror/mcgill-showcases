# Learning Guide

1. Run `make train-demo` and inspect `artifacts/metrics.json`.
2. Start the service with `make dev` and hit `/health`, `/predict`, and `/metrics`.
3. Run `make openapi-check` before any intentional contract update with `make export-openapi`.
4. Toggle OTel settings to discuss instrumentation rollout strategy.
