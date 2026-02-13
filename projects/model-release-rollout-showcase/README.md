# Model Release Rollout Showcase

Simulate canary rollout, promotion/hold/rollback decisions, and model registry updates.

## Learning outcomes
- Compare champion vs challenger metrics under canary traffic.
- Apply decision thresholds for promote/hold/rollback.
- Produce auditable rollout and rollback artifacts.

## Quickstart
```bash
cd projects/model-release-rollout-showcase
make sync
make run
make verify
```

## Key outputs
- `artifacts/registry/model_versions.json`
- `artifacts/rollout/canary_eval.csv`
- `artifacts/rollout/decision_log.json`
- `artifacts/rollout/rollback_plan.md`
