# XAI Fairness Audit Showcase

Learn explainability and fairness auditing with a reproducible tabular classification workflow.

## Learning outcomes
- Produce global feature importance with permutation methods.
- Inspect local feature contributions for individual predictions.
- Compute subgroup fairness metrics and disparity gaps.
- Compare baseline, pre-, in-, and post-processing mitigation strategies.
- Generate optional SHAP and LIME outputs for explainability comparison.

## Quickstart
```bash
cd projects/xai-fairness-audit-showcase
make sync
make run
make run-mitigations
make verify
```

Advanced optional run:
```bash
make sync-explainability
make run-explainability
```

## Key outputs
- `artifacts/explainability/global_importance.csv`
- `artifacts/explainability/local_explanations_sample.csv`
- `artifacts/fairness/group_metrics.csv`
- `artifacts/fairness/disparity_summary.md`
- `artifacts/mitigation/mitigation_tradeoff_table.csv`
- `artifacts/explainability/shap_status.txt`
- `artifacts/explainability/lime_status.txt`
