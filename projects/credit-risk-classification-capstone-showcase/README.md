# Credit Risk Classification Capstone Showcase

Reproducible script-first credit-risk capstone project with lightweight demo notebooks.

## Learning outcomes
- Build a binary default-risk target from loan status categories.
- Perform robust missingness diagnostics and categorical/numeric profiling.
- Compare up/down sampling and SMOTE methods while every logistic model uses balanced class weights (`none` means no resampling).
- Select baseline or ensemble models and an F1 threshold on validation rows, then evaluate the locked choice on test rows.
- Inspect validation precision/recall/F1 tradeoffs; these scores do not establish calibration or deployment decision costs.

## Quickstart
```bash
cd projects/credit-risk-classification-capstone-showcase
make sync
make run
make verify
```

Optional profiling run:
```bash
make sync-profiling
make run
```

## Key outputs
- `artifacts/diagnostics/feature_type_summary.csv`
- `artifacts/diagnostics/class_balance_train.csv`
- `artifacts/models/strategy_comparison.csv`
- `artifacts/models/best_model_summary.json`
- `artifacts/eval/metrics_summary.csv`
- `artifacts/eval/threshold_analysis.csv`
- `artifacts/splits/split_manifest.json`
- `artifacts/manifest.json`

## Clean notebook entry points
- `notebooks/01_data_diagnostics.ipynb`
- `notebooks/02_modeling_thresholds.ipynb`

These notebooks are intentionally lightweight and consume pipeline artifacts instead of duplicating training logic.
