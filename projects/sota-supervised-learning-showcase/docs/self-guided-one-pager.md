# Self-Guided One Pager

Use this one page if you want the shortest path through the project.

## Run

```bash
cd projects/sota-supervised-learning-showcase
uv sync --extra dev
uv run python scripts/run_showcase.py
uv run pytest
```

## Open These Files in Order

1. `artifacts/binary_metrics.csv`
2. `artifacts/multiclass_metrics.csv`
3. `artifacts/multilabel_metrics.csv`
4. `artifacts/multioutput_metrics.csv`
5. `artifacts/classification_benchmark.csv`
6. `artifacts/model_selection_summary.json`
7. `artifacts/regression_benchmark.csv`

## What to Notice

- Binary metrics: precision vs recall tradeoff.
- Multi-class metrics: macro vs weighted F1 differences.
- Multi-label: one sample can have multiple true labels.
- Multi-output: model predicts many values together.
- Benchmark table: simple model vs ensemble gains.
- Model selection summary: best hyperparameter and curve behavior.
- Regression benchmark: every model should beat baseline.

## 5 Questions to Check Your Understanding

1. Why can high accuracy still be misleading for imbalanced data?
2. When would you choose macro F1 over weighted F1?
3. Why might stacking beat a single decision tree?
4. What does a flat validation curve suggest?
5. Why is baseline comparison mandatory in regression?

## Next Step

Take one concept from `docs/domain-use-cases.md` and write how you would apply it in your own domain.
Then open `notebooks/02_domain_case_studies.ipynb` and grade your task/metric/tradeoff choices.
Then complete `notebooks/03_error-analysis-workbench.ipynb` to make a threshold-based deployment decision.
