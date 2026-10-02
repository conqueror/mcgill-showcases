# AutoML HPO Showcase

Benchmark hyperparameter optimization strategies on the same candidate space under fixed trial budgets.

## Learning outcomes
- Compare grid, random, TPE (Optuna), and optional Hyperopt searches.
- Track best score and average score per strategy.
- Understand validation performance as the trial budget grows; elapsed time and compute cost are not measured.
- Log experiments with optional MLflow backend.

## Quickstart
```bash
cd projects/automl-hpo-showcase
make sync
make run
make run-budget
make verify
```

Advanced optional run:
```bash
make sync-advanced
make run-advanced
```

## Key outputs
- `artifacts/hpo/trials.csv`
- `artifacts/hpo/strategy_comparison.csv`
- `artifacts/hpo/best_configs.json`
- `artifacts/hpo/cost_vs_score.csv`
- `artifacts/hpo/cost_vs_score.png`
- `artifacts/hpo/mlflow_status.txt`
