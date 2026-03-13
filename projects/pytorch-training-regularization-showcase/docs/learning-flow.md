# Learning Flow

## Step-by-Step Flow

1. Start with `artifacts/baseline_metrics.json` to see the run-level outcomes.
2. Open `artifacts/training_history.csv` and identify when validation performance peaks.
3. Compare `artifacts/optimizer_comparison.csv` to see how update rules change convergence.
4. Read `artifacts/learning_rate_schedule_comparison.csv` to understand how step size affects late training.
5. Inspect `artifacts/regularization_ablation.csv` to compare dropout, batch norm, and weight decay.
6. Finish with `artifacts/gradient_health_report.md` and `artifacts/error_analysis.csv` to connect training mechanics to concrete mistakes.

## Why This Order Works

The learning path starts with one stable baseline, then changes one training ingredient at a time:

- optimizer,
- scheduler,
- regularization,
- error analysis.

That keeps the experiments interpretable instead of turning the project into a black-box tuning exercise.
