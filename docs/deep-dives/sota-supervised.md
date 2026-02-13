# Supervised Learning Deep Dive

Project: `projects/sota-supervised-learning-showcase`

## Why This Deep Dive

Use this project when you want one end-to-end supervised workflow that covers:

- class imbalance handling,
- binary, multiclass, multilabel, and multioutput tasks,
- threshold-aware evaluation,
- model selection curves,
- regression baselines.

## Quickstart

```bash
cd projects/sota-supervised-learning-showcase
make sync
make run
make test
```

Optional model boosters:

```bash
make sync-boosting
```

## What You Should Inspect

| Artifact | Why it matters |
|---|---|
| `artifacts/binary_metrics.csv` | compare precision/recall/F1 under imbalance strategies |
| `artifacts/pr_curves.csv` and `artifacts/roc_curves.csv` | check threshold behavior and score ranking quality |
| `artifacts/classification_benchmark.csv` | compare model families (tree/boosting/ensemble) |
| `artifacts/validation_curve.csv` and `artifacts/learning_curve.csv` | diagnose underfitting vs overfitting |
| `artifacts/regression_benchmark.csv` | verify advanced regressors beat simple baselines |

## Example: Fast Artifact Inspection

```bash
cd projects/sota-supervised-learning-showcase
python - <<'PY'
import pandas as pd
bench = pd.read_csv("artifacts/classification_benchmark.csv")
print(bench.sort_values("pr_auc", ascending=False).head(5).to_string(index=False))
PY
```

## Example: Threshold Decision Framing

```python
# Pseudocode for converting model scores into policy:
# if score >= threshold:
#     approve_action()
# else:
#     hold_or_review()
#
# Choose threshold by business tradeoff:
# - higher threshold -> higher precision, lower recall
# - lower threshold -> higher recall, lower precision
```

## How To Interpret Outputs

1. If `ROC-AUC` is strong but `PR-AUC` is weak, class imbalance is likely dominating practical quality.
2. If train score is high and validation score plateaus early, increase regularization or simplify model.
3. If ensemble gains are marginal, prefer simpler models for explainability and maintenance.
4. If baseline regressors are close to advanced models, feature quality may be the bottleneck.

## Next Step

Move to [MLOps Drift Deep Dive](mlops-drift.md) to operationalize similar supervised models in a production-style workflow.
