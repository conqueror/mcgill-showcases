# Causal Inference Deep Dive

Project: `projects/causalml-kaggle-showcase`

## Why This Deep Dive

Use this project when you need to move from prediction to intervention decisions:

- ATE and subgroup treatment effect reasoning,
- uplift ranking for targeted actions,
- Qini and uplift-at-k evaluation,
- budget-aware policy simulation,
- confounding checks and interpretability.

## Quickstart

```bash
cd projects/causalml-kaggle-showcase
make sync
make download
make pipeline
make policy
make confounding
make verify
```

## Key Terms In Practice

- `ATE`: average effect across all users.
- `CATE` / `tau(x)`: estimated effect for user segments defined by features.
- Uplift score: model estimate used to rank who to treat first under budget constraints.

## Example: Connect Tau To Targeting

If a model predicts:

- User A: `tau(x)=0.07`
- User B: `tau(x)=0.01`

and budget allows one contact, policy should prioritize user A because expected incremental impact is higher.

## Example: Policy Run

```bash
cd projects/causalml-kaggle-showcase
python scripts/policy_simulator.py
```

Expected outputs include:

- `artifacts/policy_simulation.csv`
- `artifacts/policy_recommendations.csv`
- `artifacts/figures/policy_incremental_conversions.png`

## Example: Read Top Policy Rows

```bash
cd projects/causalml-kaggle-showcase
python - <<'PY'
import pandas as pd
df = pd.read_csv("artifacts/policy_recommendations.csv")
print(df.head(8).to_string(index=False))
PY
```

## How To Interpret Outputs

1. Positive ATE does not imply everyone benefits; inspect subgroup uplift.
2. Favor models that improve uplift-at-k and Qini, not only raw predictive metrics.
3. If propensity overlap is weak, treat causal claims as lower confidence.
4. Policy recommendations should include budget sensitivity, not one fixed threshold.

## Next Step

Cross-check production readiness with [MLOps Drift Deep Dive](mlops-drift.md), especially monitoring and retraining policy design.
