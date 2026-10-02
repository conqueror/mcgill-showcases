# Business Dataset Notes

This file explains how the business mode dataset is built and why each preprocessing step exists.

## What Data Is Used

Business mode expects a Lending Club-style CSV (`loan.csv`).

Default lookup:
- repository root `loan.csv`

You can override path:
- `--business-csv-path /path/to/loan.csv`

## Target Labels

You learn a binary risk task from resolved outcomes only.

- `0` (non-risky):
  - `Fully Paid`
  - `Does not meet the credit policy. Status:Fully Paid`
- `1` (risky):
  - `Charged Off`
  - `Does not meet the credit policy. Status:Charged Off`

Open statuses are excluded because they are not final outcomes.

## Feature Preparation Steps

1. Select mixed numeric + categorical features.
2. Parse strings into numbers:
   - `term`: `36 months` -> `36`
   - `int_rate`: `13.56%` -> `13.56`
   - `revol_util`: `83.7%` -> `83.7`
   - `emp_length`: `< 1 year` -> `0.5`
3. Fill numeric missing values with training-row medians (zero for a column missing throughout training).
4. Fill categorical missing values with `Unknown`.
5. One-hot encode using training-row categories; unseen test categories have all-zero indicators.
6. Fit standardization on training rows and transform both splits.
7. The stratified sample is drawn before these preprocessing steps. The pipeline uses the same seeded split to choose preprocessing fit rows and held-out evaluation rows; clustering and anomaly detection still fit and score the full sample.

## Why This Is Useful

You get realistic practice with:
- noisy fields,
- mixed data types,
- partial labels,
- class imbalance,
- practical runtime constraints.

## Useful Runtime Controls

- `--business-read-rows`: how many raw CSV rows to load.
- `--business-sample-size`: final sample size after filtering.
- `--labeled-fraction`: how much labeled data you keep in semi-supervised runs.
- `--active-learning-rounds`: how many label acquisition rounds you simulate.
- `--active-learning-query-size`: how many new labels you request each round.
