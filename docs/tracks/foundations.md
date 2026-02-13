# Foundations Track

This track builds core ML execution discipline: data understanding, feature preparation, robust splitting, and evaluation quality.

## Recommended Sequence

1. `projects/sota-supervised-learning-showcase`
2. `projects/eda-leakage-profiling-showcase`
3. `projects/feature-engineering-dimred-showcase`
4. `projects/sota-unsupervised-semisup-showcase`

## Core Skills Covered

- Train/validation/test discipline.
- Stratified, group-aware, and time-aware split strategies.
- Univariate and bivariate EDA.
- Missingness diagnostics and leakage checks.
- Feature encoding, selection, and dimensionality reduction.

## Evidence Artifacts To Inspect

- `artifacts/splits/split_manifest.json`
- `artifacts/eda/univariate_summary.csv`
- `artifacts/eda/bivariate_vs_target.csv`
- `artifacts/leakage/leakage_report.csv`
- `artifacts/selection/selection_scores.csv`

## Suggested Reflection Prompts

- Which split strategy is most defensible for this dataset and why?
- Which feature engineering step improved generalization most?
- Which leakage check would fail first if pipeline order changed?
