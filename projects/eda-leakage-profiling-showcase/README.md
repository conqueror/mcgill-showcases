# EDA Leakage Profiling Showcase

Practice complete supervised data diagnostics before modeling: profiling, univariate/bivariate analysis, missingness, split strategies, and leakage checks.

## Learning outcomes
- Generate profiling and missingness artifacts.
- Compare stratified, group, and time-based split manifests.
- Inspect cross-validation split integrity via a stratified K-fold manifest.
- Produce leakage diagnostics before model training.

## Quickstart
```bash
cd projects/eda-leakage-profiling-showcase
make sync
make run
make verify
```

Optional profiling extras:
```bash
make sync-profiling
make run
```

## Key outputs
- `artifacts/eda/univariate_summary.csv`
- `artifacts/eda/bivariate_vs_target.csv`
- `artifacts/eda/missingness_summary.csv`
- `artifacts/leakage/leakage_report.csv`
- `artifacts/splits/split_manifest.json`
- `artifacts/splits/group_split_manifest.json`
- `artifacts/splits/timeseries_split_manifest.json`
- `artifacts/splits/cv_split_manifest.json`
