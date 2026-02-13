# Feature Engineering and Dimensionality Reduction Showcase

Practice encoding, scaling, feature selection, and embedding methods with reproducible artifacts.

## Learning outcomes
- Compare one-hot vs ordinal encoding effects.
- Score features using mutual information and L1 sparsity.
- Evaluate PCA/t-SNE (and optional UMAP) representation quality.
- Generate profiling, missingness, leakage, and split-contract artifacts.
- Explore optional advanced feature engineering libraries.

## Quickstart
```bash
cd projects/feature-engineering-dimred-showcase
make sync
make run
make run-dimred
make verify
```

Advanced optional run:
```bash
make sync-advanced
make run-advanced
```

## Key outputs
- `artifacts/features/feature_matrix_summary.csv`
- `artifacts/selection/selection_scores.csv`
- `artifacts/dimred/embedding_quality_metrics.csv`
- `artifacts/dimred/embedding_plots.png`
- `artifacts/eda/profile_status.txt`
- `artifacts/advanced/summary.json`
