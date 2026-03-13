# Foundations Track

This track builds core ML execution discipline: math intuition, data understanding, feature preparation, robust splitting, and evaluation quality.

## Recommended Sequence

1. `projects/deep-learning-math-foundations-showcase`
2. `projects/neural-network-foundations-showcase`
3. `projects/pytorch-training-regularization-showcase`
4. `projects/sota-supervised-learning-showcase`
5. `projects/eda-leakage-profiling-showcase`
6. `projects/feature-engineering-dimred-showcase`
7. `projects/sota-unsupervised-semisup-showcase`

## Core Skills Covered

- Vectors, matrices, gradients, entropy, and optimization traces.
- Perceptrons, activations, backpropagation, and initialization choices.
- PyTorch tensors, training loops, optimizers, schedulers, and regularization tradeoffs.
- Train/validation/test discipline.
- Stratified, group-aware, and time-aware split strategies.
- Univariate and bivariate EDA.
- Missingness diagnostics and leakage checks.
- Feature encoding, selection, and dimensionality reduction.

## Evidence Artifacts To Inspect

- `artifacts/vector_operations.csv`
- `artifacts/gradient_descent_trace.csv`
- `artifacts/activation_comparison.csv`
- `artifacts/decision_boundary_summary.csv`
- `artifacts/optimizer_comparison.csv`
- `artifacts/regularization_ablation.csv`
- `artifacts/splits/split_manifest.json`
- `artifacts/eda/univariate_summary.csv`
- `artifacts/eda/bivariate_vs_target.csv`
- `artifacts/leakage/leakage_report.csv`
- `artifacts/selection/selection_scores.csv`

## Suggested Reflection Prompts

- Which split strategy is most defensible for this dataset and why?
- Where does model capacity start to help, and where does it start to overfit?
- Which optimizer or regularizer improved learning without hiding what the model was doing?
- Which feature engineering step improved generalization most?
- Which leakage check would fail first if pipeline order changed?
