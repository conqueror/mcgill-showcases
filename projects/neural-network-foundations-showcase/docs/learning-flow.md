# Learning Flow

## Step-by-Step Flow

1. Start with `artifacts/activation_comparison.csv` and identify which functions saturate and which keep growing.
2. Read `artifacts/loss_function_comparison.csv` to see how the same prediction can look mild or severe under different losses.
3. Open `artifacts/backprop_gradient_trace.csv` and narrate how output error becomes per-layer gradients.
4. Compare `artifacts/initialization_comparison.csv` to understand why zero initialization stalls learning symmetry.
5. Use `artifacts/training_curves.csv` and `artifacts/underfit_overfit_examples.csv` to identify fitting regimes.
6. Finish with `artifacts/decision_boundaries.png` and `artifacts/decision_boundary_summary.csv` to connect network structure to geometric behavior.

## Why This Order Works

The project moves from single-neuron behavior to whole-network behavior. That keeps the learner focused on one idea at a time:

- nonlinearities change what a layer can express,
- losses define what counts as a costly mistake,
- backpropagation turns cost into parameter updates,
- initialization and capacity change whether training succeeds,
- decision boundaries reveal what the model has actually learned.
