# Neural Network Foundations Showcase

Learn what a neural network is doing before framework engineering takes over.

This project is a self-guided bridge between math foundations and PyTorch-based training workflows. It keeps every network component explicit so you can connect neurons, activations, loss values, and gradient updates to visible behavior on toy classification problems.

## What You Should Learn

By working through this showcase, you should be able to:

- explain how a perceptron turns features into a weighted sum and a prediction,
- describe why hidden layers make nonlinear decision boundaries possible,
- compare sigmoid, tanh, ReLU, and leaky ReLU in plain language,
- connect loss functions to the shape of prediction errors,
- trace how backpropagation converts output error into layer-wise gradients,
- recognize underfitting, good fit, and overfitting from training curves and validation gaps.

## Prerequisites

- Python 3.11+
- `uv`
- comfort reading basic Python and NumPy
- completion of `projects/deep-learning-math-foundations-showcase` is recommended

No external datasets or GPUs are required.

## Quickstart

```bash
cd projects/neural-network-foundations-showcase
make sync
make run
make verify
make test
```

## Key Artifacts

- `artifacts/activation_comparison.csv`
- `artifacts/loss_function_comparison.csv`
- `artifacts/backprop_gradient_trace.csv`
- `artifacts/initialization_comparison.csv`
- `artifacts/underfit_overfit_examples.csv`
- `artifacts/training_curves.csv`
- `artifacts/decision_boundary_summary.csv`
- `artifacts/decision_boundaries.png`
- `artifacts/summary.md`
- `artifacts/manifest.json`

## How To Learn This Project

1. Start with `docs/learning-flow.md`.
2. Run the project end to end.
3. Compare `activation_comparison.csv` and `loss_function_comparison.csv` before reading the code.
4. Open `backprop_gradient_trace.csv` and explain what each layer gradient means.
5. Use `decision_boundaries.png` next to `decision_boundary_summary.csv` to connect model capacity to geometry.
6. Check your understanding with `docs/checkpoint-answer-key.md`.

## Makefile Commands

```bash
make sync
make run
make smoke
make verify
make test
make ruff
make ty
make lint
make quality
make check
```

## Common Failure Modes

- `uv` is missing:
  Install `uv` first, then rerun `make sync`.
- `make verify` fails:
  Run `make run` first so the expected artifacts exist.
- The XOR plots do not make intuitive sense:
  Read `docs/concept-learning-map.md` and compare the perceptron panel against the hidden-layer panel.
- Training feels too abstract:
  Inspect `src/neural_network_foundations_showcase/backprop.py` and rerun the project after changing one activation function.

## Suggested Next Projects

- `projects/pytorch-training-regularization-showcase`
- `projects/sota-supervised-learning-showcase` if you want a stronger evaluation workflow afterward

## Project Structure

```text
neural-network-foundations-showcase/
├── README.md
├── Makefile
├── pyproject.toml
├── docs/
├── scripts/
├── src/neural_network_foundations_showcase/
├── tests/
├── artifacts/
└── data/
```
