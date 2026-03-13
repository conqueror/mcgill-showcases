# PyTorch Training Regularization Showcase

Learn how to build, debug, and improve a real neural network training pipeline in PyTorch.

This project is the framework-engineering step of the deep-learning series. It keeps the code short enough to read in one sitting while still covering tensors, `nn.Module`, training loops, optimizers, schedulers, dropout, batch norm, early stopping, and regularization tradeoffs.

## What You Should Learn

By working through this showcase, you should be able to:

- explain how tensors, `Dataset`s, and `DataLoader`s feed a training loop,
- read a minimal `nn.Module` and understand how logits are produced,
- compare SGD, Adam, and RMSprop on the same classification problem,
- explain what schedulers, dropout, batch norm, and weight decay do,
- diagnose training stability using gradient norms and validation curves,
- interpret test-set errors from example-level error analysis.

## Prerequisites

- Python 3.11+
- `uv`
- basic PyTorch familiarity helps, but the docs assume you are still learning
- completion of `projects/neural-network-foundations-showcase` is strongly recommended

The default run uses an offline-friendly dataset path. A `fashion_mnist` option is included for a more realistic dataset once you want it.

## Quickstart

```bash
cd projects/pytorch-training-regularization-showcase
make sync
make smoke
make verify
make test
```

## Key Artifacts

- `artifacts/baseline_metrics.json`
- `artifacts/training_history.csv`
- `artifacts/optimizer_comparison.csv`
- `artifacts/learning_rate_schedule_comparison.csv`
- `artifacts/regularization_ablation.csv`
- `artifacts/gradient_health_report.md`
- `artifacts/error_analysis.csv`
- `artifacts/summary.md`
- `artifacts/manifest.json`

## How To Learn This Project

1. Start with `docs/learning-flow.md`.
2. Run `make smoke` for the fastest end-to-end path.
3. Inspect `baseline_metrics.json` and `training_history.csv` before comparing experiments.
4. Read `optimizer_comparison.csv`, then `learning_rate_schedule_comparison.csv`, then `regularization_ablation.csv`.
5. Use `gradient_health_report.md` and `error_analysis.csv` to connect metrics to failure modes.
6. Check your understanding with `docs/checkpoint-answer-key.md`.

## Makefile Commands

```bash
make sync
make run
make smoke
make run-optimizers
make run-regularization
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
  Run `make smoke` or `make run` first so the expected artifacts exist.
- Training looks unstable:
  Open `artifacts/gradient_health_report.md` and compare the optimizer and scheduler tables.
- FashionMNIST download fails:
  Use the default dataset path or `--dataset synthetic` while offline.

## Suggested Next Projects

- `projects/sota-supervised-learning-showcase`
- `projects/sota-unsupervised-semisup-showcase` if you want a broader PyTorch practice project afterward

## Project Structure

```text
pytorch-training-regularization-showcase/
├── README.md
├── Makefile
├── pyproject.toml
├── docs/
├── scripts/
├── src/pytorch_training_regularization_showcase/
├── tests/
├── artifacts/
└── data/
```
