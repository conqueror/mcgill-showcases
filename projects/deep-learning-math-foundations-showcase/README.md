# Deep Learning Math Foundations Showcase

Learn the essential math behind deep learning through small, runnable Python examples and inspectable artifacts.

This project is designed as a self-guided bridge between lecture-style math content and the mechanics of later neural network showcases.

## What You Should Learn

By working through this showcase, you should be able to:

- explain vectors and matrices in model terms,
- connect derivatives and partial derivatives to gradients,
- interpret probability and uncertainty with simple simulations,
- explain entropy, cross-entropy, and KL divergence in plain language,
- read a gradient descent trace and describe why loss decreases.

## Prerequisites

- Python 3.11+
- `uv`
- basic comfort reading Python functions

No external datasets or GPUs are required.

## Quickstart

```bash
cd projects/deep-learning-math-foundations-showcase
make sync
make run
make verify
make test
```

## Key Artifacts

- `artifacts/vector_operations.csv`
- `artifacts/matrix_transformations.csv`
- `artifacts/derivative_examples.csv`
- `artifacts/gradient_descent_trace.csv`
- `artifacts/probability_simulations.csv`
- `artifacts/information_theory_summary.md`
- `artifacts/summary.md`
- `artifacts/manifest.json`

## How To Learn This Project

1. Start with `docs/learning-flow.md`.
2. Run the project end to end.
3. Open one artifact at a time and explain it in plain language.
4. Use `docs/concept-learning-map.md` to connect each file to the concept it teaches.
5. Use `docs/code-examples.md` to modify the examples and rerun them.
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
- A math result looks unfamiliar:
  Read the matching explanation in `docs/concept-learning-map.md` and `docs/code-examples.md`.

## Suggested Next Projects

- `projects/neural-network-foundations-showcase` once it exists
- `projects/sota-supervised-learning-showcase` if you want a stronger model-evaluation bridge first

## Project Structure

```text
deep-learning-math-foundations-showcase/
├── README.md
├── Makefile
├── pyproject.toml
├── docs/
├── scripts/
├── src/deep_learning_math_foundations_showcase/
├── tests/
├── artifacts/
└── data/
```
