# Learning Flow

## What You Should Learn

By the end of this flow, you should be able to:

- explain the role of vectors and matrices in model computation,
- describe derivatives as sensitivity,
- connect probability summaries to uncertainty,
- explain why entropy and cross-entropy matter for learning,
- interpret a simple gradient descent optimization trace.

## Step-by-Step Flow

1. Run the showcase and inspect the generated artifact files.
2. Read `artifacts/vector_operations.csv` and `artifacts/matrix_transformations.csv`.
3. Read `artifacts/derivative_examples.csv` and explain each derivative in words.
4. Read `artifacts/probability_simulations.csv` and describe what the simulations summarize.
5. Read `artifacts/information_theory_summary.md` and connect the definitions to loss intuition.
6. Read `artifacts/gradient_descent_trace.csv` and explain why the loss keeps decreasing.
7. Finish with `artifacts/summary.md` and answer the checkpoint questions.

## Commands

```bash
cd projects/deep-learning-math-foundations-showcase
uv sync --extra dev
uv run python scripts/run_showcase.py
uv run python scripts/verify_artifacts.py
uv run pytest
```

## Checkpoint Questions

- Why does the dot product feel like a weighted combination?
- Why is the derivative of `x^2` larger at `x = 2` than at `x = 1`?
- What does sample variance tell you that sample mean does not?
- Why does cross-entropy grow when confident predictions are wrong?
- Why does a smaller `x` value lower the loss in the `x^2` optimization example?
