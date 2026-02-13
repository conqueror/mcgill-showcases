# Learning Flow (Self-Guided)

## What You Should Learn

By the end of this flow, you should be able to:
- identify the correct supervised learning task type,
- choose metrics that match the problem risk,
- compare simple vs ensemble models with evidence,
- diagnose underfitting and overfitting,
- explain why baseline comparisons are required.

## Step-by-Step Flow

1. `10-15 min` Run the project and inspect generated artifacts.
2. `20 min` Study binary classification and imbalanced-class handling.
3. `15 min` Compare OvR and OvO for multi-class classification.
4. `15 min` Study multi-label and multi-output tasks.
5. `20 min` Compare tree and ensemble benchmarks.
6. `15 min` Analyze validation and learning curves.
7. `15 min` Compare regression baseline, linear model, and gradient boosting.
8. `10 min` Write a short model-selection conclusion.

## Commands

```bash
cd projects/sota-supervised-learning-showcase
uv sync --extra dev
uv run python scripts/run_showcase.py
uv run pytest
```

## Checkpoint Questions by Step

### Step 1: Pipeline run
- Can you explain what each artifact file represents?
- Can you identify which files are classification vs regression outputs?

### Step 2: Binary + imbalance
- If recall increases and precision drops, what changed in model behavior?
- Which sampling strategy performs best for your risk profile?

### Step 3: OvR vs OvO
- Which one has higher macro F1 in your run?
- Why might that happen for this feature space?

### Step 4: Multi-label + multi-output
- Why are these not the same problem?
- In multi-output denoising, what does MAE per pixel mean practically?

### Step 5: Ensemble benchmark
- Which model wins on macro F1?
- Is the winner worth its added complexity?

### Step 6: Model selection curves
- At what depth does validation performance peak?
- Does adding more training data still improve validation score?

### Step 7: Regression comparison
- Which model has the lowest RMSE?
- Did every model beat the baseline?

### Step 8: Conclusion
Write 5-8 lines:
- chosen model,
- chosen metric,
- why simpler alternatives were not enough,
- what you would improve next.

## How to Learn Intuitively (Not Memorization)

- Predict first, run second: write your expected outcome before opening each artifact.
- Compare before/after: use one controlled change (for example, sampling strategy) and observe impact.
- Explain in plain language: if you cannot explain a metric in one sentence, revisit the section.
- Transfer the idea: map each concept to a non-digit domain using `docs/domain-use-cases.md`.

## After You Attempt All Questions

- Check your understanding with `docs/checkpoint-answer-key.md`.
- Repeat the flow in notebook form: `notebooks/01_learning_journey.ipynb`.
- Practice domain decisions in notebook form: `notebooks/02_domain_case_studies.ipynb`.
- Practice threshold/confusion-matrix deployment decisions: `notebooks/03_error-analysis-workbench.ipynb`.
