# Checkpoint Answer Key

Use this only after you attempt the questions in:
- `docs/learning-flow.md`
- `docs/self-guided-one-pager.md`

## Step 1: Pipeline run

Q: Can you explain what each artifact file represents?  
A:  
- `binary_metrics.csv`: core binary classification scores for each sampling strategy.  
- `pr_curves.csv`: precision-recall points across thresholds.  
- `roc_curves.csv`: true-positive vs false-positive tradeoff across thresholds.  
- `multiclass_metrics.csv`: OvR and OvO comparison.  
- `multilabel_metrics.csv`: F1 scores for each label and overall averages.  
- `multioutput_metrics.csv`: per-pixel denoising error.  
- `classification_benchmark.csv`: performance across tree/ensemble classifiers.  
- `validation_curve.csv` and `learning_curve.csv`: model-selection diagnostics.  
- `regression_benchmark.csv`: baseline vs linear vs gradient boosting regression.  
- `model_selection_summary.json`: compact best-configuration summary.

Q: Can you identify which files are classification vs regression outputs?  
A: Everything above is classification-focused except `regression_benchmark.csv`; model-selection files are classification diagnostics.

## Step 2: Binary + imbalance

Q: If recall increases and precision drops, what changed?  
A: The model predicts more positives. It catches more true positives but also raises false positives.

Q: Which sampling strategy is best for your risk profile?  
A:  
- High false-negative cost: prefer higher recall strategy.  
- High false-positive cost: prefer higher precision strategy.  
- Balanced risk: choose stronger F1 with acceptable precision and recall.

## Step 3: OvR vs OvO

Q: Which one has higher macro F1 in this run?  
A: Compare `f1_macro` in your `multiclass_metrics.csv`; both strategies use the same scaled logistic regression estimator.

Q: Why might that happen?  
A: OvO fits a classifier for each pair of classes; OvR fits each class against all others. Check your results before attributing a difference to these training tasks.

## Step 4: Multi-label + multi-output

Q: Why are these not the same problem?  
A:  
- Multi-label: multiple class tags per sample.  
- Multi-output: multiple output values (often continuous or structured vectors) per sample.

Q: In multi-output denoising, what does MAE per pixel mean?  
A: Average absolute error between predicted and true pixel intensities across all pixels.

## Step 5: Ensemble benchmark

Q: Which model wins on macro F1?  
A: Find the largest `f1_macro` in your `classification_benchmark.csv`. The winner can change with the split, installed versions, and optional boosters.

Q: Is the winner worth added complexity?  
A: Only if gain is meaningful for your use case and latency/maintenance costs are acceptable.

## Step 6: Model selection curves

Q: At what depth does validation performance peak?  
A: Find the largest `validation_mean` in your `validation_curve.csv`; `model_selection_summary.json` records that depth.

Q: Does more data still help?  
A: Compare `validation_mean` across train sizes in your `learning_curve.csv` to see whether the score improves and whether the gains shrink.

## Step 7: Regression comparison

Q: Which model has lowest RMSE in this run?  
A: Find the smallest `rmse` in your `regression_benchmark.csv`.

Q: Did every model beat baseline?  
A: Compare each model's `rmse` with `baseline_dummy_mean` in your run; a smaller value beats the baseline on that split.

## Step 8: Conclusion rubric

A strong conclusion includes:
- chosen model,
- chosen metric,
- why baseline and simpler alternatives were not enough,
- one next experiment (feature engineering, threshold tuning, or calibration),
- one deployment concern (latency, fairness, drift, monitoring).
