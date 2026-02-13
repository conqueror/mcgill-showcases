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
A: Typically OvO (SVC pipeline) is higher in this project’s current run.

Q: Why might that happen?  
A: Pairwise boundaries can be easier for SVC to separate than one-vs-rest boundaries in this feature space.

## Step 4: Multi-label + multi-output

Q: Why are these not the same problem?  
A:  
- Multi-label: multiple class tags per sample.  
- Multi-output: multiple output values (often continuous or structured vectors) per sample.

Q: In multi-output denoising, what does MAE per pixel mean?  
A: Average absolute error between predicted and true pixel intensities across all pixels.

## Step 5: Ensemble benchmark

Q: Which model wins on macro F1?  
A: In the current run, `voting_hard` is strongest, followed closely by `stacking`.

Q: Is the winner worth added complexity?  
A: Only if gain is meaningful for your use case and latency/maintenance costs are acceptable.

## Step 6: Model selection curves

Q: At what depth does validation performance peak?  
A: Around `max_depth = 12` in the current run.

Q: Does more data still help?  
A: Validation score still improves toward larger train sizes, but gains are smaller at the upper range.

## Step 7: Regression comparison

Q: Which model has lowest RMSE in this run?  
A: `linear_regression` currently edges out gradient boosting.

Q: Did every model beat baseline?  
A: Yes. Both linear and gradient boosting significantly beat `DummyRegressor`.

## Step 8: Conclusion rubric

A strong conclusion includes:
- chosen model,
- chosen metric,
- why baseline and simpler alternatives were not enough,
- one next experiment (feature engineering, threshold tuning, or calibration),
- one deployment concern (latency, fairness, drift, monitoring).
