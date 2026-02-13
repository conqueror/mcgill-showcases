# Concept Learning Map

Use this map when you want to connect an idea to the exact code and artifact that demonstrates it.

## Classification Concepts

### 1. Imbalanced classes
- Code: `src/sota_supervised_showcase/data.py`
- Function: `rebalance_binary_training_data`
- Strategies:
  - `upsample_minority`
  - `downsample_majority`
- Check artifact: `artifacts/binary_metrics.csv`
- Intuition:
  - If one class is rare, raw accuracy can look good while the model still misses important cases.

### 2. Binary classification
- Code: `src/sota_supervised_showcase/classification.py`
- Function: `evaluate_binary_classification`
- Task in this project:
  - `digit == 0` vs `digit != 0`
- Metrics in output:
  - accuracy, precision, recall, F1, PR AUC, ROC AUC
- Check artifacts:
  - `artifacts/binary_metrics.csv`
  - `artifacts/pr_curves.csv`
  - `artifacts/roc_curves.csv`

### 3. Multi-class classification (OvR and OvO)
- Code: `src/sota_supervised_showcase/classification.py`
- Function: `evaluate_multiclass_strategies`
- Models compared:
  - OvR + Logistic Regression
  - OvO + SVC
- Check artifact: `artifacts/multiclass_metrics.csv`
- Intuition:
  - OvR asks, "Is this class vs all others?"
  - OvO asks, "Between each pair of classes, who wins?"

### 4. Multi-label classification
- Code: `src/sota_supervised_showcase/classification.py`
- Function: `evaluate_multilabel_classification`
- Labels used:
  - `is_large_digit` (>= 5)
  - `is_odd_digit` (odd/even)
- Check artifact: `artifacts/multilabel_metrics.csv`
- Intuition:
  - One example can be in multiple categories at the same time.

### 5. Multi-output prediction
- Code: `src/sota_supervised_showcase/classification.py`
- Function: `evaluate_multioutput_denoising`
- Task:
  - Add noise to image pixels and predict clean pixels.
- Check artifact: `artifacts/multioutput_metrics.csv`
- Intuition:
  - The model predicts a full vector output, not a single class.

### 6. Model evaluation and benchmarking
- Code: `src/sota_supervised_showcase/classification.py`
- Functions:
  - `evaluate_binary_classification`
  - `build_classification_benchmark`
- Models included:
  - baseline, decision tree, bagging, boosting, random forest, voting, stacking
  - optional: XGBoost / LightGBM
- Check artifact: `artifacts/classification_benchmark.csv`

### 7. Model selection
- Code: `src/sota_supervised_showcase/classification.py`
- Function: `build_model_selection_summary`
- Check artifacts:
  - `artifacts/validation_curve.csv`
  - `artifacts/learning_curve.csv`
  - `artifacts/model_selection_summary.json`
- Intuition:
  - Validation curve helps tune hyperparameters.
  - Learning curve helps decide whether more data could help.

## Regression Concepts

### 1. Baseline regression
- Code: `src/sota_supervised_showcase/regression.py`
- Model: `DummyRegressor`
- Intuition:
  - Baseline tells you the minimum bar a real model must beat.

### 2. Linear regression
- Code: `src/sota_supervised_showcase/regression.py`
- Model: `LinearRegression`
- Intuition:
  - Useful when relationship is close to linear and interpretability matters.

### 3. Gradient boosting regression
- Code: `src/sota_supervised_showcase/regression.py`
- Models:
  - `GradientBoostingRegressor`
  - `manual_gradient_boosting_example`
- Intuition:
  - Build a sequence of weak models that fix previous errors.

### 4. Regression evaluation
- Check artifact: `artifacts/regression_benchmark.csv`
- Metrics:
  - MAE, MSE, RMSE, R2
- Intuition:
  - RMSE emphasizes larger errors.
  - MAE is easier to explain in original target units.
