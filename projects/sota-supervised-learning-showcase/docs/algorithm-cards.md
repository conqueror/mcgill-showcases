# Algorithm Cards: What It Is, Why It Helps, Where It Fits

Use these cards as quick intuition while you read the metrics and run experiments.

## 1. DummyClassifier (baseline)
- What it is: a trivial classifier (for example, always predicts the most common class).
- Why it matters: it is your minimum bar. If your real model cannot beat this, it is not useful.
- Think of it like: "always guess the majority answer."
- Common domain use: baseline for fraud, churn, and diagnosis classification tasks.

## 2. Logistic Regression
- What it is: a linear classifier with probabilistic output.
- Why it matters: fast, interpretable, and great first serious model.
- Think of it like: weighted voting by features.
- Common domain use: credit risk, customer churn, email spam detection.

## 3. One-vs-Rest (OvR)
- What it is: one binary model per class, each class vs all others.
- Why it matters: simple way to extend binary models to multi-class tasks.
- Think of it like: each class has its own "security guard" deciding if an item belongs.
- Common domain use: ticket routing, document category assignment.

## 4. One-vs-One (OvO)
- What it is: one model for each pair of classes.
- Why it matters: can perform well when pairwise boundaries are clean.
- Think of it like: many mini-duels, then majority winner.
- Common domain use: image or signal classification with clear class boundaries.

## 5. KNN for Multi-Label
- What it is: predicts labels based on similar nearby examples.
- Why it matters: naturally supports multiple labels for one sample.
- Think of it like: "show me similar past cases and copy their tags."
- Common domain use: skill tagging, content tagging, legal clause tagging.

## 6. KNN for Multi-Output
- What it is: predicts an entire vector output, not one label.
- Why it matters: demonstrates structured prediction (many outputs together).
- Think of it like: restore a noisy picture by borrowing patterns from similar clean images.
- Common domain use: denoising, multi-zone forecasting, multi-sensor prediction.

## 7. Decision Tree
- What it is: a sequence of if-then splits.
- Why it matters: easy to explain and visualize.
- Think of it like: asking a checklist of questions until a decision is reached.
- Common domain use: underwriting rules, clinical triage logic, operational rules.

## 8. Bagging
- What it is: train many models on sampled data and average predictions.
- Why it matters: reduces instability of single trees.
- Think of it like: ask many independent voters and average their answers.
- Common domain use: noisy tabular data with overfitting risk.

## 9. Random Forest
- What it is: bagging + random feature selection at each split.
- Why it matters: strong default model for many tabular problems.
- Think of it like: committee of trees, each looking at slightly different evidence.
- Common domain use: risk scoring, quality control, customer analytics.

## 10. AdaBoost
- What it is: sequentially re-weights hard examples so later models focus on mistakes.
- Why it matters: turns weak learners into a stronger ensemble.
- Think of it like: study hardest questions more after each quiz.
- Common domain use: classification where many cases are easy but edge cases matter.

## 11. Gradient Boosting (Classification)
- What it is: each new model learns residual errors from previous models.
- Why it matters: often strong on tabular data.
- Think of it like: iterative error-correction.
- Common domain use: churn prediction, claim approval risk, demand risk flags.

## 12. Voting Classifier
- What it is: combine different model predictions by majority vote.
- Why it matters: robust when models make different kinds of errors.
- Think of it like: panel decision.
- Common domain use: production systems combining simple and complex models.

## 13. Stacking Classifier
- What it is: train a meta-model on predictions from base models.
- Why it matters: learns when each base model should be trusted.
- Think of it like: a manager deciding whose opinion to trust per case.
- Common domain use: high-stakes classification where incremental gains matter.

## 14. XGBoost (optional)
- What it is: optimized gradient boosting implementation.
- Why it matters: excellent tabular performance in many competitions and industry settings.
- Think of it like: highly engineered boosting pipeline.
- Common domain use: risk, ranking, and structured tabular prediction.

## 15. LightGBM (optional)
- What it is: fast histogram-based gradient boosting.
- Why it matters: speed and memory efficiency on large tabular data.
- Think of it like: boosting optimized for large-scale runs.
- Common domain use: very large customer/product datasets.

## 16. DummyRegressor (baseline)
- What it is: predicts mean target value.
- Why it matters: baseline for every regression model.
- Think of it like: "guess the average every time."
- Common domain use: baseline for price, demand, and duration prediction.

## 17. Linear Regression
- What it is: linear relationship between features and target.
- Why it matters: simple, interpretable, and often strong baseline.
- Think of it like: each feature pushes prediction up or down by a fixed amount.
- Common domain use: pricing, capacity planning, cost estimation.

## 18. Gradient Boosting Regressor
- What it is: residual-correcting tree ensemble for continuous targets.
- Why it matters: captures nonlinear relationships.
- Think of it like: repeated correction of previous prediction errors.
- Common domain use: demand, revenue, and energy-load forecasting.

## 19. Validation Curve
- What it is: shows performance vs one hyperparameter.
- Why it matters: helps find useful complexity level.
- Think of it like: tuning difficulty until practice and exam performance align.

## 20. Learning Curve
- What it is: shows performance as data size grows.
- Why it matters: tells you whether more data may still help.
- Think of it like: checking if more practice still improves exam score.
