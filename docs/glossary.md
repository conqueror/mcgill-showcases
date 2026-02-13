# Glossary

This glossary defines the core concepts used across `mcgill-showcases`.

For each term:
- Definition: what it is.
- Why it matters: why students and contributors should care.
- Where to see it: one or more in-repo projects/artifacts.

## Data Profiling And Quality

## yData Profiling
Definition: An automated EDA report that summarizes schema, missingness, distributions, and simple correlations.  
Why it matters: It gives a fast first-pass risk scan before modeling.  
Where to see it: `projects/eda-leakage-profiling-showcase`, `projects/credit-risk-classification-capstone-showcase`.

## Univariate Analysis
Definition: Per-feature analysis of distribution, central tendency, spread, and missingness.  
Why it matters: It detects outliers, constant columns, and suspicious value ranges early.  
Where to see it: `artifacts/eda/univariate_summary.csv` in supervised showcases.

## Bivariate Analysis
Definition: Relationship analysis between one feature and the target (correlation, target mean by category, etc.).  
Why it matters: It exposes predictive signal and potential proxy leakage.  
Where to see it: `artifacts/eda/bivariate_vs_target.csv`.

## Missingness Matrix / Missingness Summary
Definition: Visualization or tabular report of where and how often values are missing.  
Why it matters: Missingness can encode process bias or break model assumptions.  
Where to see it: `artifacts/eda/missingness_summary.csv`, optional `missingness_matrix.png`.

## Feature Type Inference
Definition: Classification of columns into logical types such as numeric, categorical, and datetime.  
Why it matters: Correct preprocessing choices depend on accurate feature typing.  
Where to see it: `artifacts/diagnostics/feature_type_summary.csv`.

## Data Leakage
Definition: Information in training features that would not be available at prediction time.  
Why it matters: Leakage inflates offline metrics and causes production failure.  
Where to see it: `artifacts/leakage/leakage_report.csv`.

## Distribution Shift
Definition: A change in data distribution between training and production contexts.  
Why it matters: Shift degrades model reliability over time.  
Where to see it: `projects/mlops-drift-production-showcase`.

## Covariate Shift
Definition: Shift in feature distribution while the target mechanism remains comparatively stable.  
Why it matters: Model recalibration or retraining may be needed even if label behavior changes slowly.  
Where to see it: drift artifacts in `projects/mlops-drift-production-showcase`.

## Concept Drift
Definition: Change in the relationship between features and target over time.  
Why it matters: Historic patterns become less predictive, requiring model updates.  
Where to see it: retraining decision flow in `projects/mlops-drift-production-showcase`.

## Imputation
Definition: Filling missing values using rules such as median, mean, or most-frequent category.  
Why it matters: Models and encoders generally require complete matrices.  
Where to see it: `projects/feature-engineering-dimred-showcase`.

## Splitting And Validation

## Train/Validation/Test Split
Definition: Three-way data partition: training for fitting, validation for model/threshold selection, test for final unbiased evaluation.  
Why it matters: It separates tuning decisions from final reporting.  
Where to see it: split manifests across supervised and forecasting showcases.

## Stratified Split
Definition: Split preserving target class proportions across train/val/test sets.  
Why it matters: It stabilizes metric estimates for imbalanced classification.  
Where to see it: `projects/sota-supervised-learning-showcase`, `projects/credit-risk-classification-capstone-showcase`.

## Group Split
Definition: Split that keeps related records (same group/query/entity) in a single partition.  
Why it matters: Prevents leakage across related entities and avoids optimistic scores.  
Where to see it: `projects/learning-to-rank-foundations-showcase`.

## Time-Ordered Split
Definition: Split based on chronology (past -> present -> future).  
Why it matters: Prevents future information leakage into training.  
Where to see it: `projects/nyc-demand-forecasting-foundations-showcase`.

## Cross-Validation (CV)
Definition: Repeated train/validation procedures across folds to estimate performance variability.  
Why it matters: Reduces reliance on one lucky split.  
Where to see it: `artifacts/splits/cv_split_manifest.json` in split-focused workflows.

## Stratified K-Fold
Definition: K-fold CV preserving class balance in each fold.  
Why it matters: Improves comparability under class imbalance.  
Where to see it: split utilities in `shared/python/ml_core/splits.py`.

## No-Overlap Check
Definition: Validation that train, validation, and test partitions do not share row identity/index.  
Why it matters: Overlap invalidates reported metrics.  
Where to see it: `no_overlap_checks_passed` in split manifests.

## Feature Engineering And Representation

## One-Hot Encoding
Definition: Categorical encoding creating one binary feature per category.  
Why it matters: Works well for linear/tree models with manageable cardinality.  
Where to see it: `projects/feature-engineering-dimred-showcase`.

## Label/Ordinal Encoding
Definition: Mapping categories to integer IDs.  
Why it matters: Useful for tree models or ordered categories, but risky for linear models when order is artificial.  
Where to see it: feature preprocessing in feature engineering and credit risk workflows.

## Entity Embeddings
Definition: Dense learned vectors representing high-cardinality categorical entities.  
Why it matters: Can capture latent similarity better than sparse one-hot vectors.  
Where to see it: advanced artifacts in `projects/feature-engineering-dimred-showcase`.

## FeatureTools
Definition: Automated feature engineering library for relational/deep feature synthesis.  
Why it matters: Speeds up candidate feature generation for tabular tasks.  
Where to see it: advanced status outputs in feature engineering showcase.

## tsfresh
Definition: Time-series feature extraction library creating statistical descriptors from sequences.  
Why it matters: Adds rich temporal descriptors for forecasting/classification pipelines.  
Where to see it: advanced status outputs in feature engineering showcase.

## autofeat
Definition: Automated nonlinear feature construction and selection tooling.  
Why it matters: Finds transformed features that simple manual pipelines may miss.  
Where to see it: advanced status outputs in feature engineering showcase.

## RFECV
Definition: Recursive feature elimination with cross-validation for subset selection.  
Why it matters: Reduces dimensionality while protecting validation quality.  
Where to see it: selection outputs in feature engineering showcase.

## PCA
Definition: Linear projection into principal components maximizing explained variance.  
Why it matters: Compresses correlated numeric features and supports visualization.  
Where to see it: dimensionality reduction outputs in feature engineering showcase.

## t-SNE
Definition: Nonlinear embedding optimized for local neighborhood preservation in 2D/3D.  
Why it matters: Useful for visual cluster inspection, not primary supervised features.  
Where to see it: dimensionality reduction comparisons in feature engineering showcase.

## UMAP
Definition: Nonlinear manifold projection preserving local and some global structure.  
Why it matters: Often faster and more scalable than t-SNE for exploratory visualization.  
Where to see it: dimensionality reduction comparisons in feature engineering showcase.

## Correlation Matrix
Definition: Pairwise correlation table for numeric features.  
Why it matters: Highlights redundancy and multicollinearity risk.  
Where to see it: `artifacts/eda/correlation_matrix.csv`.

## Imbalanced Learning

## Class Imbalance
Definition: Unequal frequency of target classes (e.g., fraud/non-fraud).  
Why it matters: Naive accuracy can look high while minority class performance is poor.  
Where to see it: class balance outputs in supervised and credit-risk showcases.

## Class Weighting
Definition: Increasing loss contribution of minority examples during training.  
Why it matters: Often a low-friction baseline for imbalance mitigation.  
Where to see it: strategy comparisons in credit-risk and supervised showcases.

## Over-Sampling
Definition: Increasing minority samples through replication or synthesis.  
Why it matters: Helps models see enough minority patterns.  
Where to see it: imbalance strategy outputs in supervised/credit-risk workflows.

## Under-Sampling
Definition: Reducing majority samples to rebalance classes.  
Why it matters: Can improve minority recall but may lose information.  
Where to see it: imbalance strategy outputs in supervised/credit-risk workflows.

## SMOTE
Definition: Synthetic Minority Over-sampling Technique creating synthetic minority examples via nearest neighbors.  
Why it matters: Provides richer minority coverage than naive duplication.  
Where to see it: optional imbalance methods in shared utilities and capstone workflows.

## SMOTETomek
Definition: SMOTE over-sampling followed by Tomek link cleaning.  
Why it matters: Balances classes and removes ambiguous border pairs.  
Where to see it: `shared/python/ml_core/imbalance.py`.

## SMOTEENN
Definition: SMOTE over-sampling followed by Edited Nearest Neighbors cleaning.  
Why it matters: Often improves minority signal quality after synthetic expansion.  
Where to see it: `shared/python/ml_core/imbalance.py`.

## Models And Optimization

## Baseline Model
Definition: Simple reference model used before advanced tuning.  
Why it matters: Prevents over-engineering and provides a sanity benchmark.  
Where to see it: model benchmark outputs in supervised and credit-risk showcases.

## LightGBM
Definition: Gradient boosting framework optimized for efficiency and strong tabular performance.  
Why it matters: Common strong baseline for tabular regression/classification/ranking.  
Where to see it: supervised, ranking, and forecasting tracks.

## XGBoost
Definition: Gradient boosting implementation with robust regularization and mature ecosystem.  
Why it matters: Reliable benchmark in many tabular tasks.  
Where to see it: classification benchmark outputs in supervised workflows.

## CatBoost
Definition: Gradient boosting implementation with strong categorical feature handling.  
Why it matters: Useful when categorical structure dominates signal.  
Where to see it: classification benchmark outputs in supervised workflows.

## Stacking
Definition: Ensemble technique training a meta-model over predictions of base models.  
Why it matters: Can improve robustness when models capture complementary patterns.  
Where to see it: ensemble strategy comparisons in supervised workflows.

## Hyperparameter Optimization (HPO)
Definition: Systematic search over model configuration space.  
Why it matters: Reduces manual trial-and-error and improves reproducibility.  
Where to see it: `projects/automl-hpo-showcase`.

## Grid Search
Definition: Exhaustive evaluation over a fixed hyperparameter grid.  
Why it matters: Transparent and reproducible, but expensive at scale.  
Where to see it: strategy comparison in AutoML showcase.

## Random Search
Definition: Random sampling of hyperparameter configurations.  
Why it matters: Often more efficient than dense grids in high-dimensional search spaces.  
Where to see it: strategy comparison in AutoML showcase.

## TPE (Tree-structured Parzen Estimator)
Definition: Bayesian optimization method modeling promising vs non-promising regions.  
Why it matters: Improves sample efficiency for HPO.  
Where to see it: advanced HPO strategy runs in AutoML showcase.

## Evaluation And Decisioning

## ROC-AUC
Definition: Area under the ROC curve across classification thresholds.  
Why it matters: Measures ranking quality of scores over all thresholds.  
Where to see it: supervised and credit-risk evaluation outputs.

## PR-AUC
Definition: Area under precision-recall curve.  
Why it matters: More informative than ROC-AUC on highly imbalanced targets.  
Where to see it: supervised and credit-risk evaluation outputs.

## Threshold Analysis
Definition: Metrics computed across multiple decision thresholds.  
Why it matters: Converts model scores into policy-aware decisions.  
Where to see it: `artifacts/eval/threshold_analysis.csv`.

## Learning Curve
Definition: Performance trend as training data size increases.  
Why it matters: Distinguishes data scarcity from model capacity issues.  
Where to see it: learning/validation diagnostics in supervised workflows.

## RMSE
Definition: Root mean squared error; emphasizes larger errors.  
Why it matters: Useful when large misses are especially costly.  
Where to see it: forecasting and demand API metrics outputs.

## MAE
Definition: Mean absolute error; average absolute deviation.  
Why it matters: Interpretable average error in original units.  
Where to see it: forecasting and demand API metrics outputs.

## R²
Definition: Proportion of variance explained by regression model.  
Why it matters: Quick fit-quality indicator, but should not be used alone.  
Where to see it: regression-oriented supervised evaluations.

## sMAPE
Definition: Symmetric mean absolute percentage error for forecast accuracy.  
Why it matters: Scale-aware metric common in demand forecasting contexts.  
Where to see it: forecasting metrics in NYC demand showcase.

## Calibration
Definition: Agreement between predicted probabilities and observed outcomes.  
Why it matters: Critical for risk scoring and threshold policy reliability.  
Where to see it: threshold and probability analysis in classification showcases.

## Explainability, Fairness, And Causal ML

## SHAP
Definition: Shapley-value-based feature attribution framework.  
Why it matters: Explains local and global model behavior with additive attributions.  
Where to see it: `projects/xai-fairness-audit-showcase`.

## LIME
Definition: Local surrogate explanation method around individual predictions.  
Why it matters: Useful for case-by-case interpretability.  
Where to see it: explainability outputs in XAI/fairness showcase.

## Fairness Audit
Definition: Evaluation of metric disparities across protected or relevant subgroups.  
Why it matters: Detects unequal error distribution and policy harm risk.  
Where to see it: `projects/xai-fairness-audit-showcase`.

## ATE
Definition: Average treatment effect across all units.  
Why it matters: Baseline causal estimate for intervention impact.  
Where to see it: `projects/causalml-kaggle-showcase`.

## CATE
Definition: Conditional average treatment effect for a subgroup/segment.  
Why it matters: Supports targeted policy and personalization decisions.  
Where to see it: `projects/causalml-kaggle-showcase`.

## Counterfactual
Definition: The unobserved outcome under an alternative treatment/action for the same unit.  
Why it matters: Core concept behind causal effect estimation.  
Where to see it: causal notebooks and policy simulation outputs.

## Uplift Modeling
Definition: Modeling incremental outcome caused by treatment versus control.  
Why it matters: Improves intervention targeting efficiency.  
Where to see it: causal uplift workflows and Qini analysis outputs.

## MLOps And Productionization

## OpenAPI Contract
Definition: Machine-readable API schema describing request/response shapes.  
Why it matters: Prevents client/server drift and supports contract-first development.  
Where to see it: `openapi.json` in ranking and demand API showcases.

## Contract Drift
Definition: Mismatch between checked-in OpenAPI schema and runtime API schema.  
Why it matters: Breaks client integrations if unchecked.  
Where to see it: API openapi-check commands in ranking/demand showcases.

## Structured Logging
Definition: JSON or key-value logging with consistent fields (trace IDs, route, status).  
Why it matters: Enables reliable observability and debugging.  
Where to see it: ranking and demand API productization showcases.

## RED Metrics
Definition: Rate, Errors, Duration service telemetry model.  
Why it matters: Core monitoring lens for API reliability.  
Where to see it: demand API observability metrics endpoints.

## Prometheus Metrics
Definition: Pull-based metrics exposed via `/metrics` in Prometheus format.  
Why it matters: Standard, low-friction production telemetry interface.  
Where to see it: `projects/demand-api-observability-showcase`.

## OpenTelemetry (OTel)
Definition: Standard for traces, metrics, and logs instrumentation across services.  
Why it matters: Supports end-to-end tracing and consistent observability semantics.  
Where to see it: optional instrumentation hooks in demand API observability showcase.

## Canary Rollout
Definition: Progressive release where new model/version serves a small traffic subset first.  
Why it matters: Limits blast radius of regressions.  
Where to see it: `projects/model-release-rollout-showcase`.

## Rollback
Definition: Reverting to a previous known-good model/service version.  
Why it matters: Essential safety mechanism for production incidents.  
Where to see it: rollout decision workflows and registry artifacts.

## Model Registry
Definition: Versioned catalog of model artifacts and metadata.  
Why it matters: Enables traceability, promotion control, and reproducible serving.  
Where to see it: production and rollout showcase artifacts.

## Forecasting, Ranking, And Policy Terms

## Horizon
Definition: Future time span over which forecasts are generated/evaluated.  
Why it matters: Different horizons imply different uncertainty and use cases.  
Where to see it: demand forecasting workflows.

## Query Group
Definition: Set of candidate items ranked together for one ranking request/context.  
Why it matters: Ranking loss and metrics are computed per group/query.  
Where to see it: `projects/learning-to-rank-foundations-showcase`.

## NDCG
Definition: Normalized Discounted Cumulative Gain, a rank-quality metric emphasizing top positions.  
Why it matters: Better reflects ranking usefulness than plain accuracy.  
Where to see it: ranking metrics artifacts and model evaluation logs.

## Regret (Bandits)
Definition: Cumulative performance gap between chosen actions and an oracle best action policy.  
Why it matters: Measures exploration policy cost over time.  
Where to see it: `projects/rl-bandits-policy-showcase`.

## Exploration vs Exploitation
Definition: Tradeoff between trying uncertain actions and choosing currently best-known action.  
Why it matters: Central decision in online learning and bandit policies.  
Where to see it: RL bandits strategy comparisons.
