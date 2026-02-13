# Aspect Coverage Matrix

This matrix maps the requested ML workflow aspects to concrete showcase projects, commands, and artifacts.

| Aspect | Where It Is Implemented | How to Run | Evidence Artifact(s) |
|---|---|---|---|
| Data profiling (`ydata-profiling`) | `projects/eda-leakage-profiling-showcase`, `projects/feature-engineering-dimred-showcase` | `make sync-profiling && make run` | `artifacts/eda/profile_status.txt` |
| Univariate analysis | EDA + feature engineering showcases via shared EDA utilities | `make run` | `artifacts/eda/univariate_summary.csv` |
| Bivariate analysis vs target | EDA + feature engineering showcases | `make run` | `artifacts/eda/bivariate_vs_target.csv` |
| Missing information visualization | EDA + feature engineering showcases | `make sync-profiling && make run` | `artifacts/eda/missingness_summary.csv`, `artifacts/eda/missing_plot_status.txt` |
| Train/val/test split enforcement | Shared split contract for supervised projects | `make check-contracts` | `artifacts/splits/split_manifest.json` with `train_rows`, `val_rows`, `test_rows` |
| Split strategy coverage (stratified/group/time/CV) | EDA leakage showcase + shared split helpers | `make run` in EDA showcase | `artifacts/splits/group_split_manifest.json`, `artifacts/splits/timeseries_split_manifest.json`, `artifacts/splits/cv_split_manifest.json` |
| Data type analysis (numeric/categorical) | Feature engineering preprocessing pipelines | `make run` | `artifacts/features/feature_matrix_summary.csv` |
| Categorical encodings (One-Hot + Label/Ordinal) | `projects/feature-engineering-dimred-showcase` | `make run` | Encoded matrix from preprocessing pipeline |
| Entity embeddings | Advanced FE runner (embedding proxy output) | `make run-advanced` | `artifacts/advanced/entity_embeddings.csv` |
| Advanced feature engineering (`featuretools`, `tsfresh`, `autofeat`) | Advanced FE runner with optional dependencies | `make sync-advanced && make run-advanced` | `artifacts/advanced/featuretools_status.txt`, `artifacts/advanced/tsfresh_status.txt`, `artifacts/advanced/autofeat_status.txt` |
| Distribution shift / drift | `projects/mlops-drift-production-showcase` | `make run && make run-drift` | Drift monitor outputs in `artifacts/drift/` |
| Time-aware demand forecasting | `projects/nyc-demand-forecasting-foundations-showcase` | `make run` | `artifacts/eval/metrics_summary.csv`, `artifacts/splits/time_split_manifest.json` |
| Imbalanced dataset handling | `projects/sota-supervised-learning-showcase`, `projects/credit-risk-classification-capstone-showcase` | `make run` | Strategy comparison outputs and threshold-aware metrics |
| Correlations/distributions/densities | EDA + feature engineering showcases | `make run` | `artifacts/eda/correlation_matrix.csv` |
| Information leakage analysis | Shared leakage utilities in supervised pipelines | `make run` | `artifacts/leakage/leakage_report.csv` |
| Imputation techniques | Feature engineering preprocessing | `make run` | Pipeline uses median + most-frequent imputers |
| Over/Under sampling + SMOTE hybrids | Supervised showcase data utilities | `make sync-boosting` (for optional libs), `make run` | Strategy-level metrics and logs |
| Dimensionality reduction / feature subset selection | Feature engineering + dimred showcase | `make run && make run-dimred` | `artifacts/selection/selection_scores.csv`, `artifacts/dimred/embedding_quality_metrics.csv` |
| SoTA modeling (XGBoost/LightGBM/CatBoost/Deep/Stacking) | Supervised showcase classification benchmark | `make sync-boosting && make run` | `artifacts/classification_benchmark.csv` |
| Learning-to-rank modeling (LambdaRank + NDCG) | `projects/learning-to-rank-foundations-showcase` | `make run` | `artifacts/eval/ranking_metrics.json`, `artifacts/splits/group_split_manifest.json` |
| Overfitting/bias-aware evaluation (ROC/PR/Learning/Threshold, RMSE/MAE/R²) | Supervised, EDA, and related evaluation pipelines | `make run` | `artifacts/eval/metrics_summary.csv`, `artifacts/eval/threshold_analysis.csv`, learning/validation curve artifacts |
| Explainability (SHAP/LIME) | `projects/xai-fairness-audit-showcase` | `make sync-explainability && make run-explainability` | `artifacts/explainability/shap_status.txt`, `artifacts/explainability/lime_status.txt` |
| Hyperparameter optimization (HyperOpt/Optuna) | `projects/automl-hpo-showcase` | `make run-advanced` | `artifacts/hpo/trials.csv`, `artifacts/hpo/strategy_comparison.csv` |
| Experiment tracking (MLflow) | AutoML and MLOps showcases | `make run-advanced` (AutoML), `make run-tracking` (MLOps) | `artifacts/hpo/mlflow_status.txt`, `artifacts/tracking/mlflow_status.txt` |
| Productionization examples | MLOps serving + ranking API productization + demand API observability + rollout/systems showcases | `make serve` (MLOps), `make dev` + `make export-openapi` (ranking API / demand API) | `openapi.json`, `artifacts/registry/model_versions.json`, `http_requests_total` metrics endpoint output, rollout decision logs, serving and monitoring artifacts |

## Contract Enforcement

- `make check-contracts` now bootstraps missing supervised artifacts in quick mode and validates:
  - split manifests,
  - EDA summaries,
  - leakage reports,
  - evaluation outputs,
  - experiment logs.
- CI uses the same contract verifier path (`shared/scripts/verify_supervised_contract.py --bootstrap-missing`) to avoid clean-checkout failures.
