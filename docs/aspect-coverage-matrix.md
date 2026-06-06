# Aspect Coverage Matrix

This matrix maps the requested ML workflow aspects to concrete showcase projects, commands, and artifacts.

| Aspect | Where It Is Implemented | How to Run | Evidence Artifact(s) |
|---|---|---|---|
| Linear algebra for model intuition | `projects/deep-learning-math-foundations-showcase` | `make run` | `artifacts/vector_operations.csv`, `artifacts/matrix_transformations.csv` |
| Calculus and gradient intuition | `projects/deep-learning-math-foundations-showcase` | `make run` | `artifacts/derivative_examples.csv`, `artifacts/gradient_descent_trace.csv` |
| Probability and uncertainty basics | `projects/deep-learning-math-foundations-showcase` | `make run` | `artifacts/probability_simulations.csv` |
| Entropy / cross-entropy / KL divergence | `projects/deep-learning-math-foundations-showcase` | `make run` | `artifacts/information_theory_summary.md` |
| Perceptrons, activations, and forward-pass intuition | `projects/neural-network-foundations-showcase` | `make run` | `artifacts/activation_comparison.csv`, `artifacts/decision_boundary_summary.csv` |
| Backpropagation and initialization | `projects/neural-network-foundations-showcase` | `make run` | `artifacts/backprop_gradient_trace.csv`, `artifacts/initialization_comparison.csv` |
| Underfitting vs overfitting | `projects/neural-network-foundations-showcase` | `make run` | `artifacts/underfit_overfit_examples.csv`, `artifacts/training_curves.csv` |
| PyTorch training loops and optimizer comparison | `projects/pytorch-training-regularization-showcase` | `make run` | `artifacts/baseline_metrics.json`, `artifacts/optimizer_comparison.csv`, `artifacts/learning_rate_schedule_comparison.csv` |
| Regularization (dropout, batch norm, weight decay, early stopping) | `projects/pytorch-training-regularization-showcase` | `make run` | `artifacts/regularization_ablation.csv`, `artifacts/gradient_health_report.md` |
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
| Text chunking and document preparation for NLP systems | `projects/modern-nlp-pipeline-showcase` | `make run` | `artifacts/data/corpus_overview.csv`, `artifacts/data/topic_distribution.csv` |
| Sparse lexical vs dense semantic text representations | `projects/modern-nlp-pipeline-showcase` | `make run` | `artifacts/classification/metrics_summary.csv`, `artifacts/retrieval/retrieval_metrics.csv` |
| Retrieval-grounded QA on research text | `projects/modern-nlp-pipeline-showcase` | `make run` | `artifacts/generation/qa_outputs.csv`, `artifacts/retrieval/retrieval_examples.json` |
| Query-focused summarization from retrieved evidence | `projects/modern-nlp-pipeline-showcase` | `make run` | `artifacts/generation/query_summaries.json`, `artifacts/summary.md` |
| SoTA modeling (XGBoost/LightGBM/CatBoost/Deep/Stacking) | Supervised showcase classification benchmark | `make sync-boosting && make run` | `artifacts/classification_benchmark.csv` |
| Learning-to-rank modeling (LambdaRank + NDCG) | `projects/learning-to-rank-foundations-showcase` | `make run` | `artifacts/eval/ranking_metrics.json`, `artifacts/splits/group_split_manifest.json` |
| MDP framing for sequential decision systems | `projects/student-support-rl-showcase` | `make run-mdp` | `artifacts/concepts/mdp_spec.md`, `artifacts/concepts/concept_map.csv`, `artifacts/mdp/sample_episodes.csv` |
| Contextual bandit exploration with context-dependent regret | `projects/student-support-rl-showcase` | `make run-bandit` | `artifacts/bandit/reward_trace.csv`, `artifacts/bandit/regret_trace.csv` |
| Tabular Q-learning and Bellman intuition | `projects/student-support-rl-showcase` | `make run-q-learning` | `artifacts/q_learning/training_curve.csv`, `artifacts/q_learning/q_table.csv` |
| Exact optimal Q* by dynamic programming (planning vs. learning) | `projects/student-support-rl-showcase` | `make run-dp` | `artifacts/dp/optimal_action_values.csv`, `artifacts/dp/q_learning_gap.csv` |
| On-policy TD control (SARSA) vs off-policy Q-learning | `projects/student-support-rl-showcase` | `make run-sarsa` | `artifacts/sarsa/training_curve.csv`, `artifacts/sarsa/q_table.csv` |
| Policy-gradient control from scratch (tabular REINFORCE) | `projects/student-support-rl-showcase` | `make run-reinforce` | `artifacts/policy_gradient/training_curve.csv` |
| Same-environment DRL comparison for Q-learning vs DQN vs PPO | `projects/student-support-rl-showcase` | `make sync-drl && make run-drl-optional` | `artifacts/drl_optional/rl_family_comparison.csv`, `artifacts/drl_optional/scenario_rollups.csv`, `artifacts/drl_optional/policy_gradient_notes.md` |
| Reward hacking audit and reward redesign | `projects/student-support-rl-showcase` | `make run-reward-check` | `artifacts/reward/reward_hacking_report.md`, `artifacts/reward/reward_spec_bad.md`, `artifacts/reward/reward_spec_good.md` |
| Offline policy evaluation with governance framing | `projects/student-support-rl-showcase` | `make run-eval` | `artifacts/eval/policy_comparison.csv`, `artifacts/eval/scenario_results.csv`, `artifacts/governance/safety_controls.md`, `artifacts/governance/offline_eval_plan.md`, `artifacts/business/deploy_shadow_reject_memo.md` |
| Overfitting/bias-aware evaluation (ROC/PR/Learning/Threshold, RMSE/MAE/R²) | Supervised, EDA, and related evaluation pipelines | `make run` | `artifacts/eval/metrics_summary.csv`, `artifacts/eval/threshold_analysis.csv`, learning/validation curve artifacts |
| Explainability (SHAP/LIME) | `projects/xai-fairness-audit-showcase` | `make sync-explainability && make run-explainability` | `artifacts/explainability/shap_status.txt`, `artifacts/explainability/lime_status.txt` |
| Hyperparameter optimization (HyperOpt/Optuna) | `projects/automl-hpo-showcase` | `make run-advanced` | `artifacts/hpo/trials.csv`, `artifacts/hpo/strategy_comparison.csv` |
| Autonomous agent experiment loops | `projects/autoresearch` | `make run` | `artifacts/overview/platform_comparison.csv`, `artifacts/analysis/decision_scenarios.csv`, `artifacts/agent/codex_macos.md` |
| Agent framework routing, tools, guardrails, traces, evals, A2A, sessions, memory, skills, and harness evidence | `projects/agentic-course-assistant-showcase` | `make smoke` | `artifacts/course_assistant_response.md`, `artifacts/agent_trace.json`, `artifacts/resource_matches.csv`, `artifacts/concepts/agentic_concepts.csv`, `artifacts/evals/agent_judge_rubric.json` |
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
