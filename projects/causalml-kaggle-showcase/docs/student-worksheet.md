# Student Worksheet: Causal Inference Learning Project

Use this worksheet while you run the notebooks and scripts.

How to use it:
1. Fill the "Before running" part before executing cells.
2. Fill the "After running" part with metrics and plots.
3. Write one decision you would take based on your result.
4. Add one domain-transfer example in another field.

## Quick command checklist
```bash
uv sync --extra dev
uv run python scripts/download_kaggle_dataset.py
uv run python scripts/run_pipeline.py
uv run python scripts/policy_simulator.py
uv run python scripts/confounding_checks.py
uv run python scripts/verify_learning_artifacts.py
```

## Notebook 01: Counterfactuals and ATE
File: `notebooks/01_counterfactuals_and_ate.ipynb`

Before running:
- What do you think the treatment effect sign will be? (`positive`, `negative`, `near zero`)
- Why?

After running:
- Empirical ATE: `__________`
- Bootstrap CI: `__________`
- One-sentence interpretation: `________________________________________`

Decision:
- If you were a decision-maker, would you treat everyone? Why/why not?

Domain transfer:
- Write the same idea for healthcare, education, or product growth.

## Notebook 02: S/T/X/R Learners
File: `notebooks/02_meta_learners_s_t_x_r.ipynb`

Before running:
- Which learner do you expect to rank top users best?
- Why?

After running:
- Best learner by `uplift@30%`: `__________`
- Top 2 learners by policy metric: `__________, __________`
- One-sentence interpretation: `________________________________________`

Decision:
- Which learner would you pick for initial deployment and why?

Domain transfer:
- How would this ranking change if treatment and control sizes were very imbalanced?

## Notebook 03: Uplift Tree Interpretability
File: `notebooks/03_uplift_tree_interpretability.ipynb`

Before running:
- What kind of segment splits do you expect (time/day/exposure)?

After running:
- Most interesting split from tree text: `______________________________`
- Highest-uplift bin: `__________`
- One-sentence interpretation: `________________________________________`

Decision:
- Write one segment-targeting rule you would test in production.

Domain transfer:
- Write one interpretable segmentation rule for a non-marketing domain.

## Notebook 04: Qini Curves and Targeting Policy
File: `notebooks/04_qini_and_targeting_policy.ipynb`

Before running:
- Do you expect the same model to win at every budget level?

After running:
- Model with best Qini AUC: `__________`
- Best model at 10% budget: `__________`
- Best model at 50% budget: `__________`

Decision:
- If your budget is fixed at 20%, which model would you use and why?

Domain transfer:
- Give one example where a small budget and large budget would require different strategies.

## Notebook 05: Capstone Policy Simulation
File: `notebooks/05_capstone_policy_simulation.ipynb`

Before running:
- Predict which model has highest incremental conversions at medium budget.

After running:
- Best model at each budget (copy table or summarize):
  - 10%: `__________`
  - 20%: `__________`
  - 30%: `__________`
  - 40%: `__________`
  - 50%: `__________`

Decision:
- Write a policy memo in 3 lines:
  - Model choice: `__________`
  - Budget: `__________`
  - Expected impact: `__________`

Domain transfer:
- If each treatment has a cost, how would your decision change?

## Notebook 06: Confounding and Conditions
File: `notebooks/06_confounding_conditions.ipynb`

Before running:
- Do you think treatment assignment is predictable from features?

After running:
- Propensity AUC: `__________`
- Overlap share: `__________`
- Number of features with `|SMD| > 0.10`: `__________`

Decision:
- Is your causal claim strong, moderate, or weak? Explain briefly.

Domain transfer:
- Give one confounder example in healthcare or finance and how to measure it.

## Notebook 07: SHAP Interpretability
File: `notebooks/07_shap_interpretability.ipynb`

Before running:
- Which feature do you think will be the strongest uplift driver?

After running:
- Top 3 SHAP features:
  1. `__________`
  2. `__________`
  3. `__________`
- One-sentence interpretation for each.

Decision:
- What feature-based targeting rule would you propose next?

Domain transfer:
- Write one SHAP-driven insight in a different domain.

## Final reflection
1. In your own words, explain the difference between ATE and CATE.
2. What is one mistake someone can make with uplift models?
3. What diagnostics increased your trust in the results?
4. What diagnostics reduced your trust in the results?
5. If you had one more week, what experiment would you run next?

## Self-assessment rubric
Mark each from 1 (not confident) to 5 (very confident):
- Counterfactual reasoning: `1 2 3 4 5`
- ATE/CATE intuition: `1 2 3 4 5`
- Model comparison and selection: `1 2 3 4 5`
- Policy decision-making: `1 2 3 4 5`
- Confounding diagnostics: `1 2 3 4 5`
- SHAP interpretation: `1 2 3 4 5`
