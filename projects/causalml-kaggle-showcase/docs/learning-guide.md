# Learning Guide (Self-Study, 90-120 Minutes)

## What you should learn
By the end of this guide, you should be able to:
- Explain counterfactual thinking in plain language.
- Distinguish ATE, CATE, and ITE with examples.
- Train and compare S/T/X/R learners and uplift trees.
- Read Qini curves and choose a model for a real budget.
- Run confounding diagnostics and explain when causal claims are weak.
- Use SHAP to interpret which features drive uplift predictions.

## Learning flow
1. `10 min` Refresh core terms (`counterfactual`, `ATE`, `CATE`, `uplift`).
2. `15 min` Explore the dataset columns and treatment/control setup.
3. `20 min` Compute baseline ATE and check subgroup variation intuition.
4. `25 min` Train S/T/X/R learners and compare uplift-at-k.
5. `15 min` Fit an uplift tree and inspect split logic.
6. `15 min` Use Qini curves and policy simulation to choose top-k targeting.
7. `10 min` Run confounding checks and reflect on assumptions.
8. `10 min` Run SHAP interpretation and summarize model drivers.

## Command sequence
```bash
make sync
make download
make pipeline
make policy
make confounding
make check
make verify
```

## Notebook sequence
1. `notebooks/01_counterfactuals_and_ate.ipynb`
2. `notebooks/02_meta_learners_s_t_x_r.ipynb`
3. `notebooks/03_uplift_tree_interpretability.ipynb`
4. `notebooks/04_qini_and_targeting_policy.ipynb`
5. `notebooks/05_capstone_policy_simulation.ipynb`
6. `notebooks/06_confounding_conditions.ipynb`
7. `notebooks/07_shap_interpretability.ipynb`

## Worksheet
- `docs/student-worksheet.md`
- Fill one section per notebook.
- Use the final reflection + rubric to identify weak spots and revisit the related notebook.

## Checkpoint questions
- If ATE is positive, does that mean everyone benefits from treatment?
- Which model gives the best ranking at 10% budget vs 50% budget?
- What would make you distrust a causal estimate on observational data?
- Which 2-3 features appear to drive uplift the most, and why might that make domain sense?

## Common confusion points
- High predictive accuracy is not the same as high uplift quality.
- A global ATE can hide useful or harmful subgroup effects.
- Good-looking plots do not guarantee causal validity unless assumptions are checked.

## Practice extensions
- Compare RandomForest base learners vs gradient boosting base learners.
- Add sensitivity checks for hidden confounders.
- Introduce treatment costs and compare expected net value across policies.
