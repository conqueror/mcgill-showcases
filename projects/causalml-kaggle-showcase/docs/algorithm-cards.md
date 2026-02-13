# Algorithm Cards (Why Each Method Is Included)

## Naive empirical ATE
- Purpose: establish a baseline impact from the A/B test.
- Why: students must see the "average effect" before moving to heterogeneous effects.

## S-Learner
- Purpose: one model takes treatment as a feature.
- Why: simplest conceptually and computationally.
- Trade-off: may under-emphasize treatment interactions if the model focuses on dominant covariates.

## T-Learner
- Purpose: separate models for treatment and control outcomes.
- Why: intuitive decomposition for students.
- Trade-off: unstable when one group has much less data.

## X-Learner
- Purpose: pseudo-outcome based meta-learner with propensity weighting.
- Why: useful when treatment groups are imbalanced and for sharper CATE estimates.
- Trade-off: more moving parts and heavier dependence on propensity quality.

## R-Learner
- Purpose: orthogonalization/residualization strategy related to Double ML logic.
- Why: clean bridge from your lecture's DML ideas to practical code.
- Trade-off: harder to explain; sensitive to nuisance model quality.

## Uplift Tree (KL)
- Purpose: directly optimize treatment effect heterogeneity in tree splits.
- Why: highly interpretable for step-by-step learning and segment discussions.
- Trade-off: can overfit if depth/min-samples are not controlled.
