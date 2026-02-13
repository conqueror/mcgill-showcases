# Ranking Track

This track teaches group-aware ranking model construction and production-grade ranking API patterns.

## Recommended Sequence

1. `projects/learning-to-rank-foundations-showcase`
2. `projects/ranking-api-productization-showcase`
3. `projects/model-release-rollout-showcase` (optional extension)

## Core Skills Covered

- Query/group-aware splitting and training.
- Relevance labeling and lambda-based ranking objectives.
- Ranking metrics (NDCG@k).
- Ranking service endpoint contracts and schema checks.

## Evidence Artifacts To Inspect

- `artifacts/eval/ranking_metrics.json`
- `artifacts/splits/group_split_manifest.json`
- `artifacts/eval/test_rankings_top10.csv`
- `openapi.json` in ranking API showcase

## Suggested Reflection Prompts

- Why is grouped splitting mandatory for ranking tasks?
- How would you detect ranking quality regressions before release?
- What API schema fields are essential for safe ranking requests?
