# Learning-to-Rank Foundations Showcase

Portable ranking showcase focused on grouped training fundamentals and NDCG evaluation.

## Learning outcomes
- Build grouped ranking data (`query/group`) with relevance labels.
- Separate raw train/validation/test seasons, fit medians and category columns on training seasons, and keep each query's rows contiguous.
- Train a LightGBM LambdaRank model.
- Evaluate mean query NDCG@5 and NDCG@10 with gains `2^relevance - 1`, matching the explicit LightGBM gains. Positive singleton queries score 1; queries with no positive relevance also score 1, following LightGBM's zero-IDCG convention.

## Quickstart
```bash
cd projects/learning-to-rank-foundations-showcase
make sync
make run
make verify
```

## Key outputs
- `artifacts/data/ranking_dataset_sample.csv`
- `artifacts/data/feature_schema.json`
- `artifacts/model/model.txt`
- `artifacts/model/model_meta.json`
- `artifacts/eval/ranking_metrics.json`
- `artifacts/eval/test_rankings_top10.csv`
- `artifacts/splits/group_split_manifest.json`
- `artifacts/manifest.json`
