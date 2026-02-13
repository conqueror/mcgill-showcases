# Learning-to-Rank Foundations Showcase

Portable ranking showcase focused on grouped training fundamentals and NDCG evaluation.

## Learning outcomes
- Build grouped ranking data (`query/group`) with relevance labels.
- Enforce train/validation/test separation by group (season-style split).
- Train a LightGBM LambdaRank model.
- Evaluate ranking quality with NDCG@5 and NDCG@10.

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
