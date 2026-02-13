# Learning Guide

1. Run `make run` and inspect `artifacts/splits/group_split_manifest.json`.
2. Review `artifacts/eval/ranking_metrics.json` and explain NDCG@5 vs NDCG@10.
3. Compare predicted ranking vs observed points in `artifacts/eval/test_rankings_top10.csv`.
4. Change `--seed` or dataset size and discuss metric stability.
