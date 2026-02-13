# Project Deep Dives

These pages bring project-level details directly into the docs site so students can see concrete commands, outputs, and interpretation patterns without leaving MkDocs.

## Included Deep Dives

| Deep Dive | Focus | Time | Primary Artifacts |
|---|---|---|---|
| [Supervised Learning](sota-supervised.md) | classification/regression foundations, imbalance handling, model evaluation | 90-150 min | metrics tables, curves, learning diagnostics |
| [Causal Inference](causal-inference.md) | ATE/CATE/`tau(x)`, uplift modeling, targeting policies | 120-180 min | Qini curves, uplift-at-k, policy simulations |
| [MLOps Drift](mlops-drift.md) | training, drift detection, retrain decisions, local serving | 90-150 min | drift report, policy decision JSON, API outputs |

## How To Use These Pages

1. Pick one deep dive and run the quickstart exactly once.
2. Validate artifact generation.
3. Use the "How to interpret" checklist to turn outputs into decisions.
4. Move to the next deep dive only after you can explain current outputs in plain language.

## Source Project Documents

- Supervised project source: [`projects/sota-supervised-learning-showcase/README.md`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/sota-supervised-learning-showcase/README.md)
- Causal project source: [`projects/causalml-kaggle-showcase/README.md`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/causalml-kaggle-showcase/README.md)
- MLOps project source: [`projects/mlops-drift-production-showcase/README.md`](https://github.com/conqueror/mcgill-showcases/blob/main/projects/mlops-drift-production-showcase/README.md)
