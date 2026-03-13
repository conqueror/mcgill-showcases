# Concept Learning Map

## Concept Map

| Concept | Code Anchor | Artifact | Question To Ask |
| --- | --- | --- | --- |
| Tensors and loaders | `src/pytorch_training_regularization_showcase/data.py` | `baseline_metrics.json` | What shape enters the model each batch? |
| `nn.Module` design | `src/pytorch_training_regularization_showcase/models.py` | `training_history.csv` | How do hidden layers transform inputs into logits? |
| Metrics | `src/pytorch_training_regularization_showcase/evaluation.py` | `error_analysis.csv` | Which examples fail and with what confidence? |
| Training loop | `src/pytorch_training_regularization_showcase/training.py` | `training_history.csv` | When does validation stop improving? |
| Optimizers and schedulers | `src/pytorch_training_regularization_showcase/experiments.py` | `optimizer_comparison.csv`, `learning_rate_schedule_comparison.csv` | Which update rule learns fastest or most smoothly? |
| Regularization | `src/pytorch_training_regularization_showcase/regularization.py` | `regularization_ablation.csv` | Which technique helps generalization on this dataset? |
| Gradient health | `src/pytorch_training_regularization_showcase/training.py` | `gradient_health_report.md` | Are some parameters receiving almost no signal? |

## Study Hint

When an experiment wins, ask whether it improved:

- optimization speed,
- stability,
- or generalization.

Those are different outcomes, and the artifacts separate them on purpose.
