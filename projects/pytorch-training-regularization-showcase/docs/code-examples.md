# Code Examples

## Code Examples

### 1. Build a dataset bundle

```python
from pytorch_training_regularization_showcase import data

bundle = data.build_dataset_bundle("synthetic", batch_size=32, quick=True)
batch_features, batch_targets = next(iter(bundle.train_loader))
print(batch_features.shape, batch_targets.shape)
```

This is the first abstraction to understand: loaders turn raw tensors into repeatable mini-batches.

### 2. Build the classifier

```python
from pytorch_training_regularization_showcase import models

model = models.build_classifier(
    input_dim=16,
    num_classes=3,
    hidden_dims=(24, 12),
    dropout=0.2,
    batch_norm=True,
)
print(model)
```

The model is intentionally small so you can still read every layer.

### 3. Train one run

```python
from pytorch_training_regularization_showcase import data, training

bundle = data.build_dataset_bundle("synthetic", batch_size=32, quick=True)
result = training.train_classifier(
    bundle,
    training.TrainingConfig(epochs=4, learning_rate=0.02, hidden_dims=(24, 12)),
)
print(result.history)
```

Change `optimizer_name`, `scheduler_name`, or `weight_decay` one at a time and rerun.
