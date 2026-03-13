# Code Examples

## Code Examples

### 1. Inspect one activation directly

```python
import numpy as np
from neural_network_foundations_showcase import activations

inputs = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
print(activations.sigmoid(inputs))
print(activations.relu(inputs))
```

Try replacing `sigmoid` with `tanh` or `leaky_relu` and explain what changes.

### 2. Build a perceptron and run a forward pass

```python
from neural_network_foundations_showcase import data, networks

dataset = data.make_toy_dataset("linearly_separable", samples_per_class=6)
network = networks.build_network(layer_sizes=(2, 1), init_strategy="xavier")
probabilities = networks.predict_proba(network, dataset.features)
print(probabilities[:5])
```

This is the smallest useful network in the project: one weighted sum and one sigmoid output.

### 3. Train a hidden-layer network on XOR

```python
from neural_network_foundations_showcase import data, training

dataset = data.make_toy_dataset("xor", samples_per_class=32)
result = training.train_network(
    dataset,
    training.TrainingConfig(layer_sizes=(2, 8, 1), epochs=150, learning_rate=0.25),
)
print(result.history.tail())
```

If you switch `layer_sizes` to `(2, 1)`, the model loses the hidden layer and the XOR task becomes much harder.
