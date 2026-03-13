# Code Examples

## Code Examples

These snippets are short on purpose. The goal is to make each concept easy to edit and rerun.

### Vector operations

```python
import numpy as np

left = np.array([1.0, 2.0])
right = np.array([3.0, 4.0])
print(left + right)
print(np.dot(left, right))
```

### Derivative at a point

```python
from deep_learning_math_foundations_showcase.calculus import derivative_at_point

print(derivative_at_point("x**2", symbol="x", point=2.0))
```

### Bernoulli probability estimate

```python
from deep_learning_math_foundations_showcase.probability import estimate_bernoulli_probability

print(estimate_bernoulli_probability(seed=7, trials=1000, p=0.3))
```

### Entropy and cross-entropy

```python
from deep_learning_math_foundations_showcase.information_theory import entropy, cross_entropy

print(entropy([0.5, 0.5]))
print(cross_entropy([1.0, 0.0], [0.8, 0.2]))
```

### Gradient descent trace

```python
from deep_learning_math_foundations_showcase.optimization import run_gradient_descent_trace

print(run_gradient_descent_trace().head())
```
