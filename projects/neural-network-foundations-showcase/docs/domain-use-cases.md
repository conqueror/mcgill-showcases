# Domain Use Cases

## Domain Use Cases

- Credit approval:
  A perceptron can separate simple low-risk and high-risk cases when the boundary is close to linear, but richer behaviors require hidden layers.
- Quality control:
  Activation choices matter when sensor inputs can be negative, sparse, or highly skewed.
- Marketing response prediction:
  Loss functions affect how heavily the model punishes confident but wrong probability forecasts.
- Fraud or anomaly detection:
  Underfitting misses subtle patterns, while overfitting memorizes noise and fails on new cases.

## Transfer Question

For your own problem, ask:

1. Is the boundary likely linear or nonlinear?
2. Is the cost of a bad prediction symmetric?
3. Do training and validation move together, or does the validation signal break away?
