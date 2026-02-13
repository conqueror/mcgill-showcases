# Diagrams

These diagrams are designed to render in GitHub and Markdown viewers with Mermaid support.

## 1. Supervised Learning Task Picker

```mermaid
flowchart TD
    A[Start with prediction goal] --> B{What should be predicted?}
    B -->|Yes/No outcome| C[Binary classification]
    B -->|One class among many| D[Multi-class classification]
    B -->|Multiple labels at once| E[Multi-label classification]
    B -->|Vector of outputs| F[Multi-output prediction]
    B -->|Continuous value| G[Regression]
```

## 2. Metric Selection Guide

```mermaid
flowchart TD
    A[Choose evaluation metric] --> B{Task type}
    B -->|Binary classification| C{Which error is costly?}
    C -->|False positives| D[Prioritize precision]
    C -->|False negatives| E[Prioritize recall]
    C -->|Need balance| F[Use F1]
    B -->|Imbalanced ranking focus| H[Use PR AUC]
    B -->|Binary separability focus| I[Use ROC AUC]
    B -->|Multi-class| J[Use macro and weighted F1]
    B -->|Regression| K[Use MAE RMSE R2]
```

## 3. Ensemble Intuition

```mermaid
flowchart LR
    A[Single model] --> A1[Can be unstable]
    B[Bagging and Random Forest] --> B1[Reduce variance]
    C[Boosting] --> C1[Reduce bias by correcting residual errors]
    D[Voting and Stacking] --> D1[Combine diverse strengths]
```

## 4. Learning Loop

```mermaid
flowchart TD
    A[Run model] --> B[Read metrics and errors]
    B --> C[Form hypothesis]
    C --> D[Change one thing]
    D --> E[Run again]
    E --> F{Improved?}
    F -->|Yes| G[Keep and document]
    F -->|No| H[Revert and test new hypothesis]
    H --> C
```

## 5. Validation and Learning Curve Reading

```mermaid
flowchart TD
    A[Validation and learning curves] --> B{Train score high, validation low?}
    B -->|Yes| C[High variance / overfitting]
    B -->|No| D{Both scores low?}
    D -->|Yes| E[High bias / underfitting]
    D -->|No| F[Reasonable fit]
    C --> G[Tune regularization or reduce complexity]
    E --> H[Increase model capacity or features]
    F --> I[Check robustness and deployment constraints]
```
