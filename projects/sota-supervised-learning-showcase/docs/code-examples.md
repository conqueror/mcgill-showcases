# Documented Code Examples

Each example includes:
- what to run,
- what to observe,
- how to transfer the idea to another domain.

## 1. Imbalanced Binary Classification

```python
from sota_supervised_showcase.data import (
    build_binary_target,
    load_digits_split,
    rebalance_binary_training_data,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

split = load_digits_split()
y_train_binary = build_binary_target(split.y_train, positive_digit=0)
y_test_binary = build_binary_target(split.y_test, positive_digit=0)

for strategy in ("none", "upsample_minority", "downsample_majority"):
    x_balanced, y_balanced = rebalance_binary_training_data(
        split.x_train, y_train_binary, strategy=strategy
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    model.fit(x_balanced, y_balanced)
    score = model.score(split.x_test, y_test_binary)
    print(f"{strategy:>20} | accuracy={score:.3f}")
```

What to observe:
- Accuracy can stay high even if minority recall is weak.

Transfer idea:
- Fraud detection, disease detection, failure prediction.

## 2. OvR vs OvO Multi-Class Strategies

```python
from sota_supervised_showcase.data import load_digits_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

split = load_digits_split()

ovr = OneVsRestClassifier(LogisticRegression(max_iter=2000, random_state=42))
ovo = OneVsOneClassifier(make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale")))

for name, model in (("OvR", ovr), ("OvO", ovo)):
    model.fit(split.x_train, split.y_train)
    pred = model.predict(split.x_test)
    print(
        f"{name}: "
        f"f1_micro={f1_score(split.y_test, pred, average='micro'):.3f}, "
        f"f1_macro={f1_score(split.y_test, pred, average='macro'):.3f}"
    )
```

What to observe:
- Micro and macro F1 can tell different stories.

Transfer idea:
- Ticket routing, product type classification, image category assignment.

## 3. Multi-Label Classification

```python
from sota_supervised_showcase.data import build_multilabel_targets, load_digits_split
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

split = load_digits_split()
y_train_multilabel = build_multilabel_targets(split.y_train)
y_test_multilabel = build_multilabel_targets(split.y_test)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(split.x_train, y_train_multilabel)
pred = model.predict(split.x_test)

print("macro F1:", f1_score(y_test_multilabel, pred, average="macro"))
print("weighted F1:", f1_score(y_test_multilabel, pred, average="weighted"))
```

What to observe:
- One example can be predicted as multiple labels at once.

Transfer idea:
- Content tagging, skill matching, policy tagging.

## 4. Stacking Ensemble

```python
from sota_supervised_showcase.data import load_digits_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

split = load_digits_split()

stack = StackingClassifier(
    estimators=[
        ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))),
        ("rf", RandomForestClassifier(n_estimators=180, random_state=42)),
        ("svc", make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma="scale"))),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
    ],
    final_estimator=LogisticRegression(max_iter=2000, random_state=42),
    n_jobs=-1,
)

stack.fit(split.x_train, split.y_train)
pred = stack.predict(split.x_test)
print("Stacking macro F1:", f1_score(split.y_test, pred, average="macro"))
```

What to observe:
- Combining diverse models often improves consistency.

Transfer idea:
- Risk scoring and recommendation systems where one model misses some patterns.

## 5. Manual Gradient Boosting Intuition (Regression)

```python
from sklearn.tree import DecisionTreeRegressor

# Step 1: first weak learner
tree_1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_1.fit(X_train, y_train)

# Step 2: fit residuals from step 1
residual_1 = y_train - tree_1.predict(X_train)
tree_2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_2.fit(X_train, residual_1)

# Step 3: fit residuals from step 2
residual_2 = residual_1 - tree_2.predict(X_train)
tree_3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_3.fit(X_train, residual_2)

# Final prediction = sum of staged learners
y_pred = tree_1.predict(X_test) + tree_2.predict(X_test) + tree_3.predict(X_test)
```

What to observe:
- Each stage tries to correct the previous stage's errors.

Transfer idea:
- Price prediction, demand forecasting, and delivery-time estimation.
