from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_drift_showcase.serve import create_app


def test_health_and_predict_endpoints() -> None:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    features = np.array([[0.0, 0.2], [1.0, 1.2], [0.1, 0.3], [1.1, 1.3]])
    target = np.array([0, 1, 0, 1])
    model.fit(features, target)

    app = create_app(model=model)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    pred = client.post("/predict", json={"features": [0.5, 0.6]})
    assert pred.status_code == 200
    body = pred.json()
    assert set(body) == {"prediction", "probability"}
