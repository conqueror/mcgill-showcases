from pathlib import Path

from fastapi.testclient import TestClient

from demand_api_observability_showcase.api.app import create_app
from demand_api_observability_showcase.model.demo_training import train_demo_model
from demand_api_observability_showcase.settings import Settings


def _train_demo_model(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifacts"
    train_demo_model(artifact_dir)
    return artifact_dir / "model.joblib"


def test_predict_200_when_model_present(tmp_path: Path) -> None:
    model_path = _train_demo_model(tmp_path)

    settings = Settings(model_path=model_path)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/predict",
            json={"pickup_zone_id": 161, "pickup_datetime": "2026-02-06T15:00:00"},
            headers={"x-trace-id": "trace-123"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["pickup_zone_id"] == 161
        assert isinstance(payload["predicted_pickups"], float)
        assert payload["model_version"] == "demo-nyc-demand-v1"
        assert response.headers["x-trace-id"] == "trace-123"


def test_metrics_endpoint_enabled(tmp_path: Path) -> None:
    model_path = _train_demo_model(tmp_path)
    settings = Settings(model_path=model_path, prometheus_enabled=True)

    with TestClient(create_app(settings)) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "http_requests_total" in response.text
