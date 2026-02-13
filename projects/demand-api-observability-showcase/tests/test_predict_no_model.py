from pathlib import Path

from fastapi.testclient import TestClient

from demand_api_observability_showcase.api.app import create_app
from demand_api_observability_showcase.settings import Settings


def test_predict_returns_503_when_model_missing(tmp_path: Path) -> None:
    settings = Settings(model_path=tmp_path / "missing.joblib")
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/predict",
            json={"pickup_zone_id": 161, "pickup_datetime": "2026-02-06T15:00:00"},
        )
    assert response.status_code == 503
