from fastapi.testclient import TestClient

from demand_api_observability_showcase.api.app import create_app


def test_health_ok() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert "version" in payload
