"""
FastAPI endpoint integration tests.

These tests verify the API contract: valid requests return proper responses,
invalid requests return appropriate error codes, and the health endpoint is always up.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    def test_predict_valid_payload(self, client):
        """Valid telemetry data should return predictions."""
        payload = {
            "records": [
                {
                    "machineID": 1,
                    "volt": 170.0,
                    "rotate": 450.0,
                    "pressure": 95.0,
                    "vibration": 40.0,
                    "volt_mean_3h": 170.0,
                    "volt_std_3h": 2.0,
                    "volt_mean_12h": 169.0,
                    "volt_std_12h": 3.0,
                    "volt_mean_24h": 168.0,
                    "volt_std_24h": 4.0,
                }
            ]
        }
        response = client.post("/predict", json=payload)
        # May fail if model not loaded — that's expected in CI without training
        if response.status_code == 503:
            pytest.skip("Model not loaded — run training first")
        assert response.status_code == 200

    def test_predict_empty_records(self, client):
        """Empty records list should return 422."""
        response = client.post("/predict", json={"records": []})
        assert response.status_code == 422
