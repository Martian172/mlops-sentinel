"""End-to-end tests for the multi-tenant monitoring API (/api/v1)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sentinel.api.registry import registry
from sentinel.dashboard.app import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_registry():
    # start each test with a fresh registry
    for mid in registry.ids():
        registry.remove(mid)
    yield


def _baseline():
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, size=(200, 2)).tolist()


class TestModelLifecycle:
    def test_register_and_fetch(self, client):
        r = client.post("/api/v1/models", json={
            "model_name": "churn",
            "feature_names": ["a", "b"],
            "task_type": "classification",
            "baseline_data": _baseline(),
        })
        assert r.status_code == 201
        body = r.json()
        mid = body["model_id"]
        assert body["predictions_url"].endswith(f"{mid}/predictions")

        got = client.get(f"/api/v1/models/{mid}")
        assert got.status_code == 200
        assert got.json()["model_name"] == "churn"

    def test_list_and_delete(self, client):
        mid = client.post("/api/v1/models", json={"model_name": "m1"}).json()["model_id"]
        listing = client.get("/api/v1/models").json()
        assert listing["count"] == 1
        assert client.delete(f"/api/v1/models/{mid}").status_code == 200
        assert client.get("/api/v1/models").json()["count"] == 0

    def test_unknown_model_404(self, client):
        assert client.get("/api/v1/models/does-not-exist").status_code == 404


class TestPredictions:
    def test_log_and_performance(self, client):
        mid = client.post("/api/v1/models", json={
            "model_name": "clf", "feature_names": ["a", "b"],
            "baseline_data": _baseline(),
        }).json()["model_id"]

        for i in range(20):
            r = client.post(f"/api/v1/models/{mid}/predictions", json={
                "features": {"a": i, "b": i * 2},
                "prediction": i % 2, "actual": i % 2, "latency_ms": 10.0,
            })
            assert r.status_code == 201

        perf = client.get(f"/api/v1/models/{mid}/performance").json()
        assert perf["total_predictions"] == 20
        assert perf["accuracy"] == 1.0

    def test_batch_and_drift(self, client):
        mid = client.post("/api/v1/models", json={
            "model_name": "clf", "feature_names": ["a", "b"],
            "baseline_data": _baseline(),
        }).json()["model_id"]

        rng = np.random.default_rng(1)
        records = [
            {"features": {"a": float(rng.normal()), "b": float(rng.normal())},
             "prediction": 1}
            for _ in range(30)
        ]
        r = client.post(f"/api/v1/models/{mid}/predictions/batch",
                        json={"records": records})
        assert r.status_code == 201
        assert r.json()["count"] == 30

        drift = client.get(f"/api/v1/models/{mid}/drift").json()
        assert "drift_score" in drift

    def test_update_actual(self, client):
        mid = client.post("/api/v1/models", json={"model_name": "clf"}).json()["model_id"]
        rid = client.post(f"/api/v1/models/{mid}/predictions", json={
            "features": {"a": 1}, "prediction": 1,
        }).json()["record_id"]
        r = client.patch(f"/api/v1/models/{mid}/predictions/{rid}", json={"actual": 1})
        assert r.status_code == 200
        assert client.patch(f"/api/v1/models/{mid}/predictions/nope",
                            json={"actual": 1}).status_code == 404


class TestAlertsAndRules:
    def test_rule_triggers_alert(self, client):
        mid = client.post("/api/v1/models", json={
            "model_name": "clf", "feature_names": ["a", "b"],
            "baseline_data": _baseline(),
            "alert_rules": [{"name": "low-acc", "metric": "accuracy",
                             "threshold": 0.9, "comparison": "lt",
                             "severity": "CRITICAL"}],
        }).json()["model_id"]

        # log 50 wrong predictions -> accuracy 0 -> rule fires on the
        # periodic alert check (every 50 predictions)
        for i in range(50):
            client.post(f"/api/v1/models/{mid}/predictions", json={
                "features": {"a": i, "b": i}, "prediction": 1, "actual": 0,
            })
        client.get(f"/api/v1/models/{mid}/performance")  # refresh perf cache
        alerts = client.get(f"/api/v1/models/{mid}/alerts").json()
        assert alerts["count"] >= 1


class TestAuth:
    def test_token_enforced_when_set(self, client, monkeypatch):
        monkeypatch.setenv("SENTINEL_API_TOKEN", "secret123")
        assert client.get("/api/v1/models").status_code == 401
        ok = client.get("/api/v1/models", headers={"X-API-Key": "secret123"})
        assert ok.status_code == 200
        bearer = client.get("/api/v1/models",
                            headers={"Authorization": "Bearer secret123"})
        assert bearer.status_code == 200
