"""
Monitor YOUR model through the Sentinel API — no Python import of Sentinel
required on your side, just HTTP.

Start a server first (in another terminal):
    python run_demo.py            # serves the API at http://127.0.0.1:8001

Then run this:
    python examples/api_client.py
"""
import sys

import numpy as np
import httpx

# Make stdout UTF-8 tolerant on legacy Windows codepages.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:8001"
# If the server sets SENTINEL_API_TOKEN, add: headers={"X-API-Key": "..."}
client = httpx.Client(base_url=BASE, timeout=10.0)


def main() -> None:
    # 1. Register your model with a baseline (your training data)
    rng = np.random.default_rng(42)
    baseline = rng.normal(0, 1, size=(500, 3)).tolist()
    resp = client.post("/api/v1/models", json={
        "model_name": "my-production-model",
        "feature_names": ["age", "income", "score"],
        "task_type": "classification",
        "baseline_data": baseline,
        "alert_rules": [
            {"name": "low-accuracy", "metric": "accuracy",
             "threshold": 0.85, "comparison": "lt", "severity": "CRITICAL"},
        ],
    }).json()
    model_id = resp["model_id"]
    print(f"Registered -> model_id = {model_id}")

    # 2. Stream predictions from your serving loop (here: simulate drift)
    print("Streaming 120 predictions (drift ramps up)...")
    for i in range(120):
        drift = min(i / 120, 1.0)
        features = {
            "age": float(rng.normal(0 + 2 * drift, 1)),   # slowly shifts
            "income": float(rng.normal(0 + 2 * drift, 1)),
            "score": float(rng.normal(0, 1)),             # stays put
        }
        actual = int(rng.random() < 0.5)
        correct = rng.random() < (0.95 - 0.3 * drift)
        prediction = actual if correct else 1 - actual
        client.post(f"/api/v1/models/{model_id}/predictions", json={
            "features": features, "prediction": prediction,
            "actual": actual, "latency_ms": float(rng.gamma(2, 6)),
        })

    # 3. Pull reports back
    drift = client.get(f"/api/v1/models/{model_id}/drift").json()
    perf = client.get(f"/api/v1/models/{model_id}/performance").json()
    alerts = client.get(f"/api/v1/models/{model_id}/alerts").json()

    print(f"\nDrift score : {drift['drift_score']:.3f}  (drifted: {drift['is_drifted']})")
    print(f"Drifted     : {drift['drifted_features']}")
    print(f"Accuracy    : {perf['accuracy']}")
    print(f"Alerts fired: {alerts['count']}")
    print(f"\nOpen {BASE} to watch it live, or {BASE}/docs for the API console.")


if __name__ == "__main__":
    main()
