"""
Launch the MLOps Sentinel dashboard with a live simulated model.

Creates a ModelMonitor for a fake fraud-detection model, streams simulated
predictions into it in a background thread (with gradual data drift so the
dashboard has something interesting to show), and serves the dashboard.

Usage:
    python run_demo.py            # dashboard at http://127.0.0.1:8001
"""
import random
import threading
import time

import numpy as np
import uvicorn

from sentinel.core.alerts import AlertManager, AlertRule
from sentinel.core.monitor import ModelMonitor
from sentinel.dashboard.app import app, set_monitor

# --- 1. Baseline (training) data: what the model was trained on -----------
rng = np.random.default_rng(42)
N = 1000
baseline = np.column_stack([
    rng.normal(40, 10, N),          # age ~ N(40, 10)
    rng.normal(65_000, 15_000, N),  # income ~ N(65k, 15k)
    rng.normal(650, 80, N),         # credit_score ~ N(650, 80)
])

# --- 2. Alert rules --------------------------------------------------------
alert_mgr = AlertManager()
alert_mgr.add_rule(AlertRule(
    name="low-accuracy", metric="accuracy", threshold=0.85,
    comparison="lt", severity="CRITICAL",
))
alert_mgr.add_rule(AlertRule(
    name="high-drift", metric="drift_score", threshold=0.6,
    comparison="gt", severity="WARNING",
))

monitor = ModelMonitor(
    model_name="fraud-detector-v2",
    task_type="classification",
    baseline_data=baseline,
    feature_names=["age", "income", "credit_score"],
    alert_manager=alert_mgr,
    drift_check_interval=50,
    alert_check_interval=50,
)
set_monitor(monitor)


# --- 3. Simulated production traffic ---------------------------------------
def simulate_traffic() -> None:
    """Log a prediction every ~0.5s. After 2 minutes, drift sets in:
    customers get younger and richer, and model accuracy degrades."""
    start = time.time()
    while True:
        minutes_elapsed = (time.time() - start) / 60
        # Healthy for the first 2 minutes, then drift ramps 0 → 1 over 2 min
        drift = min(max(minutes_elapsed - 2, 0.0) / 2, 1.0)

        features = {
            "age": float(np.random.normal(40 - 12 * drift, 10)),
            "income": float(np.random.normal(65_000 + 20_000 * drift, 15_000)),
            "credit_score": float(np.random.normal(650, 80)),
        }
        actual = random.choice([0, 0, 0, 1])  # ~25% positive class
        # Model is right ~95% of the time initially, degrading with drift
        p_correct = 0.95 - 0.25 * drift
        prediction = actual if random.random() < p_correct else 1 - actual

        monitor.log_prediction(
            features=features,
            prediction=prediction,
            actual=actual,
            latency_ms=float(np.random.gamma(2, 6)),
        )
        time.sleep(0.5)


threading.Thread(target=simulate_traffic, daemon=True).start()

if __name__ == "__main__":
    print("MLOps Sentinel demo dashboard: http://127.0.0.1:8001")
    print("Simulated predictions are streaming in; drift begins after ~2 minutes.")
    uvicorn.run(app, host="127.0.0.1", port=8001)
