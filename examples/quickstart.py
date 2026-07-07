"""MLOps Sentinel — Quickstart Example."""
import random

import numpy as np

from sentinel.core.alerts import AlertManager, AlertRule
from sentinel.core.monitor import ModelMonitor

# 1. Configure alerts
alert_mgr = AlertManager()
alert_mgr.add_rule(AlertRule(
    name="low-accuracy",
    metric="accuracy",
    threshold=0.90,
    comparison="lt",
    severity="WARNING",
))

# 2. Create a monitor with baseline (training) data for drift detection
rng = np.random.default_rng(42)
baseline = np.column_stack([
    rng.integers(18, 70, size=500),            # age
    rng.uniform(20_000, 150_000, size=500),    # income
    rng.integers(300, 850, size=500),          # credit_score
])
monitor = ModelMonitor(
    model_name="fraud-detector-v2",
    task_type="classification",
    baseline_data=baseline,
    feature_names=["age", "income", "credit_score"],
    alert_manager=alert_mgr,
)

# 3. Simulate predictions
print("Logging 100 predictions...")
for i in range(100):
    features = {
        "age": random.randint(18, 70),
        "income": random.uniform(20_000, 150_000),
        "credit_score": random.randint(300, 850),
    }
    prediction = random.choice([0, 1])
    actual = random.choice([0, 1])
    monitor.log_prediction(features=features, prediction=prediction, actual=actual)

# 4. Get reports
print("\n=== Performance Report ===")
perf = monitor.get_performance_report()
for k, v in perf.items():
    print(f"  {k}: {v}")

print("\n=== Drift Report ===")
drift = monitor.get_drift_report()
if drift is None:
    print("  Not enough data for drift detection yet.")
else:
    print(f"  Overall drift score: {drift.drift_score:.4f}")
    print(f"  Is drifted: {drift.is_drifted}")
    print(f"  Drifted features: {drift.drifted_features}")

print(f"\nHealth score: {monitor.get_health_score():.2f}")
