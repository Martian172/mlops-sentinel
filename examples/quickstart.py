"""MLOps Sentinel — Quickstart Example."""
import random
from sentinel.core.monitor import ModelMonitor
from sentinel.core.alerts import AlertManager, AlertRule

# 1. Create a monitor
monitor = ModelMonitor(model_name="fraud-detector-v2", task="classification")

# 2. Optionally configure alerts
alert_mgr = AlertManager()
alert_mgr.add_rule(AlertRule(
    name="low-accuracy",
    metric="accuracy",
    threshold=0.90,
    condition="less_than",
    severity="WARNING",
))
monitor.set_alert_manager(alert_mgr)

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
print(f"  Overall drift score: {drift.get('overall_drift_score', 'N/A')}")
print(f"  Is drifted: {drift.get('is_drifted', False)}")
