# 🛡️ MLOps Sentinel — Production ML Model Monitoring & Alerting

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="MIT License"/>
  <img src="https://img.shields.io/badge/version-0.2.0-orange?style=for-the-badge" alt="Version 0.2.0"/>
  <img src="https://img.shields.io/badge/tests-passing-brightgreen?style=for-the-badge&logo=github-actions" alt="Tests Passing"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/prometheus-ready-E6522C?style=for-the-badge&logo=prometheus&logoColor=white" alt="Prometheus Ready"/>
</p>

<p align="center">
  <b>A production-grade toolkit for monitoring ML models in production.</b><br/>
  Detect data drift, track model performance, fire alerts to Slack/Email/Webhooks, and visualize everything in a beautiful real-time dashboard.
</p>

---

## Key Features

| Feature | Description |
|---|---|
| Data Drift Detection | KS-test, PSI, Jensen-Shannon divergence, chi-square for categorical features |
| Concept Drift Detection | Page-Hinkley and ADWIN algorithms for detecting shifts in model behavior |
| Performance Monitoring | Accuracy, Precision, Recall, F1, MAE, RMSE, AUC tracked over time |
| Smart Alerting | Slack webhooks, Email (SMTP), generic webhooks with configurable thresholds |
| Real-time Dashboard | Dark-theme web UI with Chart.js charts, live WebSocket updates |
| REST API | FastAPI backend for programmatic access to all monitoring data |
| Prometheus Metrics | Native /metrics endpoint compatible with Prometheus and Grafana |
| Scikit-learn Integration | Drop-in SentinelClassifier / SentinelRegressor wrappers |
| Flexible Storage | In-memory (zero config) or SQLite persistence |
| CLI | Full-featured command-line interface powered by Click and Rich |

---

## Architecture

```
+---------------------------------------------------------------------+
|                        MLOps Sentinel                               |
|                                                                     |
|  +--------------+    +--------------+    +-----------------------+  |
|  |  Your ML App |---▶|  ModelMonitor|---▶|   Storage Backend     |  |
|  |  (sklearn,   |    |              |    |  +-----------------+  |  |
|  |   pytorch,   |    | log_predict()|    |  |  InMemory/SQLite |  |  |
|  |   xgboost)   |    | get_report() |    |  +-----------------+  |  |
|  +--------------+    +------+-------+    +-----------------------+  |
|                             |                                       |
|              +--------------+--------------+                        |
|              v              v              v                        |
|  +---------------+ +--------------+ +---------------------------+  |
|  | DriftDetector | |MetricsCollect| |     AlertManager          |  |
|  |               | |              | |  +------+ +----------+    |  |
|  | KS-test       | | Prometheus   | |  |Slack | |  Email   |    |  |
|  | PSI           | | Gauges       | |  +------+ +----------+    |  |
|  | ADWIN         | | Histograms   | |  +------------------+     |  |
|  | Page-Hinkley  | | Counters     | |  |  Webhook         |     |  |
|  +---------------+ +------+-------+ |  +------------------+     |  |
|                           |         +---------------------------+   |
|                           v                                         |
|  +------------------------------------------------------------------+|
|  |                   FastAPI Dashboard Server                       ||
|  |  GET /api/metrics  GET /api/drift  GET /api/alerts  WS /ws      ||
|  |  +----------------------------------------------------------+   ||
|  |  |  Dark Web Dashboard  (Chart.js + WebSocket)               |   ||
|  |  +----------------------------------------------------------+   ||
|  |  GET /metrics  <---- Prometheus / Grafana                        ||
|  +------------------------------------------------------------------+|
+---------------------------------------------------------------------+
```

---

## Quick Start

### Installation

```bash
pip install mlops-sentinel
```

Or install from source:

```bash
git clone https://github.com/yourorg/mlops-sentinel.git
cd mlops-sentinel
pip install -e ".[dev]"
```

### 30-Second Example

```python
from sentinel import ModelMonitor
import numpy as np

# Initialize monitor with your baseline (training) data
baseline_features = np.random.randn(1000, 5)
monitor = ModelMonitor(
    model_name="fraud-detector-v2",
    baseline_data=baseline_features,
    feature_names=["amount", "merchant_cat", "hour", "country_code", "velocity"]
)

# Log predictions as they happen in production
monitor.log_prediction(
    features={"amount": 142.5, "merchant_cat": 3, "hour": 14,
               "country_code": 1, "velocity": 0.8},
    prediction=0,
    actual=0,          # optional - when ground truth arrives later
    latency_ms=12.4    # optional
)

# Get reports
drift_report = monitor.get_drift_report()
print(f"Drift detected: {drift_report.is_drifted}")
print(f"Drift score: {drift_report.drift_score:.4f}")

perf_report = monitor.get_performance_report()
print(f"Accuracy: {perf_report['accuracy']:.2%}")
print(f"F1 Score: {perf_report['f1']:.4f}")
```

### Start the Dashboard

**Try it in 10 seconds** — launch the dashboard with a simulated fraud-detection
model streaming live predictions (data drift kicks in after ~2 minutes so you
can watch Sentinel catch it):

```bash
python run_demo.py
```

Then open http://127.0.0.1:8001 in your browser.

To serve the dashboard for your own model, inject your monitor and run the app:

```python
import uvicorn
from sentinel.dashboard.app import app, set_monitor

set_monitor(my_monitor)  # your ModelMonitor instance
uvicorn.run(app, host="127.0.0.1", port=8001)
```

> Note: `sentinel monitor start` serves the dashboard without a monitor
> attached, so it will show no data until your process calls `set_monitor()`.

### CLI Commands

```bash
# Start monitoring server
sentinel monitor start --port 8080 --host 0.0.0.0

# View drift report
sentinel report drift --model fraud-detector-v2

# View performance report
sentinel report performance --model fraud-detector-v2

# List recent alerts
sentinel alerts list --limit 20

# Send a test alert to configured channels
sentinel alerts test --severity WARNING
```

---

## Scikit-learn Integration

```python
from sklearn.ensemble import RandomForestClassifier
from sentinel.integrations.sklearn import SentinelClassifier

clf = RandomForestClassifier(n_estimators=100)
monitored_clf = SentinelClassifier(
    model=clf,
    model_name="churn-predictor",
    baseline_data=X_train
)

monitored_clf.fit(X_train, y_train)

# predict() automatically logs to Sentinel
predictions = monitored_clf.predict(X_test)
probabilities = monitored_clf.predict_proba(X_test)
```

---

## Alerting Setup

```python
from sentinel import ModelMonitor
from sentinel.core.alerts import AlertManager, SlackAlertChannel, EmailAlertChannel, AlertRule

alert_manager = AlertManager()
alert_manager.add_channel(SlackAlertChannel(
    webhook_url="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
))
alert_manager.add_channel(EmailAlertChannel(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    username="alerts@yourcompany.com",
    password="your-app-password",
    recipients=["mlops-team@yourcompany.com"]
))

alert_manager.add_rule(AlertRule(
    name="high_drift",
    metric="drift_score",
    threshold=0.15,
    severity="CRITICAL",
    comparison="gt"
))
alert_manager.add_rule(AlertRule(
    name="accuracy_drop",
    metric="accuracy",
    threshold=0.85,
    severity="WARNING",
    comparison="lt"
))

monitor = ModelMonitor(
    model_name="my-model",
    alert_manager=alert_manager
)
```

---

## Prometheus + Grafana Integration

Add to your prometheus.yml:

```yaml
scrape_configs:
  - job_name: 'mlops-sentinel'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

Available Prometheus metrics:

| Metric | Type | Description |
|---|---|---|
| sentinel_predictions_total | Counter | Total predictions logged |
| sentinel_prediction_latency_seconds | Histogram | Prediction latency distribution |
| sentinel_accuracy | Gauge | Current model accuracy |
| sentinel_drift_score | Gauge | Current drift score per feature |
| sentinel_alerts_total | Counter | Total alerts fired |
| sentinel_error_rate | Gauge | Current error/failure rate |

### Docker Compose (Sentinel + Prometheus + Grafana)

```bash
cd docker
docker-compose up -d
```

Services:
- Sentinel Dashboard: http://localhost:8080
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/sentinel123)

---

## Project Structure

```
mlops-sentinel/
├── sentinel/
│   ├── core/
│   │   ├── monitor.py
│   │   ├── drift.py
│   │   ├── alerts.py
│   │   └── metrics.py
│   ├── dashboard/
│   │   ├── app.py
│   │   └── templates/index.html
│   ├── cli/
│   │   └── commands.py
│   ├── integrations/
│   │   └── sklearn.py
│   └── storage/
│       └── backend.py
├── examples/
├── tests/
├── docker/
└── .github/workflows/
```

---

## License

MIT License - see LICENSE for details.

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md first.

---

Made with love for the MLOps community
