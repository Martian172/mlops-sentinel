# Changelog

## [0.2.0] - 2024-01-01

### Added
- `ModelMonitor` — unified monitoring class for classification & regression
- `DriftDetector` — KS test, PSI, Jensen-Shannon divergence drift detection
- `AlertManager` — Slack, Email, Webhook alert channels with rule engine
- `MetricsCollector` — Prometheus metrics integration
- Beautiful dark-mode real-time dashboard (FastAPI + Chart.js)
- Scikit-learn drop-in wrappers: `SentinelClassifier`, `SentinelRegressor`
- SQLite and in-memory storage backends
- Docker Compose stack with Prometheus + Grafana
- Rich CLI with `sentinel monitor`, `sentinel report`, `sentinel alerts`
- GitHub Actions CI for Python 3.10 and 3.11
