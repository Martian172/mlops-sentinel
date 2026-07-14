# Changelog

## [0.4.0] - 2026-07-13

### Added
- **Multi-tenant monitoring API (`/api/v1`)** — Sentinel is now
  monitoring-as-a-service. Register a model, stream predictions over HTTP
  (single or batch), attach ground truth later, and pull drift / performance /
  alert reports back — from any language, no Python import required. One
  server watches any number of isolated models via a thread-safe registry.
  - `POST /api/v1/models`, `GET/DELETE /api/v1/models/{id}`,
    `POST /api/v1/models/{id}/predictions[/batch]`,
    `PATCH …/predictions/{rid}`, `GET …/drift|performance|alerts|health`,
    `POST …/alert-rules`.
  - Optional bearer-token / `X-API-Key` auth via `SENTINEL_API_TOKEN`.
  - Interactive Swagger console at `/docs`; full reference in `API.md`;
    runnable client in `examples/api_client.py`.
- 8 new API tests (31 total), exercised end-to-end with FastAPI TestClient.
- **Self-documenting dashboard** — every metric, chart, and panel now has an
  ⓘ tooltip that explains, in plain English, what it means and why it matters
  (drift score, trip line, p95 latency, PSI, p-value, pie/line charts, …),
  plus a status-color legend. Hover, tap, or keyboard-focus to reveal.
- Deployed live on Render: https://mlops-sentinel.onrender.com

### Changed
- Dashboard app title/description reflect the API; `set_monitor()` now also
  registers the monitor into the multi-tenant registry so the demo model is
  reachable at `/api/v1` too.

## [0.3.0] - 2026-07-13

### Added
- **Control-room dashboard, fully live** — complete UI redesign ("precision
  instrument panel": paper/ink/international-orange, LED status lamps, SVG
  health gauge, hazard-striped status banner). Every number now comes from
  the REST API: instruments, accuracy timeline, per-feature PSI bars with the
  0.20 trip line, prediction mix, latency sparkline, feature inspection table
  (baseline μ → production μ), and a live alert log.
- **Streaming concept-drift detection** — `ModelMonitor` now feeds every
  labeled prediction into persistent ADWIN and Page-Hinkley detectors and
  fires a CRITICAL alert the moment the error rate shifts
  (`AlertManager.alert_on_concept_drift`).
- `GET /api/snapshot` — cheap point-in-time metrics snapshot for live UIs
  (no drift recompute), including latency history and class distribution.
- `sentinel alerts test` now fires a real alert through the pipeline, with
  `--slack-webhook` / `--webhook` delivery and a local rendering fallback.
- Chart.js is vendored (`sentinel/dashboard/static/`) — the dashboard works
  offline, no CDN required.
- sklearn integration tests; `SENTINEL_DRIFT_MIN` to control demo drift
  timing; `PORT` env support for cloud platforms (Render, Railway, Spaces).

### Changed
- `SentinelClassifier` / `SentinelRegressor` rewritten to match the
  documented API: `SentinelClassifier(model=..., model_name=...,
  baseline_data=...)`, with automatic baseline capture from `fit(X, y)` and
  an internally created monitor exposed as `.monitor`.
- Core dependencies slimmed (removed pandas, matplotlib, seaborn, aiohttp,
  jinja2, python-dotenv from installs); sklearn/SQLAlchemy became extras
  (`pip install mlops-sentinel[sklearn,sqlite]`); test tooling moved to
  `requirements-dev.txt`.
- All timestamps are timezone-aware UTC (`datetime.now(timezone.utc)`);
  SQLite storage restores tz-awareness on read. Zero deprecation warnings.

### Removed
- Dead per-feature ADWIN/Page-Hinkley lists in `DriftDetector` (superseded
  by the streaming detectors in `ModelMonitor`).

## [0.2.0] - 2024-01-01

### Added
- `ModelMonitor` — unified monitoring class for classification & regression
- `DriftDetector` — KS test, PSI, Jensen-Shannon divergence drift detection
- `AlertManager` — Slack, Email, Webhook alert channels with rule engine
- `MetricsCollector` — Prometheus metrics integration
- Real-time dashboard (FastAPI + Chart.js)
- Scikit-learn drop-in wrappers: `SentinelClassifier`, `SentinelRegressor`
- SQLite and in-memory storage backends
- Docker Compose stack with Prometheus + Grafana
- Rich CLI with `sentinel monitor`, `sentinel report`, `sentinel alerts`
- GitHub Actions CI for Python 3.10 and 3.11
