# Sentinel Monitoring API (`/api/v1`)

Sentinel is also a **monitoring-as-a-service API**: point any model, in any
language, at a running Sentinel server and get drift detection, performance
tracking, and alerting over plain HTTP. No need to import Sentinel in your
code — just make HTTP calls.

**Interactive console:** run a server and open **`/docs`** (Swagger UI) — you
can try every endpoint from the browser.

## Authentication

Optional. If the server sets the env var `SENTINEL_API_TOKEN`, every `/api/v1`
request must include the token:

```
Authorization: Bearer <token>        # or:
X-API-Key: <token>
```

If `SENTINEL_API_TOKEN` is unset, the API is open (local dev / public demo).

## The flow

```
1. POST /api/v1/models                            → register, get model_id
2. POST /api/v1/models/{id}/predictions           → stream predictions
3. PATCH /api/v1/models/{id}/predictions/{rid}     → attach ground truth later
4. GET  /api/v1/models/{id}/drift                  → drift report
   GET  /api/v1/models/{id}/performance            → accuracy / latency
   GET  /api/v1/models/{id}/alerts                 → fired alerts
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/models` | Register a model (name, features, task_type, optional baseline & alert rules). Returns `model_id`. |
| `GET` | `/api/v1/models` | List all monitored models + summaries. |
| `GET` | `/api/v1/models/{id}` | One model's live summary. |
| `DELETE` | `/api/v1/models/{id}` | Stop monitoring a model. |
| `POST` | `/api/v1/models/{id}/predictions` | Log one prediction. Returns `record_id`. |
| `POST` | `/api/v1/models/{id}/predictions/batch` | Log many predictions at once. |
| `PATCH` | `/api/v1/models/{id}/predictions/{rid}` | Attach ground truth to a prediction. |
| `GET` | `/api/v1/models/{id}/drift` | Full drift report (per-feature KS/PSI). |
| `GET` | `/api/v1/models/{id}/performance` | Accuracy/F1 (or MAE/RMSE) + latency. |
| `GET` | `/api/v1/models/{id}/alerts` | Recent alerts (`?limit=`, `?severity=`). |
| `GET` | `/api/v1/models/{id}/health` | Single 0–1 health score. |
| `POST` | `/api/v1/models/{id}/alert-rules` | Add an alert rule at runtime. |

## Example — curl

```bash
# 1. Register (baseline = a 2D array of your training rows)
curl -s -X POST http://localhost:8001/api/v1/models \
  -H 'Content-Type: application/json' \
  -d '{"model_name":"fraud-v2","feature_names":["age","income"],
       "baseline_data":[[40,65000],[38,62000],[45,71000]]}'
# → {"model_id":"fraud-v2-1a2b3c4d", ...}

# 2. Log a prediction
curl -s -X POST http://localhost:8001/api/v1/models/fraud-v2-1a2b3c4d/predictions \
  -H 'Content-Type: application/json' \
  -d '{"features":{"age":41,"income":66000},"prediction":0,"latency_ms":12}'
# → {"status":"logged","record_id":"...."}

# 3. Read drift
curl -s http://localhost:8001/api/v1/models/fraud-v2-1a2b3c4d/drift
```

## Example — Python (httpx)

See [`examples/api_client.py`](examples/api_client.py) for a complete,
runnable client that registers a model, streams 120 predictions with rising
drift, and prints the drift/performance/alert reports back.

## Example — JavaScript (fetch)

```js
const BASE = "http://localhost:8001";
const { model_id } = await (await fetch(`${BASE}/api/v1/models`, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ model_name: "recommender", feature_names: ["a", "b"] }),
})).json();

await fetch(`${BASE}/api/v1/models/${model_id}/predictions`, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ features: { a: 1.2, b: 3.4 }, prediction: 1 }),
});

const drift = await (await fetch(`${BASE}/api/v1/models/${model_id}/drift`)).json();
console.log(drift.drift_score, drift.drifted_features);
```

## Notes

- **Baseline is optional at registration** — you can register without one and
  set it later, but drift detection needs a baseline and ≥ 10 predictions.
- **Ground truth can arrive late** — log the prediction now, `PATCH` the actual
  whenever it becomes known. Performance metrics update automatically.
- **One server, many models** — the registry is multi-tenant; every model is
  isolated by its `model_id`.
- **The live dashboard** at `/` shows the demo model; the API models are all
  visible via `GET /api/v1/models`.
