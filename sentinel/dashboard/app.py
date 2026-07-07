"""FastAPI dashboard for MLOps Sentinel."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Lazy import to avoid circular deps
app = FastAPI(title="MLOps Sentinel Dashboard", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory monitor reference (set externally or via startup)
_monitor = None
_connected_clients: list[WebSocket] = []

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


class PredictionLog(BaseModel):
    features: dict[str, Any]
    prediction: Any
    actual: Any = None


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard UI."""
    if TEMPLATE_PATH.exists():
        return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>MLOps Sentinel Dashboard</h1><p>Template not found.</p>")


@app.get("/api/metrics")
async def get_metrics():
    """Return current model metrics."""
    if _monitor:
        return _monitor.get_performance_report()
    return {"status": "no_monitor", "message": "No monitor instance configured"}


@app.get("/api/drift")
async def get_drift():
    """Return drift detection report."""
    if _monitor:
        return _monitor.get_drift_report()
    return {"status": "no_monitor"}


@app.get("/api/alerts")
async def get_alerts():
    """Return recent alerts."""
    return {"alerts": [], "total": 0}


@app.get("/api/performance")
async def get_performance():
    """Return performance over time."""
    if _monitor:
        return {"history": _monitor.get_performance_history()}
    return {"history": []}


@app.post("/api/log")
async def log_prediction(payload: PredictionLog):
    """Log a prediction to the monitor."""
    if _monitor:
        _monitor.log_prediction(
            features=payload.features,
            prediction=payload.prediction,
            actual=payload.actual,
        )
        # Broadcast update to connected WebSocket clients
        for client in _connected_clients[:]:
            try:
                await client.send_text(json.dumps({"event": "prediction_logged"}))
            except Exception:
                _connected_clients.remove(client)
        return {"status": "logged"}
    return {"status": "no_monitor"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.2.0", "timestamp": time.time()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time dashboard updates."""
    await websocket.accept()
    _connected_clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(json.dumps({"echo": data}))
    except WebSocketDisconnect:
        _connected_clients.remove(websocket)


def set_monitor(monitor) -> None:
    """Inject a ModelMonitor instance into the dashboard."""
    global _monitor
    _monitor = monitor
