"""Versioned multi-tenant REST API (``/api/v1``).

Anyone can:
    1. POST /api/v1/models                      register a model (+ optional baseline)
    2. POST /api/v1/models/{id}/predictions     stream predictions (single or batch)
    3. PATCH /api/v1/models/{id}/predictions/{rid}   attach ground truth later
    4. GET  /api/v1/models/{id}/drift           pull a drift report
    5. GET  /api/v1/models/{id}/performance     pull accuracy / latency
    6. GET  /api/v1/models/{id}/alerts          pull fired alerts

If the ``SENTINEL_API_TOKEN`` env var is set, every /api/v1 call must send
``Authorization: Bearer <token>`` (or ``X-API-Key: <token>``). If it is unset,
the API is open (handy for local use and the public demo).
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from sentinel.api.registry import registry
from sentinel.api.schemas import (
    ActualIn,
    AlertRuleIn,
    ModelCreate,
    ModelCreated,
    PredictionBatchIn,
    PredictionIn,
    RuleAddedOut,
)
from sentinel.core.alerts import AlertRule

router = APIRouter(prefix="/api/v1", tags=["monitoring-api"])


# --------------------------------------------------------------------------
# Auth (optional — enabled only when SENTINEL_API_TOKEN is set)
# --------------------------------------------------------------------------
def require_token(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
) -> None:
    expected = os.environ.get("SENTINEL_API_TOKEN")
    if not expected:
        return  # open mode
    supplied = x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if supplied != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API token.")


def _monitor_or_404(model_id: str):
    monitor = registry.get(model_id)
    if monitor is None:
        raise HTTPException(status_code=404, detail=f"Unknown model_id '{model_id}'.")
    return monitor


# --------------------------------------------------------------------------
# Model lifecycle
# --------------------------------------------------------------------------
@router.post("/models", response_model=ModelCreated, status_code=201,
             dependencies=[Depends(require_token)])
def create_model(body: ModelCreate):
    """Register a model to monitor. Returns a ``model_id`` you use for all
    subsequent calls."""
    model_id = registry.register(
        model_name=body.model_name,
        feature_names=body.feature_names,
        task_type=body.task_type,
        baseline_data=body.baseline_data,
        alert_rules=[r.model_dump() for r in body.alert_rules] if body.alert_rules else None,
    )
    return ModelCreated(
        model_id=model_id,
        model_name=body.model_name,
        task_type=body.task_type,
        predictions_url=f"/api/v1/models/{model_id}/predictions",
    )


@router.get("/models", dependencies=[Depends(require_token)])
def list_models():
    """List every registered model and its live summary."""
    return {"models": registry.list_summaries(), "count": len(registry.ids())}


@router.get("/models/{model_id}", dependencies=[Depends(require_token)])
def get_model(model_id: str):
    monitor = _monitor_or_404(model_id)
    summary = monitor.get_summary()
    summary["model_id"] = model_id
    return summary


@router.delete("/models/{model_id}", dependencies=[Depends(require_token)])
def delete_model(model_id: str):
    if not registry.remove(model_id):
        raise HTTPException(status_code=404, detail=f"Unknown model_id '{model_id}'.")
    return {"status": "deleted", "model_id": model_id}


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------
@router.post("/models/{model_id}/predictions", status_code=201,
             dependencies=[Depends(require_token)])
def log_prediction(model_id: str, body: PredictionIn):
    """Log a single prediction. Returns a ``record_id`` you can use later to
    attach ground truth."""
    monitor = _monitor_or_404(model_id)
    record_id = monitor.log_prediction(
        features=body.features,
        prediction=body.prediction,
        actual=body.actual,
        latency_ms=body.latency_ms,
        metadata=body.metadata,
    )
    return {"status": "logged", "record_id": record_id}


@router.post("/models/{model_id}/predictions/batch", status_code=201,
             dependencies=[Depends(require_token)])
def log_predictions_batch(model_id: str, body: PredictionBatchIn):
    """Log many predictions in one call. Returns all record ids."""
    monitor = _monitor_or_404(model_id)
    ids = [
        monitor.log_prediction(
            features=r.features,
            prediction=r.prediction,
            actual=r.actual,
            latency_ms=r.latency_ms,
            metadata=r.metadata,
        )
        for r in body.records
    ]
    return {"status": "logged", "count": len(ids), "record_ids": ids}


@router.patch("/models/{model_id}/predictions/{record_id}",
              dependencies=[Depends(require_token)])
def update_actual(model_id: str, record_id: str, body: ActualIn):
    """Attach ground truth to a previously logged prediction."""
    monitor = _monitor_or_404(model_id)
    ok = monitor.update_actual(record_id, body.actual)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Unknown record_id '{record_id}'.")
    return {"status": "updated", "record_id": record_id}


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
@router.get("/models/{model_id}/drift", dependencies=[Depends(require_token)])
def get_drift(model_id: str, window: Optional[int] = None):
    monitor = _monitor_or_404(model_id)
    report = monitor.get_drift_report(window=window)
    if report is None:
        return {"status": "insufficient_data",
                "detail": "Need a baseline and >= 10 predictions."}
    return report.to_dict()


@router.get("/models/{model_id}/performance", dependencies=[Depends(require_token)])
def get_performance(model_id: str, window: Optional[int] = None,
                    window_minutes: Optional[int] = None):
    monitor = _monitor_or_404(model_id)
    return monitor.get_performance_report(window=window, window_minutes=window_minutes)


@router.get("/models/{model_id}/alerts", dependencies=[Depends(require_token)])
def get_alerts(model_id: str, limit: int = 50, severity: Optional[str] = None):
    monitor = _monitor_or_404(model_id)
    history = monitor.alert_manager.get_history(limit=limit, severity=severity)
    return {"alerts": history, "count": len(history)}


@router.get("/models/{model_id}/health", dependencies=[Depends(require_token)])
def get_health(model_id: str):
    monitor = _monitor_or_404(model_id)
    return {"model_id": model_id, "health_score": monitor.get_health_score()}


# --------------------------------------------------------------------------
# Alert rules
# --------------------------------------------------------------------------
@router.post("/models/{model_id}/alert-rules", response_model=RuleAddedOut,
             status_code=201, dependencies=[Depends(require_token)])
def add_alert_rule(model_id: str, rule: AlertRuleIn):
    monitor = _monitor_or_404(model_id)
    monitor.alert_manager.add_rule(AlertRule(**rule.model_dump()))
    return RuleAddedOut(status="added", rule=rule.name)
