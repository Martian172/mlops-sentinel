"""Pydantic request/response models for the multi-tenant API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class AlertRuleIn(BaseModel):
    name: str
    metric: str = Field(..., description="e.g. 'accuracy' or 'drift_score'")
    threshold: float
    comparison: str = Field("gt", description="gt | lt | gte | lte | eq")
    severity: str = Field("WARNING", description="INFO | WARNING | CRITICAL")
    cooldown_seconds: int = 300


class ModelCreate(BaseModel):
    model_name: str = Field(..., examples=["fraud-detector-v2"])
    feature_names: Optional[List[str]] = Field(
        None, examples=[["age", "income", "credit_score"]]
    )
    task_type: str = Field("classification", description="classification | regression")
    baseline_data: Optional[List[List[float]]] = Field(
        None, description="2D array (n_samples x n_features) of training data"
    )
    alert_rules: Optional[List[AlertRuleIn]] = None


class ModelCreated(BaseModel):
    model_id: str
    model_name: str
    task_type: str
    message: str = "Model registered. Stream predictions to "
    predictions_url: str


class PredictionIn(BaseModel):
    features: Union[Dict[str, Any], List[float]]
    prediction: Any
    actual: Any = None
    latency_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class PredictionBatchIn(BaseModel):
    records: List[PredictionIn]


class ActualIn(BaseModel):
    actual: Any


class RuleAddedOut(BaseModel):
    status: str
    rule: str
