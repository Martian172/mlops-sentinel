"""MLOps Sentinel — Production ML Model Monitoring & Alerting."""

from sentinel.core.monitor import ModelMonitor
from sentinel.core.drift import DriftDetector, DriftReport
from sentinel.core.alerts import AlertManager, Alert
from sentinel.core.metrics import MetricsCollector

__version__ = "0.3.0"
__author__ = "Martian172"

__all__ = [
    "ModelMonitor",
    "DriftDetector",
    "DriftReport",
    "AlertManager",
    "Alert",
    "MetricsCollector",
]
