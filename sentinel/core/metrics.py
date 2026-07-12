"""
Prometheus metrics collection for MLOps Sentinel.

Exposes native Prometheus metrics via the prometheus_client library and
maintains an in-process snapshot for use by the dashboard API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy import — prometheus_client is optional but recommended
try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed. Prometheus metrics will be unavailable."
    )


# ---------------------------------------------------------------------------
# MetricsSnapshot
# ---------------------------------------------------------------------------


@dataclass
class MetricsSnapshot:
    """Point-in-time snapshot of all collected metrics."""

    model_name: str
    timestamp: datetime
    total_predictions: int = 0
    predictions_per_minute: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    accuracy: Optional[float] = None
    drift_score: float = 0.0
    error_rate: float = 0.0
    alert_count: int = 0
    health_score: float = 1.0
    latency_history: List[float] = field(default_factory=list)
    prediction_counts_by_class: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "total_predictions": self.total_predictions,
            "predictions_per_minute": round(self.predictions_per_minute, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "accuracy": round(self.accuracy, 4) if self.accuracy is not None else None,
            "drift_score": round(self.drift_score, 4),
            "error_rate": round(self.error_rate, 4),
            "alert_count": self.alert_count,
            "health_score": round(self.health_score, 4),
            "prediction_counts_by_class": self.prediction_counts_by_class,
            "latency_history": [round(v, 2) for v in self.latency_history],
        }


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """
    Collects and exposes ML model operational metrics.

    Maintains both an in-process snapshot (for the dashboard) and
    Prometheus counters / gauges / histograms (for Prometheus scraping).

    Parameters
    ----------
    model_name : str
        Label attached to all Prometheus metrics.
    registry : CollectorRegistry, optional
        Custom Prometheus registry.  Defaults to a new isolated registry
        so that multiple collectors can coexist in one process.
    """

    # Prometheus latency buckets in seconds
    LATENCY_BUCKETS = (
        0.001, 0.005, 0.01, 0.025, 0.05, 0.075,
        0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0,
    )

    def __init__(
        self,
        model_name: str,
        registry: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self._start_time = datetime.now(timezone.utc)
        self._start_ts = time.monotonic()

        # In-process counters
        self._total_predictions: int = 0
        self._latencies: List[float] = []  # in ms
        self._alert_count: int = 0
        self._current_accuracy: Optional[float] = None
        self._current_drift_score: float = 0.0
        self._current_error_rate: float = 0.0
        self._class_counts: Dict[str, int] = {}

        # Prometheus metrics
        self._prom_registry = registry
        self._prom_enabled = PROMETHEUS_AVAILABLE
        if self._prom_enabled:
            self._init_prometheus(registry)

    # ------------------------------------------------------------------
    # Recording methods (called by ModelMonitor)
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        prediction: Any,
        latency_ms: Optional[float] = None,
    ) -> None:
        """Record a single prediction event."""
        self._total_predictions += 1

        # Track class distribution
        cls_key = str(prediction)
        self._class_counts[cls_key] = self._class_counts.get(cls_key, 0) + 1

        # Latency
        if latency_ms is not None:
            self._latencies.append(latency_ms)
            # Keep a rolling window of 10,000
            if len(self._latencies) > 10_000:
                self._latencies = self._latencies[-10_000:]

        # Update Prometheus
        if self._prom_enabled:
            self._prom_prediction_counter.labels(
                model=self.model_name
            ).inc()
            if latency_ms is not None:
                self._prom_latency_histogram.labels(
                    model=self.model_name
                ).observe(latency_ms / 1000.0)

    def record_error(self) -> None:
        """Record a prediction error."""
        self._current_error_rate = (
            (self._current_error_rate * (self._total_predictions - 1) + 1)
            / max(self._total_predictions, 1)
        )
        if self._prom_enabled:
            self._prom_error_rate_gauge.labels(model=self.model_name).set(
                self._current_error_rate
            )

    def record_alert(self) -> None:
        """Increment the alert counter."""
        self._alert_count += 1
        if self._prom_enabled:
            self._prom_alert_counter.labels(model=self.model_name).inc()

    def update_accuracy(self, accuracy: float) -> None:
        """Update the accuracy gauge."""
        self._current_accuracy = accuracy
        if self._prom_enabled:
            self._prom_accuracy_gauge.labels(model=self.model_name).set(accuracy)

    def update_drift_score(self, drift_score: float) -> None:
        """Update the drift score gauge."""
        self._current_drift_score = drift_score
        if self._prom_enabled:
            self._prom_drift_score_gauge.labels(model=self.model_name).set(
                drift_score
            )

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def get_snapshot(self) -> MetricsSnapshot:
        """Return a point-in-time snapshot of all metrics."""
        import numpy as np

        elapsed_mins = (time.monotonic() - self._start_ts) / 60.0 or 1.0
        ppm = self._total_predictions / elapsed_mins

        if self._latencies:
            arr = np.array(self._latencies)
            avg = float(np.mean(arr))
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
        else:
            avg = p50 = p95 = p99 = 0.0

        # Health = harmonic mean of key indicators
        indicators = []
        if self._current_accuracy is not None:
            indicators.append(self._current_accuracy)
        drift_health = max(0.0, 1.0 - self._current_drift_score)
        indicators.append(drift_health)
        health = float(np.mean(indicators)) if indicators else 1.0

        return MetricsSnapshot(
            model_name=self.model_name,
            timestamp=datetime.now(timezone.utc),
            total_predictions=self._total_predictions,
            predictions_per_minute=round(ppm, 2),
            avg_latency_ms=avg,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            accuracy=self._current_accuracy,
            drift_score=self._current_drift_score,
            error_rate=self._current_error_rate,
            alert_count=self._alert_count,
            health_score=health,
            latency_history=self._latencies[-100:],
            prediction_counts_by_class=dict(self._class_counts),
        )

    # ------------------------------------------------------------------
    # Prometheus text format
    # ------------------------------------------------------------------

    def get_prometheus_metrics(self) -> str:
        """
        Return metrics in Prometheus text exposition format.

        Falls back to a minimal hand-written format if ``prometheus_client``
        is not installed.
        """
        if self._prom_enabled:
            return generate_latest(self._prom_registry).decode("utf-8")

        # Fallback: manual exposition
        snap = self.get_snapshot()
        lines = [
            "# HELP sentinel_predictions_total Total predictions logged",
            "# TYPE sentinel_predictions_total counter",
            f'sentinel_predictions_total{{model="{self.model_name}"}} {snap.total_predictions}',
            "# HELP sentinel_accuracy Current model accuracy",
            "# TYPE sentinel_accuracy gauge",
            f'sentinel_accuracy{{model="{self.model_name}"}} {snap.accuracy or 0}',
            "# HELP sentinel_drift_score Current drift score",
            "# TYPE sentinel_drift_score gauge",
            f'sentinel_drift_score{{model="{self.model_name}"}} {snap.drift_score}',
            "# HELP sentinel_alerts_total Total alerts fired",
            "# TYPE sentinel_alerts_total counter",
            f'sentinel_alerts_total{{model="{self.model_name}"}} {snap.alert_count}',
        ]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _init_prometheus(self, registry: Optional[Any]) -> None:
        """Create Prometheus metric objects."""
        # Use a separate registry per instance to avoid duplicate-metric errors
        if registry is None:
            self._prom_registry = CollectorRegistry()
        else:
            self._prom_registry = registry

        self._prom_prediction_counter = Counter(
            "sentinel_predictions_total",
            "Total number of predictions logged by MLOps Sentinel",
            ["model"],
            registry=self._prom_registry,
        )
        self._prom_latency_histogram = Histogram(
            "sentinel_prediction_latency_seconds",
            "Prediction latency in seconds",
            ["model"],
            buckets=self.LATENCY_BUCKETS,
            registry=self._prom_registry,
        )
        self._prom_accuracy_gauge = Gauge(
            "sentinel_accuracy",
            "Current model accuracy",
            ["model"],
            registry=self._prom_registry,
        )
        self._prom_drift_score_gauge = Gauge(
            "sentinel_drift_score",
            "Current aggregate data drift score",
            ["model"],
            registry=self._prom_registry,
        )
        self._prom_alert_counter = Counter(
            "sentinel_alerts_total",
            "Total number of alerts fired",
            ["model"],
            registry=self._prom_registry,
        )
        self._prom_error_rate_gauge = Gauge(
            "sentinel_error_rate",
            "Current prediction error rate",
            ["model"],
            registry=self._prom_registry,
        )
