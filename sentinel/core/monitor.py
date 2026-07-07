"""
ModelMonitor — the central orchestrator for MLOps Sentinel.

Provides a unified interface for logging predictions, computing performance
metrics, running drift detection, and triggering alerts.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np

from sentinel.core.alerts import AlertManager, AlertRule
from sentinel.core.drift import DriftDetector, DriftReport
from sentinel.core.metrics import MetricsCollector
from sentinel.storage.backend import InMemoryStorage, PredictionRecord, StorageBackend

logger = logging.getLogger(__name__)


@dataclass
class PerformanceReport:
    """Snapshot of model performance metrics."""

    model_name: str
    timestamp: datetime
    total_predictions: int
    labeled_predictions: int
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    # Regression metrics
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    # Throughput
    avg_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    predictions_per_minute: Optional[float] = None
    error_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "total_predictions": self.total_predictions,
            "labeled_predictions": self.labeled_predictions,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape,
            "avg_latency_ms": self.avg_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "predictions_per_minute": self.predictions_per_minute,
            "error_rate": self.error_rate,
        }


class ModelMonitor:
    """
    Central monitor for a deployed ML model.

    Parameters
    ----------
    model_name : str
        Human-readable identifier for the model being monitored.
    baseline_data : np.ndarray, optional
        Reference dataset (e.g., training data) used for drift detection.
        Shape: (n_samples, n_features).
    feature_names : list of str, optional
        Names for each feature column.
    task_type : str
        Either ``"classification"`` or ``"regression"``.
    alert_manager : AlertManager, optional
        Pre-configured alert manager.  If ``None``, a silent one is used.
    storage : StorageBackend, optional
        Storage backend.  Defaults to :class:`~sentinel.storage.backend.InMemoryStorage`.
    drift_check_interval : int
        Number of predictions between automatic drift checks.
    alert_check_interval : int
        Number of predictions between automatic alert evaluations.
    """

    def __init__(
        self,
        model_name: str,
        baseline_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        task_type: str = "classification",
        alert_manager: Optional[AlertManager] = None,
        storage: Optional[StorageBackend] = None,
        drift_check_interval: int = 100,
        alert_check_interval: int = 50,
    ) -> None:
        self.model_name = model_name
        self.task_type = task_type
        self.feature_names = feature_names or []
        self.drift_check_interval = drift_check_interval
        self.alert_check_interval = alert_check_interval

        self.storage: StorageBackend = storage or InMemoryStorage()
        self.alert_manager: AlertManager = alert_manager or AlertManager()
        self.metrics_collector = MetricsCollector(model_name=model_name)

        # Drift detector — initialised once we have baseline data
        self.drift_detector: Optional[DriftDetector] = None
        if baseline_data is not None:
            self._init_drift_detector(baseline_data)

        # Internal counters
        self._prediction_count: int = 0
        self._error_count: int = 0
        self._start_time: datetime = datetime.utcnow()

        # Cache the last computed reports
        self._last_drift_report: Optional[DriftReport] = None
        self._last_perf_report: Optional[PerformanceReport] = None

        logger.info(
            "ModelMonitor initialised for '%s' (%s task)", model_name, task_type
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_baseline(self, data: np.ndarray) -> None:
        """Set or update the baseline data used for drift comparison."""
        self._init_drift_detector(data)
        logger.info("Baseline data updated (%d samples)", len(data))

    def log_prediction(
        self,
        features: Union[Dict[str, Any], np.ndarray, List[Any]],
        prediction: Any,
        actual: Optional[Any] = None,
        latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a single prediction event.

        Parameters
        ----------
        features : dict | ndarray | list
            Input features for this prediction.
        prediction : any
            The model's output value.
        actual : any, optional
            Ground-truth label (may be supplied later).
        latency_ms : float, optional
            End-to-end prediction latency in milliseconds.
        metadata : dict, optional
            Any additional key/value pairs to store.

        Returns
        -------
        str
            Unique record ID for this prediction.
        """
        record_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # Normalise features to a flat dict
        feature_dict = self._normalise_features(features)

        record = PredictionRecord(
            id=record_id,
            model_name=self.model_name,
            timestamp=timestamp,
            features=feature_dict,
            prediction=prediction,
            actual=actual,
            latency_ms=latency_ms,
            metadata=metadata or {},
        )

        try:
            self.storage.save(record)
        except Exception as exc:  # pragma: no cover
            logger.error("Storage error: %s", exc)
            self._error_count += 1

        self._prediction_count += 1

        # Update Prometheus metrics
        self.metrics_collector.record_prediction(
            prediction=prediction,
            latency_ms=latency_ms,
        )

        # Periodic drift + alert checks
        if (
            self._prediction_count % self.drift_check_interval == 0
            and self.drift_detector is not None
        ):
            self._run_drift_check()

        if self._prediction_count % self.alert_check_interval == 0:
            self._run_alert_check()

        return record_id

    def update_actual(self, record_id: str, actual: Any) -> bool:
        """
        Supply ground truth for a previously logged prediction.

        Parameters
        ----------
        record_id : str
            The ID returned by :meth:`log_prediction`.
        actual : any
            Ground-truth label.

        Returns
        -------
        bool
            ``True`` if the record was found and updated.
        """
        return self.storage.update_actual(record_id, actual)

    def get_drift_report(
        self, window: Optional[int] = None
    ) -> Optional[DriftReport]:
        """
        Run a drift detection pass and return the report.

        Parameters
        ----------
        window : int, optional
            Use only the most-recent ``window`` production records.
            If ``None``, all stored records are used.

        Returns
        -------
        DriftReport or None
            ``None`` if there is not enough data or no baseline was set.
        """
        if self.drift_detector is None:
            logger.warning("No baseline set — drift detection unavailable.")
            return None

        records = self.storage.query(model_name=self.model_name, limit=window)
        if not records:
            logger.warning("No production records yet for drift detection.")
            return None

        production_array = self._records_to_array(records)
        if production_array is None or len(production_array) < 10:
            logger.warning("Insufficient data for drift detection (need ≥ 10 samples).")
            return None

        report = self.drift_detector.detect_covariate_drift(
            production_data=production_array,
            feature_names=self.feature_names,
        )
        self._last_drift_report = report

        # Update Prometheus gauge
        self.metrics_collector.update_drift_score(report.drift_score)

        logger.info(
            "Drift report computed — score=%.4f, drifted=%s",
            report.drift_score,
            report.is_drifted,
        )
        return report

    def get_performance_report(
        self,
        window: Optional[int] = None,
        window_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculate model performance over labeled predictions.

        Parameters
        ----------
        window : int, optional
            Use only the most-recent ``window`` records.
        window_minutes : int, optional
            Use only records from the last ``window_minutes`` minutes.

        Returns
        -------
        dict
            A serialisable dict with performance metrics.
        """
        since: Optional[datetime] = None
        if window_minutes is not None:
            since = datetime.utcnow() - timedelta(minutes=window_minutes)

        records = self.storage.query(
            model_name=self.model_name,
            limit=window,
            since=since,
        )

        labeled = [r for r in records if r.actual is not None]
        report = self._compute_metrics(records=records, labeled=labeled)
        self._last_perf_report = report

        # Update Prometheus accuracy gauge
        if report.accuracy is not None:
            self.metrics_collector.update_accuracy(report.accuracy)

        logger.info(
            "Performance report — total=%d, labeled=%d, accuracy=%s",
            report.total_predictions,
            report.labeled_predictions,
            f"{report.accuracy:.4f}" if report.accuracy is not None else "N/A",
        )
        return report.to_dict()

    def get_health_score(self) -> float:
        """
        Return an overall health score in [0, 1].

        Combines accuracy (if available) and inverse drift score.
        """
        scores: List[float] = []

        perf = self.get_performance_report()
        if perf.get("accuracy") is not None:
            scores.append(perf["accuracy"])

        drift = self.get_drift_report()
        if drift is not None:
            drift_health = max(0.0, 1.0 - drift.drift_score)
            scores.append(drift_health)

        if not scores:
            return 1.0  # No data yet — optimistically healthy

        return float(np.mean(scores))

    def get_summary(self) -> Dict[str, Any]:
        """Return a high-level JSON-serialisable summary."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        return {
            "model_name": self.model_name,
            "task_type": self.task_type,
            "uptime_seconds": uptime,
            "total_predictions": self._prediction_count,
            "error_count": self._error_count,
            "health_score": self.get_health_score(),
            "started_at": self._start_time.isoformat(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_drift_detector(self, baseline_data: np.ndarray) -> None:
        """Initialise the drift detector with reference data."""
        n_features = baseline_data.shape[1] if baseline_data.ndim == 2 else 1
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

        self.drift_detector = DriftDetector(baseline_data=baseline_data)

    def _normalise_features(
        self, features: Union[Dict[str, Any], np.ndarray, List[Any]]
    ) -> Dict[str, Any]:
        """Convert various feature formats to a flat dict."""
        if isinstance(features, dict):
            return features
        arr = np.asarray(features).flatten()
        if self.feature_names and len(self.feature_names) == len(arr):
            return dict(zip(self.feature_names, arr.tolist()))
        return {f"feature_{i}": float(v) for i, v in enumerate(arr)}

    def _records_to_array(
        self, records: List[PredictionRecord]
    ) -> Optional[np.ndarray]:
        """Convert a list of prediction records to a numpy feature matrix."""
        if not records:
            return None
        rows = []
        for r in records:
            if r.features:
                rows.append(list(r.features.values()))
        if not rows:
            return None
        return np.array(rows, dtype=float)

    def _compute_metrics(
        self,
        records: List[PredictionRecord],
        labeled: List[PredictionRecord],
    ) -> PerformanceReport:
        """Compute all performance metrics from stored records."""
        total = len(records)
        n_labeled = len(labeled)

        latencies = [r.latency_ms for r in records if r.latency_ms is not None]

        avg_lat = float(np.mean(latencies)) if latencies else None
        p95_lat = float(np.percentile(latencies, 95)) if latencies else None
        p99_lat = float(np.percentile(latencies, 99)) if latencies else None

        uptime_mins = (
            (datetime.utcnow() - self._start_time).total_seconds() / 60.0
        ) or 1.0
        ppm = total / uptime_mins if total > 0 else 0.0

        error_rate = self._error_count / max(self._prediction_count, 1)

        accuracy = precision = recall = f1 = None
        mae = rmse = mape = None

        if n_labeled > 0:
            y_true = np.array([r.actual for r in labeled])
            y_pred = np.array([r.prediction for r in labeled])

            if self.task_type == "classification":
                accuracy, precision, recall, f1 = self._classification_metrics(
                    y_true, y_pred
                )
            else:
                mae, rmse, mape = self._regression_metrics(y_true, y_pred)

        return PerformanceReport(
            model_name=self.model_name,
            timestamp=datetime.utcnow(),
            total_predictions=total,
            labeled_predictions=n_labeled,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            mae=mae,
            rmse=rmse,
            mape=mape,
            avg_latency_ms=avg_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            predictions_per_minute=ppm,
            error_rate=error_rate,
        )

    @staticmethod
    def _classification_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Compute accuracy, precision, recall, F1 without sklearn dependency."""
        try:
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

            average = "binary" if len(np.unique(y_true)) == 2 else "weighted"
            acc = float(accuracy_score(y_true, y_pred))
            prec = float(
                precision_score(y_true, y_pred, average=average, zero_division=0)
            )
            rec = float(recall_score(y_true, y_pred, average=average, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
            return acc, prec, rec, f1
        except ImportError:
            # Manual fallback
            correct = np.sum(y_true == y_pred)
            acc = float(correct / len(y_true))
            return acc, None, None, None

    @staticmethod
    def _regression_metrics(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Compute MAE, RMSE, and MAPE."""
        errors = y_true - y_pred
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        # MAPE — avoid division by zero
        nonzero = y_true != 0
        if nonzero.sum() > 0:
            mape = float(
                np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100
            )
        else:
            mape = None
        return mae, rmse, mape

    def _run_drift_check(self) -> None:
        """Periodic internal drift check."""
        try:
            report = self.get_drift_report()
            if report and report.is_drifted:
                self.alert_manager.alert_on_drift(report)
        except Exception as exc:  # pragma: no cover
            logger.error("Drift check failed: %s", exc)

    def _run_alert_check(self) -> None:
        """Periodic internal alert rule evaluation."""
        try:
            if self._last_perf_report is not None:
                perf_dict = self._last_perf_report.to_dict()
                self.alert_manager.evaluate_rules(perf_dict)
        except Exception as exc:  # pragma: no cover
            logger.error("Alert check failed: %s", exc)
