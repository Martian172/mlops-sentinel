"""Tests for sentinel.core.monitor."""
import numpy as np
import pytest

from sentinel.core.drift import DriftReport
from sentinel.core.monitor import ModelMonitor


@pytest.fixture
def baseline():
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, size=(200, 2))


@pytest.fixture
def monitor(baseline):
    return ModelMonitor(
        model_name="test-model",
        task_type="classification",
        baseline_data=baseline,
        feature_names=["age", "income"],
    )


class TestModelMonitor:
    def test_init(self, monitor):
        assert monitor.model_name == "test-model"
        assert monitor.task_type == "classification"

    def test_log_prediction(self, monitor):
        monitor.log_prediction(
            features={"age": 30, "income": 50000},
            prediction=1,
            actual=1,
        )
        assert monitor.get_summary()["total_predictions"] == 1

    def test_log_multiple(self, monitor):
        for i in range(10):
            monitor.log_prediction(
                features={"age": i, "income": i * 1000}, prediction=i % 2, actual=i % 2
            )
        assert monitor.get_summary()["total_predictions"] == 10

    def test_get_performance_report(self, monitor):
        for i in range(20):
            monitor.log_prediction(
                features={"age": i, "income": i * 1000}, prediction=i % 2, actual=i % 2
            )
        report = monitor.get_performance_report()
        assert isinstance(report, dict)
        assert report["accuracy"] == 1.0
        assert report["total_predictions"] == 20

    def test_get_drift_report(self, monitor):
        rng = np.random.default_rng(7)
        for i in range(50):
            monitor.log_prediction(
                features={"age": float(rng.normal(0, 1)), "income": float(rng.normal(0, 1))},
                prediction=1,
            )
        report = monitor.get_drift_report()
        assert isinstance(report, DriftReport)
        assert 0.0 <= report.drift_score <= 1.0

    def test_drift_report_without_data(self, baseline):
        empty_monitor = ModelMonitor(model_name="empty", baseline_data=baseline)
        assert empty_monitor.get_drift_report() is None

    def test_health_score(self, monitor):
        for i in range(20):
            monitor.log_prediction(
                features={"age": i, "income": i * 1000},
                prediction=i % 2,
                actual=i % 2,
                latency_ms=10.0,
            )
        score = monitor.get_health_score()
        assert 0.0 <= score <= 1.0
