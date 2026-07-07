"""Tests for sentinel.core.monitor."""
import pytest
from sentinel.core.monitor import ModelMonitor


@pytest.fixture
def monitor():
    return ModelMonitor(model_name="test-model", task="classification")


class TestModelMonitor:
    def test_init(self, monitor):
        assert monitor.model_name == "test-model"
        assert monitor.task == "classification"

    def test_log_prediction(self, monitor):
        monitor.log_prediction(
            features={"age": 30, "income": 50000},
            prediction=1,
            actual=1,
        )
        assert monitor.prediction_count == 1

    def test_log_multiple(self, monitor):
        for i in range(10):
            monitor.log_prediction(features={"x": i}, prediction=i % 2, actual=i % 2)
        assert monitor.prediction_count == 10

    def test_get_performance_report(self, monitor):
        for i in range(20):
            monitor.log_prediction(features={"x": i}, prediction=i % 2, actual=i % 2)
        report = monitor.get_performance_report()
        assert isinstance(report, dict)
        assert "accuracy" in report or "prediction_count" in report

    def test_get_drift_report(self, monitor):
        for i in range(50):
            monitor.log_prediction(features={"age": 30 + i, "income": 50000}, prediction=1)
        report = monitor.get_drift_report()
        assert isinstance(report, dict)

    def test_reset(self, monitor):
        monitor.log_prediction(features={"x": 1}, prediction=0)
        monitor.reset()
        assert monitor.prediction_count == 0
