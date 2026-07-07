"""Tests for sentinel.core.drift."""
import numpy as np
import pytest

from sentinel.core.drift import DriftDetector, DriftReport


@pytest.fixture
def baseline_data():
    np.random.seed(42)
    return np.random.normal(0, 1, size=(500, 1))


@pytest.fixture
def production_no_drift():
    np.random.seed(99)
    return np.random.normal(0, 1, size=(200, 1))


@pytest.fixture
def production_with_drift():
    np.random.seed(7)
    return np.random.normal(2, 1, size=(200, 1))  # Shifted mean


class TestDriftDetector:
    def test_init(self, baseline_data):
        detector = DriftDetector(baseline_data=baseline_data)
        assert detector is not None
        assert detector.baseline_data.shape == (500, 1)

    def test_no_drift(self, baseline_data, production_no_drift):
        detector = DriftDetector(baseline_data=baseline_data)
        report = detector.detect_covariate_drift(
            production_data=production_no_drift,
            feature_names=["test_feature"],
        )
        assert isinstance(report, DriftReport)
        assert report.is_drifted is False

    def test_drift_detected(self, baseline_data, production_with_drift):
        detector = DriftDetector(baseline_data=baseline_data)
        report = detector.detect_covariate_drift(
            production_data=production_with_drift,
            feature_names=["shifted_feature"],
        )
        assert isinstance(report, DriftReport)
        assert report.is_drifted is True
        assert "shifted_feature" in report.drifted_features

    def test_psi_score(self, baseline_data, production_with_drift):
        detector = DriftDetector(baseline_data=baseline_data)
        report = detector.detect_covariate_drift(production_data=production_with_drift)
        psi = report.feature_stats[0].psi_score
        assert isinstance(psi, float)
        assert psi >= 0
