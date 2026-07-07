"""Tests for sentinel.core.drift."""
import pytest
import numpy as np
from sentinel.core.drift import DriftDetector


@pytest.fixture
def reference_data():
    np.random.seed(42)
    return np.random.normal(0, 1, 500)


@pytest.fixture
def current_data_no_drift(reference_data):
    np.random.seed(99)
    return np.random.normal(0, 1, 200)


@pytest.fixture
def current_data_with_drift():
    np.random.seed(7)
    return np.random.normal(2, 1, 200)  # Shifted mean


class TestDriftDetector:
    def test_init(self):
        detector = DriftDetector()
        assert detector is not None

    def test_no_drift(self, reference_data, current_data_no_drift):
        detector = DriftDetector()
        result = detector.detect_covariate_drift(
            reference=reference_data,
            current=current_data_no_drift,
            feature_name="test_feature",
        )
        assert "is_drifted" in result
        assert result["is_drifted"] is False

    def test_drift_detected(self, reference_data, current_data_with_drift):
        detector = DriftDetector()
        result = detector.detect_covariate_drift(
            reference=reference_data,
            current=current_data_with_drift,
            feature_name="shifted_feature",
        )
        assert "is_drifted" in result
        assert result["is_drifted"] is True

    def test_psi_score(self, reference_data, current_data_with_drift):
        detector = DriftDetector()
        psi = detector.calculate_psi(reference_data, current_data_with_drift)
        assert isinstance(psi, float)
        assert psi >= 0
