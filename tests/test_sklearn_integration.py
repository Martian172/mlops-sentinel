"""Tests for sentinel.integrations.sklearn."""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn")

from sklearn.linear_model import LogisticRegression, LinearRegression  # noqa: E402

from sentinel.integrations.sklearn import SentinelClassifier, SentinelRegressor  # noqa: E402


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(120, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


class TestSentinelClassifier:
    def test_fit_sets_baseline(self, data):
        X, y = data
        clf = SentinelClassifier(model=LogisticRegression(), model_name="clf-test")
        assert clf.monitor.drift_detector is None
        clf.fit(X, y)
        assert clf.monitor.drift_detector is not None

    def test_predict_logs_to_monitor(self, data):
        X, y = data
        clf = SentinelClassifier(model=LogisticRegression(), model_name="clf-test")
        clf.fit(X, y)
        preds = clf.predict(X[:25])
        assert len(preds) == 25
        assert clf.monitor.get_summary()["total_predictions"] == 25

    def test_explicit_baseline(self, data):
        X, y = data
        clf = SentinelClassifier(
            model=LogisticRegression(),
            model_name="clf-test",
            baseline_data=X,
            feature_names=["a", "b", "c"],
        )
        assert clf.monitor.drift_detector is not None
        assert clf.monitor.feature_names == ["a", "b", "c"]

    def test_delegates_to_wrapped_model(self, data):
        X, y = data
        clf = SentinelClassifier(model=LogisticRegression(), model_name="clf-test")
        clf.fit(X, y)
        assert set(clf.classes_) == {0, 1}
        assert clf.predict_proba(X[:5]).shape == (5, 2)


class TestSentinelRegressor:
    def test_predict_logs_to_monitor(self, data):
        X, _ = data
        y = X[:, 0] * 2.0 + 1.0
        reg = SentinelRegressor(model=LinearRegression(), model_name="reg-test")
        reg.fit(X, y)
        preds = reg.predict(X[:10])
        assert len(preds) == 10
        assert reg.monitor.get_summary()["total_predictions"] == 10
        assert reg.monitor.task_type == "regression"
