"""Scikit-learn integration wrappers for MLOps Sentinel.

Wrap any sklearn-compatible estimator and every ``predict()`` call is
logged to a :class:`~sentinel.core.monitor.ModelMonitor` automatically.
If no baseline is supplied, the training data passed to ``fit()`` becomes
the drift baseline — monitoring with zero extra code.

Example
-------
.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from sentinel.integrations.sklearn import SentinelClassifier

    clf = SentinelClassifier(
        model=RandomForestClassifier(n_estimators=100),
        model_name="churn-predictor",
    )
    clf.fit(X_train, y_train)      # X_train becomes the drift baseline
    clf.predict(X_test)            # each prediction is logged to Sentinel
    report = clf.monitor.get_drift_report()
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from sentinel.core.monitor import ModelMonitor


class _SentinelWrapperBase:
    """Shared plumbing for the classifier / regressor wrappers."""

    _task_type = "classification"

    def __init__(
        self,
        model: Any,
        model_name: str = "sklearn-model",
        baseline_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        monitor: Optional[ModelMonitor] = None,
    ) -> None:
        self.model = model
        self.monitor = monitor or ModelMonitor(
            model_name=model_name,
            baseline_data=(
                np.asarray(baseline_data, dtype=float)
                if baseline_data is not None
                else None
            ),
            feature_names=feature_names,
            task_type=self._task_type,
        )

    def fit(self, X, y, **kwargs):
        """Fit the wrapped model. X becomes the drift baseline if none is set."""
        result = self.model.fit(X, y, **kwargs)
        if self.monitor.drift_detector is None:
            self.monitor.set_baseline(np.asarray(X, dtype=float))
        return result

    def predict(self, X):
        """Predict and log every row to the monitor."""
        predictions = self.model.predict(X)
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        for row, pred in zip(arr, predictions):
            self.monitor.log_prediction(features=row, prediction=self._cast(pred))
        return predictions

    def score(self, X, y):
        return self.model.score(X, y)

    @staticmethod
    def _cast(value: Any) -> Any:
        return value

    def __getattr__(self, name: str):
        # Delegate everything else (classes_, feature_importances_, ...)
        return getattr(self.model, name)


class SentinelClassifier(_SentinelWrapperBase):
    """Drop-in wrapper for sklearn classifiers that auto-logs predictions."""

    _task_type = "classification"

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    @staticmethod
    def _cast(value: Any) -> Any:
        try:
            return int(value)
        except (TypeError, ValueError):
            return value


class SentinelRegressor(_SentinelWrapperBase):
    """Drop-in wrapper for sklearn regressors that auto-logs predictions."""

    _task_type = "regression"

    @staticmethod
    def _cast(value: Any) -> Any:
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
