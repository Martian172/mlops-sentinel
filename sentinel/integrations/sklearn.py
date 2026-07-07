"""Scikit-learn integration wrappers for MLOps Sentinel."""
from __future__ import annotations

from typing import Any
import numpy as np


class SentinelClassifier:
    """Drop-in wrapper for sklearn classifiers that auto-logs predictions."""

    def __init__(self, estimator: Any, monitor: Any = None) -> None:
        self.estimator = estimator
        self.monitor = monitor

    def fit(self, X, y, **kwargs):
        return self.estimator.fit(X, y, **kwargs)

    def predict(self, X):
        predictions = self.estimator.predict(X)
        if self.monitor is not None:
            for i, pred in enumerate(predictions):
                features = {f"feature_{j}": float(X[i, j]) for j in range(X.shape[1])} if hasattr(X, 'shape') else {}
                self.monitor.log_prediction(features=features, prediction=int(pred))
        return predictions

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

    def __getattr__(self, name: str):
        return getattr(self.estimator, name)


class SentinelRegressor:
    """Drop-in wrapper for sklearn regressors that auto-logs predictions."""

    def __init__(self, estimator: Any, monitor: Any = None) -> None:
        self.estimator = estimator
        self.monitor = monitor

    def fit(self, X, y, **kwargs):
        return self.estimator.fit(X, y, **kwargs)

    def predict(self, X):
        predictions = self.estimator.predict(X)
        if self.monitor is not None:
            for i, pred in enumerate(predictions):
                features = {f"feature_{j}": float(X[i, j]) for j in range(X.shape[1])} if hasattr(X, 'shape') else {}
                self.monitor.log_prediction(features=features, prediction=float(pred))
        return predictions

    def score(self, X, y):
        return self.estimator.score(X, y)

    def __getattr__(self, name: str):
        return getattr(self.estimator, name)
