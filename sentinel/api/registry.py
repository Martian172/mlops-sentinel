"""Thread-safe registry of monitored models for the multi-tenant API."""
from __future__ import annotations

import re
import threading
import uuid
from typing import Dict, List, Optional

import numpy as np

from sentinel.core.alerts import AlertManager, AlertRule
from sentinel.core.monitor import ModelMonitor


def _slug(name: str) -> str:
    """Turn a human name into a URL-safe id stem."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "model"


class MonitorRegistry:
    """
    Holds many :class:`~sentinel.core.monitor.ModelMonitor` instances keyed
    by a generated ``model_id``, so one Sentinel server can watch any number
    of independent models at once.
    """

    def __init__(self) -> None:
        self._monitors: Dict[str, ModelMonitor] = {}
        self._lock = threading.RLock()

    # -- lifecycle ----------------------------------------------------------

    def register(
        self,
        model_name: str,
        feature_names: Optional[List[str]] = None,
        task_type: str = "classification",
        baseline_data: Optional[list] = None,
        alert_rules: Optional[List[dict]] = None,
        model_id: Optional[str] = None,
    ) -> str:
        """Create and store a new monitor. Returns its ``model_id``."""
        with self._lock:
            mid = model_id or f"{_slug(model_name)}-{uuid.uuid4().hex[:8]}"

            baseline = None
            if baseline_data is not None:
                baseline = np.asarray(baseline_data, dtype=float)
                if baseline.ndim == 1:
                    baseline = baseline.reshape(-1, 1)

            alert_mgr = AlertManager()
            for rule in alert_rules or []:
                alert_mgr.add_rule(AlertRule(**rule))

            monitor = ModelMonitor(
                model_name=model_name,
                baseline_data=baseline,
                feature_names=feature_names,
                task_type=task_type,
                alert_manager=alert_mgr,
            )
            self._monitors[mid] = monitor
            return mid

    def add(self, model_id: str, monitor: ModelMonitor) -> str:
        """Register an already-built monitor under an explicit id."""
        with self._lock:
            self._monitors[model_id] = monitor
            return model_id

    def get(self, model_id: str) -> Optional[ModelMonitor]:
        with self._lock:
            return self._monitors.get(model_id)

    def remove(self, model_id: str) -> bool:
        with self._lock:
            return self._monitors.pop(model_id, None) is not None

    def ids(self) -> List[str]:
        with self._lock:
            return list(self._monitors.keys())

    def list_summaries(self) -> List[dict]:
        with self._lock:
            items = list(self._monitors.items())
        out = []
        for mid, mon in items:
            summary = mon.get_summary()
            summary["model_id"] = mid
            out.append(summary)
        return out


# Process-wide default registry used by the FastAPI app.
registry = MonitorRegistry()
