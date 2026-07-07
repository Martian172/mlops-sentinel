"""
Data Drift Detection for MLOps Sentinel.

Implements multiple drift detection algorithms:
- Kolmogorov-Smirnov test (continuous features)
- Chi-Square test (categorical features)
- Population Stability Index (PSI)
- Jensen-Shannon Divergence
- Page-Hinkley Test (concept drift)
- ADWIN (Adaptive Windowing) algorithm (concept drift)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FeatureDriftStats:
    """Drift statistics for a single feature."""

    feature_name: str
    test_type: str  # "ks", "chi2", "psi", "js"
    statistic: float
    p_value: Optional[float]
    drift_score: float  # Normalised [0, 1]
    is_drifted: bool
    baseline_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    production_mean: Optional[float] = None
    production_std: Optional[float] = None
    psi_score: Optional[float] = None
    js_divergence: Optional[float] = None


@dataclass
class DriftReport:
    """Comprehensive drift detection report."""

    model_name: str
    timestamp: datetime
    drift_score: float  # Aggregate score [0, 1]
    is_drifted: bool
    feature_stats: List[FeatureDriftStats] = field(default_factory=list)
    drifted_features: List[str] = field(default_factory=list)
    concept_drift_detected: bool = False
    concept_drift_method: Optional[str] = None
    label_drift_detected: bool = False
    label_drift_score: Optional[float] = None
    n_baseline_samples: int = 0
    n_production_samples: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "drift_score": self.drift_score,
            "is_drifted": self.is_drifted,
            "drifted_features": self.drifted_features,
            "concept_drift_detected": self.concept_drift_detected,
            "concept_drift_method": self.concept_drift_method,
            "label_drift_detected": self.label_drift_detected,
            "label_drift_score": self.label_drift_score,
            "n_baseline_samples": self.n_baseline_samples,
            "n_production_samples": self.n_production_samples,
            "feature_stats": [
                {
                    "feature": s.feature_name,
                    "test_type": s.test_type,
                    "statistic": s.statistic,
                    "p_value": s.p_value,
                    "drift_score": s.drift_score,
                    "is_drifted": s.is_drifted,
                    "baseline_mean": s.baseline_mean,
                    "production_mean": s.production_mean,
                    "psi_score": s.psi_score,
                    "js_divergence": s.js_divergence,
                }
                for s in self.feature_stats
            ],
        }


# ---------------------------------------------------------------------------
# ADWIN algorithm
# ---------------------------------------------------------------------------


class ADWIN:
    """
    Adaptive Windowing (ADWIN) algorithm for concept drift detection.

    Maintains a sliding window and detects when the mean of the data
    inside the window has changed significantly.

    Reference: Bifet & Gavalda, "Learning from Time-Changing Data with
    Adaptive Windowing", SDM 2007.
    """

    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta
        self._window: List[float] = []
        self._total: float = 0.0
        self.drift_detected: bool = False
        self._n_detections: int = 0

    def add_element(self, value: float) -> bool:
        """
        Add a new value to the window and check for drift.

        Returns
        -------
        bool
            ``True`` if drift was detected.
        """
        self._window.append(value)
        self._total += value
        self.drift_detected = False

        if len(self._window) < 30:
            return False

        self.drift_detected = self._detect_change()
        if self.drift_detected:
            self._n_detections += 1
            # Reset window to the second half
            mid = len(self._window) // 2
            self._window = self._window[mid:]
            self._total = sum(self._window)

        return self.drift_detected

    def _detect_change(self) -> bool:
        """Check whether a change point exists in the current window."""
        n = len(self._window)
        total = self._total
        running_sum = 0.0

        for i in range(1, n):
            running_sum += self._window[i - 1]
            n0 = i
            n1 = n - i
            mu0 = running_sum / n0
            mu1 = (total - running_sum) / n1

            # Hoeffding bound
            epsilon_cut = math.sqrt(
                (1.0 / (2.0 * n0) + 1.0 / (2.0 * n1))
                * math.log(4.0 * n / self.delta)
            )

            if abs(mu0 - mu1) >= epsilon_cut:
                return True

        return False

    @property
    def n_detections(self) -> int:
        return self._n_detections


# ---------------------------------------------------------------------------
# Page-Hinkley algorithm
# ---------------------------------------------------------------------------


class PageHinkley:
    """
    Page-Hinkley test for change-point detection in sequential data.

    Detects an upward shift in the mean of a sequence.

    Parameters
    ----------
    min_instances : int
        Minimum observations before testing begins.
    delta : float
        Magnitude of tolerable change (tolerance parameter).
    threshold : float
        Alarm threshold λ.
    alpha : float
        Forgetting factor for cumulative mean.
    """

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
    ) -> None:
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self._reset()

    def _reset(self) -> None:
        self._n = 0
        self._sum = 0.0
        self._mean = 0.0
        self._ph_sum = 0.0
        self._min_ph = float("inf")
        self.drift_detected = False

    def add_element(self, value: float) -> bool:
        """Add a value and return True if drift is detected."""
        self._n += 1
        self._mean = self.alpha * self._mean + (1 - self.alpha) * value
        self._ph_sum += value - self._mean - self.delta
        self._min_ph = min(self._min_ph, self._ph_sum)

        self.drift_detected = False
        if self._n >= self.min_instances:
            ph_statistic = self._ph_sum - self._min_ph
            if ph_statistic > self.threshold:
                self.drift_detected = True
                self._reset()

        return self.drift_detected


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class DriftDetector:
    """
    Multi-method drift detection engine.

    Parameters
    ----------
    baseline_data : np.ndarray
        Reference dataset with shape ``(n_samples, n_features)``.
    ks_threshold : float
        p-value threshold for the KS test.  Below this → drift.
    psi_threshold : float
        PSI threshold.  Above this → drift (0.2 is standard).
    drift_score_threshold : float
        Aggregate drift score above which :attr:`DriftReport.is_drifted`
        is set to ``True``.
    n_bins : int
        Number of bins for PSI computation.
    """

    def __init__(
        self,
        baseline_data: np.ndarray,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.20,
        drift_score_threshold: float = 0.15,
        n_bins: int = 10,
    ) -> None:
        if baseline_data.ndim == 1:
            baseline_data = baseline_data.reshape(-1, 1)
        self.baseline_data = baseline_data.astype(float)
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.drift_score_threshold = drift_score_threshold
        self.n_bins = n_bins

        # Concept drift detectors (one per feature)
        n_features = self.baseline_data.shape[1]
        self._adwin_detectors: List[ADWIN] = [ADWIN() for _ in range(n_features)]
        self._ph_detectors: List[PageHinkley] = [
            PageHinkley() for _ in range(n_features)
        ]

        logger.info(
            "DriftDetector initialised with %d baseline samples, %d features",
            len(baseline_data),
            n_features,
        )

    # ------------------------------------------------------------------
    # Covariate drift (feature distribution shift)
    # ------------------------------------------------------------------

    def detect_covariate_drift(
        self,
        production_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model_name: str = "unknown",
    ) -> DriftReport:
        """
        Detect feature distribution drift between baseline and production.

        Uses KS-test for continuous features and chi-square for binary/
        low-cardinality features.  Also computes PSI and JS divergence.

        Parameters
        ----------
        production_data : np.ndarray
            Production observations with shape ``(n_samples, n_features)``.
        feature_names : list of str, optional
            Column names.
        model_name : str
            Label for the report.

        Returns
        -------
        DriftReport
        """
        if production_data.ndim == 1:
            production_data = production_data.reshape(-1, 1)

        production_data = production_data.astype(float)
        n_features = self.baseline_data.shape[1]
        names = feature_names or [f"feature_{i}" for i in range(n_features)]

        feature_stats: List[FeatureDriftStats] = []

        for i in range(min(n_features, production_data.shape[1])):
            ref = self.baseline_data[:, i]
            prod = production_data[:, i]
            stats_obj = self._analyse_feature(ref, prod, names[i])
            feature_stats.append(stats_obj)

        # Aggregate drift score = mean of per-feature drift scores
        aggregate_score = (
            float(np.mean([s.drift_score for s in feature_stats]))
            if feature_stats
            else 0.0
        )
        drifted_features = [s.feature_name for s in feature_stats if s.is_drifted]

        report = DriftReport(
            model_name=model_name,
            timestamp=datetime.utcnow(),
            drift_score=aggregate_score,
            is_drifted=aggregate_score > self.drift_score_threshold,
            feature_stats=feature_stats,
            drifted_features=drifted_features,
            n_baseline_samples=len(self.baseline_data),
            n_production_samples=len(production_data),
        )

        logger.info(
            "Covariate drift: score=%.4f, drifted=%s, features=%s",
            aggregate_score,
            report.is_drifted,
            drifted_features,
        )
        return report

    def detect_concept_drift(
        self, error_sequence: List[float]
    ) -> Tuple[bool, str]:
        """
        Detect concept drift from a sequence of prediction errors.

        Uses ADWIN and Page-Hinkley on the scalar error stream.

        Parameters
        ----------
        error_sequence : list of float
            Sequence of per-prediction errors (e.g., 0 for correct, 1 for wrong).

        Returns
        -------
        (is_drifted, method)
        """
        adwin = ADWIN()
        ph = PageHinkley()

        adwin_detected = False
        ph_detected = False

        for val in error_sequence:
            if adwin.add_element(float(val)):
                adwin_detected = True
            if ph.add_element(float(val)):
                ph_detected = True

        if adwin_detected:
            return True, "ADWIN"
        if ph_detected:
            return True, "PageHinkley"
        return False, "none"

    def detect_label_drift(
        self,
        baseline_labels: np.ndarray,
        production_labels: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Detect drift in the output label distribution.

        Parameters
        ----------
        baseline_labels : np.ndarray
            Labels from the training / reference period.
        production_labels : np.ndarray
            Labels seen in production.

        Returns
        -------
        (is_drifted, drift_score)
        """
        stats_obj = self._analyse_feature(
            baseline_labels.astype(float),
            production_labels.astype(float),
            "label",
        )
        return stats_obj.is_drifted, stats_obj.drift_score

    # ------------------------------------------------------------------
    # Per-feature analysis helpers
    # ------------------------------------------------------------------

    def _analyse_feature(
        self,
        reference: np.ndarray,
        production: np.ndarray,
        name: str,
    ) -> FeatureDriftStats:
        """Run all tests for one feature and return consolidated stats."""
        # Remove NaNs
        ref = reference[~np.isnan(reference)]
        prod = production[~np.isnan(production)]

        if len(ref) < 5 or len(prod) < 5:
            return FeatureDriftStats(
                feature_name=name,
                test_type="insufficient_data",
                statistic=0.0,
                p_value=None,
                drift_score=0.0,
                is_drifted=False,
            )

        n_unique = len(np.unique(ref))
        is_categorical = n_unique <= 10

        if is_categorical:
            stat, p_val = self._chi_square_test(ref, prod)
            test_type = "chi2"
        else:
            stat, p_val = self._ks_test(ref, prod)
            test_type = "ks"

        psi = self._compute_psi(ref, prod)
        js = self._jensen_shannon_divergence(ref, prod)

        # Drift score = weighted combination
        primary_score = 1.0 - min(p_val or 1.0, 1.0)
        psi_score_norm = min(psi / 0.25, 1.0) if psi is not None else 0.0
        js_score_norm = min(js / 1.0, 1.0) if js is not None else 0.0

        drift_score = 0.4 * primary_score + 0.35 * psi_score_norm + 0.25 * js_score_norm

        is_drifted = (
            (p_val is not None and p_val < self.ks_threshold)
            or (psi is not None and psi > self.psi_threshold)
        )

        return FeatureDriftStats(
            feature_name=name,
            test_type=test_type,
            statistic=float(stat),
            p_value=float(p_val) if p_val is not None else None,
            drift_score=float(np.clip(drift_score, 0.0, 1.0)),
            is_drifted=is_drifted,
            baseline_mean=float(np.mean(ref)),
            baseline_std=float(np.std(ref)),
            production_mean=float(np.mean(prod)),
            production_std=float(np.std(prod)),
            psi_score=float(psi) if psi is not None else None,
            js_divergence=float(js) if js is not None else None,
        )

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    @staticmethod
    def _ks_test(
        reference: np.ndarray, production: np.ndarray
    ) -> Tuple[float, float]:
        """Two-sample KS test."""
        result = stats.ks_2samp(reference, production)
        return float(result.statistic), float(result.pvalue)

    @staticmethod
    def _chi_square_test(
        reference: np.ndarray, production: np.ndarray
    ) -> Tuple[float, Optional[float]]:
        """Chi-square test for categorical distributions."""
        all_categories = np.union1d(np.unique(reference), np.unique(production))
        ref_counts = np.array(
            [np.sum(reference == c) for c in all_categories], dtype=float
        )
        prod_counts = np.array(
            [np.sum(production == c) for c in all_categories], dtype=float
        )

        # Scale prod_counts to match reference total
        if prod_counts.sum() > 0:
            prod_counts = prod_counts * (ref_counts.sum() / prod_counts.sum())

        # Avoid zeros
        ref_counts = np.maximum(ref_counts, 1e-10)
        prod_counts = np.maximum(prod_counts, 1e-10)

        try:
            result = stats.chisquare(f_obs=prod_counts, f_exp=ref_counts)
            return float(result.statistic), float(result.pvalue)
        except Exception:
            return 0.0, 1.0

    def _compute_psi(
        self, reference: np.ndarray, production: np.ndarray
    ) -> Optional[float]:
        """
        Population Stability Index.

        PSI < 0.1 → stable, 0.1–0.2 → slight shift, > 0.2 → major shift.
        """
        try:
            min_val = min(reference.min(), production.min())
            max_val = max(reference.max(), production.max())

            if min_val == max_val:
                return 0.0

            bins = np.linspace(min_val, max_val, self.n_bins + 1)
            ref_hist, _ = np.histogram(reference, bins=bins)
            prod_hist, _ = np.histogram(production, bins=bins)

            ref_freq = ref_hist / (ref_hist.sum() + 1e-10)
            prod_freq = prod_hist / (prod_hist.sum() + 1e-10)

            # Replace zeros
            ref_freq = np.maximum(ref_freq, 1e-10)
            prod_freq = np.maximum(prod_freq, 1e-10)

            psi = float(np.sum((prod_freq - ref_freq) * np.log(prod_freq / ref_freq)))
            return psi
        except Exception:
            return None

    @staticmethod
    def _jensen_shannon_divergence(
        reference: np.ndarray, production: np.ndarray, n_bins: int = 20
    ) -> Optional[float]:
        """Jensen-Shannon divergence (returns value in [0, 1])."""
        try:
            min_val = min(reference.min(), production.min())
            max_val = max(reference.max(), production.max())
            if min_val == max_val:
                return 0.0

            bins = np.linspace(min_val, max_val, n_bins + 1)
            p, _ = np.histogram(reference, bins=bins, density=True)
            q, _ = np.histogram(production, bins=bins, density=True)

            p = np.maximum(p, 1e-10)
            q = np.maximum(q, 1e-10)
            p = p / p.sum()
            q = q / q.sum()

            m = 0.5 * (p + q)
            js = 0.5 * (
                np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))
            )
            return float(np.clip(js, 0.0, 1.0))
        except Exception:
            return None
